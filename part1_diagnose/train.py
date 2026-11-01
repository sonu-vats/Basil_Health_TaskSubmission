"""Train and evaluate a hands-vs-feet decoder on unseen subjects.

Usage:
  python train.py --channels all --seed 0
"""
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import GroupShuffleSplit

from data import build_dataset, SUBJECTS
from model import EEGNet, count_params

# Electrode groups for the channel-subset run (Task 3).
#  frontal : the prefrontal / anterior-frontal rows (Fp, AF), i.e. the sensors
#            closest to the eyes and farthest from the sensorimotor strip.
#            If these decode "fists (target at TOP) vs feet (target at BOTTOM)",
#            the prime suspect is vertical eye movement (EOG), not motor cortex.
#  central : control group of similar size over the sensorimotor strip
#            (hand areas ~C3/C4, foot area ~Cz) where genuine MI-ERD lives.
CHANNEL_GROUPS = {
    "all": None,
    "frontal": ["Fp1", "Fpz", "Fp2", "AF7", "AF3", "AFz", "AF4", "AF8"],
    "central": ["FC3", "FCz", "FC4", "C3", "Cz", "C4", "CP3", "CPz", "CP4"],
}


def _git_commit():
    """Short hash of the code that produced a result (bookkeeping only)."""
    import subprocess
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "unknown"


def set_seed(seed):
    np.random.seed(seed); torch.manual_seed(seed)


def evaluate(model, X, y, bs=256, return_loss=False):
    model.eval(); correct = 0; loss = 0.0
    with torch.no_grad():
        for i in range(0, len(y), bs):
            xb = torch.from_numpy(X[i:i + bs]); yb = torch.from_numpy(y[i:i + bs])
            logits = model(xb)
            correct += (logits.argmax(1) == yb).sum().item()
            loss += nn.functional.cross_entropy(logits, yb, reduction="sum").item()
    acc = correct / len(y)
    return (acc, loss / len(y)) if return_loss else acc


# Fixed leave-2-subjects-out folds for --cv: every subject is tested exactly once
# per seed; the next pair is the validation set, the other 6 subjects train.
CV_PAIRS = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]


def random_subject_split(groups, seed):
    """20% of subjects for test, 25% of the rest for validation (10 -> 6/2/2)."""
    idx = np.arange(len(groups))
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    tr, te = next(gss.split(idx, groups=groups))
    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
    tr_i, va_i = next(gss.split(tr, groups=groups[tr]))
    return tr[tr_i], tr[va_i], te


def fit_and_eval(X, y, groups, tr, va, te, args, verbose=True):
    """Train EEGNet on `tr`, pick the checkpoint on `va`, score `te` once."""
    assert not (set(groups[tr]) & set(groups[te]) or set(groups[tr]) & set(groups[va])
                or set(groups[va]) & set(groups[te])), "subject overlap between splits"
    print(f"subjects  train {sorted(set(groups[tr].tolist()))}  "
          f"val {sorted(set(groups[va].tolist()))}  test {sorted(set(groups[te].tolist()))}")
    set_seed(args.seed)

    # Per-channel z-scoring with statistics from the TRAINING subjects only;
    # val/test subjects are transformed with those statistics but never shape them.
    mean = X[tr].mean(axis=(0, 2), keepdims=True)
    std = X[tr].std(axis=(0, 2), keepdims=True) + 1e-6
    X = ((X - mean) / std).astype(np.float32)

    n_ch, n_t = X.shape[1], X.shape[2]
    model = EEGNet(n_ch, n_t)
    print(f"EEGNet: {count_params(model)} params")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    rng = np.random.default_rng(args.seed)

    # Checkpoint selection uses the VALIDATION subjects only (best val accuracy,
    # ties broken by lower val loss). The test subjects are scored exactly once,
    # after training, so the reported number is not the max of many test looks.
    best_val, best_state, best_ep = (-1.0, np.inf), None, -1
    for ep in range(args.epochs):
        model.train()
        order = rng.permutation(tr)
        for i in range(0, len(order), args.bs):
            b = order[i:i + args.bs]
            xb, yb = torch.from_numpy(X[b]), torch.from_numpy(y[b])
            opt.zero_grad(); loss = loss_fn(model(xb), yb); loss.backward(); opt.step()
        va_acc, va_loss = evaluate(model, X[va], y[va], return_loss=True)
        if va_acc > best_val[0] or (va_acc == best_val[0] and va_loss < best_val[1]):
            best_val, best_ep = (va_acc, va_loss), ep
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if verbose:
            print(f"epoch {ep:3d}  loss {loss.item():.3f}  val {va_acc:.3f}  val_loss {va_loss:.3f}")

    final_test = evaluate(model, X[te], y[te])        # last-epoch model, for reference
    model.load_state_dict(best_state)
    # Per-subject test accuracy (cross-subject variance is the main uncertainty).
    per_subject = {int(s): evaluate(model, X[te][groups[te] == s], y[te][groups[te] == s])
                   for s in sorted(set(groups[te].tolist()))}
    return dict(channels=args.channels, seed=args.seed, best_epoch=best_ep,
                test_acc=evaluate(model, X[te], y[te]), n_test=len(te),
                val_acc=best_val[0], last_epoch_test_acc=final_test,
                test_subjects=sorted(set(groups[te].tolist())),
                per_subject_test_acc=per_subject,
                band=list(args.band) if args.band else None,
                tag=args.tag, commit=_git_commit())


def log(result):
    print(json.dumps(result))
    with open("results.jsonl", "a") as f:
        f.write(json.dumps(result) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--channels", choices=list(CHANNEL_GROUPS), default="all")
    ap.add_argument("--subjects", type=int, nargs="+", default=SUBJECTS)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="", help="free-text label stored in results.jsonl")
    ap.add_argument("--cv", action="store_true",
                    help="5-fold leave-2-subjects-out CV (every subject tested once)")
    ap.add_argument("--band", type=float, nargs=2, default=None, metavar=("L", "H"),
                    help="override the 1-40 Hz band-pass (confound control)")
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    set_seed(args.seed)

    X, y, groups, ch_names = build_dataset(args.subjects, band=args.band)
    picks = CHANNEL_GROUPS[args.channels]
    if picks is not None:
        if not picks:
            raise ValueError(f"Fill in CHANNEL_GROUPS['{args.channels}'] first.")
        X = X[:, [ch_names.index(c) for c in picks]]
    print(f"data: {X.shape}, subjects={len(np.unique(groups))}, "
          f"class balance={np.bincount(y)}")

    if not args.cv:
        # Subject-independent evaluation: hold out 20% of SUBJECTS for test,
        # then 25% of the remaining subjects for validation (10 -> 6/2/2).
        # Splitting trials instead would put the same person (even the same
        # run) in train and test, letting the model memorise subject signatures.
        tr, va, te = random_subject_split(groups, args.seed)
        log(fit_and_eval(X, y, groups, tr, va, te, args))
        return

    accs = []
    for k, pair in enumerate(CV_PAIRS):
        val_pair = CV_PAIRS[(k + 1) % len(CV_PAIRS)]
        te = np.where(np.isin(groups, pair))[0]
        va = np.where(np.isin(groups, val_pair))[0]
        tr = np.where(~np.isin(groups, pair + val_pair))[0]
        r = fit_and_eval(X, y, groups, tr, va, te, args, verbose=False)
        r.update(cv_fold=k)
        log(r); accs.append(r["test_acc"])
    print(f"CV ({args.channels}, seed {args.seed}): mean {np.mean(accs):.3f}  "
          f"folds {[round(a, 3) for a in accs]}")


if __name__ == "__main__":
    main()
