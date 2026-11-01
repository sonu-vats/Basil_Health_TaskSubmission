"""
EEG Transformer Sprint - entry point.
Run:  python main.py                       (full comparison, seeds 0-4, fixed budget)
      python main.py --protocols fixed val (also the validation-selected protocol)
      python main.py --quick               (1 seed, smoke test)

This script:
  1. loads the cross-subject split (train subjects 1-8, test 9-10),
  2. trains every model with the SAME data, budget, optimiser and seeds,
  3. reports, per model, mean / std / best / worst over seeds for
       - final_test         : fixed 40-epoch budget on S1-8, scored once on S9-10  (headline)
       - val_selected_test  : train S1-6, pick epoch on S7-8, score once on S9-10 (optional)
       - peak_test          : the starter's max-over-epochs TEST number (diagnostic only)
  4. appends one JSON line per (model, seed, protocol) to results.jsonl,
  5. saves figure.png (t-SNE of YourModel CLS embeddings on test subjects)
     and attention_topomap.png (where YourModel's CLS token attends on the scalp).

Do not quietly edit the vanilla transformer in models.py - the contrast is the point.
"""
import argparse
import json
import os
import time
import numpy as np
import torch
import torch.nn as nn

from data import (load_data, get_channel_positions, get_channel_names,
                  get_trial_subjects, TEST_SUBJECTS)
from models import EEGNet, VanillaEEGTransformer, count_params
from train import train_model, set_seed

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results.jsonl")
VAL_SUBJECTS = [7, 8]   # carved out of the training subjects for model selection


# =========================================================================== #
#  Reference: the starter's example bias (temporal patches). Kept unchanged,
#  under its own name, so the hypothesis "patching alone helps" is tested too.
# =========================================================================== #
class TemporalPatchTransformer(nn.Module):
    def __init__(self, n_chan, n_time, n_classes=2,
                 patch=32, d_model=64, n_heads=4, depth=3, p=0.3):
        super().__init__()
        self.tokenizer = nn.Conv1d(n_chan, d_model, kernel_size=patch, stride=patch)
        n_tokens = n_time // patch
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos = nn.Parameter(torch.zeros(1, n_tokens + 1, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.cls, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=d_model * 2,
            dropout=p, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, depth)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x, return_embed=False):
        tok = self.tokenizer(x).transpose(1, 2)
        cls = self.cls.expand(x.size(0), -1, -1)
        z = self.encoder(torch.cat([cls, tok], dim=1) + self.pos)
        embed = self.norm(z[:, 0])
        logits = self.head(embed)
        return (logits, embed) if return_embed else logits


# =========================================================================== #
#  YOUR MODEL - electrode tokens (attention across the spatial axis)
#
#  Hypothesis (written before coding, see ../ANSWERS.md and README.md):
#  For bilateral fists-vs-feet the class information is a SPATIAL pattern of
#  mu/beta power (lateral hand areas vs the medial vertex), not sample-level
#  timing. If each electrode is a token, summarised by a shared conv band-power
#  stem, attention can model that inter-electrode contrast directly over 65
#  tokens instead of 482, and the CLS->electrode attention becomes a scalp map
#  I can check against neurophysiology.
#
#  The ONE change vs VanillaEEGTransformer is the token definition:
#     vanilla : token = all 64 electrodes at one time sample  (482 tokens)
#     ours    : token = one electrode's band-power time course (65 tokens)
#  The encoder (post-norm nn.TransformerEncoderLayer, d=64, 4 heads, depth 3,
#  FFN 128, dropout 0.3), CLS token, learned position/identity embedding,
#  LayerNorm and linear head are identical to the vanilla baseline.
# =========================================================================== #
class YourModel(nn.Module):
    def __init__(self, n_chan, n_time, n_classes=2, n_filters=8, kern=64,
                 win=80, d_model=64, n_heads=4, depth=3, p=0.3):
        super().__init__()
        self.n_chan, self.win = n_chan, win
        self.n_win = n_time // win                       # 481 // 80 = 6 windows of 0.5 s
        # (1) Shared temporal filter bank, applied to every electrode separately
        #     (weights shared across electrodes, NO spatial mixing here). With
        #     kern=64 (0.4 s) each filter can become a narrow band-pass inside
        #     7-30 Hz, like EEGNet's first block.
        self.temporal = nn.Conv1d(1, n_filters, kern, padding=kern // 2, bias=False)
        # (2) Power in 0.5 s windows, then log. ERD is a power change and the
        #     oscillation phase is random across trials, so a token must be
        #     phase-invariant; log makes multiplicative power changes additive.
        # (3) Project the (filters x windows) log-power map to one token.
        self.embed = nn.Linear(n_filters * self.n_win, d_model)
        # Learned electrode-identity embedding: the spatial analogue of the
        # vanilla model's learned temporal position embedding. No geometry is
        # injected (that would be a second bias); attention must discover
        # which electrodes matter from data.
        self.chan_emb = nn.Parameter(torch.zeros(1, n_chan, d_model))
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.chan_emb, std=0.02)
        nn.init.trunc_normal_(self.cls, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=d_model * 2,
            dropout=p, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, depth)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)

    def tokens(self, x):                                  # x: (B, C, T)
        B, C, T = x.shape
        h = self.temporal(x.reshape(B * C, 1, T))          # (B*C, F, T+1)
        h = h[..., : self.n_win * self.win]                # (B*C, F, n_win*win)
        h = h.reshape(B * C, h.size(1), self.n_win, self.win).pow(2).mean(-1)
        h = torch.log(h + 1e-6)                            # (B*C, F, n_win) log band power
        tok = self.embed(h.reshape(B, C, -1))              # (B, C, d)
        cls = self.cls.expand(B, -1, -1)
        return torch.cat([cls, tok + self.chan_emb], dim=1)  # (B, C+1, d)

    def forward(self, x, return_embed=False):
        z = self.encoder(self.tokens(x))
        embed = self.norm(z[:, 0])
        logits = self.head(embed)
        return (logits, embed) if return_embed else logits

    @torch.no_grad()
    def cls_attention(self, x):
        """CLS -> electrode attention of every layer, heads averaged.

        Re-runs the (post-norm) encoder layers by hand so the attention weights
        can be returned; numerically identical to forward() in eval mode.
        Returns (B, depth, C).
        """
        self.eval()
        z = self.tokens(x)
        maps = []
        for layer in self.encoder.layers:
            a, w = layer.self_attn(z, z, z, need_weights=True, average_attn_weights=True)
            maps.append(w[:, 0, 1:])                       # CLS row, electrode columns
            z = layer.norm1(z + layer.dropout1(a))
            z = layer.norm2(z + layer._ff_block(z))
        return torch.stack(maps, 1)

    def save_attention_topomap(self, X, y, pos, ch_names, path, device="cpu"):
        """Scalp maps of CLS attention (last layer), per class and feet - fists."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import mne
        att = self.cls_attention(torch.from_numpy(X).to(device))[:, -1].cpu().numpy()  # (N, C)
        info = mne.create_info(ch_names, 160.0, "eeg")
        montage = mne.channels.make_dig_montage(
            ch_pos=dict(zip(ch_names, pos.astype(float))), coord_frame="head")
        info.set_montage(montage)
        rows = [("fists", att[y == 0].mean(0)), ("feet", att[y == 1].mean(0)),
                ("feet - fists", att[y == 1].mean(0) - att[y == 0].mean(0))]
        fig, axes = plt.subplots(1, 3, figsize=(10, 3.4))
        for ax, (title, v) in zip(axes, rows):
            cmap = "RdBu_r" if "-" in title else "viridis"
            lim = np.abs(v).max() if "-" in title else None
            im, _ = mne.viz.plot_topomap(v, info, axes=ax, show=False, cmap=cmap,
                                         vlim=(-lim, lim) if lim else (None, None),
                                         names=ch_names if title == "feet - fists" else None)
            ax.set_title(f"{title}", fontsize=10)
            plt.colorbar(im, ax=ax, shrink=0.7)
        fig.suptitle("YourModel: last-layer CLS attention over electrodes, test subjects 9-10 "
                     f"(uniform = {1 / len(ch_names):.4f})", fontsize=9)
        fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
        top = np.argsort(-att.mean(0))[:8]
        print("  top-8 attended electrodes:", [ch_names[i] for i in top])
        print(f"  saved {path}")


class ElectrodeTokensNoAttention(YourModel):
    """ABLATION (not a competitor): YourModel's exact tokens, but the
    transformer is replaced by a linear read-out of all electrode tokens.
    If YourModel does no better than this, attention across electrodes is not
    what helps; the conv band-power stem is."""
    def __init__(self, n_chan, n_time, n_classes=2, p=0.3, **kw):
        super().__init__(n_chan, n_time, n_classes, p=p, **kw)
        del self.encoder, self.cls
        d = self.chan_emb.shape[-1]
        self.norm = nn.LayerNorm(d)
        self.drop = nn.Dropout(p)
        self.head = nn.Linear(n_chan * d, n_classes)

    def forward(self, x, return_embed=False):
        B, C, T = x.shape
        h = self.temporal(x.reshape(B * C, 1, T))[..., : self.n_win * self.win]
        h = torch.log(h.reshape(B * C, h.size(1), self.n_win, self.win).pow(2).mean(-1) + 1e-6)
        tok = self.norm(self.embed(h.reshape(B, C, -1)) + self.chan_emb)
        embed = tok.flatten(1)
        logits = self.head(self.drop(embed))
        return (logits, embed) if return_embed else logits


# =========================================================================== #
#  Experiment plumbing
# =========================================================================== #
def model_zoo(C, T):
    """Every model gets the same (C, T) input and is built fresh per seed."""
    return {
        # Part 1 validated configuration (kern = 0.5 s at 160 Hz, dropout 0.5).
        "EEGNet": lambda: EEGNet(C, T, kern=80, p=0.5),
        "VanillaTransformer": lambda: VanillaEEGTransformer(C, T),
        "TemporalPatch(starter)": lambda: TemporalPatchTransformer(C, T),
        "YourModel": lambda: YourModel(C, T),
        "Ablation:ElectrodeTokens-NoAttn": lambda: ElectrodeTokensNoAttention(C, T),
    }


def log_row(row):
    with open(RESULTS, "a") as f:
        f.write(json.dumps(row) + "\n")


def summarise(name, vals):
    v = np.asarray(vals)
    return (f"{name:24s} mean {v.mean():.3f}  std {v.std(ddof=1) if len(v) > 1 else 0:.3f}  "
            f"best {v.max():.3f}  worst {v.min():.3f}  (n={len(v)})")


def save_figure(model, Xte, yte, device, path, subj=None):
    """t-SNE of the CLS embeddings on the test subjects: coloured by class
    (left) and by subject (right). The right panel checks whether the embedding
    is organised by person rather than by imagined movement."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    model.eval()
    with torch.no_grad():
        logits, emb = model(torch.from_numpy(Xte).to(device), return_embed=True)
    emb, pred = emb.cpu().numpy(), logits.argmax(1).cpu().numpy()
    acc = (pred == yte).mean()
    n = len(emb)
    emb2 = TSNE(n_components=2, perplexity=min(30, max(5, n // 4)),
                init="pca", random_state=0).fit_transform(emb)
    panels = 2 if subj is not None else 1
    fig, axes = plt.subplots(1, panels, figsize=(5 * panels, 4), squeeze=False)
    ax = axes[0, 0]
    for c, name in [(0, "fists"), (1, "feet")]:
        m = yte == c
        ax.scatter(emb2[m, 0], emb2[m, 1], s=18, alpha=0.7, label=name)
    ax.legend(); ax.set_title(f"coloured by class (test acc {acc:.2f})", fontsize=9)
    if subj is not None:
        ax = axes[0, 1]
        for s_id, mk in zip(np.unique(subj), ["o", "s", "^", "D"]):
            m = subj == s_id
            ax.scatter(emb2[m, 0], emb2[m, 1], s=18, alpha=0.7, marker=mk, label=f"S{s_id}")
        ax.legend(); ax.set_title("same points coloured by subject", fontsize=9)
    fig.suptitle("YourModel CLS embeddings of held-out subjects 9-10 (t-SNE, first seed, "
                 "final-epoch weights)", fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    print(f"  saved {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--models", nargs="+", default=None, help="subset of model names")
    ap.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = all cores)")
    ap.add_argument("--protocols", nargs="+", default=["fixed"],
                    choices=["fixed", "val"])
    ap.add_argument("--quick", action="store_true", help="1 seed smoke test")
    ap.add_argument("--no-log", action="store_true")
    ap.add_argument("--figure-only", action="store_true",
                    help="retrain YourModel for the first seed only and redraw the figures")
    args = ap.parse_args()
    if args.quick:
        args.seeds = [0]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(args.threads or max(1, os.cpu_count() or 1))
    print(f"device: {device}  seeds: {args.seeds}  epochs: {args.epochs}")

    print("\nloading data...")
    data = {}
    if "fixed" in args.protocols:
        data["fixed"] = load_data()                                   # S1-8 / S9-10
    if "val" in args.protocols:
        data["val"] = load_data(val_subjects=VAL_SUBJECTS)            # S1-6 / S7-8 / S9-10
    Xte, yte = (data.get("fixed") or data["val"])[-2:]
    C, T = Xte.shape[1], Xte.shape[2]
    for k, d in data.items():
        print(f"  [{k}] " + "  ".join(f"{a.shape}" for a in d[0::2]))
        for a in d[0::2]:
            assert np.isfinite(a).all(), "non-finite input"

    zoo = model_zoo(C, T)
    names = args.models or list(zoo)
    if args.figure_only:
        names, args.seeds, args.no_log = ["YourModel"], args.seeds[:1], True
    summary, fig_model = {}, None
    for name in names:
        build = zoo[name]
        n_params = count_params(build())
        print(f"\n=== {name}  ({n_params:,} params) ===")
        res = {"final_test": [], "peak_test": [], "val_selected_test": [], "sec": []}
        for s in args.seeds:
            if "fixed" in args.protocols:
                Xtr, ytr, Xte, yte = data["fixed"]
                set_seed(s); t0 = time.time()
                model = build()
                out = train_model(model, Xtr, ytr, Xte, yte, epochs=args.epochs, device=device)
                if name == "YourModel" and s == args.seeds[0]:
                    fig_model = model          # the exact model whose number is reported
                sec = time.time() - t0
                res["final_test"].append(out["final_test"]); res["peak_test"].append(out["peak_test"])
                res["sec"].append(sec)
                print(f"  seed {s}  [fixed S1-8]  final_test {out['final_test']:.3f}  "
                      f"(diagnostic peak_test {out['peak_test']:.3f} @ep{out['peak_epoch']})  {sec:.0f}s")
                if not args.no_log:
                    log_row(dict(part="part2", model=name, seed=s, protocol="fixed_budget_S1-8",
                                 epochs=args.epochs, n_params=n_params,
                                 test_acc=out["final_test"], diag_peak_test_acc=out["peak_test"],
                                 diag_peak_epoch=out["peak_epoch"], n_test=len(yte),
                                 train_sec=round(sec, 1)))
            if "val" in args.protocols:
                Xtr, ytr, Xva, yva, Xte, yte = data["val"]
                set_seed(s)
                out = train_model(build(), Xtr, ytr, Xte, yte, Xva=Xva, yva=yva,
                                  epochs=args.epochs, device=device)
                res["val_selected_test"].append(out["val_selected_test"])
                print(f"  seed {s}  [val S7-8 ]  val_best {out['val_best_acc']:.3f} @ep{out['val_best_epoch']}"
                      f"  -> test {out['val_selected_test']:.3f}")
                if not args.no_log:
                    log_row(dict(part="part2", model=name, seed=s, protocol="val_selected_S1-6_val_S7-8",
                                 epochs=args.epochs, n_params=n_params,
                                 test_acc=out["val_selected_test"], val_acc=out["val_best_acc"],
                                 val_best_epoch=out["val_best_epoch"], n_test=len(yte)))
        summary[name] = (n_params, res)

    print("\n================ SUMMARY (test = subjects 9-10) ================")
    for name, (n_params, res) in summary.items():
        print(f"\n{name}  params={n_params:,}"
              + (f"  mean train time {np.mean(res['sec']):.0f}s/seed" if res["sec"] else ""))
        for key, label in [("final_test", "fixed budget (headline)"),
                           ("val_selected_test", "val-selected (S7-8)"),
                           ("peak_test", "peak test [OPTIMISTIC]")]:
            if res[key]:
                print("   " + summarise(label, res[key]))

    if fig_model is not None:
        print(f"\nsaving figures (YourModel, seed {args.seeds[0]}, fixed protocol, final-epoch weights)...")
        Xtr, ytr, Xte, yte = data["fixed"]
        ym = fig_model
        torch.save(ym.state_dict(), os.path.join(HERE, "yourmodel_seed%d.pt" % args.seeds[0]))
        save_figure(ym, Xte, yte, device, os.path.join(HERE, "figure.png"),
                    subj=get_trial_subjects(TEST_SUBJECTS))
        if hasattr(ym, "save_attention_topomap"):
            ym.save_attention_topomap(Xte, yte, get_channel_positions(), get_channel_names(),
                                      os.path.join(HERE, "attention_topomap.png"), device)

    print("\ndone. Best AND worst are reported above - no cherry-picking.")


if __name__ == "__main__":
    main()
