"""
Shared training / evaluation harness. Keep runs short - the point is the idea,
not the hyperparameters.

EVALUATION CHANGE vs. the starter harness
-----------------------------------------
The starter evaluated the TEST subjects (9-10) after every epoch and reported
the maximum. That is test-set model selection: the reported number is the
luckiest of `epochs` noisy looks at ~90 test trials, so it is optimistically
biased for every model (and more so for high-variance models).

Here the test set is never used to choose anything:
  * final_test        - fixed, pre-declared budget (`epochs`); the model after
                        the last epoch is scored once on the test subjects.
  * val_selected_test - if validation subjects (carved out of 1-8) are given,
                        the checkpoint with the best validation accuracy is
                        restored and scored once on the test subjects.
  * peak_test         - the starter's number (max over epochs of test acc).
                        Kept ONLY as a diagnostic to show how much the old
                        protocol inflated results. Never used for decisions.
"""
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


def make_loader(X, y, batch=64, shuffle=False):
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(ds, batch_size=batch, shuffle=shuffle)


@torch.no_grad()
def evaluate(model, loader, device):
    """Return (accuracy, mean cross-entropy)."""
    model.eval()
    correct = total = 0
    loss_sum = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss_sum += nn.functional.cross_entropy(logits, yb, reduction="sum").item()
        correct += (logits.argmax(1) == yb).sum().item()
        total += yb.numel()
    return correct / total, loss_sum / total


def train_model(model, Xtr, ytr, Xte, yte, *, Xva=None, yva=None, epochs=40,
                lr=1e-3, weight_decay=1e-3, device="cpu", verbose=False):
    """Train for a fixed budget. Returns a dict (see module docstring).

    Model selection uses ONLY the validation subjects (if provided).
    """
    model = model.to(device)
    tr = make_loader(Xtr, ytr, shuffle=True)
    te = make_loader(Xte, yte, batch=256)
    va = make_loader(Xva, yva, batch=256) if Xva is not None else None
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    hist = dict(train_loss=[], val_acc=[], val_loss=[], test_acc=[])
    best_val = (-1.0, np.inf)          # (acc, -loss) ordering: higher acc, then lower loss
    best_state, best_ep = None, -1
    for ep in range(epochs):
        model.train()
        run_loss, n = 0.0, 0
        for xb, yb in tr:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at epoch {ep}")
            loss.backward()
            opt.step()
            run_loss += loss.item() * yb.numel(); n += yb.numel()
        hist["train_loss"].append(run_loss / n)

        if va is not None:
            v_acc, v_loss = evaluate(model, va, device)
            hist["val_acc"].append(v_acc); hist["val_loss"].append(v_loss)
            if (v_acc > best_val[0]) or (v_acc == best_val[0] and v_loss < best_val[1]):
                best_val, best_ep = (v_acc, v_loss), ep
                best_state = copy.deepcopy(model.state_dict())

        # Diagnostic only (reproduces the starter's optimistic metric). Not used
        # for any decision below.
        t_acc, _ = evaluate(model, te, device)
        hist["test_acc"].append(t_acc)
        if verbose and (ep % 10 == 0 or ep == epochs - 1):
            msg = f"  epoch {ep:2d}  train_loss {hist['train_loss'][-1]:.3f}"
            if va is not None:
                msg += f"  val_acc {hist['val_acc'][-1]:.3f}"
            print(msg)

    out = dict(final_test=hist["test_acc"][-1],
               peak_test=max(hist["test_acc"]),
               peak_epoch=int(np.argmax(hist["test_acc"])),
               history=hist)
    if va is not None:
        model.load_state_dict(best_state)
        out["val_best_epoch"] = best_ep
        out["val_best_acc"] = best_val[0]
        out["val_selected_test"], _ = evaluate(model, te, device)
    return out


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
