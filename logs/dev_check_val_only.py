"""Dev sanity check on the INNER split only (train S1-6, val S7-8).
Subjects 9-10 are never loaded here. Used once to confirm YourModel trains."""
import sys, time; sys.path.insert(0, ".")
import numpy as np, torch
from data import _load_subjects
from train import train_model, set_seed
import main
fit = [1, 2, 3, 4, 5, 6]; val = [7, 8]
Xtr, ytr = _load_subjects(fit); Xva, yva = _load_subjects(val)
mu = Xtr.mean((0, 2), keepdims=True); sd = Xtr.std((0, 2), keepdims=True) + 1e-6
Xtr, Xva = (Xtr - mu) / sd, (Xva - mu) / sd
for name in sys.argv[1:]:
  for seed in (0, 1):
    set_seed(seed); t = time.time()
    m = main.model_zoo(64, Xtr.shape[2])[name]()
    out = train_model(m, Xtr, ytr, Xva, yva, Xva=Xva, yva=yva, epochs=40)   # "test" slot = val
    h = out["history"]
    print(f"{name} s{seed}: val acc by epoch (every 5): {[round(a,2) for a in h['val_acc'][::5]]}  "
          f"final {h['val_acc'][-1]:.3f}  train_loss {h['train_loss'][0]:.3f}->{h['train_loss'][-1]:.3f}  {time.time()-t:.0f}s")
