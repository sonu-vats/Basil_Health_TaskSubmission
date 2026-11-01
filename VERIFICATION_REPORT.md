# Verification Report

| Check | Status | Evidence |
| --- | --- | --- |
| Data available | Done | 60 EDF files (S1-10, runs 5/6/9/10/13/14), all read by MNE at 160 Hz, 64 ch. physionet.org was blocked from my sandbox, so files were downloaded on my machine and placed in the MNE cache path. |
| Original pipeline run before changes | Done | tag `0-original`, seeds 0-2: 0.667 / 0.750 / 0.798 |
| One commit per Part 1 fix, rerun after each | Done | 6 `fix(...)` commits, results tagged 1-5 in `part1_diagnose/results.jsonl` |
| Subject split has no overlap | Done | assert in `fit_and_eval`; subjects printed per run |
| Determinism | Done | seed 0 rerun after refactor reproduced 0.522 / epoch 20; figure model rerun reproduced identical attention ranking |
| No NaN/inf | Done | inputs asserted finite in Part 2; training loss checked each step |
| Shapes | Done | EEGNet stage-by-stage shapes checked (64x641 -> 16x1x20 -> 320 -> 2); all Part 2 models output (B, 2); YourModel attention path matches forward exactly (max diff 0.0) |
| ERD plot inspected | Done | `erd_mu.png`, before/after in `logs/figures/` |
| Frontal vs all, with band and central controls | Done | 5-fold subject CV, seeds 0 and 1 |
| Part 2 three-model comparison | Partial | EEGNet, YourModel, starter patch model, ablation: 5 seeds. Vanilla transformer: 2 seeds (0, 1); seeds 2-4 stopped at the deadline (~35 min/seed on 2 CPU cores). |
| Validation-selected Part 2 protocol | Implemented, not run | `--protocols fixed val` |
| figure.png generated from the reported seed-0 model | Done | weights saved in `part2_transformer/yourmodel_seed0.pt` |
| models.py untouched | Done | `git diff 50fd9ff -- part2_transformer/models.py` is empty |
| Permutation tests | Not done | listed as next step |
