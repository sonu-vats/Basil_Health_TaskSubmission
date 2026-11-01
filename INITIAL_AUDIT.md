# Initial Audit (written before any code change)

Scope: `EEG_DL_Exam_3hr.pdf`, `README.md`, `environment.yml`, every file in `part1_diagnose/` and `part2_transformer/`.
Status labels: **CONFIRMED** = I can point at the line and explain why it is wrong from the code, the data documentation or the data itself. **HYPOTHESIS** = suspicious, needs a run to judge. Nothing here is invented to reach a number.

---

## 1. Dataset facts I verified from the raw EDF files

I opened `S001R05.edf` and `S001R06.edf` directly with MNE before touching the code:

| Fact | Value | How checked |
| --- | --- | --- |
| Sampling rate | 160 Hz | `raw.info['sfreq']` |
| Run length | 125.0 s | `raw.n_times / 160` |
| Trial structure | `T0` rest 4.2 s, then `T1` or `T2` cue 4.1 s, repeating | annotations onset/duration |
| Trials per run | 15 (7 or 8 per class) | annotation counter |
| Last cue onset | 120.4 s, so a 5 s epoch needs 125.4 s and does not fit | annotations |
| Channel names | `Fc5.`, `C3..` etc; `eegbci.standardize` turns them into `FC5`, `C3` | `raw.ch_names` |

PhysioNet EEGMMIDB run map (from the dataset documentation, also what MNE's `eegbci` docstring states):

| Runs | Task | T1 | T2 |
| --- | --- | --- | --- |
| 3, 7, 11 | real left/right fist | left fist | right fist |
| 4, 8, 12 | imagined left/right fist | left fist | right fist |
| **5, 9, 13** | **real (executed)** both fists / both feet | both fists | both feet |
| **6, 10, 14** | **imagined** both fists / both feet | both fists | both feet |

So T1 = fists and T2 = feet in both the executed and imagined runs; only the run numbers decide execution vs imagery.

---

## 2. File-by-file description

### part1_diagnose
| File | What it does | Shapes |
| --- | --- | --- |
| `data.py` | Loads runs `[5, 9, 13]` per subject, concatenates, standardises names, sets montage, band-pass 1-40 Hz on the continuous signal, epochs `T1`/`T2` from -2 to 5 s, crops 0-4 s, converts V to µV | X `(n_trials, 64, 641)`, y `{0: hands, 1: feet}`, groups = subject id |
| `erd.py` | Band-pass raw (8-13 or 13-30 Hz), Hilbert envelope, epoch, square to power, average trials, express as % change from a baseline window, plot C3/Cz/C4 averaged over subjects | power epochs `(n, 64, 1121)` |
| `model.py` | EEGNet-8,2 at 160 Hz (temporal kernel 80 = 0.5 s), 2,898 parameters for 64x641 input | see shape table in section 4 |
| `train.py` | Builds all-subject dataset, z-scores, splits 64/16/20, trains EEGNet 30 epochs with AdamW, keeps a "best" checkpoint, appends result to `results.jsonl` | |

### part2_transformer
| File | What it does |
| --- | --- |
| `data.py` | Runs `[6, 10, 14]` (correct imagery runs), 7-30 Hz, epochs 0-3 s (481 samples), train = subjects 1-8, test = 9-10, z-score with train stats only. Event ids 2/3 are correct because `events_from_annotations` sorts `T0, T1, T2` to 1, 2, 3 and drops `BAD/EDGE boundary` by default regexp. `get_channel_positions()` returns 3D sensor coordinates. |
| `models.py` | `EEGNet` (kern 64, dropout 0.25, 2,610 params) and `VanillaEEGTransformer` (Linear(64->64) per time sample, CLS, learned positional embedding for 482 positions, 3 post-norm encoder layers, d=64, 4 heads, FFN 128; **135,746 params, of which 30,848 are positional embeddings**). |
| `train.py` | AdamW loop. `train_model` evaluates on the **test set every epoch and returns the maximum**. |
| `main.py` | Runs EEGNet once (seed 0), vanilla over 3 seeds, the `YourModel` stub (temporal patches of 32 samples, 15 tokens) over 3 seeds, then retrains seed 0 to draw a t-SNE of CLS embeddings. |

---

## 3. Data flow

```
Part 1
EDF (runs) -> concatenate -> standardize names -> montage -> filter 1-40 Hz (continuous)
   -> events T1/T2 -> Epochs [-2, 5] s -> crop [0, 4] s -> x1e6 (µV)
   -> stack subjects (X, y, groups) -> z-score (ALL trials) -> trial-level split
   -> EEGNet -> select epoch by TEST acc -> report TEST acc of that epoch

erd.py: EDF -> band-pass -> |Hilbert|^2 -> Epochs [-2, 5] -> mean over trials
   -> % change vs BASELINE (0, 0.5) s -> plot C3 / Cz / C4

Part 2
EDF runs 6/10/14 -> filter 7-30 Hz -> Epochs [0, 3] s (481) -> train S1-8 / test S9-10
   -> z-score with train stats -> model -> evaluate TEST every epoch -> report MAX
```

---

## 4. Defects and suspected issues

### Part 1 (the PDF says exactly 6 planted defects across data.py, erd.py, train.py)

| # | Status | File:line | What it does | Why it is wrong | Expected effect |
| --- | --- | --- | --- | --- | --- |
| D1 | CONFIRMED | `data.py:15` | `RUNS = [5, 9, 13]` | Those are **executed** fists/feet runs. The docstring, the comment and the task say **imagined**; imagery is runs 6, 10, 14. | Up: real movement gives stronger, more stereotyped ERD plus EMG and movement artefacts. Wrong task altogether. |
| D2 | CONFIRMED | `train.py:58-60` | z-score mean/std computed on all trials of all subjects, before the split | Test (and val) data shape the preprocessing. | Small up-bias / optimistic; mostly a methodological contamination. |
| D3 | CONFIRMED | `train.py:64-66` | `train_test_split` on trial indices; `groups` is computed but never used | Comment claims "subject-independent". Trials of the same subject and even the same run land in train and test, so the model can memorise subject/session signatures. | Strong up-bias. |
| D4 | CONFIRMED | `train.py:84-88, 91-93` | Keeps the checkpoint with the best **test** accuracy and reports test accuracy of that checkpoint; validation is computed and ignored | Test set is used for model selection, so the reported number is the max of ~30 noisy test evaluations. | Up-bias, larger with small test sets. |
| D5 | CONFIRMED | `erd.py:18` | `BASELINE = (0.0, 0.5)` labelled "rest period before the imagery starts" | t = 0 is cue onset. 0-0.5 s is after the cue, contains the visual evoked response and the start of ERD. The pre-cue rest is -2..0 s (inside T0). | Biases ERD toward 0 (shrinks it) and can flip sign; interpretation, not accuracy. |
| D6 | HYPOTHESIS (to resolve with data) | candidates below | | | |

Candidates for the sixth defect, to be resolved by running the code rather than guessing:
- `data.py:19` `TMAX = 5.0` is longer than the 4.1 s trial; the last trial of every run cannot fit (I already saw `TOO_SHORT` in `drop_log` for S001R06), so data is silently discarded. Effect: fewer training trials.
- `data.py:18` 1-40 Hz keeps the < 7 Hz band where eye movements live; Part 2 uses 7-30 Hz. This is related to Task 3 rather than an obvious planted bug.
- Label mapping (`T1=1 -> hands`, `T2=2 -> feet`, `y = code - 1`) is **correct** per PhysioNet. Not a defect.
- Filtering before epoching and Hilbert on continuous data in `erd.py` are **correct** (no edge effects). Not defects.
- `colin27_1005` is the MNE >= 1.10 name of the old `standard_1005`. Not a defect.

### Part 2 (not stated as buggy, but the PDF asks for a fair comparison)
| # | Status | Where | Issue | Effect |
| --- | --- | --- | --- | --- |
| P2-1 | CONFIRMED | `train.py:47-51` | Reports `max` over epochs of **test** accuracy | Same leak as D4; every model is reported at its luckiest epoch on 2 subjects (~90 trials, 95% CI about +/-10 pts). |
| P2-2 | CONFIRMED | `main.py:127-133` | EEGNet run with 1 seed, transformers with 3 seeds | Unequal comparison; best-of-3 vs best-of-1. |
| P2-3 | CONFIRMED | `main.py:155-159` | Figure model is a separate retrain whose state is the last epoch, not the reported model | Figure and table come from different models. |
| P2-4 | NOTE | `main.py` stub | `YourModel` already implements temporal patching | Picking it unchanged would not be "my own bet". |
| P2-5 | NOTE | `models.py` | EEGNet here is kern 64 / dropout 0.25, Part 1 is kern 80 / dropout 0.5 | PDF asks to carry the Part 1 configuration as the reference. |

---

## 5. Experimental risks
- **Tiny test sets.** Part 2 test = 2 subjects (~90 trials). One trial = ~1.1 pts of accuracy. Chance 50% has a 95% interval of roughly 40-60%. Any difference under ~10 pts between models is noise unless it holds across seeds and folds.
- **Cross-subject variance dominates** model variance. A single subject split can swing accuracy by 10-20 pts.
- **Visual cue confound** (Task 3): top vs bottom targets drive vertical eye movements; frontal channels (Fp1/Fp2/AF) pick up EOG, especially below 7 Hz.
- **Data access.** `physionet.org` is blocked from my sandboxes; data has to be fetched on the user's machine.
- **CPU only.** Vanilla transformer has 482x482 attention per head per layer.

## 6. Correction strategy
1. Run the untouched pipeline once to record the original (inflated) number.
2. Fix D1..D6 one per commit, rerunning `train.py` after each fix and logging the result with the commit hash.
3. Train the trusted baseline with subject-level splits; also report a 5-fold leave-2-subjects-out estimate with several seeds so the number has an error bar.
4. ERD with a pre-cue baseline; inspect the PNG before writing about it.
5. Frontal vs all channels, plus a control (central channels only, and frontal with 7-30 Hz) to separate EOG from neural signal.
6. Part 2: remove test-set selection (fixed budget with validation subjects carved out of 1-8), same seeds and budget for all models, then one inductive bias.
