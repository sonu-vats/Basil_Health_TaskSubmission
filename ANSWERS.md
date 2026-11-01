# ANSWERS

Everything below comes from runs I actually executed (CPU, MNE 1.13, torch 2.15). Raw rows are in `results.jsonl` (combined), `part1_diagnose/results.jsonl` and `part2_transformer/results.jsonl`; `python summarize_results.py` regenerates every table. Test sets are small (2 subjects = 90 trials, so one trial = 1.1 points and a 95% interval around 55% is roughly ±10 points); I treat differences under ~10 points on a single split as noise.

---

# PART 1 · Diagnose & Trust

## Task 1: the six defects

Accuracy columns: mean of seeds 0/1/2 after each fix, applied cumulatively in this order, one commit each (`git log`). Before fix 2 the split is by trial; after it, the seeds hold out subjects [3,9], [3,10], [2,5].

| # | Commit | File:line (original) | What was wrong (one line) | How it changed the result | Before -> after |
| --- | --- | --- | --- | --- | --- |
| 1 | `f5e6fa7` | `data.py:15` `RUNS=[5,9,13]` | Those are the **executed** fists/feet runs; imagery is runs 6/10/14. | **Up**: real movement gives stronger ERD plus EMG/movement artefacts; it was answering a different, easier question. | 0.738 -> 0.643 |
| 2 | `542abfd` | `train.py:64-66` `train_test_split` on trials | `groups` was ignored, so the same subject (even the same run) sat in train and test. | **Up**: the model can memorise person/session signatures and still be "right" on test. | 0.643 -> 0.591 |
| 3 | `7bfe1fd` | `train.py:57-60` z-score over all data | Mean/std came from every trial, including test subjects. | **Up, tiny**: preprocessing leak; channel means over 450 trials barely move, so the effect was within noise. | 0.591 -> 0.603 |
| 4 | `0c1346f` | `train.py:84-93` best epoch by **test** acc | Checkpoint chosen on test (val computed and ignored), then test reported. | **Up**: the number is the max of 30 noisy looks at the test set. | 0.603 -> 0.556 |
| 5 | `5a4ab4e` | `erd.py:18` `BASELINE=(0.0, 0.5)` | Reference window sits **after** the cue (t=0), inside the trial. | Interpretation: the reference contains the cue-evoked burst (+50-85% at ~0.25 s) and ERD onset, so ERD is distorted and even the pre-cue rest looked like about -10% "ERD". Now -1.5..-0.5 s (in the 4.2 s T0 rest). | figure, see Task 2 |
| 6 | `ab039e7` | `data.py:19` `TMAX=5.0` | Epoch longer than the 4.1 s trial; the last cue of each run ends exactly at the end of the recording, so MNE silently dropped the last trial of every run (`BAD boundary` x2, `TOO_SHORT` x1 per subject in `drop_log`). | **Down / noisier**: 30 of 450 trials (6.7%) thrown away, always the final trial of a run. TMAX=4.0 (end of the analysed crop) keeps all 450. | 0.556 -> 0.567 (n_test 84 -> 90) |

Evidence used: the PhysioNet run table (#1); reading the code path of `groups` (#2) and of the normalisation and checkpoint logic (#3, #4); the annotation timing in the EDF files (cue at t=0, T0 rest 4.2 s, trial 4.1 s, last onset 118.9 s in a 123.0 s run) and MNE's `drop_log` (#5, #6). I also checked and **rejected** other candidates: the T1=fists / T2=feet mapping is correct; filtering and the Hilbert transform run on the continuous signal (no edge effects); `colin27_1005` is just the new MNE name of `standard_1005`; 1-40 Hz is a design choice (it matters for Task 3, see below).

**Overall before vs after:** the original pipeline printed **0.667 / 0.750 / 0.798** (mean 0.738). The corrected pipeline on the same seeds prints **0.522 / 0.467 / 0.711** (mean 0.567). Because a single 2-subject split swings by 20+ points depending on who is held out, I added 5-fold leave-2-subjects-out cross-validation (`--cv`, each subject tested once per seed, normalisation and checkpointing inside each fold):

| Trusted baseline (EEGNet, 1-40 Hz, all 64 ch) | Result |
| --- | --- |
| 5-fold CV, seeds 0 and 1 (10 fold-scores) | **0.590 ± 0.07** (seed 0: 0.580, seed 1: 0.600) |
| per subject (mean of seeds 0 and 1) | S1 0.50, S2 0.67, S3 0.50, S4 0.62, S5 0.52, S6 0.60, S7 0.69, S8 0.72, S9 0.47, S10 0.61 |

So the number I trust is **about 59% cross-subject, with a spread of roughly 47-72% across people**, and (Task 3) most of it is not motor imagery.

## Task 2: ERD (`erd_mu.png`)

With the pre-cue reference, mu power at **C3** drops for fists (about -13% over 0.5-3.5 s) but stays near baseline for feet (about -0.5%), and this hands-vs-feet difference is the only consistent one (8 of 10 subjects, t = 2.45); **C4** shows ERD for both (hands -8.6%, feet -5.8%); **Cz** shows a similar, small ERD for both (feet -6.1%, hands -3.5%, not consistent across subjects). That matches the somatotopy in part: fists imagery desynchronises the lateral hand area, and feet imagery leaves the hand area relatively spared (consistent with "focal ERD / surround ERS"), but the expected focal foot ERD at the vertex is weak. **Cz is weaker than expected** because the foot representation lies on the medial wall deep in the interhemispheric fissure: its sources are oriented tangentially and the two hemispheres' sources face each other and partly cancel, so little field reaches Cz radially, while Cz also picks up hand-area ERD by volume conduction (no Laplacian/average reference here). Naive subjects, ~22 trials per class per subject and the difficulty of foot imagery shrink it further. Every channel also shows a sharp **+50-85% burst at ~0.25 s** that is time-locked to the cue and larger for feet at C3/Cz/C4 and much larger for fists at Fpz (73% vs 35%): it is the evoked response to the visual target (and the eye movement to it), not induced mu activity. Beta (`part1_diagnose/erd_beta.png`) is noisier with the same pattern (C3 hands -8.8% vs feet -4.0%).

## Task 3: what is the model using? (`--channels frontal` vs `all`)

`frontal` = Fp1, Fpz, Fp2, AF7, AF3, AFz, AF4, AF8 (nearest the eyes, farthest from motor cortex). I added a same-size `central` control (FC3, FCz, FC4, C3, Cz, C4, CP3, CPz, CP4) and a band control.

| EEGNet, 5-fold CV over subjects (seeds 0 and 1) | 1-40 Hz | 8-30 Hz |
| --- | --- | --- |
| all 64 channels | 0.590 (0.580 / 0.600) | **0.511** (0.511 / 0.511) |
| frontal 8 channels | **0.632** (0.656 / 0.609) | **0.497** (0.516 / 0.478) |
| central 9 channels | 0.609 (0.602 / 0.616) | **0.516** (0.549 / 0.484) |

Eight prefrontal channels decode fists vs feet **at least as well as all 64 channels** (on seed 0 frontal beat all in 5 of 5 folds), and the frontal performance **collapses to chance once the slow band below 8 Hz is removed**. Neural mu/beta ERD lives at 8-30 Hz over the central strip, so this pattern says the decoder is mostly exploiting a slow, frontal, cue-locked signal. The obvious non-brain candidate is **vertical eye movement (EOG)**: the fists target is at the top and the feet target at the bottom, and `confound.py` (`part1_diagnose/frontal_confound.png`) shows exactly the ocular signature: within 0.2-0.6 s of the cue the Fp channels swing about +15 µV for top targets and about -20 µV for bottom targets, reverse sign when the eyes return (~1-1.3 s), and the feet-minus-fists difference is largest at the frontal pole and decays toward the back of the head. Even the all-channel and the central (motor-strip) models drop to chance at 8-30 Hz, so the slow band is carrying the information everywhere (ocular potentials spread back over the scalp, plus cue-evoked potentials). The honest summary is: **the 59% baseline is real in the sense of being leak-free, but it is mostly an eye-movement / visual-cue decoder, not a motor-imagery decoder.** (A smaller, consistent centro-parietal difference in the same window is probably the visual evoked response to upper vs lower field targets: neural, but visual, not motor.)

## Task 4: short answers

1. **45 trials, 58%?** 58% of 45 is 26 correct; under a fair coin P(>=26/45) = 0.19 (one-sided binomial), and the Wilson 95% interval is 43-71%, so it is **not** above chance (you would need 29/45 = 64%). I would confirm with a label-permutation test (shuffle labels within the subject, retrain/re-score many times, compare 58% to that null), which also handles class imbalance and the fact that trials are not independent.
2. **Depthwise layer:** for each of the F1 = 8 learned temporal (band-pass) filters it learns D = 2 weightings of all 64 electrodes, i.e. 16 frequency-specific **spatial filters** (like filter-bank CSP): e.g. a mu filter that contrasts the hand areas (C3/C4) with the vertex (Cz). Nothing stops it from learning a frontal EOG pattern instead, which is what Task 3 suggests it did.
3. **Stop the cue signal:** experimentally, present the class cue at the **centre of the screen** (or counterbalance/randomise top vs bottom independently of the class) with a fixation requirement, so gaze direction carries no label information. Analytically, remove ocular components (ICA/regression using Fp channels as EOG reference), drop Fp/AF, high-pass at 8 Hz and start the decoding window after the saccade (e.g. 1-4 s), and require frontal-only decoding to be at chance as a control.

---

# PART 2 · Build & Defend

## Evaluation change (before any model work)
The starter `train_model` scored subjects 9-10 after every epoch and **reported the maximum**; EEGNet got 1 seed, the transformers 3. That is test-set selection. Now every model gets the same data (7-30 Hz, 0-3 s, 481 samples), the same fixed 40-epoch budget, AdamW (lr 1e-3, wd 1e-3), batch 64 and seeds, and **the test subjects are scored once, on the final model**. The old max-over-epochs number is still logged as a diagnostic: it overstated every model by 3-10 points. A validation-selected protocol (train S1-6, choose the epoch on S7-8) is implemented (`--protocols fixed val`) but I did not run it for every model because of compute (the vanilla transformer takes ~40 min per seed on this 2-core CPU). EEGNet uses the Part 1 configuration (kernel 80 = 0.5 s, dropout 0.5). `models.py` is untouched.

## (1) Hypothesis (committed in `203c689` before the model was written)
> For bilateral fists-vs-feet the class information is a spatial pattern of mu/beta power (lateral hand areas vs the medial vertex), not sample-level timing; making each **electrode** a token, summarised by a shared conv band-power stem, lets attention model that inter-electrode contrast directly over 65 tokens instead of 482, which should generalise at least as well as the vanilla transformer with fewer parameters and lower seed variance, with CLS attention concentrated on the sensorimotor strip rather than frontal sites.

## (2) What I changed and why it targets a real weakness of attention on EEG
The vanilla model makes one token per **time sample**: a linear mix of 64 channels at one instant. Mu/beta ERD is a change in **oscillation power**, and the phase of the oscillation is random from trial to trial, so a single sample carries almost no class information; attention has to discover "power in a band at certain electrodes" by comparing 482 noisy, phase-random tokens, with 31 k positional parameters, on 360 trials. And the bilateral contrast is spatial (medial vs lateral, anterior-posterior), which the vanilla model can only express through one fixed linear projection at the input.

`YourModel` changes **only the token definition**:
- each electrode's 3 s signal -> shared `Conv1d(1, 8, 64)` filter bank (0.4 s kernels can become narrow band-passes inside 7-30 Hz; weights shared across electrodes, **no spatial mixing**) -> power in six 0.5 s windows -> log -> `Linear(48 -> 64)`, plus a learned electrode-identity embedding (the spatial analogue of the vanilla's learned time-position embedding; no geometry injected);
- encoder, CLS token, LayerNorm, head, dropout are **identical** to the vanilla model (post-norm, d=64, 4 heads, depth 3, FFN 128, p=0.3);
- sequence 65 instead of 482: attention matrix 4,225 vs 232,324 entries per head per layer (55x fewer), ~0.5 M vs ~30 M attention MACs per layer, 108 k vs 136 k parameters, ~3 vs ~40 min per run on my CPU;
- attention now runs **across electrodes**, so CLS->electrode weights are a scalp map I can check (`attention_topomap.png`).

Before running it on the test set I did one development check on validation subjects 7-8 only: the model overfit (train loss 0.76 -> 0.28, validation 0.51-0.59). Two variants (power relative to the scalp mean; d=32/depth 2) were not better on validation, so I kept the pre-registered model unchanged. A heavily regularised logistic regression on log band-power reached 0.63 on the same validation subjects, so the token representation does carry signal; the problem is fitting it with 108 k parameters on 8 people.

## (3) What happened (fixed budget, train S1-8, test S9-10 once; seeds 0-4)

| Model | Params | Seeds | Test acc per seed | Mean ± SD | Best | Worst | Old harness peak-test (optimistic) | Train time/run |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EEGNet | 2,738 | 5 | 0.544 / 0.578 / 0.611 / 0.522 / 0.500 | 0.551 ± 0.044 | 0.611 | 0.500 | 0.589 | 2.2 min |
| VanillaTransformer | 135,746 | 2 | 0.500 / 0.500 | 0.500 ± 0.000 | 0.500 | 0.500 | 0.594 | 35.1 min |
| TemporalPatch(starter) | 232,898 | 5 | 0.456 / 0.489 / 0.556 / 0.522 / 0.500 | 0.504 ± 0.037 | 0.556 | 0.456 | 0.598 | 0.2 min |
| YourModel | 108,482 | 5 | 0.511 / 0.511 / 0.578 / 0.544 / 0.522 | 0.533 ± 0.028 | 0.578 | 0.511 | 0.587 | 3.6 min |
| Ablation:ElectrodeTokens-NoAttn | 16,066 | 5 | 0.544 / 0.522 / 0.567 / 0.556 / 0.500 | 0.538 ± 0.027 | 0.567 | 0.500 | 0.571 | 2.1 min |

Vanilla transformer: only seeds 0 and 1 finished before my deadline (about 35-40 min per seed on a 2-core CPU); seeds 2-4 were stopped, not discarded. On the seeds every model shares (0 and 1): EEGNet 0.561, YourModel 0.511, no-attention ablation 0.533, vanilla 0.500, temporal patch 0.472. Chance interval for 90 test trials is roughly 0.40-0.60.

**Reading it honestly:**
- **No attention model beats EEGNet**, and **my bet did not beat the vanilla transformer in a meaningful way**: all transformer variants sit within ~5 points of chance on subjects 9-10, inside the noise band of a 90-trial test set. The hypothesis' accuracy prediction is not supported.
- What the prior did buy: **compute** (a run takes ~4 min instead of ~40) and a seed spread similar to EEGNet (SD 0.028 vs 0.043 over 5 seeds), but not accuracy. With only 2 vanilla seeds I cannot claim a variance reduction relative to the vanilla model.
- The **no-attention ablation** (same electrode tokens, linear read-out, 16 k params) is as good as YourModel: the attention layers add nothing measurable on top of the band-power stem at this data size.
- The **attention map falsifies the neurophysiological prediction**: last-layer CLS attention is diffuse (max about 2x uniform) and its feet-minus-fists difference peaks at right temporal/lateral sites (C6, TP8, T8), not at C3/Cz/C4. Combined with near-chance accuracy, I read this as the model latching onto subject-specific, partly muscular/edge-electrode power differences that do not transfer to new people.
- Part 1 explains the ceiling: with 7-30 Hz input (the ocular slow band already filtered out), Part 1's EEGNet was also at chance cross-subject (all-channel 8-30 Hz CV 0.511). There is very little transferable mu/beta signal for any model to find in 8 naive subjects, so a prior that helps attention organise power features cannot create it.

## (4) The figure (`figure.png`)
t-SNE of the seed-0 YourModel CLS embeddings for the 90 test trials shows two clean clusters, but each contains both true classes and both subjects in equal measure; colouring the same points by the **model's prediction** shows the clusters are simply its two decision regions (87/90 points), so the network confidently splits the new subjects' trials along a feature that is not fists vs feet. That is a useful warning: a crisp t-SNE separation is not evidence of generalisation; only the held-out accuracy (0.51 for this seed) is.

## Limitations and next experiments
- 2 test subjects, 5 seeds, 1 fixed budget; differences of a few points are noise. The vanilla model's cost limited how many protocols I could run.
- No permutation test on the Part 2 numbers yet; all are within the chance interval for n=90.
- Next: (a) remove the ocular confound (ICA / Fp regression, centre-cue design) and repeat Part 1, (b) regularisation aimed at the 8-subject wall for the electrode-token model (smaller d, attention dropout, subject-wise Euclidean alignment of covariance, mixup across subjects), (c) geometry-aware electrode embeddings from the 3D positions (would also let the model transfer across montages), (d) a validation-selected comparison for all models, (e) visualise EEGNet's depthwise filters as topomaps to see whether it uses central or frontal patterns.
