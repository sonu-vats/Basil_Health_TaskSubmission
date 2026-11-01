# Interview Preparation

How to use this: each answer starts with the one-sentence version I would say first, then the detail I can add if asked. Numbers refer to my actual runs (see `ANSWERS.md` and `results.jsonl`).

---

## 1. The six Part 1 bugs (how I found each, which way it moved accuracy)

| # | Bug | How I knew | Direction |
| --- | --- | --- | --- |
| 1 | `data.py` loads runs 5/9/13 | PhysioNet run table: 5/9/13 are **executed** fists/feet, 6/10/14 are **imagined**. The docstring and the task say imagery. | **Up.** Real movement gives bigger ERD plus EMG and movement artefacts. 0.738 -> 0.643 (mean of 3 seeds). |
| 2 | `train.py` splits trials, not subjects | `groups` was computed and never used; the comment promised "subject-independent". | **Up.** Same person (same run, even neighbouring trials) in train and test. 0.643 -> 0.591. |
| 3 | z-score stats computed on all data | Mean/std computed before the split. | **Up, tiny.** 0.591 -> 0.603 (noise level). A correctness fix; the effect is small because 64 channel means over hundreds of trials barely move. |
| 4 | Best checkpoint chosen on **test** accuracy | `if te_acc > best_acc` and validation accuracy printed but unused. | **Up.** Reported number = max of 30 noisy test looks. 0.603 -> 0.556. |
| 5 | ERD baseline = 0-0.5 s | t = 0 is the cue. 0-0.5 s holds the cue-evoked burst (+50-85 % at ~0.25 s) and the start of ERD. With that reference the pre-cue rest itself looked like -10 % "ERD". | Interpretation only: shrinks/distorts ERD. |
| 6 | `TMAX = 5.0` silently drops the last trial of every run | The last cue ends exactly when the recording ends (118.9 + 4.1 = 123.0 s). MNE's `drop_log` showed `BAD boundary`, `BAD boundary`, `TOO_SHORT` for trials 15, 30, 45 of every subject. | **Down / noisier.** 30 of 450 trials (6.7 %) lost. 0.556 -> 0.567 (noise level). |

What I would say about #6 honestly: the five others are unambiguous. #6 is the one remaining silent defect I could prove from the data; I checked and rejected the other candidates (label mapping is correct per PhysioNet, filtering is done on continuous data, Hilbert on continuous data, `colin27_1005` is just the new name of `standard_1005`, 1-40 Hz is a design choice that matters for the confound rather than a bug).

**Seed-level caveat I volunteer:** after fix 2 the test set is 2 subjects (90 trials). One trial = 1.1 points. Single-split differences under ~10 points are noise, which is why I added a 5-fold leave-2-subjects-out CV for the final number.

---

## 2. Why is subject-level splitting important?
- The question being asked is "will this work on a new person?". Only a subject-level split answers it.
- EEG has huge between-subject differences (skull thickness, cap placement, impedance, individual alpha frequency, strategy). A trial split lets the model learn "this is subject 3's head" and still get credit.
- Trials inside a run are also correlated in time (slow drifts, electrode gel drying), so neighbouring trials in train and test leak even within a subject.

## 3. What is data leakage?
Any path by which information from the evaluation data influences training or model choice. Here there were three kinds: **sample leakage** (same subject in train and test), **preprocessing leakage** (normalisation statistics from test), **selection leakage** (choosing the checkpoint on test). All three make the number optimistic; none crashes.

## 4. Why can test-set checkpoint selection be misleading?
- Test accuracy on 90 trials has a standard error of about 5 points at 50-60 %. Taking the max over 30 epochs picks the epoch where noise happened to favour you.
- I measured it in Part 2: the starter's "peak test" number was 3-10 points higher than the honest final-epoch number for every model (e.g. temporal-patch transformer: peak 0.60 average vs 0.50 honest).
- Fix: choose on validation subjects, or fix the budget in advance, and look at test once.

## 5. What is mu ERD?
- Mu = 8-13 Hz rhythm over sensorimotor cortex, an "idling" rhythm produced by synchronised thalamo-cortical activity when the area is not processing movement.
- Preparing, executing **or imagining** a movement desynchronises that population, so band power drops: event-related desynchronisation, ERD% = (P(t) - P_rest) / P_rest x 100, negative.
- ERS (synchronisation) is a power increase, e.g. the post-movement beta rebound, or "surround ERS" in areas not involved in the task.

## 6. Why do C3, Cz and C4 matter?
- 10-20 system: C3 sits over the left hand area, C4 over the right hand area, Cz on the vertex over the medial (leg/foot) area in the interhemispheric fissure.
- Both-fists imagery should desynchronise C3 **and** C4 (bilateral); both-feet imagery should desynchronise Cz and often leave hand areas unchanged or even enhanced (focal ERD / surround ERS).
- My ERD (pre-cue baseline, 10 subjects, 0.5-3.5 s): C3 hands -12.9 % vs feet -0.5 % (8/10 subjects in the expected direction); Cz feet -6.1 % vs hands -3.5 % (small, not consistent); C4 hands -8.6 % vs feet -5.8 %.

**Why Cz is weaker than expected:** the foot area is on the medial wall deep in the interhemispheric fissure; its dipoles point sideways (tangential to the scalp at Cz) and the two hemispheres' sources point toward each other and partly cancel, so little radial field reaches Cz. Cz also picks up the bilateral hand-area ERD by volume conduction, so both classes show ERD there. Add a common reference (no Laplacian/CAR), naive subjects and only ~22 trials per class per subject, and the contrast shrinks. Foot imagery is also known to be harder for naive users.

## 7. Why can frontal channels classify fists vs feet?
- In these runs the fists target is at the **top** of the screen and the feet target at the **bottom**. People look at the target. Vertical eye movement is perfectly correlated with the label.
- 8 frontal channels (Fp1, Fpz, Fp2, AF7, AF3, AFz, AF4, AF8), far from motor cortex, decoded **better** than all 64 channels: 5-fold subject CV 0.632 vs 0.590 (frontal won 5/5 folds on seed 0). That is the red flag.
- Control: filter at 8-30 Hz (the mu/beta band where ERD lives) and frontal drops to 0.497, all-channel to 0.511, central to 0.516, i.e. chance. The decodable information was in the slow band (< 8 Hz), where eye movements and cue-evoked potentials live.
- `confound.py` shows the mechanism: after the cue, Fp channels go **positive for top targets and negative for bottom targets** (+/-15-20 µV within 0.2-0.6 s), then reverse when the eyes return; the class difference is largest at the frontal pole and decays toward the back of the head, exactly the gradient of an ocular source.

## 8. What is EOG contamination?
The eye is an electrical dipole (cornea about +, retina about -). Rotating the eye moves that dipole, so electrodes near the eyes see large slow potential shifts (tens to hundreds of µV) that dwarf cortical signals (a few µV). Looking up makes Fp electrodes more positive, looking down more negative. Blinks add large positive spikes at Fp. Saccades also produce a brief spike potential from the eye muscles (broadband, up to beta/gamma), so 8-30 Hz filtering reduces but does not remove ocular information.

## 9. What does EEGNet's depthwise convolution learn?
`Conv2d(F1, F1*D, (n_ch, 1), groups=F1)`: for each of the F1 = 8 temporal filters (each a learned band-pass, e.g. mu or beta), it learns D = 2 **spatial filters**, each a weighted sum over all 64 electrodes, applied only to that band. So the 16 outputs are frequency-specific spatial filters, like filter-bank CSP: for this task you would hope for a mu/beta filter weighting C3 and C4 against Cz (hand vs foot area). It can just as well learn a frontal (EOG) pattern if that separates the classes. The weights can be plotted as topomaps (1,024 parameters, 64 per filter).

## 10. Why divide by sqrt(d)?
If query and key components are roughly independent with zero mean and unit variance, q·k is a sum of d_k terms, so its variance is d_k. Dividing by sqrt(d_k) keeps the logits at unit scale regardless of head size. Without it, logits grow with d_k, softmax saturates to one-hot, its gradient (p(1-p)) goes to ~0, and training stalls or becomes unstable. Here d_k = 64 / 4 heads = 16, so unscaled logits would have ~4x the standard deviation.

## 11. Why is attention quadratic?
Every token computes a score with every other token: QK^T is (n x d)(d x n) -> n x n. Time O(n^2 d), memory O(n^2) per head per layer (the attention matrix must be stored for backward, and with attention dropout the fused kernels cannot skip it).

## 12. The transformer's sequence length (Part 2)
- Window 0-3 s at 160 Hz inclusive = 481 samples, +1 CLS = **482 tokens**.
- Attention matrix 482 x 482 = 232,324 entries per head per layer; 4 heads x 3 layers = 2.8 M entries per trial, ~11 MB fp32 per trial, ~0.7 GB per batch of 64 before dropout masks.
- MACs per layer for QK^T and AV: 2 x 482^2 x 64 ≈ 30 M, vs ≈16 M for all the linear layers. Attention dominates.
- Measured on my 2-core CPU: ~5 s per training batch for vanilla vs ~0.3 s for my 65-token model.
- My model: 64 electrodes + CLS = **65 tokens**, 4,225 entries per head per layer (55x fewer). Starter temporal patches: 15 + 1 = 16 tokens.

## 13. Why are temporal patches useful?
They give attention a locality prior: each token already summarises 32 samples (200 ms), so the model does not spend capacity learning that adjacent samples are related, the O(T^2) surface shrinks from 482^2 to 16^2, and positional parameters shrink. Limits: patches are still linear mixtures of all channels at all lags, not phase-invariant (a power change is invisible to a linear patch embedding unless later layers square it), and the starter version has 232 k parameters (131 k in the patch conv), so it can overfit faster. In my runs it scored 0.504 on average (5 seeds) on the honest metric, at chance, though its old peak-test number looked like 0.60. That is why I moved away from temporal patches, my first idea, as the bet: it is the starter's own example, it does not address the phase/power issue, and it says nothing about the spatial (medial vs lateral) structure of this task.

## 14. Why might EEGNet beat a transformer on small EEG data?
- EEGNet has ~2.7 k parameters vs 108-233 k; with 360 training trials from 8 people that is the difference between fitting and memorising.
- Its structure **is** the right prior: temporal band-pass, then spatial filter, then power-like pooling (ELU + average pooling approximates an envelope). A transformer has to learn all of that from data it does not have.
- BatchNorm and heavy dropout also regularise strongly. Attention's strength (input-dependent, long-range interactions) needs data to pay off.

## 15. What is an inductive bias?
An assumption built into the model that restricts or favours some solutions before seeing data. Convolution assumes locality and translation invariance; EEGNet's depthwise layer assumes "a frequency band, then a fixed spatial filter". Good biases trade flexibility for sample efficiency; wrong ones cap performance.

## 16. Why is anterior-posterior structure important here?
Both classes are bilateral. Left vs right hand differ by hemisphere (lateralisation), but fists vs feet differ by **where along the motor strip** the ERD happens: lateral hand areas (C3/C4) vs the medial vertex (Cz), and along the anterior-posterior axis (premotor/SMA vs central/parietal). A prior that only encodes left/right symmetry would add nothing. My bet works on the electrode axis without hard-coding any direction, so attention can in principle learn the medial-vs-lateral contrast.

## 17. What if we had 80 training subjects?
- Overfitting pressure drops ~10x; between-subject variability is better sampled. Larger models, including attention, start to pay off; EEGNet's advantage from its tiny parameter count shrinks.
- My electrode-token idea should matter **more**, not less, in one respect: input-dependent spatial weighting can adapt to each person's slightly different source location, which a fixed depthwise filter cannot. I would also add 3D sensor positions so it transfers across montages.
- Temporal patching mostly buys compute and stability; with 80 subjects its regularising value matters less.
- The ocular confound does **not** go away with more data: more data makes it easier to learn. That needs a design/analysis fix.

## 18. How to interpret the t-SNE plot
- It is a 2-D non-linear projection of the 64-d CLS embeddings of the 90 test trials (subjects 9-10), from the seed-0 YourModel whose number is reported.
- t-SNE preserves local neighbourhoods, not distances or cluster sizes; separation can be exaggerated or hidden depending on perplexity, and the axes mean nothing.
- It shows what the model learned, not that it generalises: the test accuracy is the evidence.
- My figure has two crisp clusters. Each contains both true classes and both subjects; colouring by the **prediction** shows the clusters are the model's two decision regions (87 of 90 points). So the model is confident and wrong half the time: it splits new people's trials on some feature (the predicted-fists cluster has higher overall signal power) that is not the label. Crisp t-SNE clusters != class separation.

## 19. Limitations of cross-subject generalisation here
- Only 10 subjects, 45 trials each; test = 2 people. Confidence intervals are ±10 points.
- Naive users, no feedback, no subject calibration; many people are "BCI-illiterate" for foot imagery.
- Ocular/visual confound correlated with the label.
- No artefact rejection, no re-referencing, one fixed window.
- The official split (train 1-8, test 9-10) measures two specific people; S9 is hard in every model I ran.

## 20. What I would investigate next
1. Remove the confound and remeasure: ICA/regression of ocular components (Fp channels as EOG proxy), drop Fp/AF, start the window after 1 s; frontal-only decoding should fall to chance.
2. Permutation tests (label shuffling within subject) for every headline number.
3. For the transformer: stronger regularisation targeted at the 8-subject wall (smaller d, attention dropout, subject-wise mixup / Euclidean alignment), then geometry-aware electrode embeddings (3D positions) instead of learned IDs.
4. Look at EEGNet's depthwise weights as topomaps to see whether it uses central or frontal patterns.
5. Inter-trial-variance ERD (subtract the evoked response) to separate induced ERD from the cue-evoked burst.

---

## Decision log (for "why did you do X?")

### Part 1
| Change | Why | Alternative considered | Evidence | Limitation |
| --- | --- | --- | --- | --- |
| Runs 6/10/14 | Imagery is the task | Keep executed runs and note it | PhysioNet run table; accuracy drop 0.738 -> 0.643 | None |
| GroupShuffleSplit 6/2/2 subjects | New-person question | Leave-one-subject-out | groups unused before | 2 test subjects -> noisy |
| z-score on train only | Prevent preprocessing leak | Per-subject z-score (also leak-free, and reduces subject shift) | Tiny measured effect | Global stats ignore subject scale differences |
| Checkpoint on validation (ties -> lower val loss) | Test touched once | Fixed epoch budget (also logged as last_epoch_test_acc) | 0.603 -> 0.556 | 2 val subjects -> noisy selection (seed 1 chose epoch 1) |
| ERD reference -1.5..-0.5 s | Inside rest, away from previous trial's rebound and anticipation | -2..0 s | Rest now flat at 0 %, burst and ERD visible | Cue timing is regular, anticipation cannot be fully excluded |
| TMAX 4.0 s | Keep all 450 trials | 4.5 s (still drops trials of 123 s runs, verified) | drop_log empty | ERD plot now ends at 4 s |
| `--cv` | One split is not trustworthy | More seeds on one split | Per-split swings of 20 points | 5x compute |
| frontal = Fp/AF rows | Closest to eyes, far from motor strip | F-row too (closer to SMA) | confound.py polarity reversal | Fp/AF still pick up some frontal cortex |

### Part 2
| Change | Why | Alternative considered | Evidence | Limitation |
| --- | --- | --- | --- | --- |
| Removed test-peak reporting | Selection leak | Validation subjects 7-8 (implemented as optional protocol) | peak vs final gap of 5-10 points | Fixed budget may be sub-optimal for some models |
| Same 5 seeds, same budget for all | Fair comparison | Starter: 1 seed EEGNet, 3 transformer | — | 5 seeds still a small sample |
| EEGNet with Part 1 config (kern 80, p 0.5) | PDF asks to carry the validated baseline | Starter config (kern 64, p 0.25) | — | Not re-tuned for 7-30 Hz / 3 s |
| Electrode tokens (my bet) | Task signal is spatial (medial vs lateral power); per-timestep tokens are phase-sensitive and 482 long | Temporal patches (the starter's own example, kept as reference), 3D-geometry positional encoding, neighbourhood attention bias | log band-power + logistic regression reaches 0.63 on validation subjects, so the tokens carry signal; attention maps | 108 k params still overfit 8 subjects |
| Encoder identical to vanilla | Isolate the one change | Smaller encoder (tried on validation only, not better) | dev log | Not the best-performing configuration |
| No-attention ablation | Tells whether attention or the stem helps | — | see ANSWERS | Linear read-out is a different capacity |
