# EEG Transformer Sprint — Starter Kit

You have 90 minutes. This kit removes all the plumbing so you spend the time on
one thing: **giving self-attention a fighting chance on tiny EEG data, and
understanding why your idea does (or doesn't) help.**

## Setup (first ~10 minutes)

```bash
pip install -r requirements.txt
python data.py      # sanity check: downloads ~50 MB, prints array shapes
python main.py      # runs both baselines + the starter YourModel + saves figure.png
```

Everything runs on CPU in well under a minute per model. The first run
downloads the PhysioNet EEGBCI data and caches it.

## The task

**Both fists vs. both feet** (imagined movement), cross-subject: train on
subjects 1–8, test on held-out subjects 9–10.

This is a **bilateral** task. The discriminative signal is largely
anterior–posterior (hand/arm motor area vs. the medial foot representation near
the vertex, ~Cz), **not** the classic left/right lateralised mu/beta ERD. A
spatial prior that only encodes left vs. right will not help you here.

## What's in the box

| File | What it is |
|------|------------|
| `data.py` | Loader: download, 7–30 Hz filter, epoch, cross-subject split, per-channel z-score (train stats only). Also `get_channel_positions()` for 3D electrode coordinates. |
| `models.py` | `EEGNet` (conv-prior baseline) and `VanillaEEGTransformer` (per-timestep tokens, **no** EEG prior — the thing to beat/understand). |
| `train.py` | Shared train/eval loop, seed control, data loaders. |
| `main.py` | Entry point. Runs the baselines, runs **your** model across seeds, reports best + worst, saves `figure.png`. |

## Your job — in `main.py`

Implement **one** inductive bias in `YourModel` (or replace it entirely). The
starter version already shows one legal example: learned **temporal patch**
tokens instead of per-timestep tokens, which shrinks the O(T²) attention
surface. Swap in your own bet.

Do **not** quietly edit the vanilla transformer in `models.py` — the whole point
is the contrast between "attention with no help" and "attention with your one
idea."

## What to hand in

- This repo with your `YourModel` filled in. `python main.py` must reproduce
  your numbers end to end, seed printed.
- A short note (half a page — edit the section below is fine) with:
  1. **Hypothesis** — one line, written *before* you coded it.
  2. **What you changed and why** it targets a real weakness of attention on EEG.
  3. **What happened** — best *and* worst run across seeds, no cherry-picking.
  4. One sentence on what `figure.png` shows.
- A live code walkthrough at the end: share your screen, walk us through the
  file, answer questions.

Accuracy is **not** scored. We're reading your code and your answers for how you
reason about EEG and attention under a clock. A clear negative result beats a
lucky spike you can't explain.

---

## Your write-up (fill this in)

**Hypothesis:** (fixed before YourModel was implemented or trained)
For bilateral fists-vs-feet the class information is a *spatial* pattern of
mu/beta power (lateral hand areas vs the medial vertex, an anterior-posterior /
medial-lateral contrast), not sample-level timing; making each **electrode** a
token, summarised by a shared conv band-power stem, lets attention model that
inter-electrode contrast directly over 65 tokens instead of 482 time samples,
which should generalise across subjects at least as well as the vanilla
transformer with fewer parameters and lower seed variance, and its
CLS-to-electrode attention should concentrate on the sensorimotor strip
(C3/Cz/C4 region) rather than on frontal (ocular) sites.

**What I changed and why:** one token per electrode instead of one per time
sample (65 vs 482 tokens), each built by a shared conv filter bank -> log power
in 0.5 s windows -> linear map, plus a learned electrode-identity embedding;
encoder/CLS/head identical to the vanilla model. Full reasoning in
`../ANSWERS.md` (Part 2).

**Results (best / worst / mean across seeds):** see the table in `../ANSWERS.md`
(fixed 40-epoch budget, test subjects scored once, seeds 0-4) and
`results.jsonl`.

**Figure:** `figure.png` = t-SNE of YourModel CLS embeddings on subjects 9-10,
coloured by true class, predicted class and subject; the two clusters are the
model's decision regions, not the classes (see `../ANSWERS.md`).
