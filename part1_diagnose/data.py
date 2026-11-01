"""Data loading for the hands-vs-feet motor imagery task.

Dataset: PhysioNet EEG Motor Movement/Imagery (EEGBCI), 64 channels, 160 Hz.
Task: imagined opening/closing of BOTH FISTS (T1) vs BOTH FEET (T2).
"""
import numpy as np
import mne
from mne.datasets import eegbci

# Built-in template montage (renamed in recent MNE versions).
MONTAGE = ("colin27_1005" if "colin27_1005" in mne.channels.get_builtin_montages()
           else "standard_1005")

# Runs containing imagined both-fists vs both-feet trials.
# PhysioNet EEGMMIDB: 5/9/13 = EXECUTED fists/feet, 6/10/14 = IMAGINED fists/feet.
RUNS = [6, 10, 14]
SUBJECTS = list(range(1, 11))

L_FREQ, H_FREQ = 1.0, 40.0   # broadband filter applied to the continuous recording
# Epoch window around the cue (s). A trial is 4.1 s of cue then 4.2 s of rest,
# and the LAST cue of a run ends exactly when the recording ends (e.g. onset
# 118.9 s in a 123.0 s run). TMAX=5.0 therefore ran past the end of the run (or
# across the run boundary after concatenation) and MNE silently dropped the last
# trial of every run (TOO_SHORT / BAD boundary): 3 of 45 trials per subject,
# 30 of 450 overall. TMAX = 4.0 = end of the analysed crop keeps every trial.
TMIN, TMAX = -2.0, 4.0
CROP = (0.0, 4.0)            # part of each trial the network sees (s)
EVENT_ID = dict(hands=1, feet=2)


def load_raw(subject, runs=RUNS):
    """Download (first call only) and load the requested runs as one Raw object."""
    files = eegbci.load_data(subject, runs)
    raws = [mne.io.read_raw_edf(f, preload=True, verbose="error") for f in files]
    raw = mne.concatenate_raws(raws)
    eegbci.standardize(raw)              # "C3.." -> "C3", "Fcz" -> "FCz", ...
    raw.set_montage(MONTAGE)
    if raw.info["sfreq"] != 160:
        raise ValueError(f"S{subject:03d}: unexpected sfreq {raw.info['sfreq']}")
    return raw


def get_events(raw):
    events, _ = mne.events_from_annotations(
        raw, event_id=dict(T1=1, T2=2), verbose="error")
    return events


def epoch_subject(subject, band=None):
    """Return X (n_trials, 64, n_times) in microvolts, y (0=hands, 1=feet), ch_names."""
    raw = load_raw(subject)
    l_freq, h_freq = band if band is not None else (L_FREQ, H_FREQ)
    raw.filter(l_freq, h_freq, fir_design="firwin", verbose="error")
    events = get_events(raw)
    epochs = mne.Epochs(raw, events, EVENT_ID, TMIN, TMAX, baseline=None,
                        preload=True, verbose="error")
    n_drop = sum(bool(r) for r in epochs.drop_log)
    if n_drop:
        print(f"  S{subject:03d}: {n_drop} of {len(events)} epochs dropped "
              f"({sorted(set(sum(map(list, epochs.drop_log), [])))})")
    X = epochs.copy().crop(*CROP).get_data(copy=True) * 1e6
    y = epochs.events[:, 2] - 1
    return X.astype(np.float32), y.astype(np.int64), epochs.ch_names


def build_dataset(subjects=SUBJECTS, band=None):
    """Stack all subjects. `groups` holds the subject id of every trial.
    `band=(l, h)` overrides the default 1-40 Hz filter (used for confound checks)."""
    Xs, ys, gs = [], [], []
    for s in subjects:
        X, y, ch_names = epoch_subject(s, band=band)
        Xs.append(X); ys.append(y); gs.append(np.full(len(y), s))
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(gs), ch_names
