"""Task 3 evidence: is the frontal decoding neural or ocular?

Top target (fists) vs bottom target (feet) should drive opposite VERTICAL eye
movements. The eyeball is a dipole (cornea +, retina -), so looking up makes the
prefrontal electrodes (Fp1/Fpz/Fp2, just above the eyes) more positive and
looking down makes them more negative. Motor-imagery ERD, in contrast, is a
POWER change in the 8-30 Hz band over the central strip, with no fixed polarity.

This script plots, for the imagery runs:
  (a) class-averaged slow waveforms (1-40 Hz, no band-power) at Fpz/Fp1/Fp2
      vs C3/Cz/C4: an ocular confound shows a polarity split frontally that is
      LARGER than centrally and decays with distance from the eyes;
  (b) the per-electrode class difference (feet - fists) averaged 0.2-1.5 s
      after the cue on a scalp map.

Usage: python confound.py   ->  frontal_confound.png
"""
import numpy as np
import mne
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import build_dataset, SUBJECTS, CROP, MONTAGE

FRONT = ["Fp1", "Fpz", "Fp2"]
CENTRAL = ["C3", "Cz", "C4"]
WIN = (0.2, 0.6)   # s after cue: initial gaze shift towards the target (saccade latency ~200 ms)


def main():
    X, y, groups, ch = build_dataset(SUBJECTS)          # (N, 64, T) in µV, 1-40 Hz
    t = np.arange(X.shape[2]) / 160.0 + CROP[0]
    # Per-subject average first, then grand average (each subject counts once).
    diff_sub, wave = [], {c: [] for c in (0, 1)}
    for s in np.unique(groups):
        m = groups == s
        for c in (0, 1):
            wave[c].append(X[m & (y == c)].mean(0))
        diff_sub.append(X[m & (y == 1)].mean(0) - X[m & (y == 0)].mean(0))
    wave = {c: np.mean(v, 0) for c, v in wave.items()}
    diff_sub = np.array(diff_sub)                            # (n_sub, 64, T)

    fig = plt.figure(figsize=(13, 6.5))
    for i, name in enumerate(FRONT + CENTRAL):
        ax = fig.add_subplot(2, 4, i + 1 + (i >= 3))
        k = ch.index(name)
        ax.plot(t, wave[0][k], label="fists (target TOP)")
        ax.plot(t, wave[1][k], label="feet (target BOTTOM)")
        ax.axvline(0, color="k", lw=0.6); ax.axhline(0, color="k", lw=0.4)
        ax.axvspan(*WIN, color="0.9", zorder=0)
        ax.set_title(name); ax.set_xlabel("time from cue (s)")
        if i in (0, 3):
            ax.set_ylabel("µV (1-40 Hz, trial average)")
        if i == 0:
            ax.legend(fontsize=7)
    # Scalp map of the class difference in the gaze window, mean over subjects.
    w = (t >= WIN[0]) & (t <= WIN[1])
    d = diff_sub[:, :, w].mean(axis=(0, 2))
    info = mne.create_info(ch, 160.0, "eeg")
    info.set_montage(MONTAGE)
    ax = fig.add_subplot(2, 4, 4)
    lim = np.abs(d).max()
    im, _ = mne.viz.plot_topomap(d, info, axes=ax, show=False, cmap="RdBu_r", vlim=(-lim, lim))
    ax.set_title(f"feet - fists, {WIN[0]}-{WIN[1]} s (µV)", fontsize=9)
    plt.colorbar(im, ax=ax, shrink=0.7)
    # Effect size per electrode across subjects (consistency of the sign).
    t_stat = diff_sub[:, :, w].mean(2).mean(0) / (diff_sub[:, :, w].mean(2).std(0, ddof=1)
                                                  / np.sqrt(len(diff_sub)))
    ax = fig.add_subplot(2, 4, 8)
    im, _ = mne.viz.plot_topomap(t_stat, info, axes=ax, show=False, cmap="RdBu_r",
                                 vlim=(-np.abs(t_stat).max(), np.abs(t_stat).max()))
    ax.set_title("across-subject t (feet - fists)", fontsize=9)
    plt.colorbar(im, ax=ax, shrink=0.7)
    fig.suptitle("Imagery runs 6/10/14, subjects 1-10: slow frontal deflection follows the "
                 "target position", fontsize=10)
    fig.tight_layout(); fig.savefig("frontal_confound.png", dpi=120)
    order = np.argsort(-np.abs(t_stat))
    print("electrodes with the most consistent feet-fists difference (|t|):")
    for k in order[:10]:
        print(f"  {ch[k]:4s}  mean diff {d[k]:+6.2f} µV   t = {t_stat[k]:+5.2f}")
    print("saved frontal_confound.png")


if __name__ == "__main__":
    main()
