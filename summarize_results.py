"""Print the result tables used in ANSWERS.md straight from the result logs,
and write the combined top-level results.jsonl (Part 1 + Part 2 rows).

Usage: python summarize_results.py
"""
import json
import os
from collections import OrderedDict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
P1 = os.path.join(HERE, "part1_diagnose", "results.jsonl")
P2 = os.path.join(HERE, "part2_transformer", "results.jsonl")


def load(path):
    return [json.loads(l) for l in open(path)] if os.path.exists(path) else []


def stats(v):
    v = np.asarray(v, float)
    sd = v.std(ddof=1) if len(v) > 1 else 0.0
    return f"{v.mean():.3f} ± {sd:.3f} (best {v.max():.3f}, worst {v.min():.3f}, n={len(v)})"


def part1(rows):
    print("\n## Part 1 - single subject split (seeds 0,1,2)")
    g = OrderedDict()
    for r in rows:
        if r.get("cv_fold") is None and not r["tag"].startswith(("smoke", "repro")):
            g.setdefault((r["tag"], r["channels"]), []).append(r)
    print("| stage | channels | per-seed test acc | mean |")
    print("| --- | --- | --- | --- |")
    for (tag, ch), rs in g.items():
        a = [r["test_acc"] for r in rs]
        print(f"| {tag} | {ch} | {' / '.join(f'{x:.3f}' for x in a)} | {np.mean(a):.3f} |")

    print("\n## Part 1 - 5-fold leave-2-subjects-out CV")
    g = OrderedDict()
    for r in rows:
        if r.get("cv_fold") is not None:
            g.setdefault(r["tag"], {}).setdefault(r["seed"], []).append(r)
    print("| config | per-seed CV mean | folds (seed 0) | mean over seeds x folds | per-subject (mean over seeds) |")
    print("| --- | --- | --- | --- | --- |")
    for tag, by_seed in g.items():
        seed_means = {s: np.mean([r["test_acc"] for r in rs]) for s, rs in by_seed.items()
                      if len(rs) == 5}
        allf = [r["test_acc"] for s, rs in by_seed.items() if len(rs) == 5 for r in rs]
        f0 = [f"{r['test_acc']:.2f}" for r in by_seed.get(0, [])]
        subj = {}
        for s, rs in by_seed.items():
            if len(rs) == 5:
                for r in rs:
                    for k, v in r["per_subject_test_acc"].items():
                        subj.setdefault(int(k), []).append(v)
        subj_s = " ".join(f"S{k}:{np.mean(v):.2f}" for k, v in sorted(subj.items()))
        if allf:
            print(f"| {tag} | {', '.join(f's{s}={m:.3f}' for s, m in seed_means.items())} | "
                  f"{' '.join(f0)} | {np.mean(allf):.3f} ± {np.std(allf, ddof=1):.3f} | {subj_s} |")


def part2(rows):
    print("\n## Part 2 - fixed 40-epoch budget, train S1-8, test S9-10 once")
    g = OrderedDict()
    for r in rows:
        if r["protocol"].startswith("fixed"):
            g.setdefault(r["model"], []).append(r)
    common = None
    for rs in g.values():
        s = {r["seed"] for r in rs}
        common = s if common is None else common & s
    common = sorted(common or [])
    print(f"seeds common to all models: {common}")
    print("| model | params | honest test acc (common seeds) | all seeds run | diagnostic peak-test (old harness) | sec/run |")
    print("| --- | --- | --- | --- | --- | --- |")
    for m, rs in g.items():
        rs = sorted(rs, key=lambda r: r["seed"])
        a_common = [r["test_acc"] for r in rs if r["seed"] in common]
        a_all = [r["test_acc"] for r in rs]
        peak = [r["diag_peak_test_acc"] for r in rs if r["seed"] in common]
        sec = np.mean([r.get("train_sec", np.nan) for r in rs])
        per = ", ".join(f"s{r['seed']}={r['test_acc']:.3f}" for r in rs)
        print(f"| {m} | {rs[0]['n_params']:,} | {stats(a_common)} | {per} | "
              f"{np.mean(peak):.3f} | {sec:.0f} |")


def main():
    r1, r2 = load(P1), load(P2)
    part1(r1)
    if r2:
        part2(r2)
    with open(os.path.join(HERE, "results.jsonl"), "w") as f:
        for r in r1:
            f.write(json.dumps({"part": "part1", **r}) + "\n")
        for r in r2:
            f.write(json.dumps(r) + "\n")
    print(f"\nwrote combined results.jsonl ({len(r1)} part-1 rows, {len(r2)} part-2 rows)")


if __name__ == "__main__":
    main()
