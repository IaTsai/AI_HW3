"""Combine Pong + LunarLander multi-seed numbers into a single compact table.

Reports mean and 95% bootstrap CI across seeds (B=10000 resamples). With
n=3 the CI is wide -- this is the honest interval for small-n estimates.
"""
import json
import os
import numpy as np


RES = "/mnt/nvme0n1/ia313553058/Others/AI_3/results"
RNG = np.random.default_rng(0)


def load(p):
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def stats(history, k=100):
    if history is None or "ep_return" not in history:
        return None, None
    r = np.asarray(history["ep_return"], dtype=float)
    if len(r) < k:
        return None, None
    ma = np.convolve(r, np.ones(k) / k, mode="valid")
    return float(ma.max()), float(ma[-1])


def bootstrap_ci(xs, B=10000, alpha=0.05):
    """Percentile bootstrap CI for the mean of small-n samples."""
    xs = np.asarray(xs, dtype=float)
    n = len(xs)
    if n == 0:
        return (float("nan"), float("nan"))
    idx = RNG.integers(0, n, size=(B, n))
    means = xs[idx].mean(axis=1)
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return (lo, hi)


def aggregate(paths, k=100):
    bests, finals = [], []
    for p in paths:
        h = load(p)
        b, f = stats(h, k)
        if b is not None: bests.append(b)
        if f is not None: finals.append(f)
    mb, sb = (np.mean(bests), np.std(bests)) if bests else (np.nan, np.nan)
    mf, sf = (np.mean(finals), np.std(finals)) if finals else (np.nan, np.nan)
    blo, bhi = bootstrap_ci(bests)
    flo, fhi = bootstrap_ci(finals)
    return (mb, sb, mf, sf, len(bests), blo, bhi, flo, fhi)


def main():
    out = "/mnt/nvme0n1/ia313553058/Others/AI_3/report/tables/multiseed_combined.tex"

    # Pong: 3 seeds (42, 0, 7). seed=42 ran 2M steps; seeds 0 and 7 ran 1.5M.
    pong_rows = []
    for label, subs in [
        ("Vanilla DQN", ["pong_vanilla", "pong_vanilla_s0", "pong_vanilla_s7"]),
        ("Double DQN", ["pong_double", "pong_double_s0", "pong_double_s7"]),
        ("Dueling Double DQN", ["pong_dueling", "pong_dueling_s0", "pong_dueling_s7"]),
    ]:
        paths = [f"{RES}/{s}/history.json" for s in subs]
        pong_rows.append((label, *aggregate(paths)))

    # LunarLander: 3 seeds (42, 0, 7)
    lander_rows = []
    for label, subs in [
        ("DQN (Double)", ["lander_double", "lander_double_s0", "lander_double_s7"]),
        ("DQN (Dueling Double)", ["lander_dueling", "lander_dueling_s0", "lander_dueling_s7"]),
    ]:
        paths = [f"{RES}/{s}/history.json" for s in subs]
        lander_rows.append((label, *aggregate(paths)))

    def fmt_row(label, mb, sb, mf, sf, n, blo, bhi, flo, fhi, bold=False):
        # show mean and 95% bootstrap CI for best-train; std for final-train
        if bold:
            lhs = f"\\textbf{{{label}}}"
            cell_best = f"$\\mathbf{{{mb:.2f}}}$ [{blo:.2f},\\,{bhi:.2f}]"
        else:
            lhs = label
            cell_best = f"${mb:.2f}$ [{blo:.2f},\\,{bhi:.2f}]"
        return f"  & {lhs} & {cell_best} & ${mf:.2f}\\pm{sf:.2f}$ & {n} \\\\"

    s = []
    s.append(r"\begin{table}[h]\centering\footnotesize")
    s.append(r"\caption{Multi-seed summary across both tasks. Pong: 3 seeds (42, 0, 7); "
             r"seed~42 ran 2\,M steps, seeds 0 and 7 ran 1.5\,M each. "
             r"LunarLander: 3 seeds (42, 0, 7), each 500\,K steps. "
             r"\emph{Best train} is the mean of best 100-ep MA across seeds with 95\% percentile "
             r"bootstrap CI (B$=$10$^{4}$); \emph{Final train} is mean$\pm$std of final 100-ep MA. "
             r"Bold = mean best $\geq$ ``solved'' threshold (LunarLander only).}")
    s.append(r"\label{tab:multiseed}")
    s.append(r"\begin{tabular}{llccc}")
    s.append(r"\toprule")
    s.append(r"Task & Variant & Best train (mean, 95\% CI) & Final train (mean$\pm$std) & $n$ \\")
    s.append(r"\midrule")
    s.append(r"\multirow{3}{*}{Pong}")
    for row in pong_rows:
        s.append(fmt_row(*row, bold=False))
    s.append(r"\midrule")
    s.append(r"\multirow{2}{*}{LunarLander}")
    for row in lander_rows:
        s.append(fmt_row(*row, bold=row[1] >= 200))
    s.append(r"\bottomrule")
    s.append(r"\end{tabular}")
    s.append(r"\end{table}")
    with open(out, "w") as f:
        f.write("\n".join(s) + "\n")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
