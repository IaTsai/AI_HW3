"""Post-process a history.json so the per-episode arrays are sorted by env
step and each step appears at most once. Necessary because crash+resume runs
duplicate (step, return) pairs spanning the lost progress."""
import argparse
import json
import os
import sys


def clean(history):
    pairs = list(zip(history.get("step", []),
                      history.get("ep_return", []),
                      history.get("ep_len", [])))
    if not pairs:
        return history
    # Sort by step, then dedup keeping the LAST occurrence per step (later
    # entries reflect the most recent training pass through that env step).
    pairs.sort(key=lambda t: t[0])
    seen = {}
    for s, r, l in pairs:
        seen[s] = (r, l)
    out_steps = sorted(seen.keys())
    history["step"] = out_steps
    history["ep_return"] = [seen[s][0] for s in out_steps]
    history["ep_len"] = [seen[s][1] for s in out_steps]
    # Same for loss_step/loss/q_mean
    lpairs = list(zip(history.get("loss_step", []),
                       history.get("loss", []),
                       history.get("q_mean", [])))
    if lpairs:
        lpairs.sort(key=lambda t: t[0])
        lseen = {}
        for s, l, q in lpairs:
            lseen[s] = (l, q)
        out_l = sorted(lseen.keys())
        history["loss_step"] = out_l
        history["loss"] = [lseen[s][0] for s in out_l]
        history["q_mean"] = [lseen[s][1] for s in out_l]
    # Eval
    epairs = list(zip(history.get("eval_step", []),
                       history.get("eval_mean", []),
                       history.get("eval_std", [])))
    if epairs:
        epairs.sort(key=lambda t: t[0])
        eseen = {}
        for s, m, d in epairs:
            eseen[s] = (m, d)
        out_e = sorted(eseen.keys())
        history["eval_step"] = out_e
        history["eval_mean"] = [eseen[s][0] for s in out_e]
        history["eval_std"] = [eseen[s][1] for s in out_e]
    return history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", help="history.json files to clean (in-place)")
    args = parser.parse_args()
    for p in args.paths:
        if not os.path.exists(p):
            print(f"skip (missing): {p}")
            continue
        with open(p) as f:
            h = json.load(f)
        before = len(h.get("step", []))
        h2 = clean(h)
        after = len(h2.get("step", []))
        with open(p, "w") as f:
            json.dump(h2, f)
        print(f"{p}: {before} -> {after} episodes")


if __name__ == "__main__":
    main()
