#!/usr/bin/env python3
"""Compact overview of a set of W&B runs, sized for pasting into a chat.

Prints a markdown table to stdout and writes wandb_summary.csv. With --curves it
also writes wandb_history.csv (long format, downsampled) for plotting or for
questions about trajectories rather than endpoints.

Examples:
    python tools/wandb_summary.py me/veda-geom
    python tools/wandb_summary.py me/veda-geom --name latrec
    python tools/wandb_summary.py me/veda-geom --at-hours 24 --at-step 400000
    python tools/wandb_summary.py me/veda-geom --runs 3k9fja2b 8xn1qq0p --curves

Note on history: W&B returns only rows where every requested key is present, so
train and val metrics logged at different steps cannot be fetched in one call.
This pulls one metric at a time, which is slower but correct.
"""

from __future__ import annotations

import argparse
import sys

try:
    import pandas as pd
    import wandb
except ImportError as exc:
    sys.exit(f"missing dependency ({exc.name}); pip install wandb pandas")


# Metric -> higher is better. This order is the column order.
METRICS = {
    "val-validity": True,
    "val-fc-validity": True,
    "val-molecule-stability": True,
    "val-atom-stability": True,
    "val-opt-energy-validity": True,
    "val-uniqueness": True,
    "val-opt-rmsd": False,
    "val-strain-per-atom": False,
    "val-energy-per-atom": False,
    "train-loss": False,
    "train-type-loss": False,
    "train-bond-loss": False,
    "train-coord-loss": False,
}

CONFIG_KEYS = [
    "dataset", "n_layers", "num_parameters",
    "self_condition", "latent_recursion", "latrec_transfer",
    "latrec_n", "latrec_k", "latrec_p",
    "lr", "batch_cost", "acc_batches",
]


def fetch(run, metric, samples):
    """Downsampled history for one metric, with step and runtime alongside."""
    frame = run.history(keys=[metric, "_step", "_runtime"], samples=samples, pandas=True)
    if frame is None or frame.empty or metric not in frame:
        return None
    return frame.dropna(subset=[metric])


def summarise(frame, metric, higher_is_better, at_step, at_hours):
    """Final value, best value, and the value at a step or wall-clock budget."""
    series = frame[metric]
    out = {
        f"{metric}": series.iloc[-1],
        f"{metric} (best)": series.max() if higher_is_better else series.min(),
    }
    if at_step is not None:
        upto = frame[frame["_step"] <= at_step]
        out[f"{metric} @{at_step//1000}k"] = upto[metric].iloc[-1] if len(upto) else float("nan")
    if at_hours is not None:
        upto = frame[frame["_runtime"] <= at_hours * 3600]
        out[f"{metric} @{at_hours}h"] = upto[metric].iloc[-1] if len(upto) else float("nan")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("project", help="entity/project, e.g. bjarke/veda-geom")
    ap.add_argument("--runs", nargs="+", help="explicit run ids; overrides --name")
    ap.add_argument("--name", help="keep runs whose display name contains this")
    ap.add_argument("--state", default="finished,running", help="comma-separated run states to keep")
    ap.add_argument("--at-step", type=int, help="also report each metric at this global step")
    ap.add_argument("--at-hours", type=float, help="also report each metric at this wall-clock hour")
    ap.add_argument("--samples", type=int, default=500, help="history points per metric per run")
    ap.add_argument("--curves", action="store_true", help="also write wandb_history.csv")
    args = ap.parse_args()

    api = wandb.Api()
    runs = [api.run(f"{args.project}/{r}") for r in args.runs] if args.runs else list(api.runs(args.project))

    states = {s.strip() for s in args.state.split(",")}
    runs = [r for r in runs if r.state in states]
    if args.name:
        runs = [r for r in runs if args.name in r.name]
    if not runs:
        sys.exit("no runs matched")

    rows, curves = [], []
    for run in runs:
        print(f"fetching {run.name} ...", file=sys.stderr)
        row = {"run": run.name, "id": run.id, "state": run.state}
        row.update({k: run.config.get(k) for k in CONFIG_KEYS})
        row["hours"] = round((run.summary.get("_runtime") or 0) / 3600, 1)
        row["steps"] = run.summary.get("_step")

        for metric, higher in METRICS.items():
            frame = fetch(run, metric, args.samples)
            if frame is None:
                continue
            row.update(summarise(frame, metric, higher, args.at_step, args.at_hours))
            if args.curves:
                tidy = frame[["_step", "_runtime", metric]].rename(columns={metric: "value"})
                tidy.insert(0, "metric", metric)
                tidy.insert(0, "run", run.name)
                curves.append(tidy)
        rows.append(row)

    table = pd.DataFrame(rows).set_index("run")
    table = table.dropna(axis=1, how="all")
    table.to_csv("wandb_summary.csv")
    try:
        print(table.to_markdown(floatfmt=".4g"))
    except ImportError:  # to_markdown needs tabulate
        print(table.to_string(float_format=lambda v: f"{v:.4g}"))
    print(f"\nwrote wandb_summary.csv ({len(table)} runs, {len(table.columns)} columns)", file=sys.stderr)

    if curves:
        history = pd.concat(curves, ignore_index=True)
        history.to_csv("wandb_history.csv", index=False)
        print(f"wrote wandb_history.csv ({len(history)} rows)", file=sys.stderr)


if __name__ == "__main__":
    main()
