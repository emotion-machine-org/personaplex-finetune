"""One-time exporter that produced the wandb snapshot in this directory.

Kept here as provenance — not part of any user-facing workflow. The
exported data is the canonical artifact; this script is the recipe.

Reproduce (from repo root):
    python docs/history/wandb/_export.py \\
        --entity emotion-machine \\
        --project adhery-demo --project companion-plex \\
        --output docs/history/wandb

Output layout:
    <output>/<project>/<run_id>/
        config.yaml      run.config at launch
        summary.json     run.summary final values
        history.parquet  full unsampled metric history (run.scan_history())
    <output>/index.json  list of all exported runs (machine-readable)
    <output>/summary.csv 1-row-per-run digest (spreadsheet-friendly)
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pandas as pd
import wandb
import yaml

DIGEST_METRICS = [
    "train/loss", "loss", "eval/loss", "eval_loss",
    "train/lr", "lr",
    "_step", "_runtime",
]


def export_run(run, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "config.yaml").open("w") as fh:
        yaml.safe_dump(dict(run.config), fh, sort_keys=False)

    summary = {k: _coerce(v) for k, v in dict(run.summary).items()}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    rows = list(run.scan_history())
    if rows:
        pd.DataFrame(rows).to_parquet(out_dir / "history.parquet", index=False)

    meta = {
        "id": run.id,
        "name": run.name,
        "state": run.state,
        "created_at": str(run.created_at),
        "n_steps": len(rows),
        "url": run.url,
        "tags": list(run.tags or []),
    }
    for m in DIGEST_METRICS:
        v = summary.get(m)
        if isinstance(v, (int, float)):
            meta[m] = v
    return meta


def _coerce(v):
    if isinstance(v, (str, int, float, bool, type(None))):
        return v
    if isinstance(v, dict):
        return {str(k): _coerce(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_coerce(x) for x in v]
    return str(v)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entity", required=True)
    ap.add_argument("--project", action="append", required=True, help="repeat for multiple")
    ap.add_argument("--output", type=Path, default=Path("docs/history/wandb"))
    ap.add_argument("--limit", type=int, default=None, help="cap runs per project (debug)")
    args = ap.parse_args()

    api = wandb.Api()
    args.output.mkdir(parents=True, exist_ok=True)
    index = []

    for project in args.project:
        proj_dir = args.output / project
        runs = list(api.runs(f"{args.entity}/{project}"))
        if args.limit:
            runs = runs[: args.limit]
        print(f"=== {args.entity}/{project} ({len(runs)} runs) ===")
        for run in runs:
            print(f"  {run.id}  {run.state:>10}  {run.name!r}")
            try:
                meta = export_run(run, proj_dir / run.id)
                meta["project"] = project
                index.append(meta)
            except Exception as exc:
                print(f"    ! failed: {exc}")

    (args.output / "index.json").write_text(json.dumps(index, indent=2, default=str))

    cols = ["project", "id", "name", "state", "created_at", "n_steps", "url", "tags", *DIGEST_METRICS]
    with (args.output / "summary.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in sorted(index, key=lambda x: x["created_at"]):
            row = dict(r)
            row["tags"] = ",".join(row.get("tags") or [])
            w.writerow(row)

    print(f"\n{len(index)} runs exported → {args.output}/{{index.json,summary.csv}}")


if __name__ == "__main__":
    main()
