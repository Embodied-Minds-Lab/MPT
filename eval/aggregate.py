import argparse
import csv
import json
import math
import os
import sys
from collections import OrderedDict
from typing import Any, Dict, Iterable, List, Optional, Tuple


def load_dumps(paths: Iterable[str]) -> List[Dict[str, Any]]:
    recs = []
    for p in paths:
        with open(p) as f:
            for ln, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    recs.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"WARNING: {p}:{ln}: skipped unparsable line ({e})", file=sys.stderr)
    return recs


def get_metric(rec: Dict[str, Any], name: str) -> Optional[float]:
    cur: Any = rec
    for part in name.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    if isinstance(cur, bool):
        return float(cur)
    if isinstance(cur, (int, float)):
        v = float(cur)
        return v if math.isfinite(v) else None
    return None


def instance_id(rec: Dict[str, Any], key: str) -> Optional[str]:
    inst = rec.get("instance") or {}
    v = inst.get(key, rec.get(key))
    return None if v is None else str(v)


def resolve_run_field(recs: List[Dict[str, Any]], run_field: str) -> str:
    if run_field in ("run", "rep"):
        return run_field
    runs = {r.get("run", 0) for r in recs}
    reps = {r.get("rep", 0) for r in recs}
    if len(runs) > 1 and len(reps) > 1:
        return "run+rep"
    return "run" if len(runs) > 1 else "rep"


def run_index(rec: Dict[str, Any], field: str):
    if field == "run+rep":
        return (int(rec.get("run", 0)), int(rec.get("rep", 0)))
    return int(rec.get(field, 0))


def _label(idx) -> str:
    return f"{idx[0]}_{idx[1]}" if isinstance(idx, tuple) else str(idx)


def _mean(xs: List[float]) -> Optional[float]:
    return sum(xs) / len(xs) if xs else None


def _se(xs: List[float]) -> Optional[float]:
    n = len(xs)
    if n < 2:
        return None
    m = sum(xs) / n
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return math.sqrt(var) / math.sqrt(n)


def aggregate(recs: List[Dict[str, Any]], key: str, metrics: List[str], run_field: str = "auto"
              ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Returns (per-instance rows, summary). See the module docstring for the definitions."""
    field = resolve_run_field(recs, run_field)
    # dedup by (instance, run index): the last record wins
    table: "OrderedDict[str, OrderedDict[Any, Dict[str, Any]]]" = OrderedDict()
    skipped = 0
    replaced = 0
    for r in recs:
        iid = instance_id(r, key)
        if iid is None:
            skipped += 1
            continue
        by_run = table.setdefault(iid, OrderedDict())
        idx = run_index(r, field)
        if idx in by_run:
            replaced += 1
        by_run[idx] = r
    if skipped:
        print(f"WARNING: {skipped} records have no instance key '{key}' and were skipped", file=sys.stderr)
    if replaced:
        print(f"WARNING: {replaced} duplicate (instance, run) records were replaced by a later record "
              f"(run index = {field})", file=sys.stderr)

    rows = []
    for iid, by_run in table.items():
        row: Dict[str, Any] = {"instance_id": iid, "n_runs": len(by_run)}
        for m in metrics:
            vals = OrderedDict((idx, get_metric(rec, m)) for idx, rec in sorted(by_run.items()))
            present = [v for v in vals.values() if v is not None]
            row[f"{m}_mean"] = _mean(present)
            for idx, v in vals.items():
                row[f"{m}_run{_label(idx)}"] = v
        rows.append(row)

    n_runs = [row["n_runs"] for row in rows]
    summary: Dict[str, Any] = {"key": key, "run_field": field, "n_instances": len(rows),
                               "n_duplicates_replaced": replaced,
                               "runs_per_instance": {"min": min(n_runs) if n_runs else 0,
                                                     "max": max(n_runs) if n_runs else 0},
                               "metrics": {}}
    for m in metrics:
        means = [row[f"{m}_mean"] for row in rows if row[f"{m}_mean"] is not None]
        pooled = [v for by_run in table.values() for rec in by_run.values()
                  for v in [get_metric(rec, m)] if v is not None]
        summary["metrics"][m] = {"mean": _mean(means), "se": _se(means), "n_instances": len(means),
                                 "runs_per_instance": summary["runs_per_instance"],
                                 "pooled_mean": _mean(pooled), "n_episodes": len(pooled)}
    return rows, summary


def write_outputs(rows: List[Dict[str, Any]], summary: Dict[str, Any], out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    cols: List[str] = []
    for row in rows:
        for c in row:
            if c not in cols:
                cols.append(c)
    with open(os.path.join(out_dir, "per_instance.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in rows:
            w.writerow({c: ("" if row.get(c) is None else row.get(c)) for c in cols})
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)


def _fmt(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{x:.4g}"


def markdown(summary: Dict[str, Any]) -> str:
    rpi = summary["runs_per_instance"]
    lines = ["| metric | mean ± SE (n_instances) | pooled mean | runs/instance (min-max) |",
             "|---|---|---|---|"]
    for m, s in summary["metrics"].items():
        lines.append(f"| {m} | {_fmt(s['mean'])} ± {_fmt(s['se'])} ({s['n_instances']}) | "
                     f"{_fmt(s['pooled_mean'])} | {rpi['min']}-{rpi['max']} |")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", action="append", required=True, help="per-episode JSONL (repeatable)")
    ap.add_argument("--key", required=True, help="instance key: seed | episode_id | qid")
    ap.add_argument("--metrics", required=True, help="comma-separated metric names / dotted paths")
    ap.add_argument("--out", default=None, help="directory for per_instance.csv and summary.json")
    ap.add_argument("--run-field", default="auto", choices=["auto", "run", "rep"])
    ap.add_argument("--expect-instances", type=int, default=None,
                    help="warn when the number of instances differs from this")
    a = ap.parse_args(argv)
    recs = load_dumps(a.dump)
    if not recs:
        sys.exit(f"no records in {a.dump}")
    metrics = [m.strip() for m in a.metrics.split(",") if m.strip()]
    rows, summary = aggregate(recs, a.key, metrics, a.run_field)
    rpi = summary["runs_per_instance"]
    if rpi["min"] != rpi["max"]:
        short = [r["instance_id"] for r in rows if r["n_runs"] < rpi["max"]]
        print(f"WARNING: unequal runs per instance ({rpi['min']}-{rpi['max']}); "
              f"fewer runs for: {', '.join(short[:20])}{' ...' if len(short) > 20 else ''}", file=sys.stderr)
    if a.expect_instances is not None and summary["n_instances"] != a.expect_instances:
        print(f"WARNING: {summary['n_instances']} instances, expected {a.expect_instances}", file=sys.stderr)
    for m, s in summary["metrics"].items():
        if s["n_episodes"] == 0:
            print(f"WARNING: metric '{m}' was not found in any record", file=sys.stderr)
    if a.out:
        write_outputs(rows, summary, a.out)
        print(f"wrote {os.path.join(a.out, 'per_instance.csv')} and {os.path.join(a.out, 'summary.json')}")
    print(f"{summary['n_instances']} instances (key={a.key}, run index={summary['run_field']}), "
          f"{len(recs)} records read")
    print(markdown(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
