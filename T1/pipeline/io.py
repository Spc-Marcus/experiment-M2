"""
T1/pipeline/io.py — Thread-safe CSV and JSON I/O for experiment results.

The CSV header is defined here as the single source of truth; every other
module that needs column names imports ``CSV_HEADER`` from here.

Public API
----------
CSV_HEADER        : list[str]  — the canonical column order
init_csv(path)    — create / overwrite the results CSV with the header
append_csv_row(path, row) — append one result row (thread-safe)
write_json_log(log_dir, run_id, data) -> str  — write a per-run JSON log
"""

import csv
import json
import os
import threading
from typing import Any, Dict

# ── Canonical CSV header ───────────────────────────────────────────────────────
CSV_HEADER = [
    "instance_id", "m", "n", "base_dens", "gamma", "solver",
    "seed", "heuristic", "time", "status", "objective", "area",
    "density", "gap",
]

# ── Thread-safety for concurrent CSV writes ───────────────────────────────────
_csv_lock = threading.Lock()


def init_csv(csv_path: str) -> None:
    """Create (or overwrite) the results CSV and write the standard header."""
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=CSV_HEADER).writeheader()


def append_csv_row(csv_path: str, row: Dict[str, Any]) -> None:
    """Append one result row to *csv_path* (thread-safe via module-level lock)."""
    with _csv_lock:
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=CSV_HEADER).writerow(row)


def write_json_log(log_dir: str, run_id: str, data: Dict[str, Any]) -> str:
    """
    Serialise *data* as pretty-printed JSON in ``<log_dir>/<safe_run_id>.json``.

    Characters in *run_id* that are not alphanumeric or in ``-_.`` are
    replaced with ``_`` to produce a safe filename.

    Parameters
    ----------
    log_dir : str  — directory to write to (created if absent)
    run_id  : str  — unique run identifier
    data    : dict — payload to serialise (``default=str`` handles unknowns)

    Returns
    -------
    str : absolute path of the written file
    """
    os.makedirs(log_dir, exist_ok=True)
    safe_id = "".join(
        c if c.isalnum() or c in ("-", "_", ".") else "_" for c in run_id
    )
    path = os.path.join(log_dir, f"{safe_id}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    return path
