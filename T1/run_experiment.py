#!/usr/bin/env python3
"""
T1/run_experiment.py — CLI entry point for the T1 experiment pipeline.

This file is intentionally thin: it only handles argument parsing and
wires the components together.  All logic lives in ``T1/pipeline/``.

Usage
-----
    python T1/run_experiment.py [config_file]
    python T1/run_experiment.py --dry-run [config_file]
    python T1/run_experiment.py --quick-check [config_file]
    python T1/run_experiment.py --log-level DEBUG [config_file]

If *config_file* is omitted, ``T1/config.arg`` is used when present.
"""

import argparse
import logging
import os
import sys
import time
import traceback

# ── Path setup (must happen before pipeline imports) ──────────────────────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS_DIR)
for _p in (_ROOT, _THIS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from pipeline import config as cfg_module          # noqa: E402
from pipeline.discovery import discover_heuristics, discover_solvers  # noqa: E402
from pipeline.planner import plan_runs, print_plan  # noqa: E402
from pipeline.runner import execute_pipeline, run_quick_check  # noqa: E402
from utils.env_info import collect as collect_env  # noqa: E402


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="T1 reproducible experiment pipeline for biclustering.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python T1/run_experiment.py --quick-check
  python T1/run_experiment.py --dry-run
  python T1/run_experiment.py T1/config.arg
  python T1/run_experiment.py --log-level DEBUG T1/config.arg
""",
    )
    p.add_argument(
        "config", nargs="?", default=None,
        help="Path to config.arg (default: T1/config.arg)",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print planned runs without executing.",
    )
    p.add_argument(
        "--quick-check", action="store_true",
        help="Run minimal 5x5 validation pipeline.",
    )
    p.add_argument(
        "--log-level", default="INFO",
        help="Logging level: DEBUG, INFO, WARNING, ERROR (default: INFO)",
    )
    return p


def main() -> None:
    args = _build_arg_parser().parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ── Locate and parse config ────────────────────────────────────────────
    config_path = args.config
    if config_path is None:
        default_cfg = os.path.join(_THIS_DIR, "config.arg")
        if os.path.exists(default_cfg):
            config_path = default_cfg
        else:
            logging.warning("No config.arg found; all built-in defaults will be used.")

    raw = cfg_module.parse_file(config_path) if config_path else {}
    cfg = cfg_module.build(raw)

    # CLI flags override config file values
    if args.dry_run:
        cfg["dry_run"] = True
    if args.quick_check:
        cfg["quick_check"] = True

    if cfg["_assumptions"]:
        logging.info("Active assumptions: %s", cfg["_assumptions"])

    # ── Output paths ───────────────────────────────────────────────────────
    output_dir: str = cfg["output_dir"]
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(_ROOT, output_dir)
    timestamp = int(time.time())
    csv_path = os.path.join(output_dir, f"results_{timestamp}.csv")
    log_dir = os.path.join(output_dir, "logs")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    env_info = collect_env(_ROOT)
    logging.info("git=%s  python=%s", env_info["git_hash"], env_info["python_version"])

    # ── quick_check mode ───────────────────────────────────────────────────
    if cfg["quick_check"]:
        try:
            run_quick_check(cfg, csv_path, log_dir, env_info)
        except Exception as exc:
            diag = os.path.join(output_dir, "quick_check_diagnostic.txt")
            with open(diag, "w", encoding="utf-8") as fh:
                fh.write(f"quick_check FAILED\n\n{traceback.format_exc()}")
            logging.error("quick_check FAILED: %s  Diagnostic: %s", exc, diag)
            sys.exit(1)
        return

    # ── dry_run mode ───────────────────────────────────────────────────────
    if cfg["dry_run"]:
        all_solvers = discover_solvers(_ROOT)
        all_heuristics = discover_heuristics(_ROOT)
        runs = plan_runs(cfg, all_solvers, all_heuristics, _ROOT)
        print_plan(runs)
        return

    # ── Full pipeline ──────────────────────────────────────────────────────
    execute_pipeline(cfg, csv_path, log_dir, env_info)
    logging.info("Done. Results CSV: %s", csv_path)
    print(f"Done. Results: {csv_path}")


if __name__ == "__main__":
    main()
