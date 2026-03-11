# T1/pipeline — internal modules for the T1 experiment pipeline.
#
# Module map
# ----------
# config.py    — parse and build the key=value configuration
# discovery.py — discover and resolve solvers / heuristics
# metrics.py   — objective / area / density / gap computation
# io.py        — thread-safe CSV row append + JSON log writing
# executor.py  — execute one exact-solver or heuristic run
# planner.py   — build the flat run plan (dry-run support)
# runner.py    — orchestrate the full pipeline + quick_check
