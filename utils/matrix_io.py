"""
utils/matrix_io.py — Load binary matrices from CSV files.

Features
--------
- Auto-detects the field separator (comma, semicolon, tab, space).
- Auto-detects a header row (non-numeric first line → skipped).
- Returns a NumPy int array.

Reusable across any tool that loads 0/1 binary matrices.
"""

import csv
import logging
from typing import Optional

import numpy as np


def load_csv_matrix(path: str, sep: Optional[str] = None) -> np.ndarray:
    """
    Load a binary (0/1) matrix from a CSV file.

    Auto-detects the field separator and header presence when *sep* is None.
    Skips rows that cannot be converted to integers.

    Parameters
    ----------
    path : str
        Path to the CSV file.
    sep : str, optional
        Field separator.  If None, auto-detected from the first line.

    Returns
    -------
    numpy.ndarray of shape (m, n), dtype int.
    Returns a (0, 0) array when no data rows are found.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    """
    with open(path, "r", encoding="utf-8") as f:
        first_line = f.readline().strip()

    # ── Auto-detect separator ────────────────────────────────────────────────
    detected_sep: str = sep or ","
    if sep is None:
        for candidate in (",", ";", "\t", " "):
            if candidate in first_line:
                detected_sep = candidate
                break

    # ── Auto-detect header ───────────────────────────────────────────────────
    has_header = False
    try:
        [float(v.strip()) for v in first_line.split(detected_sep) if v.strip()]
    except ValueError:
        has_header = True

    if detected_sep != "," or has_header:
        logging.info(
            "matrix_io: auto-detected sep=%r, header=%s for %s",
            detected_sep,
            has_header,
            path,
        )

    # ── Read rows ────────────────────────────────────────────────────────────
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=detected_sep)
        if has_header:
            next(reader)
        for row in reader:
            if not row:
                continue
            try:
                rows.append([int(float(v.strip())) for v in row if v.strip()])
            except ValueError:
                logging.warning("matrix_io: skipping non-numeric row in %s", path)
                continue

    if not rows:
        return np.zeros((0, 0), dtype=int)

    return np.array(rows, dtype=int)
