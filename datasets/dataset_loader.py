from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat

logger = logging.getLogger("strokeai.datasets")


class DatasetLoader:
    """Load and summarize MATLAB datasets from a raw-data directory."""

    def __init__(self) -> None:
        self._logger = logger

    def load_datasets(self, data_dir: str | Path) -> list[dict[str, Any]]:
        """Read all .mat files from a directory and return dataset summaries.

        The loader intentionally does not parse variable payloads yet. It validates
        that the directory exists, discovers MAT files, and returns a summary for
        each file. Missing or unreadable files are surfaced with clear errors.
        """
        path = Path(data_dir).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Data directory not found: {path}")
        if not path.is_dir():
            raise NotADirectoryError(f"Data path is not a directory: {path}")

        mat_files = sorted(path.rglob("*.mat"))
        if not mat_files:
            self._logger.warning("No .mat files found under %s", path)
            return []

        summaries: list[dict[str, Any]] = []
        for mat_file in mat_files:
            summary = self._summarize_mat_file(mat_file)
            summaries.append(summary)
        return summaries

    def _summarize_mat_file(self, file_path: Path) -> dict[str, Any]:
        if not file_path.exists():
            return {
                "status": "error",
                "path": str(file_path),
                "error": f"File not found: {file_path}",
            }

        if not file_path.is_file():
            return {
                "status": "error",
                "path": str(file_path),
                "error": f"Not a regular file: {file_path}",
            }

        try:
            matlab_data = loadmat(file_path)
        except Exception as exc:  # pragma: no cover - defensive path
            return {
                "status": "error",
                "path": str(file_path),
                "error": f"Unable to read MATLAB file: {exc}",
            }

        keys = [key for key in matlab_data.keys() if not key.startswith("__")]
        structure = []
        for key in keys:
            value = matlab_data[key]
            if isinstance(value, np.ndarray):
                shape = tuple(int(dim) for dim in value.shape)
                structure.append({"name": key, "shape": shape, "dtype": str(value.dtype)})
            else:
                structure.append({"name": key, "type": type(value).__name__})

        return {
            "status": "ok",
            "path": str(file_path),
            "keys": keys,
            "structure": structure,
        }
