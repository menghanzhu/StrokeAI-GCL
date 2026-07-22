from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
from scipy.io import loadmat


logger = logging.getLogger("strokeai.datasets")


class StrokeGaitAdapter:
    """Load local stroke gait .mat files and convert usable movement data into a DataFrame."""

    def __init__(self, data_dir: str | Path | None = None) -> None:
        self.data_dir = Path(data_dir) if data_dir is not None else Path("data/raw")
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def discover_mat_files(self) -> list[Path]:
        """Return .mat files found under the configured data directory."""
        if not self.data_dir.exists():
            return []
        return sorted(self.data_dir.rglob("*.mat"))

    def inspect_file(self, file_path: str | Path) -> dict[str, Any]:
        """Print and return a summary of the available keys and dataset structure for a .mat file."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"MAT file not found: {path}")

        logger.info("Inspecting MAT file: %s", path)
        try:
            if h5py.is_hdf5(path):
                with h5py.File(path, "r") as handle:
                    return self._describe_h5(handle, path)

            matlab_data = loadmat(path)
            return self._describe_matlab(matlab_data, path)
        except Exception as exc:  # pragma: no cover - defensive path
            raise RuntimeError(f"Unable to inspect MAT file {path}: {exc}") from exc

    def load_to_dataframe(self, file_path: str | Path) -> pd.DataFrame:
        """Load a MAT file and convert usable movement data to a pandas DataFrame."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"MAT file not found: {path}")

        logger.info("Loading movement data from %s", path)
        try:
            if h5py.is_hdf5(path):
                with h5py.File(path, "r") as handle:
                    return self._load_from_h5(handle)

            matlab_data = loadmat(path)
            return self._load_from_matlab(matlab_data)
        except Exception as exc:  # pragma: no cover - defensive path
            raise RuntimeError(f"Unable to load MAT file {path}: {exc}") from exc

    def export_dataframe(self, dataframe: pd.DataFrame, output_path: str | Path | None = None) -> Path:
        """Export a DataFrame to CSV while preserving missing values as NaN."""
        output = Path(output_path or "data/processed/stroke_gait.csv")
        output.parent.mkdir(parents=True, exist_ok=True)
        dataframe.to_csv(output, index=False)
        logger.info("Exported processed dataset to %s", output)
        return output

    def _describe_h5(self, handle: h5py.File, path: Path) -> dict[str, Any]:
        def walk(name: str, obj: Any) -> None:
            if isinstance(obj, h5py.Dataset):
                structure[name] = {"shape": obj.shape, "dtype": str(obj.dtype)}
            elif isinstance(obj, h5py.Group):
                structure[name] = {"children": list(obj.keys())}

        structure: dict[str, Any] = {}
        handle.visititems(walk)
        print(f"[inspect] {path}")
        print("Available keys:")
        for key in handle.keys():
            print(f"- {key}")
        print("Dataset structure:")
        for key, value in structure.items():
            print(f"- {key}: {value}")
        return {"path": str(path), "keys": list(handle.keys()), "structure": structure}

    def _describe_matlab(self, matlab_data: dict[str, Any], path: Path) -> dict[str, Any]:
        print(f"[inspect] {path}")
        print("Available keys:")
        for key in matlab_data.keys():
            print(f"- {key}")
        structure: dict[str, Any] = {}
        for key, value in matlab_data.items():
            if isinstance(value, np.ndarray):
                structure[key] = {"shape": value.shape, "dtype": str(value.dtype)}
            else:
                structure[key] = {"type": type(value).__name__}
        print("Dataset structure:")
        for key, value in structure.items():
            print(f"- {key}: {value}")
        return {"path": str(path), "keys": list(matlab_data.keys()), "structure": structure}

    def _load_from_h5(self, handle: h5py.File) -> pd.DataFrame:
        records: list[dict[str, Any]] = []
        for key in handle.keys():
            dataset = handle[key]
            if isinstance(dataset, h5py.Dataset):
                values = np.array(dataset)
                if values.ndim == 1:
                    records.append({key: values.tolist()})
                else:
                    records.append({key: values})
        if not records:
            raise ValueError("No usable datasets found in the HDF5 file.")

        frame = pd.DataFrame(records)
        return frame.replace({np.nan: np.nan})

    def _load_from_matlab(self, matlab_data: dict[str, Any]) -> pd.DataFrame:
        usable_columns: dict[str, list[Any]] = {}
        for key, value in matlab_data.items():
            if key.startswith("__"):
                continue
            if isinstance(value, np.ndarray):
                if value.ndim == 0:
                    usable_columns[key] = [value.item()]
                elif value.ndim == 1:
                    usable_columns[key] = value.tolist()
                elif value.ndim == 2:
                    usable_columns[key] = [row.tolist() for row in value]
                else:
                    usable_columns[key] = [value.tolist()]
            else:
                usable_columns[key] = [value]

        if not usable_columns:
            raise ValueError("No usable variables were found in the MATLAB file.")

        return pd.DataFrame.from_dict(usable_columns, orient="columns")
