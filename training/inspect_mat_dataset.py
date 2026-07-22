from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from external.Readin_MatlabStruct_toPython.PythonMatlabScript.MatToPy import MatToPySTD


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect a local MATLAB .mat dataset using the integrated reader utility")
    parser.add_argument("file_path", type=str, help="Path to a local .mat file")
    args = parser.parse_args()

    path = Path(args.file_path).expanduser().resolve()
    if not path.exists():
        print(f"MAT file not found: {path}")
        return 1

    reader = MatToPySTD()
    try:
        data = reader.import_data(str(path))
        if data is None:
            print("No data could be imported from the provided MAT file.")
            return 1

        print(f"Inspecting: {path}")
        print("Available keys:")
        for key in data.keys():
            print(f"- {key}")

        print("\nStructure and data types:")
        for key, value in data.items():
            if hasattr(value, "shape"):
                print(f"- {key}: shape={value.shape}, dtype={getattr(value, 'dtype', type(value).__name__)}")
            elif isinstance(value, dict):
                print(f"- {key}: dict with keys={list(value.keys())}")
            else:
                print(f"- {key}: type={type(value).__name__}")

        return 0
    except Exception as exc:
        print(f"Error inspecting MAT file: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
