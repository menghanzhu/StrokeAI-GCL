from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.stroke_gait_adapter import StrokeGaitAdapter


def main() -> int:
    parser = argparse.ArgumentParser(description="Load local stroke gait MAT files and export a CSV dataset")
    parser.add_argument("--input", type=str, default="data/raw", help="Directory containing .mat files")
    parser.add_argument("--output", type=str, default="data/processed/stroke_gait.csv", help="Output CSV path")
    args = parser.parse_args()

    adapter = StrokeGaitAdapter(data_dir=args.input)
    mat_files = adapter.discover_mat_files()
    if not mat_files:
        print("No .mat files found in the provided input directory.")
        return 1

    combined_frames = []
    for mat_path in mat_files:
        try:
            adapter.inspect_file(mat_path)
            frame = adapter.load_to_dataframe(mat_path)
            if not frame.empty:
                combined_frames.append(frame)
        except Exception as exc:
            print(f"Skipping {mat_path}: {exc}")
            continue

    if not combined_frames:
        print("No usable datasets could be converted from the provided MAT files.")
        return 1

    combined_df = combined_frames[0]
    for frame in combined_frames[1:]:
        combined_df = combined_df.merge(frame, how="outer", left_index=True, right_index=True)

    combined_df = combined_df.reset_index(drop=True)
    adapter.export_dataframe(combined_df, args.output)
    print(f"Processed dataset saved to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
