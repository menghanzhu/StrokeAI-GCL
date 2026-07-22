from __future__ import annotations

from typing import Final

FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    "body_center_x",
    "body_center_y",
    "body_tilt_angle",
    "head_tilt_angle",
    "shoulder_symmetry",
    "hip_symmetry",
    "knee_angle",
    "stride_length",
    "walking_speed",
)

LIVE_FEATURE_COLUMNS: Final[tuple[str, ...]] = FEATURE_COLUMNS
OFFLINE_FEATURE_COLUMNS: Final[tuple[str, ...]] = FEATURE_COLUMNS


def get_feature_columns() -> list[str]:
    """Return the shared feature column names as a mutable list."""
    return list(FEATURE_COLUMNS)
