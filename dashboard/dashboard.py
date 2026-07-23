from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st


def render_dashboard(
    feature_row: dict[str, float],
    prediction: int,
) -> None:
    """Render a compact summary for a single prediction."""
    st.subheader("Latest analysis")

    feature_frame = pd.DataFrame([feature_row])
    st.dataframe(
        feature_frame,
        use_container_width=True,
    )

    status = (
        "Movement difference detected"
        if prediction == -1
        else "Within expected range"
    )

    st.metric("Status", status)

    if prediction == -1:
        st.warning(
            "The current posture pattern differs from the "
            "session baseline. This is not a medical diagnosis."
        )
    else:
        st.success(
            "The observed pattern is consistent with the "
            "current session baseline."
        )


def render_live_metrics(
    metrics: dict[str, Any],
) -> None:
    """Render the latest technical movement metrics."""
    if not metrics:
        st.info("No live metrics are available.")
        return

    movement_speed = _as_float(
        metrics.get("movement_speed")
    )
    body_tilt = _as_float(
        metrics.get("body_tilt")
    )
    posture_change = _as_float(
        metrics.get("posture_change")
    )
    pose_confidence = _as_float(
        metrics.get("pose_confidence")
    )
    asymmetry = _as_float(
        metrics.get(
            "current_asymmetry",
            max(
                _as_float(
                    metrics.get("shoulder_symmetry")
                ),
                _as_float(
                    metrics.get("hip_symmetry")
                ),
            ),
        )
    )
    risk_score = _as_float(
        metrics.get("risk_score")
    )

    first_column, second_column, third_column = (
        st.columns(3)
    )

    with first_column:
        st.metric(
            "Movement speed",
            f"{movement_speed:.4f}",
        )
        st.metric(
            "Pose confidence",
            f"{pose_confidence:.2f}",
        )

    with second_column:
        st.metric(
            "Body tilt",
            f"{body_tilt:.1f}°",
        )
        st.metric(
            "Posture change",
            f"{posture_change:.2f}",
        )

    with third_column:
        st.metric(
            "Left-right difference",
            f"{asymmetry:.3f}",
        )
        st.metric(
            "Attention score",
            f"{risk_score:.0f}/100",
        )

    monitoring_status = str(
        metrics.get(
            "monitoring_status",
            "Not available",
        )
    )
    risk_state = str(
        metrics.get(
            "risk_state",
            "Not available",
        )
    )

    status_column, risk_column = st.columns(2)

    with status_column:
        st.write(
            "**Monitoring status:**",
            monitoring_status,
        )

    with risk_column:
        st.write(
            "**Risk state:**",
            risk_state,
        )

    if "both_arms_raised" in metrics:
        arm_column, difference_column = st.columns(2)

        with arm_column:
            st.metric(
                "Both arms raised",
                (
                    "Yes"
                    if bool(
                        metrics.get(
                            "both_arms_raised",
                            False,
                        )
                    )
                    else "No"
                ),
            )

        with difference_column:
            st.metric(
                "Arm height difference",
                (
                    f"{_as_float(
                        metrics.get(
                            'arm_height_difference',
                            1.0,
                        )
                    ):.3f}"
                ),
            )

    reasons = list(
        metrics.get("reasons", []) or []
    )

    if reasons:
        st.write("**Recorded reasons:**")

        for reason in reasons:
            st.write(f"- {reason}")


def render_movement_chart(
    history: list[dict[str, Any]],
) -> None:
    """Render movement values collected during the session."""
    if not history:
        st.info("No movement history is available.")
        return

    rows: list[dict[str, Any]] = []

    for index, record in enumerate(history):
        timestamp = record.get(
            "timestamp",
            record.get("time", index),
        )

        rows.append(
            {
                "Sample": index + 1,
                "Timestamp": timestamp,
                "Movement speed": _as_float(
                    record.get("movement_speed")
                ),
                "Body tilt": _as_float(
                    record.get("body_tilt")
                ),
                "Left-right difference": _as_float(
                    record.get(
                        "current_asymmetry",
                        max(
                            _as_float(
                                record.get(
                                    "shoulder_symmetry"
                                )
                            ),
                            _as_float(
                                record.get(
                                    "hip_symmetry"
                                )
                            ),
                        ),
                    )
                ),
                "Attention score": _as_float(
                    record.get("risk_score")
                ),
            }
        )

    frame = pd.DataFrame(rows)

    chart_columns = [
        "Movement speed",
        "Left-right difference",
    ]

    st.line_chart(
        frame.set_index("Sample")[chart_columns],
        use_container_width=True,
    )

    with st.expander(
        "Show movement history table",
        expanded=False,
    ):
        st.dataframe(
            frame,
            use_container_width=True,
            hide_index=True,
        )


def render_recent_alerts(
    history: list[dict[str, Any]],
    maximum_alerts: int = 8,
) -> None:
    """Render the most recent anomaly records."""
    if not history:
        st.info("No alert history is available.")
        return

    alerts: list[dict[str, Any]] = []

    for index, record in enumerate(history):
        risk_state = str(
            record.get("risk_state", "")
        )
        reasons = list(
            record.get("reasons", []) or []
        )
        risk_score = _as_float(
            record.get("risk_score")
        )

        is_alert = (
            risk_state == "Possible anomaly"
            or risk_score >= 45.0
            or bool(reasons)
        )

        if not is_alert:
            continue

        alerts.append(
            {
                "Sample": index + 1,
                "Status": (
                    risk_state
                    or "Movement difference"
                ),
                "Attention score": risk_score,
                "Reasons": (
                    "; ".join(str(reason) for reason in reasons)
                    if reasons
                    else "No specific reason recorded"
                ),
            }
        )

    if not alerts:
        st.success(
            "No movement alerts were recorded."
        )
        return

    recent_alerts = alerts[-maximum_alerts:]
    recent_alerts.reverse()

    alert_frame = pd.DataFrame(recent_alerts)

    st.dataframe(
        alert_frame,
        use_container_width=True,
        hide_index=True,
    )


def _as_float(
    value: Any,
    default: float = 0.0,
) -> float:
    """Safely convert an unknown value to float."""
    try:
        if value is None:
            return default

        return float(value)
    except (TypeError, ValueError):
        return default