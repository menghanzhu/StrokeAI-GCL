from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st


def render_dashboard(feature_row: dict[str, float], prediction: int) -> None:
    """Render a compact healthcare-style summary card for the latest frame."""
    st.subheader("Latest analysis")
    feature_frame = pd.DataFrame([feature_row])
    st.dataframe(feature_frame, use_container_width=True)

    status = "Anomaly detected" if prediction == -1 else "Within expected range"
    st.metric("Status", status)

    if prediction == -1:
        st.warning("The current posture pattern differs from the baseline. Consider a follow-up evaluation.")
    else:
        st.success("The observed pattern is consistent with the baseline model.")


def render_live_metrics(metrics: dict[str, Any]) -> None:
    """Render the continuously updated streaming metrics."""
    st.metric("Movement speed", f"{metrics.get('movement_speed', 0.0):.2f}")
    st.metric("Body center displacement", f"{metrics.get('body_center_displacement', 0.0):.3f}")
    st.metric("Posture change", f"{metrics.get('posture_change', 0.0):.3f}")
    st.metric("Body tilt", f"{metrics.get('body_tilt', 0.0):.2f}°")
    st.metric("Shoulder symmetry", f"{metrics.get('shoulder_symmetry', 0.0):.3f}")
    st.metric("Hip symmetry", f"{metrics.get('hip_symmetry', 0.0):.3f}")
    st.metric("Pose confidence", f"{metrics.get('pose_confidence', 0.0):.2f}")


def render_movement_chart(history: list[dict[str, Any]]) -> None:
    """Render a rolling chart for smoothed movement speed."""
    if not history:
        st.info("No movement history yet.")
        return

    chart_frame = pd.DataFrame(history)
    chart_frame = chart_frame.dropna(subset=["timestamp"])
    if chart_frame.empty:
        return

    chart_frame = chart_frame.sort_values("timestamp")
    chart_frame["timestamp"] = chart_frame["timestamp"].astype(float)
    chart_view = chart_frame[["timestamp", "smoothed_speed"]].copy()
    chart_view = chart_view.set_index("timestamp")
    st.line_chart(chart_view, use_container_width=True)


def render_recent_alerts(history: list[dict[str, Any]]) -> None:
    """Render recent monitoring alerts and reasons."""
    if not history:
        st.info("No alert history yet.")
        return

    for entry in history[-8:][::-1]:
        reasons = entry.get("reasons") or []
        status = entry.get("monitoring_status", "Monitoring")
        if reasons:
            st.warning(f"{status}: {', '.join(reasons)}")
        else:
            st.info(f"{status}: baseline remains stable")
