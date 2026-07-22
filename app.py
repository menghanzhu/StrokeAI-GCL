from __future__ import annotations

import logging
from typing import Any

import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer

from camera.video_processor import VideoProcessor
from dashboard.dashboard import (
    render_live_metrics,
    render_movement_chart,
    render_recent_alerts,
)
from utils.logging_utils import configure_logging
from utils.pipeline import StrokeAIPipeline


configure_logging(logging.INFO)
logger = logging.getLogger("strokeai.app")

st.set_page_config(
    page_title="StrokeAI",
    page_icon="🧠",
    layout="wide",
)


def init_state() -> None:
    """Initialize ordinary Streamlit session state."""
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = StrokeAIPipeline()

    if "history" not in st.session_state:
        st.session_state.history = []

    if "latest_metrics" not in st.session_state:
        st.session_state.latest_metrics = None


@st.fragment(run_every=1.0)
def render_realtime_panels(webrtc_ctx: Any) -> None:
    """Refresh metrics and history once per second."""
    processor = webrtc_ctx.video_processor

    metrics: dict[str, Any] | None = None
    history: list[dict[str, Any]] = []

    if processor is not None:
        metrics = processor.latest_metrics
        history = list(processor.history_records)

    left_column, middle_column, right_column = st.columns(
        [1.0, 1.1, 0.95],
        gap="large",
    )

    with left_column:
        st.subheader("Live camera")

        if webrtc_ctx.state.playing:
            st.success(
                "Live annotated video is displayed in the camera panel above."
            )
            st.caption(
                "Movement features are extracted in memory. "
                "Raw video is not saved."
            )
        else:
            st.info("Press START above to begin monitoring.")

    with middle_column:
        st.subheader("Current metrics")

        if metrics is not None:
            render_live_metrics(metrics)

        elif webrtc_ctx.state.playing:
            st.info(
                "Waiting for valid pose data. "
                "Keep your head, shoulders, torso and hips visible."
            )

        else:
            st.info("Start monitoring to begin live analysis.")

    with right_column:
        st.subheader("Monitoring status")

        if metrics is None:
            if webrtc_ctx.state.playing:
                st.metric("Status", "Initializing")
            else:
                st.metric("Status", "Stopped")

            st.metric("Risk state", "Not available")
            st.metric("Pose confidence", "N/A")
            st.metric("Risk score", "N/A")

        else:
            status = metrics.get(
                "monitoring_status",
                "Not available",
            )
            risk_state = metrics.get(
                "risk_state",
                "Not available",
            )
            pose_confidence = metrics.get(
                "pose_confidence",
                0.0,
            )
            risk_score = metrics.get("risk_score")

            st.metric("Status", status)
            st.metric("Risk state", risk_state)

            st.metric(
                "Pose confidence",
                f"{pose_confidence:.2f}",
            )

            st.metric(
                "Risk score",
                "N/A"
                if risk_score is None
                else f"{risk_score:.1f}",
            )

            reasons = metrics.get("reasons", [])

            if reasons:
                st.warning(
                    "Observed signals: "
                    + ", ".join(reasons)
                )

            elif status == "Calibrating":
                st.caption(
                    "The system is building your "
                    "personal movement baseline."
                )

            elif status == "Monitoring":
                st.success(
                    "Monitoring is active. "
                    "No significant anomaly detected."
                )

            elif status == "No person detected":
                st.info(
                    "No person detected. "
                    "Please move into the camera view."
                )

    st.divider()
    st.subheader("Rolling movement chart")

    if history:
        render_movement_chart(history)

    elif webrtc_ctx.state.playing:
        st.info(
            "Collecting movement samples. "
            "The chart will appear shortly."
        )

    else:
        st.info(
            "Start monitoring to collect movement history."
        )

    st.subheader("Recent alert log")

    if history:
        render_recent_alerts(history)
    else:
        st.info("No alerts recorded.")


def main() -> None:
    """Render the StrokeAI dashboard."""
    init_state()

    pipeline: StrokeAIPipeline = st.session_state.pipeline

    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(
                135deg,
                #f4f9ff 0%,
                #eef6ff 100%
            );
        }

        .block-container {
            padding-top: 1rem;
            padding-bottom: 2rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("StrokeAI")

    st.caption(
        "Real-time posture and movement monitoring "
        "using browser webcam input"
    )

    st.caption(
        "Research prototype only — this system does not diagnose stroke."
    )

    st.caption(
        "Video is processed in memory. Raw video frames are not saved."
    )

    with st.sidebar:
        st.header("Controls")

        st.info(
            "Use the START and STOP controls in the "
            "video panel to control real-time monitoring."
        )

        st.markdown(
            """
            **Recommended camera position**

            - Keep your head and shoulders visible
            - Step back so your torso and hips are visible
            - Use good lighting
            - Face the camera directly
            """
        )

        st.divider()

        with st.expander(
            "Fallback snapshot test",
            expanded=False,
        ):
            st.caption(
                "Use this only if the live browser camera is unavailable."
            )

            if st.button(
                "Capture snapshot",
                key="capture_snapshot",
                use_container_width=True,
            ):
                run_capture(pipeline)

            if st.button(
                "Release fallback camera",
                key="release_fallback_camera",
                use_container_width=True,
            ):
                release_camera(pipeline)

    st.subheader("Live browser camera")

    webrtc_ctx = webrtc_streamer(
        key="strokeai-live-monitor",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=VideoProcessor,
        media_stream_constraints={
            "video": True,
            "audio": False,
        },
        async_processing=False,
    )

    if webrtc_ctx.state.playing:
        st.success(
            "Live monitoring is active. "
            "Metrics refresh once per second."
        )
    else:
        st.info(
            "Press START in the video panel to begin monitoring."
        )

    render_realtime_panels(webrtc_ctx)

    with st.expander(
        "Fallback snapshot results",
        expanded=False,
    ):
        render_snapshot_results()


def run_capture(
    pipeline: StrokeAIPipeline,
) -> None:
    """Run one fallback snapshot analysis."""
    try:
        result = pipeline.process_frame()

    except (RuntimeError, ValueError) as exc:
        logger.exception(
            "Fallback capture failed: %s",
            exc,
        )
        st.error(str(exc))
        return

    status = result.get("status")

    if status == "error":
        st.error(
            result.get(
                "message",
                "Camera processing failed.",
            )
        )
        return

    st.session_state.snapshot_frame = result.get("frame")
    st.session_state.snapshot_annotated = result.get(
        "annotated_frame"
    )

    if status == "no_pose":
        st.session_state.snapshot_message = result.get(
            "message",
            "No pose detected.",
        )
        st.session_state.snapshot_metrics = None
        st.warning(st.session_state.snapshot_message)
        return

    st.session_state.snapshot_message = (
        "Fallback snapshot analyzed successfully."
    )

    st.session_state.snapshot_metrics = result.get(
        "metrics"
    )

    st.session_state.history = result.get(
        "history",
        [],
    )

    st.success(
        "Fallback snapshot analyzed successfully."
    )


def render_snapshot_results() -> None:
    """Render the latest fallback snapshot result."""
    frame = st.session_state.get("snapshot_frame")
    annotated = st.session_state.get(
        "snapshot_annotated"
    )
    metrics = st.session_state.get(
        "snapshot_metrics"
    )
    message = st.session_state.get(
        "snapshot_message"
    )

    if frame is None and annotated is None:
        st.info("No fallback snapshot has been captured.")
        return

    if message:
        st.caption(message)

    first_column, second_column = st.columns(2)

    with first_column:
        st.markdown("**Original frame**")

        if frame is not None:
            st.image(
                frame,
                channels="BGR",
                width="stretch",
            )

    with second_column:
        st.markdown("**Pose analysis**")

        if annotated is not None:
            st.image(
                annotated,
                channels="BGR",
                width="stretch",
            )

    if metrics is not None:
        st.markdown("**Snapshot metrics**")
        render_live_metrics(metrics)


def release_camera(
    pipeline: StrokeAIPipeline,
) -> None:
    """Release fallback OpenCV camera resources."""
    pipeline.release_camera()

    for key in (
        "snapshot_frame",
        "snapshot_annotated",
        "snapshot_metrics",
        "snapshot_message",
    ):
        st.session_state.pop(key, None)

    st.session_state.history = []

    st.success("Fallback camera released.")


if __name__ == "__main__":
    main()