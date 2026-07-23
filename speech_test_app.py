from __future__ import annotations

import time

import av
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer

from audio.realtime_audio_processor import (
    RealtimeAudioProcessor,
)


st.set_page_config(
    page_title="StrokeAI Speech Test",
    page_icon="🎤",
)


if "audio_processor" not in st.session_state:
    st.session_state.audio_processor = (
        RealtimeAudioProcessor(
            recording_seconds=6.0,
            minimum_recording_seconds=2.0,
        )
    )


processor: RealtimeAudioProcessor = (
    st.session_state.audio_processor
)


def audio_frame_callback(
    frame: av.AudioFrame,
) -> av.AudioFrame:
    """Pass each microphone frame to the shared processor."""
    return processor.process_frame(frame)


st.title("StrokeAI Speech Module Test")

st.info(
    'Please read aloud: "Today is a beautiful day."'
)

webrtc_ctx = webrtc_streamer(
    key="speech-test",
    mode=WebRtcMode.SENDRECV,
    audio_frame_callback=audio_frame_callback,
    media_stream_constraints={
        "video": False,
        "audio": True,
    },
    rtc_configuration={
        "iceServers": [
            {
                "urls": [
                    "stun:stun.l.google.com:19302",
                ]
            }
        ]
    },
    async_processing=False,
)

button_column_1, button_column_2 = st.columns(2)

with button_column_1:
    if st.button(
        "Start 6-second recording",
        type="primary",
        use_container_width=True,
        disabled=not webrtc_ctx.state.playing,
    ):
        processor.start_recording()
        st.rerun()

with button_column_2:
    if st.button(
        "Reset",
        use_container_width=True,
    ):
        processor.reset()
        st.rerun()


@st.fragment(run_every=0.5)
def render_audio_status() -> None:
    duration = processor.recorded_duration
    volume = processor.current_volume

    st.metric(
        "Recorded duration",
        f"{duration:.1f} seconds",
    )

    st.metric(
        "Microphone level",
        f"{volume:.4f}",
    )

    if processor.recording_active:
        st.warning(
            "Recording... Please keep speaking."
        )

        progress = min(
            1.0,
            duration / processor.recording_seconds,
        )

        st.progress(
            progress,
            text=(
                f"Recording {duration:.1f} / "
                f"{processor.recording_seconds:.1f} seconds"
            ),
        )

    elif processor.analysis_in_progress:
        st.info("Analysing your recording...")

    error = processor.latest_error

    if error:
        st.error(error)

    result = processor.latest_result

    if result:
        score = float(
            result["difference_score"]
        )

        st.success("Speech analysis complete.")

        result_col1, result_col2 = st.columns(2)

        with result_col1:
            st.metric(
                "Speech-pattern difference score",
                f"{score:.1f}%",
            )

        with result_col2:
            st.metric(
                "Result",
                result["result_level"],
            )

        st.caption(
            result["important_notice"]
        )

        with st.expander(
            "Show speech timing details"
        ):
            features = result["features"]

            st.write(
                "Total duration:",
                f"{features['total_duration']:.2f}s",
            )
            st.write(
                "Voiced duration:",
                f"{features['voiced_duration']:.2f}s",
            )
            st.write(
                "Voiced ratio:",
                f"{features['voiced_ratio']:.2f}",
            )
            st.write(
                "Speech start delay:",
                f"{features['speech_start_delay']:.2f}s",
            )
            st.write(
                "Pause count:",
                int(features["pause_count"]),
            )
            st.write(
                "Longest pause:",
                f"{features['longest_pause']:.2f}s",
            )


render_audio_status()