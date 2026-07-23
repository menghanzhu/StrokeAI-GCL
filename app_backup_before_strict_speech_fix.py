from __future__ import annotations

import json
import logging
import time
from typing import Any

import av
import streamlit as st
import streamlit.components.v1 as components
from streamlit_webrtc import WebRtcMode, webrtc_streamer

from audio.realtime_audio_processor import RealtimeAudioProcessor
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
    page_title="StrokeAI Wellness Check",
    page_icon="🧠",
    layout="wide",
)

TOTAL_STEPS = 5


def init_state() -> None:
    """Initialize Streamlit session state."""
    defaults: dict[str, Any] = {
        "check_started": False,
        "check_step": 0,
        "step_started_at": time.time(),
        "completed_steps": [],
        "spoken_messages": [],
        "latest_metrics": None,
        "history": [],
        "step_completed": False,
        "step_success_message": "",
        "correct_pose_started_at": None,
        "step_completed_at": None,
        "last_instruction_spoken_at": {},
        "instruction_repeat_seconds": 8.0,
        "completion_pause_seconds": 3.0,
        "step_timeout_seconds": 30.0,
        "skipped_steps": [],
        "last_anomaly_spoken_at": 0.0,
        "anomaly_voice_cooldown_seconds": 20.0,
        "active_anomaly_started_at": None,
        "anomaly_hold_seconds": 2.0,
        "snapshot_frame": None,
        "snapshot_annotated": None,
        "snapshot_metrics": None,
        "snapshot_message": None,
        "speech_result": None,
        "speech_recording_started": False,
        "speech_step_initialized": False,
        "speech_prompt_started_at": None,
        "speech_recording_ready_at": None,
        "speech_attempts": 0,
        "speech_prompt": "Today is a beautiful day.",
        "webrtc_desired_playing_state": None,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def get_audio_processor() -> RealtimeAudioProcessor:
    """Create and reuse the microphone processor for the current session."""
    if "audio_processor" not in st.session_state:
        st.session_state.audio_processor = RealtimeAudioProcessor(
            recording_seconds=6.0,
            minimum_recording_seconds=2.0,
        )

    return st.session_state.audio_processor


def build_audio_frame_callback(
    processor: RealtimeAudioProcessor,
):
    """Return the WebRTC callback that forwards microphone frames."""

    def audio_frame_callback(frame: av.AudioFrame) -> av.AudioFrame:
        return processor.process_frame(frame)

    return audio_frame_callback


def apply_accessible_style() -> None:
    """Apply a simple, high-contrast, older-adult-friendly interface."""
    st.markdown(
        """
        <style>
        .stApp {
            background: #FAFAF7;
        }

        .block-container {
            max-width: 1180px;
            padding-top: 1.5rem;
            padding-bottom: 3rem;
        }

        h1 {
            font-size: 46px !important;
            line-height: 1.2 !important;
            color: #26332D;
        }

        h2 {
            font-size: 34px !important;
            color: #26332D;
        }

        h3 {
            font-size: 28px !important;
            color: #26332D;
        }

        p, li, label {
            font-size: 21px !important;
            line-height: 1.55 !important;
        }

        div.stButton > button {
            min-height: 72px;
            font-size: 25px;
            font-weight: 700;
            border-radius: 18px;
            border: none;
        }

        div.stButton > button[kind="primary"] {
            background: #2E7D5B;
            color: white;
        }

        .welcome-card {
            max-width: 820px;
            margin: 3rem auto 2rem auto;
            padding: 3rem;
            text-align: center;
            background: white;
            border-radius: 28px;
            box-shadow: 0 10px 35px rgba(0, 0, 0, 0.08);
        }

        .welcome-icon {
            font-size: 74px;
            margin-bottom: 0.5rem;
        }

        .welcome-subtitle {
            font-size: 25px;
            color: #405048;
        }

        .privacy-note {
            margin-top: 1.5rem;
            color: #66736D;
            font-size: 18px !important;
        }

        .step-card {
            background: white;
            border-left: 8px solid #4B8F71;
            border-radius: 22px;
            padding: 2rem 2.2rem;
            margin: 1rem 0 1.5rem 0;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.06);
        }

        .step-number {
            color: #2E7D5B;
            font-size: 19px;
            font-weight: 700;
            margin-bottom: 0.5rem;
        }

        .instruction {
            font-size: 27px !important;
            font-weight: 650;
            color: #26332D;
        }

        .helper-text {
            color: #5E6B65;
            font-size: 20px !important;
        }

        .friendly-status {
            background: #EEF7F1;
            border: 2px solid #84B99B;
            border-radius: 20px;
            padding: 1.4rem;
            margin-bottom: 1rem;
        }

        .calibration-status {
            background: #FFF6DF;
            border: 2px solid #E6B75B;
            border-radius: 20px;
            padding: 1.4rem;
            margin-bottom: 1rem;
        }

        .completion-card {
            background: #EAF7EF;
            border: 3px solid #5BAA72;
            border-radius: 28px;
            padding: 2.5rem;
            text-align: center;
            margin: 1rem 0 2rem 0;
        }

        .care-card {
            background: #FFF5DF;
            border: 2px solid #E0A642;
            border-radius: 22px;
            padding: 1.6rem;
            margin-top: 1rem;
        }

        .emergency-card {
            background: #FDECEC;
            border: 3px solid #C94A4A;
            border-radius: 22px;
            padding: 1.6rem;
            margin-top: 1rem;
        }

        .simple-result {
            background: white;
            border-radius: 22px;
            padding: 1.5rem;
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.05);
        }

        .speech-card {
            background: #F4F0FF;
            border: 2px solid #9A83C7;
            border-radius: 22px;
            padding: 1.7rem;
            margin: 1rem 0;
        }

        .speech-sentence {
            font-size: 31px !important;
            font-weight: 750;
            text-align: center;
            color: #302548;
            padding: 1rem;
        }

        [data-testid="stMetricValue"] {
            font-size: 34px;
        }

        [data-testid="stSidebar"] {
            background: #F2F4F1;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def speak_message(
    text: str,
    message_key: str | None = None,
    minimum_interval_seconds: float = 5.0,
) -> None:
    """Speak through one shared browser speech engine across all reruns."""
    if not text.strip():
        return

    if message_key is None:
        message_key = text

    safe_text = json.dumps(text)
    safe_key = json.dumps(f"strokeai_voice_{message_key}")
    interval_ms = int(max(0.0, minimum_interval_seconds) * 1000)

    components.html(
        f"""
        <script>
        (() => {{
            /*
             * Every Streamlit HTML component is an iframe. Using the iframe's
             * own speech engine can produce two voices at the same time.
             * Always use the top application window so every prompt shares
             * one speech queue and one lock.
             */
            const host = window.parent;
            const synth = host.speechSynthesis;
            const storage = host.localStorage;
            const text = {safe_text};
            const storageKey = {safe_key};
            const minimumInterval = {interval_ms};
            const now = Date.now();

            if (!synth) {{
                return;
            }}

            const previousTime = Number(storage.getItem(storageKey) || "0");
            if (now - previousTime < minimumInterval) {{
                return;
            }}

            storage.setItem(storageKey, String(now));

            function choosePreferredVoice(voices) {{
                const preferredFemaleNames = [
                    "Samantha", "Ava", "Allison", "Susan",
                    "Victoria", "Karen", "Moira", "Tessa",
                    "Fiona", "Zoe", "Jenny", "Aria",
                    "Emma", "Sonia", "Serena", "Veena",
                    "Google UK English Female",
                    "Microsoft Aria", "Microsoft Jenny",
                    "Microsoft Zira", "Microsoft Sonia"
                ];

                for (const preferredName of preferredFemaleNames) {{
                    const match = voices.find((voice) =>
                        voice.name.toLowerCase().includes(
                            preferredName.toLowerCase()
                        )
                    );
                    if (match) return match;
                }}

                const englishVoice = voices.find((voice) =>
                    (voice.lang || "").toLowerCase().startsWith("en")
                );
                return englishVoice || voices[0] || null;
            }}

            function playSpeech() {{
                const voices = synth.getVoices();
                const selectedVoice = choosePreferredVoice(voices);

                /* One shared queue means cancel removes every older prompt. */
                synth.cancel();

                const utterance = new SpeechSynthesisUtterance(text);
                if (selectedVoice) {{
                    utterance.voice = selectedVoice;
                    utterance.lang = selectedVoice.lang || "en-US";
                }} else {{
                    utterance.lang = "en-US";
                }}
                utterance.rate = 0.88;
                utterance.pitch = 1.08;
                utterance.volume = 0.95;
                synth.speak(utterance);
            }}

            if (synth.getVoices().length > 0) {{
                setTimeout(playSpeech, 120);
            }} else {{
                let played = false;
                const playOnce = () => {{
                    if (played) return;
                    played = true;
                    playSpeech();
                }};
                synth.addEventListener("voiceschanged", playOnce, {{ once: true }});
                setTimeout(playOnce, 1200);
            }}
        }})();
        </script>
        """,
        height=0,
    )


def speak_once(
    message_key: str,
    text: str,
) -> None:
    """Speak a one-time message during the current assessment."""
    spoken_messages = list(
        st.session_state.spoken_messages
    )

    if message_key in spoken_messages:
        return

    spoken_messages.append(message_key)
    st.session_state.spoken_messages = spoken_messages

    speak_message(
        text=text,
        message_key=f"once_{message_key}",
        minimum_interval_seconds=3600.0,
    )

def get_friendly_anomaly_reason(reasons: list[str]) -> str:
    """Convert technical anomaly reasons into calm spoken language."""
    friendly_reasons = {
        "large body tilt change": "Your body position changed suddenly.",
        "sudden body-center vertical drop": (
            "We noticed a sudden downward movement."
        ),
        "prolonged low body position": (
            "Your body remained in a lower position."
        ),
        "large left/right asymmetry": (
            "We noticed a larger difference between your left and right sides."
        ),
        "unusually low movement relative to the session baseline": (
            "Your movement became lower than earlier in this check."
        ),
    }

    converted = [
        friendly_reasons[reason]
        for reason in reasons
        if reason in friendly_reasons
    ]

    if not converted:
        return "The current movement pattern is different from earlier."

    return " ".join(converted[:2])


def speak_anomaly_warning(
    metrics: dict[str, Any] | None,
) -> None:
    """Speak a calm warning only when an anomaly persists."""
    if metrics is None:
        st.session_state.active_anomaly_started_at = None
        return

    risk_state = str(metrics.get("risk_state", ""))
    reasons = list(metrics.get("reasons", []) or [])
    current_time = time.time()

    if risk_state != "Possible anomaly":
        st.session_state.active_anomaly_started_at = None
        return

    if st.session_state.active_anomaly_started_at is None:
        st.session_state.active_anomaly_started_at = current_time
        return

    anomaly_duration = (
        current_time
        - st.session_state.active_anomaly_started_at
    )

    if anomaly_duration < st.session_state.anomaly_hold_seconds:
        return

    time_since_last_warning = (
        current_time
        - st.session_state.last_anomaly_spoken_at
    )

    if (
        time_since_last_warning
        < st.session_state.anomaly_voice_cooldown_seconds
    ):
        return

    reason_text = get_friendly_anomaly_reason(reasons)

    speak_message(
        text=(
            "We noticed a movement difference. "
            + reason_text
            + " Please remain calm and continue only if you feel comfortable. "
            + "This system cannot provide a medical diagnosis."
        ),
        message_key="persistent_anomaly_warning",
        minimum_interval_seconds=float(
            st.session_state.anomaly_voice_cooldown_seconds
        ),
    )

    st.session_state.last_anomaly_spoken_at = current_time

def has_step_timed_out() -> bool:
    """Return True when the current step exceeds its time limit."""
    if st.session_state.step_completed:
        return False

    elapsed = time.time() - st.session_state.step_started_at
    return elapsed >= st.session_state.step_timeout_seconds


def retry_current_step() -> None:
    """Restart validation and spoken guidance for the current step."""
    current_step = st.session_state.check_step

    if current_step == 5:
        processor = st.session_state.get("audio_processor")
        if processor is not None:
            processor.reset()
        st.session_state.speech_recording_started = False
        st.session_state.speech_step_initialized = False
        st.session_state.speech_prompt_started_at = None
        st.session_state.speech_recording_ready_at = None
        st.session_state.speech_result = None
        st.session_state.speech_attempts += 1

    reset_step_validation()

    spoken_times = st.session_state.last_instruction_spoken_at
    spoken_times.pop(current_step, None)
    st.session_state.last_instruction_spoken_at = spoken_times

    speak_message(
        "Let us try this check again. "
        + get_step_voice_instruction(current_step)
    )


def skip_current_step() -> None:
    """Record the current step as incomplete and continue."""
    current_step = st.session_state.check_step

    if current_step not in st.session_state.skipped_steps:
        st.session_state.skipped_steps.append(current_step)

    if current_step == 5:
        processor = st.session_state.get("audio_processor")
        if processor is not None:
            processor.reset()
        st.session_state.speech_recording_started = False
        st.session_state.speech_step_initialized = False
        st.session_state.speech_prompt_started_at = None
        st.session_state.speech_recording_ready_at = None
        st.session_state.speech_result = None

    speak_message(
        "That is okay. We will move to the next check."
    )

    st.session_state.check_step = current_step + 1
    reset_step_validation()


def reset_step_validation() -> None:
    """Reset validation state for the current or next step."""
    st.session_state.step_completed = False
    st.session_state.step_success_message = ""
    st.session_state.correct_pose_started_at = None
    st.session_state.step_completed_at = None
    st.session_state.step_started_at = time.time()


def move_to_next_step() -> None:
    """Automatically advance to the next step."""
    current_step = st.session_state.check_step

    if current_step not in st.session_state.completed_steps:
        st.session_state.completed_steps.append(current_step)

    st.session_state.check_step = current_step + 1
    reset_step_validation()


def reset_check() -> None:
    """Reset the full guided check."""
    st.session_state.check_started = False
    st.session_state.check_step = 0
    st.session_state.completed_steps = []
    st.session_state.skipped_steps = []
    st.session_state.spoken_messages = []
    st.session_state.latest_metrics = None
    st.session_state.history = []
    st.session_state.last_instruction_spoken_at = {}
    st.session_state.last_anomaly_spoken_at = 0.0
    st.session_state.active_anomaly_started_at = None
    st.session_state.speech_result = None
    st.session_state.speech_recording_started = False
    st.session_state.speech_step_initialized = False
    st.session_state.speech_prompt_started_at = None
    st.session_state.speech_recording_ready_at = None
    st.session_state.speech_attempts = 0
    st.session_state.webrtc_desired_playing_state = None

    processor = st.session_state.get("audio_processor")
    if processor is not None:
        processor.reset()

    reset_step_validation()


def hold_condition(
    condition: bool,
    required_seconds: float,
) -> bool:
    """Return True only if a condition stays valid continuously."""
    if not condition:
        st.session_state.correct_pose_started_at = None
        return False

    if st.session_state.correct_pose_started_at is None:
        st.session_state.correct_pose_started_at = time.time()
        return False

    elapsed = time.time() - st.session_state.correct_pose_started_at
    return elapsed >= required_seconds


def mark_step_complete(
    success_message: str,
    voice_message: str,
) -> None:
    """Mark a step complete and play encouragement."""
    if st.session_state.step_completed:
        return

    st.session_state.step_completed = True
    st.session_state.step_success_message = success_message
    st.session_state.correct_pose_started_at = None
    st.session_state.step_completed_at = time.time()

    speak_once(
        f"step_success_{st.session_state.check_step}",
        voice_message,
    )


def get_step_content(step: int) -> tuple[str, str, str]:
    """Return visible instruction content for each step."""
    content = {
        1: (
            "Get comfortable",
            "Please face the camera.",
            (
                "Sit or stand naturally. Keep your head, shoulders "
                "and upper body visible."
            ),
        ),
        2: (
            "Posture check",
            "Please relax and stay still.",
            (
                "Keep your shoulders comfortable and look towards "
                "the camera while we record your posture."
            ),
        ),
        3: (
            "Arm check",
            "Please slowly raise both arms.",
            (
                "Keep both hands close to or above shoulder height "
                "and hold the position."
            ),
        ),
        4: (
            "Natural movement",
            "Please move naturally for a few seconds.",
            (
                "You may gently shift your weight or take a small step "
                "while staying in view."
            ),
        ),
        5: (
            "Speech check",
            "Please read the sentence aloud.",
            (
                'Read: "Today is a beautiful day." '
                "Speak naturally until the recording finishes."
            ),
        ),
    }

    return content.get(
        step,
        (
            "Check complete",
            "Thank you.",
            "Your wellness check has finished.",
        ),
    )


def get_step_voice_instruction(step: int) -> str:
    """Return the spoken instruction for each check."""
    instructions = {
        1: (
            "Please face the camera. "
            "Keep your head, shoulders and upper body visible."
        ),
        2: (
            "Please relax and stay still. "
            "Keep your shoulders comfortable and look towards the camera."
        ),
        3: (
            "Please slowly raise both arms. "
            "Keep both hands close to or above shoulder height."
        ),
        4: (
            "Please move naturally for a few seconds. "
            "You may gently shift your weight or take a small step."
        ),
        5: (
            "Please read the sentence shown on the screen. "
            "Recording will begin in three seconds."
        ),
    }

    return instructions.get(step, "")


def repeat_step_instruction_if_needed(
    step: int,
    media_is_playing: bool,
) -> None:
    """Repeat camera instructions without duplicating the speech prompt."""
    if (
        st.session_state.step_completed
        or not media_is_playing
        or step == 5
    ):
        return

    instruction = get_step_voice_instruction(step)
    if not instruction:
        return

    spoken_times = dict(st.session_state.last_instruction_spoken_at)
    last_spoken_at = spoken_times.get(step)
    current_time = time.time()
    should_speak = (
        last_spoken_at is None
        or current_time - last_spoken_at
        >= st.session_state.instruction_repeat_seconds
    )

    if should_speak:
        spoken_times[step] = current_time
        st.session_state.last_instruction_spoken_at = spoken_times
        speak_message(
            text=instruction,
            message_key=f"step_instruction_{step}",
            minimum_interval_seconds=max(
                1.0,
                float(st.session_state.instruction_repeat_seconds) - 0.5,
            ),
        )


def render_step_instruction(step: int) -> None:
    """Render the current check instruction."""
    title, instruction, helper = get_step_content(step)

    st.progress(
        min(step, TOTAL_STEPS) / TOTAL_STEPS,
        text=f"Step {min(step, TOTAL_STEPS)} of {TOTAL_STEPS}",
    )

    st.markdown(
        f"""
        <div class="step-card">
            <div class="step-number">
                STEP {min(step, TOTAL_STEPS)} OF {TOTAL_STEPS}
            </div>
            <h2>{title}</h2>
            <p class="instruction">{instruction}</p>
            <p class="helper-text">{helper}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def evaluate_step_progress(
    metrics: dict[str, Any] | None,
    history: list[dict[str, Any]],
) -> None:
    """Collect valid data continuously before completing each step."""
    if metrics is None or st.session_state.step_completed:
        return

    step = st.session_state.check_step
    status = str(metrics.get("monitoring_status", ""))

    pose_confidence = float(
        metrics.get("pose_confidence", 0.0) or 0.0
    )
    movement_speed = float(
        metrics.get("movement_speed", 0.0) or 0.0
    )
    posture_change = abs(
        float(metrics.get("posture_change", 0.0) or 0.0)
    )
    both_arms_raised = bool(
        metrics.get("both_arms_raised", False)
    )
    arm_height_difference = float(
        metrics.get("arm_height_difference", 1.0) or 1.0
    )

    person_visible = (
        status not in {
            "",
            "No person detected",
            "Not available",
        }
        and pose_confidence >= 0.50
    )

    if step == 1:
        correct = person_visible

        if hold_condition(
            correct,
            required_seconds=2.0,
        ):
            mark_step_complete(
                "Camera position confirmed.",
                "Great. We can see you clearly.",
            )

    elif step == 2:
        correct = (
            person_visible
            and posture_change < 20.0
            and movement_speed < 0.25
        )

        if hold_condition(
            correct,
            required_seconds=3.0,
        ):
            mark_step_complete(
                "Posture successfully recorded.",
                "Well done. Your posture check is complete.",
            )

    elif step == 3:
        correct = (
            person_visible
            and both_arms_raised
            and arm_height_difference < 0.25
        )

        if hold_condition(
            correct,
            required_seconds=2.0,
        ):
            mark_step_complete(
                "Both arms were detected in the raised position.",
                "Wonderful. You completed the arm check.",
            )

    elif step == 4:
        correct = (
            person_visible
            and movement_speed > 0.005
            and len(history) >= 3
        )

        if hold_condition(
            correct,
            required_seconds=3.0,
        ):
            mark_step_complete(
                "Natural movement successfully recorded.",
                "Excellent. Your movement check is complete.",
            )


def evaluate_speech_step(
    audio_processor: RealtimeAudioProcessor,
    media_is_playing: bool,
) -> None:
    """Prompt first, then record after the prompt has finished."""
    if (
        st.session_state.check_step != 5
        or st.session_state.step_completed
        or not media_is_playing
    ):
        return

    current_time = time.time()

    # Reset stale audio exactly once when Step 5 begins.
    if not st.session_state.speech_step_initialized:
        audio_processor.reset()
        st.session_state.speech_result = None
        st.session_state.speech_recording_started = False
        st.session_state.speech_step_initialized = True
        st.session_state.speech_prompt_started_at = current_time
        st.session_state.speech_recording_ready_at = current_time + 4.5

        speak_once(
            f"speech_prompt_attempt_{st.session_state.speech_attempts}",
            (
                "Please read the sentence shown on the screen. "
                "Recording will begin in three seconds."
            ),
        )
        return

    ready_at = st.session_state.speech_recording_ready_at
    if ready_at is not None and current_time < float(ready_at):
        return

    result = audio_processor.latest_result

    if result is not None and st.session_state.speech_recording_started:
        st.session_state.speech_result = result

        if not bool(result.get("speech_detected", False)):
            audio_processor.reset()
            st.session_state.speech_recording_started = False
            st.session_state.speech_step_initialized = False
            st.session_state.speech_prompt_started_at = None
            st.session_state.speech_recording_ready_at = None
            st.session_state.speech_attempts += 1
            speak_message(
                text=(
                    "We could not hear enough speech. "
                    "Please move a little closer to the microphone and try again."
                ),
                message_key=f"speech_retry_{st.session_state.speech_attempts}",
                minimum_interval_seconds=4.0,
            )
            return

        score = float(result.get("difference_score", 0.0) or 0.0)

        if score >= 75.0:
            voice_message = (
                "Thank you. The speech check is complete. "
                "We noticed a stronger speech pattern difference. "
                "Please remember that this is not a medical diagnosis."
            )
        elif score >= 45.0:
            voice_message = (
                "Thank you. The speech check is complete. "
                "We noticed a moderate speech pattern difference. "
                "This is not a medical diagnosis."
            )
        else:
            voice_message = (
                "Well done. Your speech check is complete."
            )

        mark_step_complete(
            "Speech sample successfully recorded and analysed.",
            voice_message,
        )

        # Stop collecting audio immediately. The next full rerun also stops
        # the browser WebRTC session, releasing camera and microphone access.
        audio_processor.stop_recording()
        st.session_state.webrtc_desired_playing_state = False
        return

    if (
        not audio_processor.recording_active
        and not audio_processor.analysis_in_progress
        and not st.session_state.speech_recording_started
    ):
        audio_processor.start_recording()
        st.session_state.speech_recording_started = True

def render_speech_status(
    audio_processor: RealtimeAudioProcessor,
    media_is_playing: bool,
) -> None:
    """Render the automatic speech recording and model result."""
    st.markdown(
        f"""
        <div class="speech-card">
            <h3>🎤 Read this sentence aloud</h3>
            <p class="speech-sentence">
                “{st.session_state.speech_prompt}”
            </p>
            <p>
                Speak naturally. The microphone recording is processed
                in memory and is not saved.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not media_is_playing:
        st.info(
            "Press START in the camera panel and allow both camera "
            "and microphone access."
        )
        return

    ready_at = st.session_state.get("speech_recording_ready_at")
    if (
        st.session_state.get("speech_step_initialized")
        and not st.session_state.speech_recording_started
        and ready_at is not None
    ):
        remaining = max(0, int(float(ready_at) - time.time()) + 1)
        if remaining > 0:
            st.info(
                f"Please get ready. Recording begins in {remaining} seconds."
            )

    duration = audio_processor.recorded_duration
    volume = audio_processor.current_volume

    duration_column, level_column = st.columns(2)

    with duration_column:
        st.metric(
            "Recorded time",
            f"{duration:.1f} seconds",
        )

    with level_column:
        st.metric(
            "Microphone level",
            f"{volume:.4f}",
        )

    if audio_processor.recording_active:
        st.warning("Recording now. Please read the sentence aloud.")
        st.progress(
            min(1.0, duration / audio_processor.recording_seconds),
            text=(
                f"Recording {duration:.1f} of "
                f"{audio_processor.recording_seconds:.1f} seconds"
            ),
        )
    elif audio_processor.analysis_in_progress:
        st.info("Analysing your speech sample...")
    elif st.session_state.speech_recording_started:
        st.info("Preparing your speech result...")

    error = audio_processor.latest_error
    if error:
        st.error(error)

    result = (
        st.session_state.get("speech_result")
        or audio_processor.latest_result
    )

    if result:
        score = float(result.get("difference_score", 0.0) or 0.0)
        result_column, score_column = st.columns(2)

        with result_column:
            st.metric(
                "Speech result",
                str(result.get("result_level", "Available")),
            )

        with score_column:
            st.metric(
                "Pattern difference score",
                f"{score:.1f}%",
            )

        st.caption(
            str(
                result.get(
                    "important_notice",
                    "This is not a stroke diagnosis.",
                )
            )
        )

        with st.expander("Show speech timing details", expanded=False):
            features = dict(result.get("features", {}) or {})
            st.write(
                "Total duration:",
                f"{float(features.get('total_duration', 0.0)):.2f}s",
            )
            st.write(
                "Voiced duration:",
                f"{float(features.get('voiced_duration', 0.0)):.2f}s",
            )
            st.write(
                "Voiced ratio:",
                f"{float(features.get('voiced_ratio', 0.0)):.2f}",
            )
            st.write(
                "Speech start delay:",
                f"{float(features.get('speech_start_delay', 0.0)):.2f}s",
            )
            st.write(
                "Pause count:",
                int(float(features.get("pause_count", 0.0))),
            )
            st.write(
                "Longest pause:",
                f"{float(features.get('longest_pause', 0.0)):.2f}s",
            )


def render_simple_status(
    metrics: dict[str, Any] | None,
    is_playing: bool,
) -> None:
    """Show plain-language status."""
    if not is_playing:
        st.info("Press START above when you are ready.")
        return

    if metrics is None:
        st.markdown(
            """
            <div class="calibration-status">
                <strong>Getting ready...</strong>
                <p>Please remain in view for a moment.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    status = metrics.get("monitoring_status", "Not available")

    if status == "No person detected":
        st.markdown(
            """
            <div class="calibration-status">
                <strong>We cannot see you clearly yet.</strong>
                <p>Please move into view and keep your upper body visible.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    elif status == "Calibrating":
        st.markdown(
            """
            <div class="calibration-status">
                <strong>Learning your usual movement...</strong>
                <p>Please relax and remain in view.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    elif status == "Possible anomaly":
        st.markdown(
            """
            <div class="care-card">
                <strong>We noticed a movement difference.</strong>
                <p>Please continue calmly. This prototype cannot diagnose.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    else:
        st.markdown(
            """
            <div class="friendly-status">
                <strong>You are doing well.</strong>
                <p>The check is continuing normally.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_live_results(metrics: dict[str, Any] | None) -> None:
    """Show the main live values."""
    st.write("")
    st.subheader("Your live results")

    if metrics is None:
        st.info("Your live results will appear once the camera detects you.")
        return

    movement_speed = float(
        metrics.get("movement_speed", 0.0) or 0.0
    )
    body_tilt = float(
        metrics.get("body_tilt", 0.0) or 0.0
    )
    asymmetry = float(
        metrics.get("current_asymmetry", 0.0) or 0.0
    )
    risk_score = metrics.get("risk_score")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Movement level", f"{movement_speed:.3f}")

    with col2:
        st.metric("Body position", f"{body_tilt:.1f}°")

    with col3:
        st.metric("Left-right difference", f"{asymmetry:.3f}")

    with col4:
        st.metric(
            "Attention score",
            "N/A"
            if risk_score is None
            else f"{float(risk_score):.0f}/100",
        )

    if st.session_state.check_step == 3:
        arm_col1, arm_col2 = st.columns(2)

        with arm_col1:
            st.metric(
                "Both arms raised",
                "Yes"
                if metrics.get("both_arms_raised", False)
                else "Not yet",
            )

        with arm_col2:
            arm_diff = float(
                metrics.get("arm_height_difference", 1.0) or 1.0
            )
            st.metric(
                "Arm height difference",
                f"{arm_diff:.3f}",
            )

    if metrics.get("risk_state") == "Possible anomaly":
        reasons = list(metrics.get("reasons", []) or [])
        friendly_reason = get_friendly_anomaly_reason(reasons)
        st.warning(
            "Movement difference detected. "
            + friendly_reason
            + " This is not a medical diagnosis."
        )



def get_step_name(step: int) -> str:
    """Return the display name for each wellness check."""
    names = {
        1: "Camera positioning",
        2: "Posture stability",
        3: "Arm raise and symmetry",
        4: "Natural movement",
        5: "Speech timing and pattern",
    }
    return names.get(step, f"Check {step}")


def get_friendly_reason(reason: str) -> str:
    """Convert technical anomaly reasons into user-friendly language."""
    reason_map = {
        "large body tilt change": "A sudden change in body position was observed.",
        "sudden body-center vertical drop": "A sudden downward movement was observed.",
        "prolonged low body position": "The body remained in a lower position for a period of time.",
        "large left/right asymmetry": "A larger difference between the left and right sides was observed.",
        "unusually low movement relative to the session baseline": "Movement became lower than earlier in the session.",
    }
    return reason_map.get(
        reason,
        "A movement pattern different from the earlier session was observed.",
    )


def collect_session_reasons(
    metrics: dict[str, Any] | None,
    history: list[dict[str, Any]],
) -> list[str]:
    """Collect unique anomaly reasons from the current session."""
    collected: list[str] = []

    if metrics is not None:
        for reason in metrics.get("reasons", []):
            if reason and reason not in collected:
                collected.append(reason)

    for entry in history:
        for reason in entry.get("reasons", []):
            if reason and reason not in collected:
                collected.append(reason)

    return collected


def calculate_session_summary(
    metrics: dict[str, Any] | None,
    history: list[dict[str, Any]],
) -> dict[str, Any]:
    """Calculate dynamic summary values for the completed session."""
    skipped_steps = list(st.session_state.get("skipped_steps", []))
    completed_steps = list(st.session_state.get("completed_steps", []))

    pose_confidences: list[float] = []
    risk_scores: list[float] = []
    movement_speeds: list[float] = []
    body_tilts: list[float] = []
    asymmetry_values: list[float] = []

    for entry in history:
        pose_confidences.append(float(entry.get("pose_confidence", 0.0) or 0.0))
        risk_scores.append(float(entry.get("risk_score", 0.0) or 0.0))
        movement_speeds.append(float(entry.get("movement_speed", 0.0) or 0.0))
        body_tilts.append(abs(float(entry.get("body_tilt", 0.0) or 0.0)))
        asymmetry_values.append(
            float(
                entry.get(
                    "current_asymmetry",
                    max(
                        float(entry.get("shoulder_symmetry", 0.0) or 0.0),
                        float(entry.get("hip_symmetry", 0.0) or 0.0),
                    ),
                )
                or 0.0
            )
        )

    if metrics is not None:
        pose_confidences.append(float(metrics.get("pose_confidence", 0.0) or 0.0))
        risk_scores.append(float(metrics.get("risk_score", 0.0) or 0.0))
        movement_speeds.append(float(metrics.get("movement_speed", 0.0) or 0.0))
        body_tilts.append(abs(float(metrics.get("body_tilt", 0.0) or 0.0)))
        asymmetry_values.append(float(metrics.get("current_asymmetry", 0.0) or 0.0))

    average_pose_confidence = (
        sum(pose_confidences) / len(pose_confidences)
        if pose_confidences
        else 0.0
    )
    maximum_risk_score = max(risk_scores) if risk_scores else 0.0
    average_movement_speed = (
        sum(movement_speeds) / len(movement_speeds)
        if movement_speeds
        else 0.0
    )
    maximum_body_tilt = max(body_tilts) if body_tilts else 0.0
    maximum_asymmetry = max(asymmetry_values) if asymmetry_values else 0.0

    completed_count = len(
        [
            step
            for step in range(1, TOTAL_STEPS + 1)
            if step in completed_steps and step not in skipped_steps
        ]
    )
    completion_rate = completed_count / TOTAL_STEPS if TOTAL_STEPS else 0.0
    sample_quality = min(1.0, len(history) / 8.0)
    assessment_quality = (
        average_pose_confidence * 0.50
        + completion_rate * 0.35
        + sample_quality * 0.15
    )
    assessment_quality_percent = int(round(max(0.0, min(1.0, assessment_quality)) * 100))

    reasons = collect_session_reasons(metrics, history)
    speech_result = st.session_state.get("speech_result")
    speech_score = (
        float(speech_result.get("difference_score", 0.0) or 0.0)
        if speech_result
        else None
    )
    speech_level = (
        str(speech_result.get("result_level", "Not available"))
        if speech_result
        else "Not available"
    )
    speech_difference_observed = (
        speech_score is not None and speech_score >= 75.0
    )

    possible_anomaly = (
        maximum_risk_score >= 45.0
        or speech_difference_observed
        or bool(reasons)
        or (
            metrics is not None
            and metrics.get("risk_state") == "Possible anomaly"
        )
    )
    incomplete = bool(skipped_steps) or completed_count < TOTAL_STEPS

    if incomplete:
        overall_level = "Incomplete assessment"
        overall_icon = "🟡"
        overall_message = (
            "Some checks were not completed, so this session cannot provide a complete summary."
        )
    elif possible_anomaly:
        overall_level = "Movement difference observed"
        overall_icon = "🟠"
        overall_message = (
            "One or more movement differences were observed during this session."
        )
    else:
        overall_level = "No significant difference observed"
        overall_icon = "🟢"
        overall_message = (
            "No significant movement difference was identified by this research prototype during this session."
        )

    return {
        "skipped_steps": skipped_steps,
        "completed_steps": completed_steps,
        "completed_count": completed_count,
        "completion_rate": completion_rate,
        "assessment_quality_percent": assessment_quality_percent,
        "average_pose_confidence": average_pose_confidence,
        "maximum_risk_score": maximum_risk_score,
        "average_movement_speed": average_movement_speed,
        "maximum_body_tilt": maximum_body_tilt,
        "maximum_asymmetry": maximum_asymmetry,
        "reasons": reasons,
        "possible_anomaly": possible_anomaly,
        "incomplete": incomplete,
        "overall_level": overall_level,
        "overall_icon": overall_icon,
        "overall_message": overall_message,
        "speech_result": speech_result,
        "speech_score": speech_score,
        "speech_level": speech_level,
        "speech_difference_observed": speech_difference_observed,
    }


def render_step_summary(summary: dict[str, Any]) -> None:
    """Render the status of every test."""
    st.subheader("Check-by-check summary")

    skipped_steps = summary["skipped_steps"]
    completed_steps = summary["completed_steps"]
    columns = st.columns(2)

    for index, step in enumerate(range(1, TOTAL_STEPS + 1)):
        with columns[index % 2]:
            if step in skipped_steps:
                icon = "⚪"
                status = "Not completed"
                message = "Not enough information was collected."
            elif step in completed_steps:
                icon = "✅"
                status = "Completed"
                messages = {
                    1: "The camera detected a clear upper-body pose.",
                    2: "A stable posture sample was recorded.",
                    3: "The arm-raise position was recorded.",
                    4: "Natural movement data was recorded.",
                    5: "A speech sample was recorded and analysed.",
                }
                message = messages.get(step, "The check was completed.")
            else:
                icon = "⚪"
                status = "Not recorded"
                message = "No completed result was available."

            st.markdown(
                f"""
                <div class="simple-result">
                    <h3>{icon} {get_step_name(step)}</h3>
                    <p><strong>{status}</strong></p>
                    <p>{message}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.write("")


def render_observation_summary(summary: dict[str, Any]) -> None:
    """Render session observations."""
    st.subheader("Session observations")
    reasons = summary["reasons"]

    if reasons:
        for reason in reasons:
            st.warning(get_friendly_reason(reason))
    else:
        st.success("No specific movement warning reason was recorded during this session.")

    speech_score = summary.get("speech_score")
    if speech_score is not None:
        if speech_score >= 75.0:
            st.warning(
                "A higher dysarthria-related speech-pattern difference "
                f"was observed ({speech_score:.1f}%)."
            )
        elif speech_score >= 45.0:
            st.info(
                "A moderate dysarthria-related speech-pattern difference "
                f"was observed ({speech_score:.1f}%)."
            )
        else:
            st.success(
                "A lower dysarthria-related speech-pattern difference "
                f"was observed ({speech_score:.1f}%)."
            )


def render_recommendation(summary: dict[str, Any]) -> None:
    """Render a recommendation appropriate to the result."""
    st.subheader("What this result means")

    if summary["incomplete"]:
        st.markdown(
            """
            <div class="care-card">
                <h3>The assessment is incomplete</h3>
                <p>
                    Some checks could not be completed. This can happen because of
                    camera position, lighting, internet delay, movement limitations,
                    or pose-detection uncertainty.
                </p>
                <p>Consider repeating the check when comfortable.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif summary["possible_anomaly"]:
        st.markdown(
            """
            <div class="care-card">
                <h3>Movement or speech-pattern differences were observed</h3>
                <p>This does not confirm a stroke or another medical condition.</p>
                <p>
                    Consider repeating the check and discussing new or persistent
                    changes with a qualified healthcare professional.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="simple-result">
                <h3>No significant warning pattern was observed</h3>
                <p>
                    This result only describes movement data collected during this
                    short session. It cannot rule out stroke or another condition.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def build_text_report(summary: dict[str, Any]) -> str:
    """Build a plain-text report for download."""
    lines = [
        "StrokeAI Wellness Check",
        "========================",
        "",
        f"Overall result: {summary['overall_level']}",
        f"Assessment quality: {summary['assessment_quality_percent']}%",
        f"Completed checks: {summary['completed_count']}/{TOTAL_STEPS}",
        "",
        "Check summary:",
    ]

    for step in range(1, TOTAL_STEPS + 1):
        if step in summary["skipped_steps"]:
            status = "Not completed"
        elif step in summary["completed_steps"]:
            status = "Completed"
        else:
            status = "Not recorded"
        lines.append(f"- {get_step_name(step)}: {status}")

    lines.extend(
        [
            "",
            "Session values:",
            f"- Average pose confidence: {summary['average_pose_confidence']:.2f}",
            f"- Highest attention score: {summary['maximum_risk_score']:.1f}/100",
            f"- Average movement level: {summary['average_movement_speed']:.3f}",
            f"- Maximum body position angle: {summary['maximum_body_tilt']:.1f} degrees",
            f"- Maximum left-right difference: {summary['maximum_asymmetry']:.3f}",
            (
                "- Speech-pattern difference score: Not available"
                if summary.get("speech_score") is None
                else f"- Speech-pattern difference score: {summary['speech_score']:.1f}%"
            ),
            f"- Speech result: {summary.get('speech_level', 'Not available')}",
            "",
            "Observations:",
        ]
    )

    if summary["reasons"]:
        for reason in summary["reasons"]:
            lines.append("- " + get_friendly_reason(reason))
    else:
        lines.append("- No specific warning reason was recorded.")

    lines.extend(
        [
            "",
            "Important:",
            "This research prototype cannot diagnose or exclude stroke or another medical condition.",
            "If sudden facial weakness, arm weakness or speech difficulty occurs, seek emergency help.",
        ]
    )
    return "\n".join(lines)


def render_completion(
    metrics: dict[str, Any] | None,
    history: list[dict[str, Any]] | None = None,
) -> None:
    """Render a dynamic session assessment report."""
    if history is None:
        history = list(st.session_state.get("history", []))

    summary = calculate_session_summary(metrics, history)

    speak_once(
        "final_completion",
        (
            "Your wellness check is complete. "
            + summary["overall_message"]
            + " This result is not a medical diagnosis."
        ),
    )

    st.markdown(
        f"""
        <div class="completion-card">
            <div style="font-size: 72px;">{summary['overall_icon']}</div>
            <h1>Wellness check complete</h1>
            <h2>{summary['overall_level']}</h2>
            <p>{summary['overall_message']}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    quality_column, completed_column, risk_column = st.columns(3)

    with quality_column:
        st.metric(
            "Assessment quality",
            f"{summary['assessment_quality_percent']}%",
            help=(
                "Based on completed checks, pose visibility and available session samples."
            ),
        )

    with completed_column:
        st.metric(
            "Checks completed",
            f"{summary['completed_count']}/{TOTAL_STEPS}",
        )

    with risk_column:
        st.metric(
            "Highest attention score",
            f"{summary['maximum_risk_score']:.0f}/100",
            help="A research-only movement signal score, not a medical diagnosis.",
        )

    st.divider()
    render_step_summary(summary)

    st.divider()
    st.subheader("Key session values")
    value_column_1, value_column_2, value_column_3 = st.columns(3)

    with value_column_1:
        st.metric(
            "Average movement level",
            f"{summary['average_movement_speed']:.3f}",
        )

    with value_column_2:
        st.metric(
            "Maximum body position",
            f"{summary['maximum_body_tilt']:.1f}°",
        )

    with value_column_3:
        st.metric(
            "Maximum left-right difference",
            f"{summary['maximum_asymmetry']:.3f}",
        )

    if summary.get("speech_score") is not None:
        st.divider()
        st.subheader("Speech check result")
        speech_col1, speech_col2 = st.columns(2)

        with speech_col1:
            st.metric(
                "Speech-pattern difference score",
                f"{summary['speech_score']:.1f}%",
            )

        with speech_col2:
            st.metric(
                "Speech result",
                summary["speech_level"],
            )

        st.caption(
            "The speech model was trained to distinguish healthy and "
            "dysarthria-related patterns in TORGO. It does not estimate "
            "stroke probability."
        )

    st.divider()
    render_observation_summary(summary)

    st.divider()
    render_recommendation(summary)

    st.markdown(
        """
        <div class="emergency-card">
            <h3>Sudden symptoms need urgent help</h3>
            <p>
                If you or someone nearby suddenly develops facial weakness,
                arm weakness or speech difficulty, call emergency services immediately.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    report_text = build_text_report(summary)
    download_column, restart_column = st.columns(2)

    with download_column:
        st.download_button(
            label="Download session summary",
            data=report_text,
            file_name="strokeai_wellness_summary.txt",
            mime="text/plain",
            use_container_width=True,
        )

    with restart_column:
        if st.button(
            "Start another wellness check",
            type="primary",
            use_container_width=True,
            key="restart_check",
        ):
            reset_check()
            st.rerun(scope="app")

    render_technical_details(metrics, history)


@st.fragment(run_every=1.0)
def render_guided_check(
    webrtc_ctx: Any,
    audio_processor: RealtimeAudioProcessor,
) -> None:
    """Refresh the guided check once per second."""
    processor = webrtc_ctx.video_processor

    metrics: dict[str, Any] | None = None
    history: list[dict[str, Any]] = []

    if processor is not None:
        metrics = processor.latest_metrics
        history = list(processor.history_records)

    if metrics is not None:
        st.session_state.latest_metrics = metrics

    if history:
        st.session_state.history = history

    evaluate_step_progress(metrics, history)
    evaluate_speech_step(
        audio_processor,
        webrtc_ctx.state.playing,
    )
    speak_anomaly_warning(metrics)

    current_step = st.session_state.check_step

    if current_step > TOTAL_STEPS:
        render_completion(metrics, history)
        return

    render_step_instruction(current_step)
    repeat_step_instruction_if_needed(
        current_step,
        webrtc_ctx.state.playing,
    )

    if has_step_timed_out():
        st.warning(
            "We have not collected enough information for this check yet."
        )

        if current_step == 5:
            st.markdown(
                """
                Please check that:

                - The browser has microphone permission
                - Your microphone is not muted
                - You are speaking close enough to the microphone
                - You read the full sentence in a natural voice
                """
            )
        else:
            st.markdown(
                """
                Please check that:

                - Your full upper body is visible
                - The room is well lit
                - You are facing the camera
                - Your movements are slow and comfortable
                """
            )

        retry_column, skip_column, stop_column = st.columns(3)

        with retry_column:
            if st.button(
                "Try again",
                type="primary",
                use_container_width=True,
                key=f"retry_step_{current_step}",
            ):
                retry_current_step()
                st.rerun()

        with skip_column:
            if st.button(
                "Skip this check",
                use_container_width=True,
                key=f"skip_step_{current_step}",
            ):
                skip_current_step()
                st.rerun()

        with stop_column:
            if st.button(
                "End the check",
                use_container_width=True,
                key=f"stop_step_{current_step}",
            ):
                reset_check()
                st.rerun()

        if current_step == 5:
            render_speech_status(
                audio_processor,
                webrtc_ctx.state.playing,
            )
        else:
            render_live_results(metrics)

        render_technical_details(metrics, history)
        return

    if st.session_state.step_completed:
        st.success(
            "✅ " + st.session_state.step_success_message
        )

        st.info(
            "Well done. The next check will begin automatically."
        )

        completed_at = (
            st.session_state.step_completed_at
            or time.time()
        )

        pause_elapsed = time.time() - completed_at
        remaining = max(
            0,
            int(
                st.session_state.completion_pause_seconds
                - pause_elapsed
            ),
        )

        if remaining > 0:
            st.caption(
                f"Next check begins in {remaining + 1} seconds..."
            )
        else:
            move_to_next_step()
            st.rerun()

        if current_step == 5:
            render_speech_status(
                audio_processor,
                webrtc_ctx.state.playing,
            )
        else:
            render_live_results(metrics)

        render_technical_details(metrics, history)
        return

    if current_step == 5:
        render_speech_status(
            audio_processor,
            webrtc_ctx.state.playing,
        )
        render_technical_details(metrics, history)
        return

    instruction_column, status_column = st.columns(
        [1.15, 0.85],
        gap="large",
    )

    with instruction_column:
        st.subheader("Camera check")

        if webrtc_ctx.state.playing:
            st.success(
                "Camera is active. Please follow the instruction above."
            )
        else:
            st.info(
                "Press START in the camera panel above when you are ready."
            )

        st.caption(
            "Your video is processed in memory. Raw video is not saved."
        )

    with status_column:
        st.subheader("How you are doing")
        render_simple_status(
            metrics,
            webrtc_ctx.state.playing,
        )

    render_live_results(metrics)
    render_technical_details(metrics, history)


def render_technical_details(
    metrics: dict[str, Any] | None,
    history: list[dict[str, Any]],
) -> None:
    """Render technical information in a collapsed section."""
    with st.expander(
        "Show technical details",
        expanded=False,
    ):
        st.caption(
            "This section is intended for researchers "
            "and demonstration reviewers."
        )

        if metrics is not None:
            render_live_metrics(metrics)
        else:
            st.info("No technical metrics are available yet.")

        st.divider()
        st.subheader("Movement history")

        if history:
            render_movement_chart(history)
        else:
            st.info(
                "Movement history will appear after data is collected."
            )

        st.subheader("Recent alert log")

        if history:
            render_recent_alerts(history)
        else:
            st.info("No alerts have been recorded.")


def render_welcome_page() -> None:
    """Render the landing page."""
    st.markdown(
        """
        <div class="welcome-card">
            <div class="welcome-icon">🧠</div>
            <h1>Welcome to StrokeAI</h1>
            <p class="welcome-subtitle">
                We will guide you through a gentle,
                simple wellness check.
            </p>
            <p>
                The check takes only a short time.
                You can stop whenever you wish.
            </p>
            <p class="privacy-note">
                No login is required. Raw video is not saved.
                This is a research prototype and not a medical diagnosis.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    _, button_column, _ = st.columns([1, 2, 1])

    with button_column:
        if st.button(
            "Start my wellness check",
            type="primary",
            use_container_width=True,
            key="start_health_check",
        ):
            st.session_state.check_started = True
            st.session_state.check_step = 1
            st.session_state.completed_steps = []
            st.session_state.spoken_messages = []
            st.session_state.last_instruction_spoken_at = {}
            st.session_state.step_completed_at = None
            st.session_state.skipped_steps = []
            st.session_state.last_anomaly_spoken_at = 0.0
            st.session_state.active_anomaly_started_at = None
            st.session_state.speech_result = None
            st.session_state.speech_recording_started = False
            st.session_state.speech_step_initialized = False
            st.session_state.speech_prompt_started_at = None
            st.session_state.speech_recording_ready_at = None
            st.session_state.speech_attempts = 0
            st.session_state.webrtc_desired_playing_state = None

            processor = st.session_state.get("audio_processor")
            if processor is not None:
                processor.reset()

            reset_step_validation()

            # Do not speak before the user starts the camera/microphone stream.
            st.rerun()


def get_fallback_pipeline() -> StrokeAIPipeline:
    """Create fallback OpenCV pipeline only when needed."""
    if "fallback_pipeline" not in st.session_state:
        st.session_state.fallback_pipeline = StrokeAIPipeline()

    return st.session_state.fallback_pipeline


def run_capture() -> None:
    """Run one fallback snapshot analysis."""
    pipeline = get_fallback_pipeline()

    try:
        result = pipeline.process_frame()
    except (RuntimeError, ValueError) as exc:
        logger.exception("Fallback capture failed: %s", exc)
        st.error(str(exc))
        return

    status = result.get("status")

    st.session_state.snapshot_frame = result.get("frame")
    st.session_state.snapshot_annotated = result.get(
        "annotated_frame"
    )

    if status == "error":
        st.error(
            result.get(
                "message",
                "Camera processing failed.",
            )
        )
        return

    if status == "no_pose":
        st.session_state.snapshot_message = result.get(
            "message",
            "No person was detected.",
        )
        st.session_state.snapshot_metrics = None
        st.warning(st.session_state.snapshot_message)
        return

    st.session_state.snapshot_message = (
        "Snapshot analyzed successfully."
    )
    st.session_state.snapshot_metrics = result.get("metrics")
    st.success("Snapshot analyzed successfully.")


def release_fallback_camera() -> None:
    """Release fallback camera resources."""
    pipeline = st.session_state.get("fallback_pipeline")

    if pipeline is not None:
        pipeline.release_camera()

    st.session_state.pop("fallback_pipeline", None)
    st.session_state.snapshot_frame = None
    st.session_state.snapshot_annotated = None
    st.session_state.snapshot_metrics = None
    st.session_state.snapshot_message = None

    st.success("Fallback camera released.")


def render_snapshot_results() -> None:
    """Render fallback snapshot results."""
    frame = st.session_state.snapshot_frame
    annotated = st.session_state.snapshot_annotated
    metrics = st.session_state.snapshot_metrics
    message = st.session_state.snapshot_message

    if frame is None and annotated is None:
        st.info("No fallback snapshot has been captured.")
        return

    if message:
        st.caption(message)

    first_column, second_column = st.columns(2)

    with first_column:
        st.markdown("**Original image**")

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
        render_live_metrics(metrics)


def main() -> None:
    """Render the application."""
    init_state()
    apply_accessible_style()

    if not st.session_state.check_started:
        render_welcome_page()
        return

    audio_processor = get_audio_processor()

    st.title("StrokeAI Wellness Check")
    st.caption(
        "Take your time. Follow one simple instruction at a time."
    )

    with st.sidebar:
        st.header("Help")

        st.markdown(
            """
            **For the best experience**

            - Sit or stand somewhere comfortable
            - Keep your head, shoulders and hips visible
            - Use good lighting
            - Move slowly and gently
            - Allow microphone access for the speech check
            """
        )

        st.divider()

        if st.button(
            "End this check",
            use_container_width=True,
            key="end_check",
        ):
            reset_check()
            st.rerun()

        with st.expander(
            "Fallback snapshot option",
            expanded=False,
        ):
            st.caption(
                "Use this only if the live camera does not work."
            )

            if st.button(
                "Capture one image",
                use_container_width=True,
                key="capture_snapshot",
            ):
                run_capture()

            if st.button(
                "Release fallback camera",
                use_container_width=True,
                key="release_fallback",
            ):
                release_fallback_camera()

            render_snapshot_results()

    st.subheader("Live camera")

    webrtc_ctx = webrtc_streamer(
        key="strokeai-live-monitor",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=VideoProcessor,
        audio_frame_callback=build_audio_frame_callback(
            audio_processor
        ),
        media_stream_constraints={
            "video": True,
            "audio": True,
        },
        desired_playing_state=(
            st.session_state.webrtc_desired_playing_state
        ),
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

    render_guided_check(webrtc_ctx, audio_processor)


if __name__ == "__main__":
    main()
