# StrokeAI

StrokeAI is an AI-powered home health monitoring prototype designed to analyze posture and movement patterns from a webcam feed. The system combines computer vision, pose estimation, feature extraction, and anomaly detection to provide a simple, interpretable view of movement behavior over time.

## Project Overview

StrokeAI is intended for early-stage experimentation in home health monitoring and digital movement assessment. It captures live video, detects human pose landmarks, derives posture and gait features, compares current behavior with a baseline profile, and presents the results in a Streamlit dashboard.

The project focuses on modularity and clarity, making it suitable for further development into a more robust health-monitoring application.

## System Architecture

The application follows a straightforward pipeline:

```mermaid
flowchart LR
    A[Webcam Input] --> B[Pose Detection<br/>MediaPipe]
    B --> C[Feature Extraction]
    C --> D[Baseline Modeling]
    D --> E[Anomaly Detection]
    E --> F[Streamlit Dashboard]
```

At a high level:
- Camera input is captured via OpenCV.
- Pose landmarks are estimated using MediaPipe Pose.
- Relevant posture and gait metrics are derived from the landmark data.
- A baseline profile is built from recent movement history.
- The current frame is compared to that baseline to produce a risk-style assessment.

## Tech Stack

- Python 3.12
- Streamlit for the interactive dashboard
- OpenCV for camera access and image processing
- MediaPipe for pose estimation
- pandas for feature and history processing
- scikit-learn for anomaly detection
- NumPy for numerical operations
- Matplotlib for baseline visualization

## Installation

### Prerequisites
- Python 3.12
- A working webcam (for live analysis)

### Setup

```bash
git clone <repository-url>
cd StrokeAI
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Start the application with:

```bash
streamlit run app.py
```

Then:
1. Open the local Streamlit URL in your browser.
2. Allow camera access if prompted.
3. Click the "Capture snapshot" button to analyze a frame.
4. Review the annotated pose view, movement metrics, and history table.

If pose landmarks are not detected reliably, improve lighting and ensure the subject is within the camera frame.

## Project Structure

```text
StrokeAI/
├── app.py                     # Streamlit app entry point
├── camera/                    # Camera capture utilities
├── pose/                      # Pose detection logic
├── features/                  # Feature extraction module
├── baseline/                  # Baseline modeling helpers
├── anomaly/                   # Anomaly detection logic
├── dashboard/                 # Dashboard support utilities
├── utils/                     # Shared logging and pipeline helpers
├── requirements.txt           # Python dependencies
└── test_anomaly.py            # Basic anomaly detector check
```

## Future Work

Possible next steps for the project include:
- Persistent storage for long-term movement history
- More sophisticated baseline modeling and personalization
- Real-time monitoring with alerting and notifications
- Integration with wearable devices or external health data sources
- Improved model validation and clinician-oriented reporting

## License

This project is intended for research and prototyping purposes and may be adapted for broader use as development continues.
