import pandas as pd

from anomaly.anomaly_detector import AnomalyDetector


if __name__ == "__main__":
    baseline = pd.DataFrame(
        {
            "walking_speed": [0.8, 0.9, 0.7, 0.8, 0.85, 0.88, 0.82],
            "body_center_y": [0.45, 0.47, 0.44, 0.46, 0.45, 0.46, 0.45],
        }
    )
    current = pd.DataFrame(
        {
            "walking_speed": [1.7, 1.8, 1.6],
            "body_center_y": [0.2, 0.18, 0.22],
        }
    )

    detector = AnomalyDetector()
    result = detector.analyze_behavior(baseline, current)
    print(result)
