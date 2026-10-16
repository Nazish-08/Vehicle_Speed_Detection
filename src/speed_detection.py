
import cv2
import csv
import math
import numpy as np

from pathlib import Path
from collections import defaultdict, deque
from ultralytics import YOLO


# ==================== PROJECT PATHS ====================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_VIDEO = BASE_DIR / "input" / "traffic.mp4"
OUTPUT_VIDEO = BASE_DIR / "output" / "output_speed.mp4"
OUTPUT_CSV = BASE_DIR / "output" / "speed_log.csv"

MODEL_PATH = "yolo11n.pt"


# ==================== ASSUMED ROAD CALIBRATION ====================
# These dimensions and points are assumptions, NOT measured values.
# The points must lie on the road surface.
#
# Assumed road width: 10.5 metres
# Assumed road length: 30 metres
#
# Incorrect points or dimensions produce incorrect speed estimates.

ROAD_WIDTH_M = 10.5
ROAD_LENGTH_M = 30.0

SOURCE_POINTS = [
    [790, 0],       # top-left
    [1370, 0],      # top-right
    [1580, 1079],   # bottom-right
    [450, 1079],    # bottom-left
]

DESTINATION_POINTS = [
    [0, 0],
    [ROAD_WIDTH_M, 0],
    [ROAD_WIDTH_M, ROAD_LENGTH_M],
    [0, ROAD_LENGTH_M],
]


# COCO class IDs: car=2, motorcycle=3, bus=5, truck=7
VEHICLE_CLASSES = [2, 3, 5, 7]


# ==================== SPEED SETTINGS ====================

HISTORY_SECONDS = 0.8
MIN_SPEED_INTERVAL = 0.4
MAX_REASONABLE_SPEED = 160.0

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.65
FONT_THICKNESS = 2


def draw_label(frame, box, label):
    """Draw readable text with a solid background."""

    height, width = frame.shape[:2]
    x1, y1, x2, y2 = map(int, box)

    (text_width, text_height), baseline = cv2.getTextSize(
        label, FONT, FONT_SCALE, FONT_THICKNESS
    )

    # Keep label horizontally inside the frame.
    label_x = max(5, min(x1, width - text_width - 10))

    # Prefer above the vehicle bounding box.
    text_y = y1 - 8

    if text_y - text_height - baseline - 6 < 0:
        text_y = min(
            height - baseline - 5,
            y1 + text_height + 12
        )

    top = max(0, text_y - text_height - baseline - 6)
    bottom = min(height - 1, text_y + baseline + 4)
    right = min(width - 1, label_x + text_width + 10)

    cv2.rectangle(
        frame,
        (label_x, top),
        (right, bottom),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        frame,
        label,
        (label_x + 5, text_y),
        FONT,
        FONT_SCALE,
        (0, 255, 0),
        FONT_THICKNESS,
        cv2.LINE_AA
    )


def main():
    if not INPUT_VIDEO.exists():
        raise FileNotFoundError(
            f"Input video not found: {INPUT_VIDEO}"
        )

    OUTPUT_VIDEO.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(INPUT_VIDEO))

    if not cap.isOpened():
        raise RuntimeError("Could not open input video.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps <= 0 or width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError("Invalid input video properties.")

    if (width, height) != (1920, 1080):
        print(
            f"WARNING: Video resolution is {width}x{height}. "
            "Review SOURCE_POINTS before using the estimates."
        )

    # Image pixels -> assumed road-plane coordinates in metres.
    homography = cv2.getPerspectiveTransform(
        np.array(SOURCE_POINTS, dtype=np.float32),
        np.array(DESTINATION_POINTS, dtype=np.float32)
    )

    writer = cv2.VideoWriter(
        str(OUTPUT_VIDEO),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Could not create output video.")

    model = YOLO(MODEL_PATH)

    # Each history entry: (timestamp, road_x, road_y)
    histories = defaultdict(deque)
    latest_speeds = {}
    csv_rows = []
    frame_number = 0

    try:
        while True:
            success, frame = cap.read()

            if not success:
                break

            timestamp = frame_number / fps
            frame_number += 1

            result = model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                classes=VEHICLE_CLASSES,
                conf=0.3,
                verbose=False
            )[0]

            # Keep detection boxes but hide YOLO's default labels/confidence.
            annotated = result.plot(labels=False, conf=False)

            if (
                result.boxes is not None
                and result.boxes.id is not None
            ):
                boxes = result.boxes.xyxy.cpu().numpy()
                ids = result.boxes.id.int().cpu().tolist()
                classes = result.boxes.cls.int().cpu().tolist()

                for box, track_id, class_id in zip(boxes, ids, classes):
                    x1, y1, x2, y2 = box

                    # Bottom-centre approximates the road contact position.
                    image_point = np.array(
                        [[[(x1 + x2) / 2, y2]]],
                        dtype=np.float32
                    )

                    road_point = cv2.perspectiveTransform(
                        image_point, homography
                    )[0][0]

                    road_x = float(road_point[0])
                    road_y = float(road_point[1])

                    points = histories[track_id]
                    points.append(
                        (timestamp, road_x, road_y)
                    )

                    # Retain only recent observations.
                    while (
                        points
                        and timestamp - points[0][0] > HISTORY_SECONDS
                    ):
                        points.popleft()

                    speed_kmh = latest_speeds.get(track_id)

                    if len(points) >= 2:
                        old_t, old_x, old_y = points[0]
                        new_t, new_x, new_y = points[-1]

                        elapsed = new_t - old_t

                        if elapsed >= MIN_SPEED_INTERVAL:
                            distance_m = math.hypot(
                                new_x - old_x,
                                new_y - old_y
                            )

                            candidate = (
                                distance_m / elapsed
                            ) * 3.6

                            if (
                                math.isfinite(candidate)
                                and 0 <= candidate <= MAX_REASONABLE_SPEED
                            ):
                                speed_kmh = candidate
                                latest_speeds[track_id] = speed_kmh

                    if speed_kmh is None:
                        label = f"ID {track_id} | Calculating..."
                        csv_speed = ""
                    else:
                        label = (
                            f"ID {track_id} | EST. "
                            f"{speed_kmh:.1f} km/h"
                        )
                        csv_speed = round(speed_kmh, 2)

                    draw_label(annotated, box, label)

                    csv_rows.append({
                        "timestamp_seconds": round(timestamp, 3),
                        "vehicle_id": track_id,
                        "vehicle_class": model.names[class_id],
                        "estimated_speed_kmh": csv_speed,
                        "speed_type": "APPROXIMATE_ASSUMED_CALIBRATION"
                    })

            writer.write(annotated)

            if frame_number % 150 == 0:
                print(f"Processed {frame_number} frames")

    finally:
        cap.release()
        writer.release()

    columns = [
        "timestamp_seconds",
        "vehicle_id",
        "vehicle_class",
        "estimated_speed_kmh",
        "speed_type"
    ]

    with open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:
        csv_writer = csv.DictWriter(
            file,
            fieldnames=columns
        )
        csv_writer.writeheader()
        csv_writer.writerows(csv_rows)

    print(f"Output video: {OUTPUT_VIDEO}")
    print(f"CSV report: {OUTPUT_CSV}")
    print("Processing completed.")
    print(
        "WARNING: Speed values use assumed calibration. "
        "They are not verified real-world speeds."
    )


if __name__ == "__main__":
    main()
