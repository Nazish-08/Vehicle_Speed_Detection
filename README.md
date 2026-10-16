# Vehicle Speed Detection Using YOLO and OpenCV

A Python-based computer vision project that detects and tracks vehicles in a video and generates an annotated output video with vehicle IDs and estimated speed information.

## Overview

This project uses YOLO for vehicle detection, ByteTrack for multi-object tracking, and OpenCV for video processing and perspective transformation.

The system processes a traffic video, tracks detected vehicles across frames, estimates their speed using an assumed road-plane calibration, and exports an annotated video and CSV report.

**Important:** Speed values are approximate because the current road dimensions and calibration points are assumed rather than measured. The results are not verified real-world speeds.

## Features

- Vehicle detection using YOLO11n
- Vehicle tracking using ByteTrack
- Detection of cars, motorcycles, buses, and trucks
- Unique tracking IDs for detected vehicles
- Approximate speed estimation in km/h
- Readable speed labels with dark backgrounds
- Annotated output video generation
- CSV report containing timestamps, vehicle IDs, classes, and estimated speeds
- Perspective transformation for road-plane coordinate estimation

## Technology Stack

- Python
- Ultralytics YOLO
- OpenCV
- NumPy
- Pandas is optional and not required by the current script
- ByteTrack
- CSV for exporting results

## Project Structure

```text
Vehicle Speed Detection/
├── input/
│   └── traffic.mp4
├── output/
│   ├── output_speed.mp4
│   └── speed_log.csv
├── src/
│   └── speed_detection.py
├── requirements.txt
├── README.md
├── .gitignore
└── .venv/
```

## Installation

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd "Vehicle Speed Detection"
```

Replace `YOUR_GITHUB_REPOSITORY_URL` with your actual GitHub repository URL.

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Add the input video

Place your video inside the `input` directory and name it:

```text
traffic.mp4
```

### 5. Run the application

```bash
python src/speed_detection.py
```

On the first run, the YOLO model may be downloaded automatically.

## Output

After successful processing, the following files are generated:

**`output/output_speed.mp4`**

The annotated video displays vehicle bounding boxes, tracking IDs, and approximate speed labels.

**`output/speed_log.csv`**

The CSV report contains:

- Timestamp in seconds
- Vehicle tracking ID
- Vehicle class
- Estimated speed in km/h
- Speed estimation method

## Speed Estimation Method

1. YOLO detects vehicles in each video frame.
2. ByteTrack associates detections across consecutive frames.
3. The bottom-centre point of each vehicle bounding box is used as an approximate road contact point.
4. A perspective transformation maps image coordinates to assumed road-plane coordinates.
5. Movement over elapsed time is converted into an estimated speed in km/h.

The current implementation assumes a road width of 10.5 metres and a selected road length of 30 metres. These values and the source points must be validated against the actual camera view.

## Limitations

- Speed accuracy depends on road calibration and camera geometry.
- Assumed road dimensions can produce incorrect speed values.
- Vehicle tracking IDs may change when tracking is lost.
- Occlusion, poor lighting, perspective distortion, and detection errors can affect results.
- The current implementation is a prototype, not a certified speed measurement system.

For reliable real-world speed estimation, use measured road dimensions, calibrate the road plane, and validate the results against an independent speed reference.

## Open Source References

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [Ultralytics Tracking Documentation](https://docs.ultralytics.com/modes/track/)
- [OpenCV](https://github.com/opencv/opencv-python)

Check the applicable licenses of the libraries and any referenced implementation before redistributing code.

## License

Choose and add an appropriate license before publishing this repository. Third-party dependencies remain subject to their respective licenses.

## Author

Nazish Shaikh