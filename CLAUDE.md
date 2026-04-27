# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DetectionAI - Qt-based C++ application for real-time object detection using YOLO11 via ONNX Runtime. The application captures video from a webcam, performs object detection inference, and displays results with bounding boxes overlaid on the video feed.

## Build System

This project uses **qmake** (Qt's build system). Typically built via **Qt Creator** (open `DetectionAI.pro` directly). For CLI builds:

```bash
qmake DetectionAI.pro
nmake debug   # Windows MSVC
make          # Linux/macOS
```

### Dependencies

Paths must be configured in `DetectionAI.pro` for your local installations:
- **OpenCV**: `OPENCV_INC` / `OPENCV_LIB`
- **ONNX Runtime**: `ORT_INC` / `ORT_LIB`
- **Qt 6 Widgets** module (declared via `QT += widgets`)

### Runtime

The ONNX model file `yolo11n.onnx` must be in the working directory at launch.

## Architecture

The application consists of a single `MainWindow` class that orchestrates the entire inference pipeline:

1. **Initialization** (`initYOLO`): Loads ONNX model, creates ONNX Runtime session, queries input/output names
2. **Frame Loop** (`onFrameUpdate`): Triggered every 30ms by QTimer
   - Captures frame from webcam via OpenCV
   - Preprocesses frame (resize, color conversion, normalization)
   - Runs inference via ONNX Runtime
   - Postprocesses outputs (confidence filtering + coordinate mapping)
   - Displays result

### Key Constants

- `INPUT_WIDTH` / `INPUT_HEIGHT`: 640x640 (YOLO default)
- `CONF_THRESHOLD`: 0.25 (confidence threshold for detections)

### YOLO11 Output Format

The model outputs shape `(1, 8400, 84)`:
- 8400 anchor boxes
- 84 channels per box: `[x, y, w, h, class_1, ..., class_80]` (COCO 80 classes)

Coordinates are center-based (x, y is center point, w/h are dimensions).

### Known Limitations

- **No NMS**: Postprocessing only filters by confidence threshold — overlapping boxes from the same object are not suppressed. Adding NMS would reduce duplicate detections.
- **Input name lifetime**: `inputNames`/`outputNames` store raw `const char*` from ONNX Runtime allocators. These must not outlive the `Ort::Session` (currently safe since both are members of `MainWindow`).
