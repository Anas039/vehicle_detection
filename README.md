# Real-Time Vehicle Detection and Tracking

This project implements a real-time vehicle detection and tracking system using OpenCV and Kalman Filter. It processes video input, detects vehicles based on contour analysis, and tracks their positions using Kalman Filters.

## Features

- **Vehicle Detection**: Detects vehicles based on contour area and bounding box size.
- **Vehicle Tracking**: Tracks detected vehicles using Kalman Filters to predict their positions.
- **Real-Time Processing**: Processes video frames in real-time and displays detection results.

## Prerequisites

- Python 3.x
- OpenCV
- NumPy

## Installation

1. **Clone the repository**:

    ```bash
    git clone https://github.com/yourusername/vehicle_detection.git
    cd vehicle_detection
    ```

2. **Install dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. **Configure Parameters**:

    Edit the `config.py` file to adjust parameters such as video path, area thresholds, bounding box sizes, etc.

2. **Run the Application**:

    ```bash
    python main.py
    ```

    Press `q` to quit the video processing window.

## Project Structure


- **`main.py`**: Entry point of the application.
- **`tracker.py`**: Contains the `VehicleKalmanTracker` class for tracking.
- **`config.py`**: Configuration parameters.
- **`utils.py`**: (Optional) Utility functions.
- **`requirements.txt`**: Python dependencies.
- **`README.md`**: Project overview and instructions.

## Configuration

Modify the `config.py` file to change the settings:

- **VIDEO_PATH**: Path to your input video file.
- **AREA_THRESHOLD**: Minimum contour area to be considered a vehicle.
- **MIN_WIDTH & MIN_HEIGHT**: Minimum size of the bounding box to filter out non-vehicle objects.
- **KERNEL_SIZE**: Structuring element size for morphological operations.
- **DISPLAY_WINDOW_NAME**: Name of the display window.
- **PREDICTION_COLOR, BOUNDING_BOX_COLOR**: Colors for drawing predicted positions and bounding boxes.
- **FONT, FONT_SCALE, FONT_COLOR, FONT_THICKNESS**: Parameters for displaying text on the video.


## Acknowledgements

- OpenCV for computer vision functionalities.
- NumPy for numerical operations.


