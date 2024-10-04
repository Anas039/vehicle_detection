# main.py

import cv2
import numpy as np
from tracker import VehicleKalmanTracker
import config

def initialize_video_capture(video_path):
    """
    Initialize the video capture object.
    Args:
        video_path (str): Path to the video file.
    Returns:
        cv2.VideoCapture: Video capture object.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")
    return cap

def initialize_background_subtractor():
    """
    Initialize the background subtractor.
    Returns:
        cv2.BackgroundSubtractor: Background subtractor object.
    """
    return cv2.createBackgroundSubtractorMOG2()

def get_structuring_element(kernel_size):
    """
    Get the structuring element for morphological operations.
    Args:
        kernel_size (tuple): Size of the structuring element.
    Returns:
        np.ndarray: Structuring element.
    """
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)

def process_frame(frame, fgbg, kernel):
    """
    Apply background subtraction and morphological operations to the frame.
    Args:
        frame (np.ndarray): The input video frame.
        fgbg (cv2.BackgroundSubtractor): Background subtractor.
        kernel (np.ndarray): Structuring element for morphological operations.
    Returns:
        np.ndarray: Processed foreground mask.
    """
    fgmask = fgbg.apply(frame)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
    return fgmask

def find_vehicle_contours(fgmask):
    """
    Find contours in the foreground mask.
    Args:
        fgmask (np.ndarray): The foreground mask.
    Returns:
        list: List of contours.
    """
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def main():
    # Initialize video capture
    try:
        cap = initialize_video_capture(config.VIDEO_PATH)
    except IOError as e:
        print(e)
        return

    # Initialize background subtractor and structuring element
    fgbg = initialize_background_subtractor()
    kernel = get_structuring_element(config.KERNEL_SIZE)

    # Initialize tracking dictionaries and counters
    vehicle_trackers = {}
    vehicle_counter = 0  # Unique vehicle ID counter
    crossed_vehicle_ids = set()  # Track vehicles that have crossed the line

    # Initialize display window
    cv2.namedWindow(config.DISPLAY_WINDOW_NAME, cv2.WINDOW_NORMAL)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("End of video stream or cannot fetch the frame.")
            break

        # Apply background subtraction and morphological operations
        fgmask = process_frame(frame, fgbg, kernel)

        # Find contours in the foreground mask
        contours = find_vehicle_contours(fgmask)

        for contour in contours:
            area = cv2.contourArea(contour)
            # Adjust the contour area threshold to filter out small objects
            if area > config.AREA_THRESHOLD:
                x, y, w, h = cv2.boundingRect(contour)
                center = (x + w // 2, y + h // 2)

                # Check if the bounding box represents a valid vehicle size
                if h > config.MIN_HEIGHT and w > config.MIN_WIDTH:
                    vehicle_id = f"vehicle_{vehicle_counter}"
                    vehicle_counter += 1

                    # Initialize a new tracker for the vehicle
                    kalman_tracker = VehicleKalmanTracker()
                    vehicle_trackers[vehicle_id] = kalman_tracker

                    # Predict the next position of the vehicle
                    predicted_x, predicted_y = kalman_tracker.predict()

                    # Correct the Kalman filter with the new measurement (center of the bounding box)
                    measurement = np.array([[np.float32(center[0])], [np.float32(center[1])]])
                    kalman_tracker.correct(measurement)

                    # Draw the predicted position
                    cv2.circle(frame, (predicted_x, predicted_y), config.PREDICTION_RADIUS, 
                               config.PREDICTION_COLOR, -1)  # Predicted position

                    # Draw the bounding box around the vehicle
                    cv2.rectangle(frame, (x, y), (x + w, y + h), config.BOUNDING_BOX_COLOR, config.BOUNDING_BOX_THICKNESS)

        # Remove trackers for vehicles that have exited the frame
        for vehicle_id in list(vehicle_trackers.keys()):
            kalman_tracker = vehicle_trackers[vehicle_id]
            predicted_position = kalman_tracker.predict()
            if predicted_position[1] < 0 or predicted_position[1] > frame.shape[0]:
                del vehicle_trackers[vehicle_id]
                crossed_vehicle_ids.discard(vehicle_id)  # Reset crossed status if applicable

        # Optionally, display the foreground mask for debugging
        cv2.imshow('Foreground Mask', fgmask)

        # Display the processed frame
        cv2.imshow(config.DISPLAY_WINDOW_NAME, frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(30) & 0xFF == ord('q'):
            print("Quitting video processing.")
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
