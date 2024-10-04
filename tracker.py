import cv2
import numpy as np

class VehicleKalmanTracker:
    """
    A class to handle vehicle tracking using Kalman Filter.
    """
    def __init__(self):
        # Initialize Kalman Filter with 4 dynamic parameters and 2 measurement parameters
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                                  [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                                 [0, 1, 0, 1],
                                                 [0, 0, 1, 0],
                                                 [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5

    def predict(self):
        """
        Predict the next position of the vehicle.
        Returns predicted coordinates
        """
        prediction = self.kalman.predict()
        predicted_x = int(prediction[0])
        predicted_y = int(prediction[1])
        return predicted_x, predicted_y

    def correct(self, measurement):
        """
        Correct the Kalman Filter with the new measurement.
        Args:
            measurement (np.ndarray): The measurement array [[x], [y]].
        """
        self.kalman.correct(measurement)
