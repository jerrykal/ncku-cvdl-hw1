import os

import cv2
import numpy as np


class Calibrator:
    def __init__(self, imgpaths):
        self.imgpaths = imgpaths
        self.calibrated = False
        self.pattern_size = (11, 8)
        self.corners = []
        self.objpoints = []
        self.imgpoints = []
        self.camera_matrix = []
        self.dist_coeffs = []
        self.rvecs = []
        self.tvecs = []

    def update_imgpaths(self, imgpaths):
        """Update image paths."""
        self.imgpaths = imgpaths
        self.calibrated = False

    def calibrate_camera(self):
        """Calibrate camera and store all relevant parameters."""
        # Prepare object points
        objp = np.zeros((np.prod(self.pattern_size), 3), np.float32)
        objp[:, :2] = np.indices(self.pattern_size).T.reshape(-1, 2)

        # Find corners
        for filepath in self.imgpaths:
            grayimg = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

            # Find chessboard corners
            found, corners = cv2.findChessboardCorners(grayimg, self.pattern_size, None)
            if found:
                self.objpoints.append(objp)
                self.imgpoints.append(corners)
                self.corners.append(corners)

        # Get image size
        grayimg = cv2.imread(self.imgpaths[0], cv2.IMREAD_GRAYSCALE)

        # Calibrate camera
        (
            _,
            self.camera_matrix,
            self.dist_coeffs,
            self.rvecs,
            self.tvecs,
        ) = cv2.calibrateCamera(
            self.objpoints, self.imgpoints, grayimg.shape[::-1], None, None  # type: ignore
        )

        # Set flag
        self.calibrated = True

    def draw_corners(self):
        """Draw corners of chessboard in each images."""
        if not self.calibrated:
            self.calibrate_camera()

        # Draw chessboard corners onto the image
        for filepath, corners in zip(self.imgpaths, self.corners):
            image = cv2.imread(filepath, cv2.IMREAD_COLOR)
            cv2.drawChessboardCorners(image, self.pattern_size, corners, True)

            # Show image
            cv2.imshow(f"Chessboard Corners: {os.path.basename(filepath)}", image)
            cv2.waitKey(0)

        cv2.destroyAllWindows()

    def print_intrinsic(self):
        """Print intrinsic matrix of camera."""
        if not self.calibrated:
            self.calibrate_camera()

        # Print intrinsic parameters on console
        print("Intrinsic:")
        print(self.camera_matrix)

    def print_extrinsic(self, idx):
        """Print extrinsic matrix of camera for a given image."""
        if not self.calibrated:
            self.calibrate_camera()

        # Get extrinsic matrix
        self.extrinsic_matrix = np.zeros((4, 4))
        self.extrinsic_matrix[:3, :3], _ = cv2.Rodrigues(self.rvecs[idx])
        self.extrinsic_matrix[:3, 3] = self.tvecs[idx].T

        # Print extrinsic parameters on console
        print("Extrinsic:")
        print(self.extrinsic_matrix)

    def print_distortion(self):
        """Print distortion coefficients of camera."""
        if not self.calibrated:
            self.calibrate_camera()

        # Print distortion coefficients on console
        print("Distortion:")
        print(self.dist_coeffs)

    def show_undistort(self):
        """Show undistorted images."""
        if not self.calibrated:
            self.calibrate_camera()

        # Show undistorted images
        for filepath in self.imgpaths:
            image = cv2.imread(filepath, cv2.IMREAD_COLOR)
            image_undistort = cv2.undistort(image, self.camera_matrix, self.dist_coeffs)  # type: ignore

            # Show image side by side
            image_concat = np.concatenate((image, image_undistort), axis=1)
            cv2.imshow(
                f"Distorted vs Undistorted Image: {os.path.basename(filepath)}",
                image_concat,
            )
            cv2.waitKey(0)

        cv2.destroyAllWindows()
