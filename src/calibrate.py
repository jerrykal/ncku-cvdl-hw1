import os
from string import ascii_uppercase
from typing import List

import cv2
import numpy as np
from cv2.typing import Point


class Calibrator:
    def __init__(self, imgpaths: List[str]) -> None:
        self.imgpaths = imgpaths
        self.calibrated = False
        self.pattern_size = (11, 8)
        self.corners = []
        self.objpoints = []
        self.imgpoints = []
        self.camera_matrix: np.ndarray = np.zeros((3, 3))
        self.dist_coeffs: np.ndarray = np.zeros((1, 5))
        self.rvecs = []
        self.tvecs = []

        # Q2
        fs = cv2.FileStorage(
            os.path.abspath(
                os.path.join(__file__, os.pardir, "Q2_lib", "alphabet_lib_onboard.txt")
            ),
            cv2.FILE_STORAGE_READ,
        )
        self.ar_alphabet = {}
        for alphabet in ascii_uppercase:
            self.ar_alphabet[alphabet] = fs.getNode(alphabet).mat()

        fs = cv2.FileStorage(
            os.path.abspath(
                os.path.join(__file__, os.pardir, "Q2_lib", "alphabet_lib_vertical.txt")
            ),
            cv2.FILE_STORAGE_READ,
        )
        self.ar_alphabet_v = {}
        for alphabet in ascii_uppercase:
            self.ar_alphabet_v[alphabet] = fs.getNode(alphabet).mat()

    def update_imgpaths(self, imgpaths: List[str]) -> None:
        """Update image paths."""
        self.imgpaths = imgpaths
        self.calibrated = False
        self.corners = []
        self.objpoints = []
        self.imgpoints = []
        self.camera_matrix: np.ndarray = np.zeros((3, 3))
        self.dist_coeffs: np.ndarray = np.zeros((1, 5))
        self.rvecs = []
        self.tvecs = []

    def calibrate_camera(self) -> None:
        """Calibrate camera and store all relevant parameters."""
        if len(self.imgpaths) == 0:
            return

        # Prepare object points
        objp = np.zeros((np.prod(self.pattern_size), 3), np.float32)
        objp[:, :2] = np.indices(self.pattern_size).T.reshape(-1, 2)

        # Find corners
        for filepath in self.imgpaths:
            grayimg = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

            # Find chessboard corners
            found, corners = cv2.findChessboardCorners(grayimg, self.pattern_size, None)
            if found:
                corners = cv2.cornerSubPix(
                    grayimg,
                    corners,
                    (5, 5),
                    (-1, -1),
                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
                )
                self.objpoints.append(objp)
                self.imgpoints.append(corners)
                self.corners.append(corners)

        grayimg = cv2.imread(self.imgpaths[0], cv2.IMREAD_GRAYSCALE)
        image_size = grayimg.shape[::-1]

        # Calibrate camera
        (
            _,
            self.camera_matrix,
            self.dist_coeffs,
            self.rvecs,
            self.tvecs,
        ) = cv2.calibrateCamera(
            self.objpoints, self.imgpoints, image_size, None, None  # type: ignore
        )

        # Set flag
        self.calibrated = True

    def draw_corners(self) -> None:
        """Draw corners of chessboard in each images."""
        if len(self.imgpaths) == 0:
            return

        if not self.calibrated:
            self.calibrate_camera()

        # Draw chessboard corners onto the image
        for filepath, corners in zip(self.imgpaths, self.corners):
            image = cv2.imread(filepath, cv2.IMREAD_COLOR)
            cv2.drawChessboardCorners(image, self.pattern_size, corners, True)

            # Show image
            cv2.imshow("Chessboard Corners", image)

            # Wait for 1 second
            cv2.waitKey(1000)

        cv2.destroyAllWindows()

    def print_intrinsic(self) -> None:
        """Print intrinsic matrix of camera."""
        if len(self.imgpaths) == 0:
            return

        if not self.calibrated:
            self.calibrate_camera()

        # Print intrinsic parameters on console
        print("Intrinsic:")
        print(self.camera_matrix)

    def print_extrinsic(self, idx: int) -> None:
        """Print extrinsic matrix of camera for a given image."""
        if len(self.imgpaths) == 0:
            return

        if not self.calibrated:
            self.calibrate_camera()

        # Get extrinsic matrix
        self.extrinsic_matrix = np.zeros((4, 4))
        self.extrinsic_matrix[:3, :3], _ = cv2.Rodrigues(self.rvecs[idx])
        self.extrinsic_matrix[:3, 3] = self.tvecs[idx].T

        # Print extrinsic parameters on console
        print("Extrinsic:")
        print(self.extrinsic_matrix)

    def print_distortion(self) -> None:
        """Print distortion coefficients of camera."""
        if len(self.imgpaths) == 0:
            return

        if not self.calibrated:
            self.calibrate_camera()

        # Print distortion coefficients on console
        print("Distortion:")
        print(self.dist_coeffs)

    def show_undistort(self) -> None:
        """Show undistorted images."""
        if len(self.imgpaths) == 0:
            return

        if not self.calibrated:
            self.calibrate_camera()

        # Show undistorted images
        for filepath in self.imgpaths:
            image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            image_undistort = cv2.undistort(image, self.camera_matrix, self.dist_coeffs)

            # Show image side by side
            image_concat = np.concatenate((image, image_undistort), axis=1)
            cv2.imshow("Distorted vs Undistorted Image", image_concat)

            # Wait for 1 second
            cv2.waitKey(1000)

        cv2.destroyAllWindows()

    def project_word(self, word: str, vertical: bool = False) -> None:
        """Project a word onto the chessboard."""
        if len(self.imgpaths) == 0:
            return

        if not self.calibrated:
            self.calibrate_camera()

        word = word.upper()
        for i, filepath in enumerate(self.imgpaths):
            image = cv2.imread(filepath, cv2.IMREAD_COLOR)

            # Project word onto the chessboard
            for j, alphabet in enumerate(word):
                translation = np.array([7 - 3 * (j % 3), 5 - 3 * (j // 3), 0])
                for line in (
                    self.ar_alphabet[alphabet]
                    if not vertical
                    else self.ar_alphabet_v[alphabet]
                ):
                    line_pts = cv2.projectPoints(
                        (line + translation).astype(np.float32),
                        self.rvecs[i],
                        self.tvecs[i],
                        self.camera_matrix,
                        self.dist_coeffs,
                    )[0]
                    cv2.line(
                        image,
                        np.ravel(line_pts[0]).astype(int),  # type: ignore
                        np.ravel(line_pts[1]).astype(int),  # type: ignore
                        (0, 0, 255),
                        10,
                    )

            # Show image
            cv2.imshow("AR", image)

            # Wait for 1 second
            cv2.waitKey(1000)

        cv2.destroyAllWindows()
