import os

from PyQt5.QtCore import QRegExp
from PyQt5.QtGui import QRegExpValidator
from PyQt5.QtWidgets import QApplication, QFileDialog, QMainWindow

import calibrate
import sift
import stereo
from ui.mainwindow import Ui_MainWindow


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self) -> None:
        super(MainWindow, self).__init__()
        self.setupUi(self)

        self.img_l_path = ""
        self.img_r_path = ""
        self.img_sift_path_1 = ""
        self.img_sift_path_2 = ""
        self.calibrator = calibrate.Calibrator([])

        # Connect signals and slots
        self.btnLoadFolder.clicked.connect(self.load_folder)
        self.btnLoadImgL.clicked.connect(self.load_image_l)
        self.btnLoadImgR.clicked.connect(self.load_image_r)

        # Q1
        self.btnQ11.clicked.connect(self.calibrator.draw_corners)
        self.btnQ12.clicked.connect(self.calibrator.print_intrinsic)
        self.btnQ13.clicked.connect(
            lambda: self.calibrator.print_extrinsic(self.cmbQ13.currentIndex())
        )
        self.btnQ14.clicked.connect(self.calibrator.print_distortion)
        self.btnQ15.clicked.connect(self.calibrator.show_undistort)

        # Set validator for Q2
        regex = QRegExp("[a-zA-Z]+")
        validator = QRegExpValidator(regex)
        self.lineEditQ2.setValidator(validator)

        # Q2
        self.btnQ21.clicked.connect(
            lambda: self.calibrator.project_word(self.lineEditQ2.text())
        )
        self.btnQ22.clicked.connect(
            lambda: self.calibrator.project_word(self.lineEditQ2.text(), vertical=True)
        )

        # Q3
        self.btnQ31.clicked.connect(
            lambda: stereo.stereo_disparity_map(self.img_l_path, self.img_r_path)
        )

        # Q4
        self.btnSIFTLoadImg1.clicked.connect(self.load_sift_image_1)
        self.btnSIFTLoadImg2.clicked.connect(self.load_sift_image_2)
        self.btnQ41.clicked.connect(lambda: sift.draw_keypoints(self.img_sift_path_1))
        self.btnQ42.clicked.connect(
            lambda: sift.draw_matched_keypoints(
                self.img_sift_path_1, self.img_sift_path_2
            )
        )

    def load_folder(self) -> None:
        """Load a folder of chessboard images for Q1 and Q2."""
        # Get image paths
        imgpaths = []
        directory = QFileDialog.getExistingDirectory()
        for filename in sorted(os.listdir(directory)):
            if filename.endswith(("jpg", "png", "jpeg", "bmp")):
                imgpaths.append(os.path.join(directory, filename))

        # Update calibrator
        self.calibrator.update_imgpaths(imgpaths)

        # Update combo box
        self.cmbQ13.clear()
        self.cmbQ13.addItems([str(i + 1) for i in range(len(imgpaths))])

    def load_image_l(self) -> None:
        """Load left stereo image for Q3."""
        self.img_l_path = QFileDialog.getOpenFileName(
            filter="Image files (*.jpg *.png *.jpeg *.bmp)"
        )[0]

    def load_image_r(self) -> None:
        """Load right stereo image for Q3."""
        self.img_r_path = QFileDialog.getOpenFileName(
            filter="Image files (*.jpg *.png *.jpeg *.bmp)"
        )[0]

    def load_sift_image_1(self) -> None:
        """Load first image for Q4."""
        self.img_sift_path_1 = QFileDialog.getOpenFileName(
            filter="Image files (*.jpg *.png *.jpeg *.bmp)"
        )[0]

    def load_sift_image_2(self) -> None:
        """Load second image for Q4."""
        self.img_sift_path_2 = QFileDialog.getOpenFileName(
            filter="Image files (*.jpg *.png *.jpeg *.bmp)"
        )[0]


if __name__ == "__main__":
    app = QApplication([])
    form = MainWindow()
    form.show()
    app.exec_()
