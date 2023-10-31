import os

from PyQt5.QtCore import QRegExp
from PyQt5.QtGui import QRegExpValidator
from PyQt5.QtWidgets import QApplication, QFileDialog, QMainWindow

import calibrate
import stereo
from ui.mainwindow import Ui_MainWindow


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self) -> None:
        super(MainWindow, self).__init__()
        self.setupUi(self)

        self.imgpaths = []
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

    def load_folder(self) -> None:
        """Load a folder of images."""
        # Get image paths
        self.imgpaths = []
        folder = QFileDialog.getExistingDirectory()
        for dirpath, _, filenames in os.walk(folder):
            for file in filenames:
                if file.endswith(("jpg", "png", "jpeg", "bmp")):
                    self.imgpaths.append(os.path.join(dirpath, file))

        # Sort image paths
        self.imgpaths.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

        # Update calibrator
        self.calibrator.update_imgpaths(self.imgpaths)

        # Update combo box
        self.cmbQ13.clear()
        self.cmbQ13.addItems([str(i + 1) for i in range(len(self.imgpaths))])

    def load_image_l(self) -> None:
        self.img_l_path = QFileDialog.getOpenFileName(
            filter="Image files (*.jpg *.png *.jpeg *.bmp)"
        )[0]

    def load_image_r(self) -> None:
        self.img_r_path = QFileDialog.getOpenFileName(
            filter="Image files (*.jpg *.png *.jpeg *.bmp)"
        )[0]


if __name__ == "__main__":
    app = QApplication([])
    form = MainWindow()
    form.show()
    app.exec_()
