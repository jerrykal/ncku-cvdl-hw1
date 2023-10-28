import os

from PyQt5.QtWidgets import QApplication, QFileDialog, QMainWindow

import calibrate
from ui.mainwindow import Ui_MainWindow


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)

        self.imgpaths = []
        self.calibrator = calibrate.Calibrator([])

        # Connect signals and slots
        self.btnLoadFolder.clicked.connect(self.load_folder)

        # Q1
        self.btnQ11.clicked.connect(self.calibrator.draw_corners)
        self.btnQ12.clicked.connect(self.calibrator.print_intrinsic)
        self.btnQ13.clicked.connect(
            lambda: self.calibrator.print_extrinsic(self.cmbQ13.currentIndex())
        )
        self.btnQ14.clicked.connect(self.calibrator.print_distortion)
        self.btnQ15.clicked.connect(self.calibrator.show_undistort)

    def load_folder(self):
        """Load a folder of images."""
        # Get image paths
        self.imgpaths = []
        folder = QFileDialog.getExistingDirectory()
        for dirpath, _, filenames in os.walk(folder):
            for file in filenames:
                if file.endswith(("jpg", "png", "jpeg", "bmp")):
                    self.imgpaths.append(os.path.join(dirpath, file))

        # Sort image paths
        self.imgpaths.sort(key=lambda x: int(x.split("/")[-1].split(".")[0]))

        # Update calibrator
        self.calibrator.update_imgpaths(self.imgpaths)

        # Update combo box
        self.cmbQ13.clear()
        self.cmbQ13.addItems([str(i + 1) for i in range(len(self.imgpaths))])


if __name__ == "__main__":
    app = QApplication([])
    form = MainWindow()
    form.show()
    app.exec_()
