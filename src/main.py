import os

import cv2
import torch
import torchvision
from matplotlib import pyplot as plt
from PIL import Image
from PyQt5.QtCore import QRegExp
from PyQt5.QtGui import QPixmap, QRegExpValidator
from PyQt5.QtWidgets import QApplication, QFileDialog, QMainWindow
from torchsummary import summary
from torchvision.transforms import v2

import calibrate
import sift
import stereo
from train_vgg19 import transforms_test, transforms_train
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

        # Load VGG19 model
        self.class_labels = [
            "airplane",
            "automobile",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        ]
        self.inference_img = None
        self.vgg19_model = torchvision.models.vgg19_bn(num_classes=10)
        self.vgg19_model.load_state_dict(
            torch.load(
                os.path.abspath(
                    os.path.join(
                        __file__, os.pardir, os.pardir, "models", "vgg19_bn.pth"
                    )
                ),
                map_location="cpu",
            )
        )
        self.vgg19_model.eval()

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

        # Q5
        self.btnVGGLoadImg.clicked.connect(self.load_inference_image)
        self.btnQ51.clicked.connect(self.show_augmented_images)
        self.btnQ52.clicked.connect(lambda: summary(self.vgg19_model, (3, 32, 32)))
        self.btnQ53.clicked.connect(self.show_loss_and_acc)
        self.btnQ54.clicked.connect(self.inference)

    def load_folder(self) -> None:
        """Load a folder of chessboard images for Q1 and Q2."""
        # Get image paths
        directory = QFileDialog.getExistingDirectory()
        if directory == "":
            return

        imgpaths = []
        for filename in sorted(
            os.listdir(directory), key=lambda x: int(x.split(".")[0])
        ):
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

    def load_inference_image(self) -> None:
        """Load an image for inference with the VGG19 model."""
        filepath = QFileDialog.getOpenFileName(
            filter="Image files (*.jpg *.png *.jpeg *.bmp)"
        )[0]
        if filepath == "":
            return

        # Load image
        self.inference_img = Image.open(filepath)

        # Show image
        qt_img = QPixmap(filepath)
        self.lblPredResult.setText("")
        self.lblInferenceImg.setPixmap(qt_img.scaled(128, 128))

    def inference(self) -> None:
        """Inference with the VGG19 model."""
        if self.inference_img is None:
            return

        pred = self.vgg19_model(transforms_test(self.inference_img).unsqueeze(0))
        pred_label = self.class_labels[int(torch.argmax(pred, dim=1).item())]

        # Show predict label
        self.lblPredResult.setText(pred_label)

        # Show predict probability
        plt.bar(self.class_labels, torch.softmax(pred, dim=1).squeeze().tolist())
        plt.title("Probability of each class")
        plt.ylabel("Probability")
        plt.ylim(0, 1)
        plt.yticks([i / 10 for i in range(0, 11)])
        plt.xlabel("Class")
        plt.xticks(rotation=45)
        plt.show()

    def show_augmented_images(self) -> None:
        """Show example of augmented images for Q5."""
        data_dir = os.path.abspath(
            os.path.join(__file__, os.pardir, os.pardir, "data", "Q5_image", "Q5_1")
        )

        # Plot augmented images
        fig = plt.figure(figsize=(32, 32))
        for i, filename in enumerate(sorted(os.listdir(data_dir))):
            if filename.endswith(("jpg", "png", "jpeg", "bmp")):
                image = Image.open(os.path.join(data_dir, filename))
                augmented = v2.ToPILImage()(transforms_train(image))

                fig.add_subplot(3, 3, i + 1)

                plt.title(filename.split(".")[0])
                plt.imshow(augmented)

        plt.show()

    def show_loss_and_acc(self) -> None:
        """Show loss and accuracy for the VGG19 model."""
        image = cv2.imread(
            os.path.abspath(
                os.path.join(__file__, os.pardir, os.pardir, "logs", "loss_and_acc.png")
            )
        )
        cv2.imshow("Loss and Accuracy", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    app = QApplication([])
    form = MainWindow()
    form.show()
    app.exec_()
