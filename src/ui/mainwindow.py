# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'assets/mainwindow.ui'
#
# Created by: PyQt5 UI code generator 5.15.10
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(900, 680)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(20, 10, 171, 191))
        self.groupBox.setObjectName("groupBox")
        self.btnLoadImgR = QtWidgets.QPushButton(self.groupBox)
        self.btnLoadImgR.setGeometry(QtCore.QRect(20, 130, 131, 41))
        self.btnLoadImgR.setObjectName("btnLoadImgR")
        self.btnLoadImgL = QtWidgets.QPushButton(self.groupBox)
        self.btnLoadImgL.setGeometry(QtCore.QRect(20, 80, 131, 41))
        self.btnLoadImgL.setObjectName("btnLoadImgL")
        self.btnLoadFolder = QtWidgets.QPushButton(self.groupBox)
        self.btnLoadFolder.setGeometry(QtCore.QRect(20, 30, 131, 41))
        self.btnLoadFolder.setObjectName("btnLoadFolder")
        self.groupBox_2 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_2.setGeometry(QtCore.QRect(210, 10, 191, 371))
        self.groupBox_2.setObjectName("groupBox_2")
        self.btnQ11 = QtWidgets.QPushButton(self.groupBox_2)
        self.btnQ11.setGeometry(QtCore.QRect(30, 30, 131, 41))
        self.btnQ11.setObjectName("btnQ11")
        self.btnQ12 = QtWidgets.QPushButton(self.groupBox_2)
        self.btnQ12.setGeometry(QtCore.QRect(30, 80, 131, 41))
        self.btnQ12.setObjectName("btnQ12")
        self.btnQ15 = QtWidgets.QPushButton(self.groupBox_2)
        self.btnQ15.setGeometry(QtCore.QRect(30, 310, 131, 41))
        self.btnQ15.setObjectName("btnQ15")
        self.groupBox_3 = QtWidgets.QGroupBox(self.groupBox_2)
        self.groupBox_3.setGeometry(QtCore.QRect(20, 130, 151, 111))
        self.groupBox_3.setObjectName("groupBox_3")
        self.btnQ13 = QtWidgets.QPushButton(self.groupBox_3)
        self.btnQ13.setGeometry(QtCore.QRect(10, 60, 131, 41))
        self.btnQ13.setObjectName("btnQ13")
        self.cmbQ13 = QtWidgets.QComboBox(self.groupBox_3)
        self.cmbQ13.setGeometry(QtCore.QRect(30, 30, 91, 32))
        self.cmbQ13.setObjectName("cmbQ13")
        self.btnQ14 = QtWidgets.QPushButton(self.groupBox_2)
        self.btnQ14.setGeometry(QtCore.QRect(30, 260, 131, 41))
        self.btnQ14.setObjectName("btnQ14")
        self.groupBox_4 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_4.setGeometry(QtCore.QRect(420, 10, 211, 191))
        self.groupBox_4.setObjectName("groupBox_4")
        self.lineEditQ2 = QtWidgets.QLineEdit(self.groupBox_4)
        self.lineEditQ2.setGeometry(QtCore.QRect(20, 30, 171, 41))
        font = QtGui.QFont()
        font.setPointSize(20)
        self.lineEditQ2.setFont(font)
        self.lineEditQ2.setText("")
        self.lineEditQ2.setMaxLength(6)
        self.lineEditQ2.setObjectName("lineEditQ2")
        self.btnQ21 = QtWidgets.QPushButton(self.groupBox_4)
        self.btnQ21.setGeometry(QtCore.QRect(20, 80, 171, 41))
        self.btnQ21.setObjectName("btnQ21")
        self.btnQ22 = QtWidgets.QPushButton(self.groupBox_4)
        self.btnQ22.setGeometry(QtCore.QRect(20, 130, 171, 41))
        self.btnQ22.setObjectName("btnQ22")
        self.groupBox_5 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_5.setGeometry(QtCore.QRect(650, 10, 221, 91))
        self.groupBox_5.setObjectName("groupBox_5")
        self.btnQ31 = QtWidgets.QPushButton(self.groupBox_5)
        self.btnQ31.setGeometry(QtCore.QRect(20, 30, 181, 41))
        self.btnQ31.setObjectName("btnQ31")
        self.groupBox_7 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_7.setGeometry(QtCore.QRect(420, 210, 211, 241))
        self.groupBox_7.setObjectName("groupBox_7")
        self.btnSIFTLoadImg1 = QtWidgets.QPushButton(self.groupBox_7)
        self.btnSIFTLoadImg1.setGeometry(QtCore.QRect(29, 30, 151, 41))
        self.btnSIFTLoadImg1.setObjectName("btnSIFTLoadImg1")
        self.btnSIFTLoadImg2 = QtWidgets.QPushButton(self.groupBox_7)
        self.btnSIFTLoadImg2.setGeometry(QtCore.QRect(30, 80, 151, 41))
        self.btnSIFTLoadImg2.setObjectName("btnSIFTLoadImg2")
        self.btnQ41 = QtWidgets.QPushButton(self.groupBox_7)
        self.btnQ41.setGeometry(QtCore.QRect(30, 130, 151, 41))
        self.btnQ41.setObjectName("btnQ41")
        self.btnQ42 = QtWidgets.QPushButton(self.groupBox_7)
        self.btnQ42.setGeometry(QtCore.QRect(30, 180, 151, 41))
        self.btnQ42.setObjectName("btnQ42")
        self.groupBox_8 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_8.setGeometry(QtCore.QRect(650, 110, 221, 491))
        self.groupBox_8.setObjectName("groupBox_8")
        self.btnVGGLoadImg = QtWidgets.QPushButton(self.groupBox_8)
        self.btnVGGLoadImg.setGeometry(QtCore.QRect(20, 30, 181, 41))
        self.btnVGGLoadImg.setObjectName("btnVGGLoadImg")
        self.btnQ51 = QtWidgets.QPushButton(self.groupBox_8)
        self.btnQ51.setGeometry(QtCore.QRect(20, 80, 181, 41))
        self.btnQ51.setObjectName("btnQ51")
        self.btnQ52 = QtWidgets.QPushButton(self.groupBox_8)
        self.btnQ52.setGeometry(QtCore.QRect(20, 130, 181, 41))
        self.btnQ52.setObjectName("btnQ52")
        self.btnQ53 = QtWidgets.QPushButton(self.groupBox_8)
        self.btnQ53.setGeometry(QtCore.QRect(20, 180, 181, 41))
        self.btnQ53.setObjectName("btnQ53")
        self.btnQ54 = QtWidgets.QPushButton(self.groupBox_8)
        self.btnQ54.setGeometry(QtCore.QRect(20, 230, 181, 41))
        self.btnQ54.setObjectName("btnQ54")
        self.label = QtWidgets.QLabel(self.groupBox_8)
        self.label.setGeometry(QtCore.QRect(20, 280, 58, 16))
        self.label.setObjectName("label")
        self.lblPredResult = QtWidgets.QLabel(self.groupBox_8)
        self.lblPredResult.setGeometry(QtCore.QRect(80, 280, 121, 16))
        self.lblPredResult.setText("")
        self.lblPredResult.setObjectName("lblPredResult")
        self.lblInferenceImg = QtWidgets.QLabel(self.groupBox_8)
        self.lblInferenceImg.setGeometry(QtCore.QRect(50, 310, 128, 128))
        self.lblInferenceImg.setAutoFillBackground(True)
        self.lblInferenceImg.setAlignment(QtCore.Qt.AlignCenter)
        self.lblInferenceImg.setObjectName("lblInferenceImg")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 900, 24))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.groupBox.setTitle(_translate("MainWindow", "Load Image"))
        self.btnLoadImgR.setText(_translate("MainWindow", "Load Image_R"))
        self.btnLoadImgL.setText(_translate("MainWindow", "Load Image_L"))
        self.btnLoadFolder.setText(_translate("MainWindow", "Load Folder"))
        self.groupBox_2.setTitle(_translate("MainWindow", "1. Calibration"))
        self.btnQ11.setText(_translate("MainWindow", "1.1 Find Corners"))
        self.btnQ12.setText(_translate("MainWindow", "1.2 Find Intrinsic"))
        self.btnQ15.setText(_translate("MainWindow", "1.5 Show Result"))
        self.groupBox_3.setTitle(_translate("MainWindow", "1.3 Find Extrinsic"))
        self.btnQ13.setText(_translate("MainWindow", "1.3 Find Extrinsic"))
        self.btnQ14.setText(_translate("MainWindow", "1.4 Find Distortion"))
        self.groupBox_4.setTitle(_translate("MainWindow", "2. Augmented Reality"))
        self.btnQ21.setText(_translate("MainWindow", "2.1 Show Words on Board"))
        self.btnQ22.setText(_translate("MainWindow", "2.2 Show Words Vertically"))
        self.groupBox_5.setTitle(_translate("MainWindow", "3. Stereo Disparity Map"))
        self.btnQ31.setText(_translate("MainWindow", "3.1 Stereo Disparity Map"))
        self.groupBox_7.setTitle(_translate("MainWindow", "4.SIFT"))
        self.btnSIFTLoadImg1.setText(_translate("MainWindow", "Load Image1"))
        self.btnSIFTLoadImg2.setText(_translate("MainWindow", "Load Image2"))
        self.btnQ41.setText(_translate("MainWindow", "4.1 Keypoints"))
        self.btnQ42.setText(_translate("MainWindow", "4.2 Matched Keypoints"))
        self.groupBox_8.setTitle(_translate("MainWindow", "5. VGG 19"))
        self.btnVGGLoadImg.setText(_translate("MainWindow", "Load Image"))
        self.btnQ51.setText(_translate("MainWindow", "5.1 Show Augmented\n"
"Images"))
        self.btnQ52.setText(_translate("MainWindow", "5.2 Show Model Structure"))
        self.btnQ53.setText(_translate("MainWindow", "5.3 Show Acc and Loss"))
        self.btnQ54.setText(_translate("MainWindow", "5.4 Inference"))
        self.label.setText(_translate("MainWindow", "Predict ="))
        self.lblInferenceImg.setText(_translate("MainWindow", "Inference Image"))
