#!/usr/local/bin/python3

import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QMenu, QVBoxLayout, QSizePolicy, QMessageBox, QWidget, \
    QPushButton, QGroupBox, QAction, QFileDialog, qApp
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import QGridLayout
from PyQt5.QtWidgets import QLabel
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import cv2

class App(QMainWindow):
    def __init__(self):
        super(App, self).__init__()

        self.title = 'Corner Detection and Segmentation'

        # Booelans to track if input images are loaded
        self.cornerLoaded = False
        self.segmentLoaded = False

        # Fix the size so boxes cannot expand
        self.setFixedSize(self.geometry().width(), self.geometry().height())

        self.initUI()

    def addImageToGroupBox(self, image, groupBox, labelString):
        # Get the height, width information
        height, width, channel = image.shape
        bytesPerLine = channel * width # 3-channel image

        qImg = QImage(image.data, width, height, bytesPerLine, QImage.Format_RGB888).rgbSwapped()

        pix = QPixmap(qImg)

        # Add image  to the widget
        label = QLabel(labelString)
        label.setPixmap(pix)
        label.setAlignment(Qt.AlignCenter)
        groupBox.layout().addWidget(label)

    def deleteItemsFromWidget(self, layout):
        # Deletes items in the given layout
        if layout is not None:
            while layout.count():
                item = layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()
                else:
                    deleteItemsFromWidget(item.layout())

    def openCornerImage(self):
        # This function is called when the user clicks File->Open corner file.
        fName = QFileDialog.getOpenFileName(self, 'Open corner file', './', 'Image files (*.png *.jpg)')

        # File open dialog has been dismissed or file could not be found
        if fName[0] is '':
            return

        # If there is an input image loaded, remove it
        if self.cornerLoaded:
            self.deleteItemsFromWidget(self.cornerGroupBox.layout())

        self.cornerImage = cv2.imread(fName[0]) # Read the image
        self.cornerLoaded = True

        self.addImageToGroupBox(self.cornerImage, self.cornerGroupBox, 'Corner image')

    def openSegmentImage(self):
        # This function is called when the user clicks File->Open MR file.
        fName = QFileDialog.getOpenFileName(self, 'Open MR file', './', 'Image files (*.png *.jpg)')

        # File open dialog has been dismissed or file could not be found
        if fName[0] is '':
            return

        # If there is an input image loaded, remove it
        if self.segmentLoaded:
            self.deleteItemsFromWidget(self.segmentGroupBox.layout())

        self.segmentImage = cv2.imread(fName[0]) # Read the image
        self.segmentLoaded = True

        self.addImageToGroupBox(self.segmentImage, self.segmentGroupBox, 'MR image')

    def createEmptyCornerGroupBox(self):
        self.cornerGroupBox = QGroupBox('Corner Detection')
        layout = QVBoxLayout()

        self.cornerGroupBox.setLayout(layout)

    def createEmptySegmentGroupBox(self):
        self.segmentGroupBox = QGroupBox('Tumor Segmentation')
        layout = QVBoxLayout()

        self.segmentGroupBox.setLayout(layout)

    def initUI(self):
        # Add menu bar
        menubar = self.menuBar()
        fileMenu = menubar.addMenu('File')
        
        # Create action buttons of the menu bar
        cornerAct = QAction('Open corner image', self)
        cornerAct.triggered.connect(self.openCornerImage)

        segmentAct = QAction('Open segment image', self) 
        segmentAct.triggered.connect(self.openSegmentImage)

        exitAct = QAction('Exit', self)        
        exitAct.triggered.connect(qApp.quit) # Quit the app

        # Add action buttons to the menu bar
        fileMenu.addAction(cornerAct)
        fileMenu.addAction(segmentAct)
        fileMenu.addAction(exitAct)

        # Create detect corners button for toolbar
        detectCornersAct = QAction('Detect Corners', self) 
        detectCornersAct.triggered.connect(self.detectCornersButtonClicked)

        # Create segmentation button for toolbar
        segmentationAct = QAction('Segment Tumor', self) 
        segmentationAct.triggered.connect(self.segmentationButtonClicked)
        
        # Create toolbar
        toolbar = self.addToolBar('Image Operations')
        toolbar.addAction(detectCornersAct)
        toolbar.addAction(segmentationAct)

        # Create empty group boxes 
        self.createEmptyCornerGroupBox()
        self.createEmptySegmentGroupBox()

        # Since QMainWindows layout has already been set, create central widget
        # to manipulate layout of main window
        wid = QWidget(self)
        self.setCentralWidget(wid)

        # Initialize layout with groupboxes
        windowLayout = QGridLayout()
        windowLayout.addWidget(self.cornerGroupBox, 0, 0)
        windowLayout.addWidget(self.segmentGroupBox, 0, 1)
        wid.setLayout(windowLayout)

        self.setWindowTitle(self.title) 
        self.showMaximized()
        self.show()

    def detectCornersButtonClicked(self):
        if not self.cornerLoaded:
            # Error: "First load corner detection image" in MessageBox
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setWindowTitle("Corner detection image is missing.")
            msg.setText('First load corner detection image!')
            msg.setStandardButtons(QMessageBox.Ok)

            msg.exec()
            return

        # Convert image to grayscale
        grayscaleImage = cv2.cvtColor(self.cornerImage, cv2.COLOR_BGR2GRAY)

        # Get rid of noise
        I = self.gaussianFiltering(grayscaleImage, 3, 1)

        return NotImplemented

    def segmentationButtonClicked(self):
        if not self.segmentLoaded:
            # Error: "First MR segmentation image" in MessageBox
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setWindowTitle("MR image is missing.")
            msg.setText('First MR segmentation image!')
            msg.setStandardButtons(QMessageBox.Ok)

            msg.exec()
            return

        return NotImplemented

    def gaussianFiltering(self, image, size, sigma):
        height, width = image.shape

        extendedSize = size-1
        start = int(extendedSize / 2)
        endH = start + height
        endW = start + width

        kernel = np.zeros((size,size), dtype=np.float64)

        for x in range(size):
            for y in range(size):
                kernel[x,y] = (1/(2*np.pi*sigma**2))*np.exp(-((x-start)**2 + (y-start)**2)/(2*sigma**2))

        extendedIm = np.zeros((height+extendedSize, width+extendedSize), dtype=np.float64)
        extendedIm[start:endH,start:endW] = image.astype(np.float64)

        I = image.copy()
        kernelSum = np.sum(kernel, dtype=np.float64)

        for h in range(height):
            for w in range(width):
                I[h, w] = np.sum(kernel*extendedIm[h:h+size,w:w+size], dtype=np.float64) / kernelSum

        return np.round(I).astype(np.uint8)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())