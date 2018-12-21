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
import random

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

        height, width = I.shape

        Ix = np.zeros((height, width), dtype=np.float64) # Gradient x
        Iy = np.zeros((height, width), dtype=np.float64) # Gradient y

        I = I.astype(np.float64)

        for h in range(1,height-1):
            for w in range(1,width-1):
                Ix[h,w] = (I[h+1,w] - I[h-1,w]) / 2 # X gradient of image pixel
                Iy[h,w] = (I[h,w+1] - I[h,w-1]) / 2 # Y gradient of image pixel

        Ix2 = Ix*Ix # square of Ix
        Iy2 = Iy*Iy # square of Iy
        Ixy = Ix*Iy # Ix * Iy

        k = 0.04
        threshold = 530000

        imageCopy = self.cornerImage.copy()

        for h in range(3,height-3):
            for w in range(3,width-3):
                G = np.zeros((2,2), dtype=np.float64)
                G[0,0] = np.sum(Ix2[h-1:h+2,w-1:w+2], dtype=np.float64)
                G[0,1] = np.sum(Ixy[h-1:h+2,w-1:w+2], dtype=np.float64)
                G[1,0] = np.sum(Ixy[h-1:h+2,w-1:w+2], dtype=np.float64)
                G[1,1] = np.sum(Iy2[h-1:h+2,w-1:w+2], dtype=np.float64)

                det = (G[0,0]*G[1,1])-(G[0,1]*G[1,0])
                trace = G[0,0] + G[1,1]
                harris = det - k*(trace*trace)

                if harris > threshold:
                    cv2.circle(imageCopy, (w,h), 2, (0,255,0), cv2.FILLED, cv2.LINE_AA, 0)

        self.deleteItemsFromWidget(self.cornerGroupBox.layout())
        self.addImageToGroupBox(imageCopy, self.cornerGroupBox, 'Corner image')

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

        # Convert image to grayscale
        grayscaleImage = cv2.cvtColor(self.segmentImage, cv2.COLOR_BGR2GRAY)

        height, width = grayscaleImage.shape

        mask = grayscaleImage.copy()
        maskThreshold = 60

        for h in range(height):
            for w in range(width):
                if(mask[h,w] < maskThreshold):
                    mask[h,w] = 0
                else:
                    mask[h,w] = 255

        kernel = np.ones((10,10), dtype=np.uint8)
        brainMask = cv2.erode(mask, kernel, iterations = 2) # Remove skull
        brainMask = cv2.dilate(brainMask, kernel, iterations = 2) # Recover size

        segmented = self.kMeansClustering(grayscaleImage, brainMask)

        tumor = segmented.copy()
        for h in range(height):
            for w in range(width):
                if segmented[h,w] == 255:
                    tumor[h,w] = 255
                else:
                    tumor[h,w] = 0

        kernel = np.ones((3,3), dtype=np.uint8)
        tumor = cv2.morphologyEx(tumor, cv2.MORPH_CLOSE, kernel)
        tumorBound = cv2.morphologyEx(tumor, cv2.MORPH_GRADIENT, kernel)

        image = cv2.cvtColor(grayscaleImage, cv2.COLOR_GRAY2RGB)

        for h in range(height):
            for w in range(width):
                if tumorBound[h,w] == 255:
                    image[h,w,0] = 255
                    image[h,w,1] = 0
                    image[h,w,2] = 0

        tumorImage = cv2.cvtColor(tumorBound, cv2.COLOR_GRAY2RGB)

        self.deleteItemsFromWidget(self.segmentGroupBox.layout())
        self.addImageToGroupBox(image, self.segmentGroupBox, 'MR image')

    def kMeansClustering(self, image, mask):
        height, width = image.shape

        # Randomly pick cluster centers (centroids)
        center1 = np.zeros(2, dtype=int)
        center2 = np.zeros(2, dtype=int)

        imageCopy = image.copy().astype(int)

        while mask[center1[0], center1[1]] == 0:
            center1[0] = random.randrange(height)
            center1[1] = random.randrange(width)

        while mask[center2[0], center2[1]] == 0:
            center2[0] = random.randrange(height)
            center2[1] = random.randrange(width)

        tolerance = int(round((height*width) * 1 / 100))
        change = tolerance+1

        segmentationImage = np.full((height,width), 2, dtype=int)
        centersTotal = np.zeros((2,3), dtype=int)
        while change > tolerance:
            change = 0
            centersTotal = np.zeros((2,3), dtype=int)
            for h in range(height):
                for w in range(width):
                    if mask[h,w] == 255:
                        differences = np.zeros(2, dtype=int)
                        differences[0] = abs(imageCopy[h,w] - imageCopy[center1[0],center1[1]])
                        differences[1] = abs(imageCopy[h,w] - imageCopy[center2[0],center2[1]])

                        if (differences[0] == differences[1]):
                            cluster = random.randrange(2)
                        else:
                            cluster = np.argmin(differences)

                        if segmentationImage[h,w] != cluster:
                            change += 1

                        centersTotal[cluster, 0] += h
                        centersTotal[cluster, 1] += w
                        centersTotal[cluster, 2] += 1
                        segmentationImage[h,w] = cluster

            center1[0] = int(round(centersTotal[0, 0] / centersTotal[0, 2]))
            center1[1] = int(round(centersTotal[0, 1] / centersTotal[0, 2]))

            center2[0] = int(round(centersTotal[1, 0] / centersTotal[1, 2]))
            center2[1] = int(round(centersTotal[1, 1] / centersTotal[1, 2]))

        colors = np.full(2, 255, dtype=np.uint8)
        if centersTotal[0,2] > centersTotal[1,2]:
            colors[0] = 127
        else:
            colors[1] = 127

        for h in range(height):
            for w in range(width):
                if mask[h,w] == 255:
                    imageCopy[h,w] = colors[segmentationImage[h,w]]
                else:
                    imageCopy[h,w] = 0

        return imageCopy.astype(np.uint8)

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