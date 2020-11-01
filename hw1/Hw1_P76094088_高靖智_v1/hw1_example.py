# -*- coding: utf-8 -*-

import sys
from hw1_ui import Ui_MainWindow
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QMainWindow, QApplication
from matplotlib.widgets import Slider, Button, RadioButtons

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        self.onBindingUI()

    # Write your code below
    # UI components are defined in hw1_ui.py, please take a look.
    # You can also open hw1.ui by qt-designer to check ui components.

    def onBindingUI(self):
        self.btn1_1.clicked.connect(self.on_btn1_1_click)
        self.btn1_2.clicked.connect(self.on_btn1_2_click)
        self.btn1_3.clicked.connect(self.on_btn1_3_click)
        self.btn1_4.clicked.connect(self.on_btn1_4_click)
        self.btn2_1.clicked.connect(self.on_btn2_1_click)
        self.btn3_1.clicked.connect(self.on_btn3_1_click)
        self.btn4_1.clicked.connect(self.on_btn4_1_click)
        self.btn4_2.clicked.connect(self.on_btn4_2_click)
        self.btn5_1.clicked.connect(self.on_btn5_1_click)
        self.btn5_2.clicked.connect(self.on_btn5_2_click)

    for p in sys.path:
        print( p )
    # button for problem 1.1
    def on_btn1_1_click(self):
        uncleroger = cv.imread( '../Q1_image/Uncle_Roger.jpg', cv.IMREAD_COLOR )
        uncleroger = cv.cvtColor(uncleroger, cv.COLOR_BGR2RGB)
        print('Height:', uncleroger.shape[0])
        print('width:', uncleroger.shape[1])
        plt.imshow(uncleroger)
        plt.xticks([]), plt.yticks([])
        plt.show()
        cv.waitKey( 0 )
        cv.destroyWindow('uncleroger')

    def on_btn1_2_click(self):
        img = cv.imread( '../Q1_image/Flower.jpg', cv.IMREAD_COLOR )
        #blue
        img[:,:,1] = 0
        img[:,:,2] = 0
        cv.imshow('blue_img', img)

        img = cv.imread( '../Q1_image/Flower.jpg', cv.IMREAD_COLOR )
        #green
        img[:,:,0] = 0
        img[:,:,2] = 0
        cv.imshow('green_img', img)

        img = cv.imread( '../Q1_image/Flower.jpg', cv.IMREAD_COLOR )
        # red
        img[:,:,0] = 0
        img[:,:,1] = 0
        cv.imshow('red_img', img)

    def on_btn1_3_click(self):
        uncleroger = cv.imread( '../Q1_image/Uncle_Roger.jpg', cv.IMREAD_COLOR )
        uncleroger = cv.cvtColor(uncleroger, cv.COLOR_BGR2RGB)
        uncleroger__f = cv.flip(uncleroger, 1)
        plt.imshow(uncleroger__f)
        plt.xticks([]), plt.yticks([])
        plt.show()

    def on_btn1_4_click(self):
        #load image
        def Change(x):
            alpha = cv.getTrackbarPos('Blend', 'Blending')/100
            dst = cv.addWeighted(img1,alpha,img2,1-alpha,0)
            cv.imshow('Blending',dst)
        img1 = cv.imread( '../Q1_image/Uncle_Roger.jpg', cv.IMREAD_COLOR )
        img2 = cv.flip(img1, 1)
        cv.namedWindow('Blending')
        cv.createTrackbar('Blend', 'Blending', 0,100,Change)
        # default image : flipped image
        dst = cv.addWeighted(img1,0,img2,1,0)
        cv.imshow('Blending',dst)

    def on_btn2_1_click(self):
        img = cv.imread( '../Q2_image/Cat.png', cv.IMREAD_COLOR )
        cv.imshow('origin', img)
        dst = cv.medianBlur(img,7)
        cv.imshow('after median filter', dst)

    def on_btn3_1_click(self):
        pass

    def on_btn4_1_click(self):
        pass

    def on_btn4_2_click(self):
        pass

    def on_btn5_1_click(self):
        # edtAngle, edtScale. edtTx, edtTy to access to the ui object
        pass

    def on_btn5_2_click(self):
        pass

    ### ### ###


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
