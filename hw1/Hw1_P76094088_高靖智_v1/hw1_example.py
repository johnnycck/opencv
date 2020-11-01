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
        dog = cv.imread( '../img/dog.bmp', cv.IMREAD_COLOR )
        dog = cv.cvtColor(dog, cv.COLOR_BGR2RGB)
        print('Height:', dog.shape[0])
        print('width:', dog.shape[1])
        plt.imshow(dog)
        plt.xticks([]), plt.yticks([])
        plt.show()
        #cv.waitKey( 0 )
        #cv.destroyWindow('image')

    def on_btn1_2_click(self):
        img = cv.imread( '../img/color.png', cv.IMREAD_COLOR )
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        #rbg = img[...,[1,0,2]]
        rbg = img[...,[2,0,1]]
        plt.imshow(rbg)
        plt.xticks([]), plt.yticks([])
        plt.show()

    def on_btn1_3_click(self):
        dog = cv.imread( '../img/dog.bmp', cv.IMREAD_COLOR )
        dog = cv.cvtColor(dog, cv.COLOR_BGR2RGB)
        dog__f = cv.flip(dog, 1)
        plt.imshow(dog__f)
        plt.xticks([]), plt.yticks([])
        plt.show()

    def on_btn1_4_click(self):
        #load image
        def Chang(x):
            hul = cv.getTrackbarPos('Blend', '4')
            dst = cv.addWeighted(img1,hul/100,img2,(100-hul)/100,0)
            cv.imshow('4',dst)
        dog = cv.imread( '../img/dog.bmp', cv.IMREAD_COLOR )
        img1 = cv.cvtColor(dog, cv.COLOR_BGR2RGB)
        img2 = cv.flip(img1, 1)
        cv.namedWindow('4')
        cv.createTrackbar('Blend', '4', 0,100,Chang)
        # fig = gcf()
        # fig.canvas.manager.window.raise_()
        dst = cv.addWeighted(img1,1,img2,0,0) #dst = img*1+dst*0+gamma
        plt.imshow('4',dst)
        plt.show()

        # #initial image alpha = 0
        # dst = cv.addWeighted(img1, 0, img2, 1, 0)
        # plt.imshow(dst)

        # #adjust the image position
        # plt.subplots_adjust(left=0.25, bottom=0.25)

        # #setting of trackbar
        # axcolor = 'lightgoldenrodyellow'
        # axamp = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)
        # bar = Slider(axamp, 'BLEND', 0, 1, 0)
        # #plt.show()
        # def update(val):
        #     alpha = bar.val
        #     beta = 1 - alpha
        #     dst.set_data(img1, alpha, img2, beta, 0)
        #     draw()
        # bar.on_changed(update)
        # plt.show()


    def on_btn2_1_click(self):
        screw = cv.imread( '../img/M8.jpg', cv.IMREAD_COLOR )
        screw = cv.cvtColor(screw, cv.COLOR_BGR2RGB)
        blur = cv.GaussianBlur(screw,(3,3),0)
        #screw_G = cv.cvtColor(screw,cv.COLOR_BGR2GRAY)
        plt.imshow(screw)
        plt.show()

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
