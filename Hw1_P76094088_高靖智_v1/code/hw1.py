# -*- coding: utf-8 -*-

import sys
from hw1_ui import Ui_MainWindow
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QMainWindow, QApplication, QLineEdit
from matplotlib.widgets import Slider, Button, RadioButtons
from scipy import signal
from scipy import misc
from scipy.ndimage import filters
import glob
import os
import random
import tensorflow as tf
from tensorflow.keras import models, optimizers, regularizers
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import load_model


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
        self.btn2_2.clicked.connect(self.on_btn2_2_click)
        self.btn2_3.clicked.connect(self.on_btn2_3_click)
        self.btn3_1.clicked.connect(self.on_btn3_1_click)
        self.btn3_2.clicked.connect(self.on_btn3_2_click)
        self.btn3_3.clicked.connect(self.on_btn3_3_click)
        self.btn3_4.clicked.connect(self.on_btn3_4_click)
        self.btn4_1.clicked.connect(self.on_btn4_1_click)
        self.btn5_1.clicked.connect(self.on_btn5_1_click)
        self.btn5_2.clicked.connect(self.on_btn5_2_click)
        self.btn5_3.clicked.connect(self.on_btn5_3_click)
        self.btn5_4.clicked.connect(self.on_btn5_4_click)
        self.btn5_5.clicked.connect(self.on_btn5_5_click)
        self.Inference.setText("(0~9999)")

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
        cv.waitKey(2)
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
        cv.waitKey(0)
        cv.destroyWindow('blue_img')
        cv.destroyWindow('green_img')
        cv.destroyWindow('red_img')

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
        cv.waitKey(0)
        cv.destroyWindow('Blending')
        
    def on_btn2_1_click(self):
        img = cv.imread( '../Q2_image/Cat.png', cv.IMREAD_COLOR )
        cv.imshow('origin', img)
        dst = cv.medianBlur(img,7)
        cv.imshow('after median filter', dst)
        cv.waitKey(0)
        cv.destroyWindow('origin')
        cv.destroyWindow('after median filter')

    def on_btn2_2_click(self):
        img = cv.imread( '../Q2_image/Cat.png', cv.IMREAD_COLOR )
        cv.imshow('origin', img)
        dst = cv.GaussianBlur(img,(3,3),0)
        cv.imshow('after gaussian blur', dst)
        cv.waitKey(0)
        cv.destroyWindow('origin')
        cv.destroyWindow('after gaussian blur')

    def on_btn2_3_click(self):
        img = cv.imread( '../Q2_image/Cat.png', cv.IMREAD_COLOR )
        cv.imshow('origin', img)
        dst = cv.bilateralFilter(img,9,90,90)
        cv.imshow('after bilateral filter', dst)
        cv.waitKey(0)
        cv.destroyWindow('origin')
        cv.destroyWindow('after bilateral filter')

    def on_btn3_1_click(self):
        img = cv.imread( '../Q3_image/Chihiro.jpg', cv.IMREAD_COLOR )
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        cv.imshow('Chihiro.jpg', img)
        cv.imshow('Grayscale', gray)
        #3*3 Gassian filter
        x, y = np.mgrid[-1:2, -1:2]
        gaussian_kernel = np.exp(-(x**2+y**2))
        
        #Normalization
        gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
        grad = signal.convolve2d(gray, gaussian_kernel) #卷積
        grad = grad/(grad.max()/255.0)
        grad = grad.astype(np.uint8) # 轉換成 uint8 不然會都白色
        cv.imshow('Gaussian Blur', grad)
        cv.imwrite('../Q3_image/Chihiro_GaussianBlur.jpg',grad)
        cv.waitKey(0)
        cv.destroyWindow('Chihiro.jpg')
        cv.destroyWindow('Grayscale')
        cv.destroyWindow('Gaussian Blur')

    def on_btn3_2_click(self):
        img = cv.imread( '../Q3_image/Chihiro_GaussianBlur.jpg', cv.IMREAD_COLOR )
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        kernelx = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
        img_sobelx = signal.convolve2d(gray, kernelx) #卷積
        img_sobelx = img_sobelx/(img_sobelx.max()/255.0)
        img_sobelx = img_sobelx.astype(np.uint8)
        cv.imshow('Sobel X', img_sobelx)
        cv.imwrite('../Q3_image/Chihiro_GaussianBlur_SobelX.jpg',img_sobelx)
        cv.waitKey(0)
        cv.destroyWindow('Sobel X')

    def on_btn3_3_click(self):
        img = cv.imread( '../Q3_image/Chihiro_GaussianBlur.jpg', cv.IMREAD_COLOR )
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        kernely = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]
        img_sobely = signal.convolve2d(gray, kernely) #卷積
        img_sobely = img_sobely/(img_sobely.max()/255.0)
        img_sobely = img_sobely.astype(np.uint8) 
        cv.imshow('Sobel Y', img_sobely)
        cv.imwrite('../Q3_image/Chihiro_GaussianBlur_SobelY.jpg',img_sobely)
        cv.waitKey(0)
        cv.destroyWindow('Sobel Y')

    def on_btn3_4_click(self):
        imgX = cv.imread( '../Q3_image/Chihiro_GaussianBlur_SobelX.jpg', cv.IMREAD_COLOR )
        imgY = cv.imread( '../Q3_image/Chihiro_GaussianBlur_SobelY.jpg', cv.IMREAD_COLOR ) 
        gradient_magnitude = np.hypot(imgX,imgY)
        gradient_magnitude = gradient_magnitude.astype(np.uint8)
        cv.imshow("Magnitude", gradient_magnitude)
        cv.waitKey(0)
        cv.destroyWindow('Magnitude')

    def on_btn4_1_click(self):
        # get the value from ui
        Angle = float(self.edtAngle.text())
        Scale = float(self.edtScale.text())
        Tx = float(self.edtTx.text())
        Ty = float(self.edtTy.text())

        # read image
        img = cv.imread('../Q4_image/Parrot.png')

        # making translation matrix
        H = np.float32([[1,0,Tx],[0,1,Ty]])

        # translate the image
        rows,cols = img.shape[:2]
        tansImg = cv.warpAffine(img,H,(rows,cols))

        # making rotate and scale matrix
        rows,cols = tansImg.shape[:2]
        M = cv.getRotationMatrix2D((130+Tx,125+Ty),Angle,Scale)

        # rotating and Scaling the image
        result = cv.warpAffine(tansImg,M,(rows,cols))

        cv.imshow('Origin Image',img)
        cv.imshow('Image RST',result)
        cv.waitKey(0)
        cv.destroyWindow('Origin Image')
        cv.destroyWindow('Image RST')
    def on_btn5_1_click(self):
        label_dict={0:"airplain",1:"automobile",2:"bird",3:"cat",4:"deer",5:"dog",6:"frog",7:"horse",8:"ship",9:"truck"}  #轉換標籤為類別名稱用

        def plot_images_labels_prediction(images,labels,prediction,idx,num=10):
            fig=plt.gcf()                                           #取得 pyplot 物件參考
            fig.set_size_inches(12, 14)                    #設定畫布大小為 12 吋*14吋
            
            for i in range(0, num):                            #依序顯示 num 個子圖
                ax=plt.subplot(5, 5, i+1)                     #建立 5*5 個子圖中的第 i+1 個
                ax.imshow(images[idx], cmap='binary')      #顯示子圖
                title= label_dict[labels[idx][0]]  
                if len(prediction) > 0:                    #有預測值就加入標題中
                    title += ",predict=" + str(prediction[idx])
                ax.set_title(title, fontsize=10)            #設定標題
                ax.set_xticks([]);                                #不顯示 x 軸刻度
                ax.set_yticks([]);                                #不顯示 y 軸刻度
                idx += 1                                              #樣本序號增量 1
            plt.show()                                                #繪製圖形

        (x_train_image, y_train_label), (x_test_image, y_test_label)=cifar10.load_data() #載入 Cifar-10 資料集
        plot_images_labels_prediction(x_train_image,y_train_label,[],random.randint(0, 49990),10)    #無預測值

    def on_btn5_2_click(self):
        weight_decay = 5e-4
        batch_size = 128
        learning_rate = 1e-2
        dropout_rate = 0.5
        epoch_num = 20
        print('################### training hyperparameters ######################')
        print('hyperparameters:')
        print('batch size: ', batch_size)
        print('learning rate: ', learning_rate)
        print('optimizer: ', optimizers.SGD)
        print('####################################################################')
    def on_btn5_3_click(self):
        weight_decay = 5e-4
        batch_size = 128
        learning_rate = 1e-2
        dropout_rate = 0.5
        epoch_num = 20
        def VGG16():
            model = models.Sequential()
            model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3), kernel_regularizer=regularizers.l2(weight_decay)))
            model.add(Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
            model.add(MaxPooling2D((2, 2)))

            model.add(Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
            model.add(Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
            model.add(MaxPooling2D((2, 2)))

            model.add(Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
            model.add(Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
            model.add(Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
            model.add(MaxPooling2D((2, 2)))

            model.add(Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
            model.add(Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
            model.add(Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
            model.add(MaxPooling2D((2, 2)))

            model.add(Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
            model.add(Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
            model.add(Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(weight_decay)))

            model.add(Flatten())  # 2*2*512
            model.add(Dense(4096, activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(4096, activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(10, activation='softmax'))

            return model
        # get model
        model = VGG16()

        # show
        print('######################## model structure ##########################')
        model.summary()
        print('###################################################################')
    def on_btn5_4_click(self):
        acc = cv.imread( '../Q5_Image/accuracy.png', cv.IMREAD_COLOR )
        loss = cv.imread( '../Q5_Image/loss.png', cv.IMREAD_COLOR )
        cv.imshow('accuracy', acc)
        cv.imshow('loss', loss)
        cv.waitKey(0)
        cv.destroyWindow('acc')
        cv.destroyWindow('loss')

    def on_btn5_5_click(self):
        Inference = int(self.Inference.text())
        (x_train_image, y_train_label), (x_test_image, y_test_label)=cifar10.load_data() #載入 Cifar-10 資料集
        testimg = x_test_image[Inference]
        cv.imshow('Test Image', testimg)
        cv.waitKey(0)
        cv.destroyWindow('Test Image')


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
