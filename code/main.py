import sys,cv2
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QMessageBox, QInputDialog, QFileDialog, QHBoxLayout, QGroupBox, \
                            QDialog, QVBoxLayout, QGridLayout,QLineEdit
from PyQt5.QtCore import pyqtSlot, Qt
from PyQt5.QtGui import QPixmap, qRgb, QImage

gray_color_table = [qRgb(i, i, i) for i in range(256)]

def generateGaussianKernel(winSize, sigma):
    """
        generates a gaussian kernel taking window size = winSize
        and standard deviation = sigma as input/control parameters. 

        Returns the gaussian kernel
    """
    kernel = np.zeros((winSize, winSize))        # generate a zero numpy kernel
    for i in range(winSize):
        for j in range(winSize):
            temp = pow(i-winSize//2,2)+pow(j-winSize//2,2)
            kernel[i,j] = np.exp(-1*temp/(2*pow(sigma,2)))
    kernel = kernel/(2*np.pi*pow(sigma,2))
    norm_factor = np.sum(kernel)               # finding the sum of the kernel generated
    kernel = kernel/norm_factor                # dividing by total sum to make the kernel matrix unit-sum
    return kernel

def boxKernel(winSize):
    """
        returns  a box kernel with window size = winSize
    """
    kernel = np.ones((winSize, winSize))/(winSize*winSize)
    return kernel

def roundAngle(angle):
    """ 
        this functions rounds-off the input angle to four values
        Input angle must be in [0,180) 
    """
    # converts angle in radian to degree
    angle = np.rad2deg(angle) % 180
    if (0 <= angle < 22.5) or (157.5 <= angle < 180):
        angle = 0
    elif (22.5 <= angle < 67.5):
        angle = 45
    elif (67.5 <= angle < 112.5):
        angle = 90
    elif (112.5 <= angle < 157.5):
        angle = 135
    return angle


class MyImageEditor(QWidget):
    def __init__(self):
        super().__init__()
        self.title = '150070031-Image-Editor'               # declare the title of editor window
        self.left = 100                                     # (next two lines) x-y co-ordinate on the screen where the editor's top left corner will lie
        self.top = 100 
        self.width = 840                                    # the width and height of the editor window
        self.height = 480
        self.number_horizontal_box_layouts = 3
        self.initUserInterface()                            # initialize the User Interface
 
    
    def initUserInterface(self):
        self.setWindowTitle(self.title)                     # sets window title as defined above
        geometry = myApp.desktop().availableGeometry()        # window size = entire screen available
        # self.setGeometry(self.left, self.top, self.width, self.height)  # sets the geometry of the window
        self.setGeometry(geometry)                          # sets the geometry of the window

        self.createGridLayout()
        windowLayout = QHBoxLayout()

        for i in range(self.number_horizontal_box_layouts):
            windowLayout.addWidget(self.horizontalGroupBox[i])
        self.setLayout(windowLayout)                   # sets the layout of editor-window to be windowLayout
        self.show()                                    # displays the editor-window on screen

        self.label = QLabel(self)                     # initialize pyqt QLabel widget for displaying original Image always
        self.label_result = QLabel(self)                      # QLabel widget for displaying modified Image after each \
                                                    # operation. Initialized to original Image in the beginning 

        self.layout[0].addWidget(self.label)                               # both the label widgets declared above added to 1st & 2nd layout resp.                             
        self.layout[1].addWidget(self.label_result)

    
    def createGridLayout(self):
        self.layout = []                               # initialize layout of the editor-window to be an empty list
        number_layouts = 3                             # first layout : for showing the original image \
                                                       # sedond layout : for showing the modified image \
                                                       # third layout : for all the pushbuttons 
        for i in range(number_layouts):
            self.layout.append(QGridLayout())          # each layout type is GridLayout.Creates a gridlayout object and appends to layout list

        self.layout[0].setColumnStretch(1, 2)
        self.layout[1].setColumnStretch(1, 2)
 
        nButtons = 11                                  # number of buttons
        buttonLabel = ['Load Image', 'Histogram Equalization', 'Gamma Correction', 'Log Transform', 'Gaussian Blur', 'Image Sharpening', 'Canny Edge Detection', 
                       'Undo Last', 'Undo All', 'Save', 'Exit']          # Labels of pushbutton
        onButtonClick = [self.loadImage, self.histEqualize, self.gammaCorrection, self.logTransform, self.blurImage, self.sharpImage, 
                        self.specialFeature, self.undoLast, self.undoAll, self.saveImage, self.myExit]    # list of functions that are executed when the 
                                                                                                # corresponding button defined in buttonLabel is clicked
        buttons = []                                             # an empty list created for storing pushbutton objects
        for i in range(nButtons):
            buttons.append(QPushButton(buttonLabel[i], self))    # Pushbutton object created with the first argument as the button label \
                                                                 # and appended in the buttons list
            buttons[i].clicked.connect(onButtonClick[i])         # Pushbutton attached with an event listener which results in corresponding \
                                                                 # function in onButtonClick list to be executed on button click
            buttons[i].setStyleSheet("color : black; background-color : lightblue;")  # sets the button text color and background color
            self.layout[2].addWidget(buttons[i],i,0)             # Pushbutton object added to layout 3 at position (i,0)

        horizontalGroupBox_Label = ["Original Image Grid", "Modified Image Grid", "Buttons Grid"]
        self.horizontalGroupBox = []
        for i in range(self.number_horizontal_box_layouts):
            self.horizontalGroupBox.append(QGroupBox(horizontalGroupBox_Label[i]))
            self.horizontalGroupBox[i].setLayout(self.layout[i])

    
    def finalDisplay(self, result):
        """
            Display Image
            Input argument 
                result : two dimensional image/single channel
        """
        # combine V-channel (self.image) with H,S channel
        self.hsvImage[:,:,2] = self.image
        self.prevImage = cv2.cvtColor(self.hsvImage, cv2.COLOR_HSV2RGB)
        self.hsvImage[:,:,2] = result
        # update the image display with the modified HSV image
        qImage = self.convertCvToQImage(cv2.cvtColor(self.hsvImage, cv2.COLOR_HSV2RGB))
        self.label_result.setPixmap(QPixmap(qImage))
        self.image = result

    
    def openFileNameDialog(self):
        fileLoadOptions = QFileDialog.Options()
        fileLoadOptions |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self, "Load Image File","","All Files (*);;jpg (*.jpg);; png (*.png)", options = fileLoadOptions)
        possibleImageTypes = ['jpg', 'png', 'jpeg', 'bmp']                             # only these image extensions are allowed

        if (fileName and (fileName.split('.')[1] in possibleImageTypes)):       # if user selects a file and it is a allowed file
            self.origImage = cv2.resize(cv2.imread(fileName, 1), (480, 360))    # reads a color image. Resizes to (image_width, image_height)=(400, 300)
            self.origImage = cv2.cvtColor(self.origImage, cv2.COLOR_BGR2RGB)    # openCV image color format when reading = BGR (by default) and \
                                                                                # converts image format from BGR to RGB for later convenience
            self.prevImage = self.origImage                                     # previous image variable for "Undo to Last" Operation
            self.hsvImage = cv2.cvtColor(self.origImage, cv2.COLOR_RGB2HSV)     # convert image format from RGB -> HSV for operations on V channel
            self.image = self.hsvImage[:, :, 2].astype(np.uint8)                # self.image = V channel of original HSV image
            qImage = self.convertCvToQImage(self.origImage)                     # convert openCV image -> pyQt QImage for displaying in Pixmap
            # set pixmap with the original loaded image in both original and modified grids
            self.label.setPixmap(QPixmap(qImage))
            self.label_result.setPixmap(QPixmap(qImage))
            # self.label_result.setPixmap(None)
        elif (fileName):                                                        # case when user selects a file with extension not in "possibleImageTypes"
            buttonReply = QMessageBox.question(self, 'Warning Message', "Wrong File Selection", QMessageBox.Ok, QMessageBox.Ok)


    
    def convertCvToQImage(self, img, copy=False):
        """
            converts opencv image to QImage. 
            source : Stack Overflow
            Input : img must be in openCV image format (np.uint8)
        """
        if img is None:
            return QImage()
        # if input image format is openCv image format np.uint8
        if img.dtype == np.uint8:
            # grayscale image or images having two dimensions [height, width]
            if len(img.shape) == 2:
                qim = QImage(img.data, img.shape[1], img.shape[0], img.strides[0], QImage.Format_Indexed8)
                qim.setColorTable(gray_color_table)
                return qim.copy() if copy else qim
            # image having three dimansions [height, width, nChannels]
            elif len(img.shape) == 3:
                # if image has three channels
                if img.shape[2] == 3:
                    qim = QImage(img.data, img.shape[1], img.shape[0], img.strides[0], QImage.Format_RGB888)
                    return qim.copy() if copy else qim
                # if image has four channels
                elif img.shape[2] == 4:
                    qim = QImage(img.data, img.shape[1], img.shape[0], img.strides[0], QImage.Format_ARGB32)
                    return qim.copy() if copy else qim

    
    @pyqtSlot()
    def loadImage(self):
        self.openFileNameDialog()             # [In-Built Function] opens a dialog box for user to choose an Image

    
    @pyqtSlot()
    def histEqualize(self):
        """
            Histrogram Equalization Function
                1. finds the histogram of input image
                2. computes the cumulative distribution of input image
                3. computes the probability distribution and scale it to 255
                4. assign the new intensity values to pixels
                5. Plots the input image histogram and equalized image histogram 
        """
        L = 256
        imgHeight, imgWidth = self.image.shape
        resultantImage = np.zeros(self.image.shape)                          # numpy array of zeros for output "histogram equalized" image
        original_img_hist = np.zeros((L, 1))                                  # variable to store the input image histogram
        equalized_img_hist = np.zeros((L, 1))                                 # variable to store the "histogram equalized" image histogram
        for i in range(L):                                                   # compute the histogram of input image
            original_img_hist[i, 0] = np.sum(self.image == i)

        cdf = np.zeros(original_img_hist.shape)
        sumHist = 0
        for i in range(L):
            sumHist = sumHist + original_img_hist[i,0]                    # finds the cumulative distribution of input image
            cdf[i,0] = sumHist
        # cdf = np.cumsum(original_img_hist)                                   
        temp = ((L-1)/(imgHeight*imgWidth))                                       # temporary variable
        for i in range(L):                                                   # compute the transform values for each intensity from [0-255] and assign it
            resultantImage[np.where(self.image == i)] = np.round(temp*cdf[i])# to the pixels locations where that intensity is present in input image
        for i in range(L):                                                
            equalized_img_hist[i, 0] = np.sum(resultantImage == i)            # compute the histogram of output image
        resultantImage = resultantImage.astype(np.uint8)                     # change output image type for display purposes

        # the following lines plots the input and output image histograms in a (6x6) plot window
        fig = plt.figure(1, figsize = (6, 6))
        plt.subplot(211); plt.plot(original_img_hist, linewidth=0.9); plt.xlabel('Intensity'); plt.ylabel('Count of Pixels'); plt.grid(True)
        plt.subplot(212); plt.plot(equalized_img_hist, linewidth=0.9); plt.xlabel('Intensity'); plt.ylabel('Count of Pixels'); plt.grid(True)
        plt.suptitle('Comparison of Original vs Equalized Image Histograms')
        plt.show()

        self.finalDisplay(resultantImage)                                    # display the output image          


    
    @pyqtSlot()
    def gammaCorrection(self):
        c = 1.0
        gamma, okPressed = QInputDialog.getDouble(self, "Get Gamma Value", "Value:")               # dialog-box for taking gamma value as input
        if okPressed:                                                                              # if user presses "OK" only then only compute 
            normImage = cv2.normalize(self.image.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX) # convert image from [0-255] -> [0-1]
            resultantImage = c*pow(normImage, gamma)                                               # output image = c*(input image)^gamma
            max_result = np.max(resultantImage)                                                    # finding the maximum intensity value  of output image
            resultantImage = (resultantImage/max_result)*255.0                                     # bring the output image to [0-255] range
            resultantImage = resultantImage.astype(np.uint8)                                       # change output image type for display purposes
            self.finalDisplay(resultantImage)                                                            # display the output image 

    
    @pyqtSlot()
    def logTransform(self):
        # c, okPressed = QInputDialog.getDouble(self, "C Value", "Value:")
        # if okPressed:
        c = 1.0
        normImage = cv2.normalize(self.image.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)  # convert image from [0-255] -> [0-1]
        resultantImage = c*np.log(1+normImage)                                                  # output image = c*ln(1 + normalized_Image)
        max_result = np.max(resultantImage)                                                     # finding the maximum intensity value  of output image
        resultantImage = (resultantImage/max_result)*255.0                                      # bring the output image to [0-255] range
        resultantImage = resultantImage.astype(np.uint8)                                        # change output image type for display purposes
        self.finalDisplay(resultantImage)                                                       # display the output image

    
    @pyqtSlot()
    def blurImage(self):
        """
            Blur Image using Gaussian Kernel with kernel size and gaussian variance
            chosen by initUserInterface
        """
        winSize, ok1Pressed = QInputDialog.getInt(self, "Kernel Window Size", "Value (odd number > 0):")      # get gaussian kernel size from user
        sigma, ok2pressed = QInputDialog.getDouble(self, "Standard Deviation", "Value (> 0):")     # get variance of gaussian kernel from user
        if (ok1Pressed and ok2pressed):                                                     # if both "OK" pressed then proceed further
            if(winSize%2!=0):                                                               # if kernel size is odd number the proceed further \
                                                                                            # else display a warning message
                imgHeight, imgWidth = self.image.shape
                mykernel = generateGaussianKernel(winSize, sigma)                           # get gaussian kernel
                # mykernel = boxKernel(winSize)
                paddedImage = np.zeros((imgHeight+(winSize//2)*2, imgWidth+(winSize//2)*2), dtype=np.uint8)   # initialize an empty zero-padded image for \
                                                                                            # convolution with gaussian kernel
                paddedImage[winSize//2:imgHeight+winSize//2, winSize//2:imgWidth+winSize//2] = self.image     # replace the central section of padded image \
                                                                                                         # with input image
                blurredImage = np.zeros(self.image.shape,dtype=np.uint8)                       # initialize empty image of size=imagesize for storing result
                for i in range(imgHeight):                                                     # convolution of zero-padded image and gaussian kernel
                    for j in range(imgWidth):
                        blurredImage[i,j] = np.round(np.sum(paddedImage[i:i+winSize, j:j+winSize]*mykernel))   # since gaussian kernel is symmetric matrix \
                                                                                        # as well as diagonal-symmetric, convolution ~ matrix multiplication
                blurredImage = blurredImage.astype(np.uint8)                            # change output image type for display purposes
                self.finalDisplay(blurredImage)                                         # display the output image
            else:
                buttonReply = QMessageBox.question(self, 'Warning Message', "Kernel Size has to be an odd number", QMessageBox.Ok, QMessageBox.Ok)

    
    @pyqtSlot()
    def sharpImage(self):
        # k = scale factor in unsharp masking; user input
        k, okPressed = QInputDialog.getDouble(self, "Scale Factor (k)", "Value (default = 5):") 
        if okPressed is None:
            k = 5
        winSize = 5         # kernel size for gaussian blurring
        sigma = 2           # standard deviation for gaussian blurring
        imgHeight, imgWidth = self.image.shape
        # kernel window of given winSize and sigma
        mykernel = generateGaussianKernel(winSize, sigma)
        # zero padded image for convolution with gaussian kernel
        paddedImage = np.zeros((imgHeight+(winSize//2)*2, imgWidth+(winSize//2)*2), dtype=np.uint8)
        paddedImage[winSize//2:imgHeight+winSize//2, winSize//2:imgWidth+winSize//2] = self.image
        blurredImage = np.zeros(self.image.shape,dtype=np.uint8)
        for i in range(imgHeight):
            for j in range(imgWidth):
                # convolution operation
                blurredImage[i,j] = np.sum(paddedImage[i:i+winSize, j:j+winSize]*mykernel)    
        # scaled version of mask computed by subtracting the blurred image from original image and scaling by k
        maskImage = k*cv2.subtract(self.image, blurredImage)
        maskImage = maskImage.astype(np.uint8)
        # sharp image computed by adding scaled mask with original image
        sharpenedImage = cv2.add(self.image, maskImage)
        sharpenedImage = sharpenedImage.astype(np.uint8)
        # display the sharpened image 
        self.finalDisplay(sharpenedImage)

        
    @pyqtSlot()
    def specialFeature(self):
        """
            Canny Edge DEtection
            Source : https://github.com/fubel/PyCannyEdge
        """
        sigma = 1.6                      # standard deviation for gaussian blurring 
        winSize = 3                      # kernel size for gaussian blurring
        strong = np.int32(255)
        weak = np.int32(50)

        lowerThreshold, ok1Pressed = QInputDialog.getInt(self, "Lower Threshold", "Value (default = 20):")
        upperThreshold, ok2Pressed = QInputDialog.getInt(self, "Upper Threshold", "Value (default = 40):")

        if ok1Pressed:
            pass
        else:
            lowerThreshold = 20

        if ok2Pressed:
            pass
        else:
            upperThreshold = 40

        # combine V-channel (self.image) with H,S channel
        self.hsvImage[:,:,2] = self.image
        img = cv2.cvtColor(cv2.cvtColor(self.hsvImage, cv2.COLOR_HSV2BGR), cv2.COLOR_BGR2GRAY)
        img = img.astype(np.int32)
        imgHeight, imgWidth = img.shape

        mykernel = generateGaussianKernel(winSize, sigma)
        # zero padded image for convolution with gaussian kernel
        paddedImage = np.zeros((imgHeight+(winSize//2)*2, imgWidth+(winSize//2)*2), dtype=np.int32)
        paddedImage[winSize//2:imgHeight+winSize//2, winSize//2:imgWidth+winSize//2] = img
        blurredImage = np.zeros(self.image.shape,dtype=np.int32)
        for i in range(imgHeight):
            for j in range(imgWidth):
                blurredImage[i,j] = np.sum(paddedImage[i:i+winSize, j:j+winSize]*mykernel)
        # kernel for finding gradient in X & Y direction
        kernelGradientX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.int32)    # vertical edges
        kernelGradientY = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.int32)    # horizontal edges

        paddedImage[winSize//2:imgHeight+winSize//2, winSize//2:imgWidth+winSize//2] = blurredImage
        # for storing the convolution result of 
        gradientX = np.zeros(img.shape, dtype=np.int32)
        gradientY = np.zeros(img.shape, dtype=np.int32)

        # convolution for finding gradients
        for i in range(imgHeight):
            for j in range(imgWidth):
                gradientX[i,j] = np.round(np.sum(paddedImage[i:i+winSize, j:j+winSize]*(-1*kernelGradientX)))
                gradientY[i,j] = np.round(np.sum(paddedImage[i:i+winSize, j:j+winSize]*(-1*kernelGradientY)))

        # gradientMagnitude = \sqrt(gx*gx + gy*gy)
        gradientMagnitude = np.hypot(gradientX, gradientY)
        gradientDirection = np.arctan2(gradientY, gradientX)

        # output image for displaying canny edges
        edgeDetectedImage = np.zeros((imgHeight, imgWidth), dtype=np.int32)

        for i in range(imgHeight):
            for j in range(imgWidth):
                # find neighbour pixels to visit from the gradient directions
                quantizedGradientDirection = roundAngle(gradientDirection[i, j])
                try:
                    if quantizedGradientDirection == 0:
                        if (gradientMagnitude[i, j] >= gradientMagnitude[i, j - 1]) and (gradientMagnitude[i, j] >= gradientMagnitude[i, j + 1]):
                            edgeDetectedImage[i,j] = gradientMagnitude[i,j]
                    elif quantizedGradientDirection == 90:
                        if (gradientMagnitude[i, j] >= gradientMagnitude[i - 1, j]) and (gradientMagnitude[i, j] >= gradientMagnitude[i + 1, j]):
                            edgeDetectedImage[i,j] = gradientMagnitude[i,j]
                    elif quantizedGradientDirection == 135:
                        if (gradientMagnitude[i, j] >= gradientMagnitude[i - 1, j - 1]) and (gradientMagnitude[i, j] >= gradientMagnitude[i + 1, j + 1]):
                            edgeDetectedImage[i,j] = gradientMagnitude[i,j]
                    elif quantizedGradientDirection == 45:
                        if (gradientMagnitude[i, j] >= gradientMagnitude[i - 1, j + 1]) and (gradientMagnitude[i, j] >= gradientMagnitude[i + 1, j - 1]):
                            edgeDetectedImage[i,j] = gradientMagnitude[i,j]
                except IndexError as e:
                    # exception when index i, j goes out of the boundary of gradientMagnitude
                    pass
        
        # get strong pixel indices
        strong_i, strong_j = np.where(edgeDetectedImage > upperThreshold)
        # get weak pixel indices
        weak_i, weak_j = np.where((edgeDetectedImage >= lowerThreshold) & (edgeDetectedImage <= upperThreshold))
        # get pixel indices set to be zero
        zero_i, zero_j = np.where(edgeDetectedImage < lowerThreshold)
        # update the values of pixel indices found above
        edgeDetectedImage[strong_i, strong_j] = strong
        edgeDetectedImage[weak_i, weak_j] = weak
        edgeDetectedImage[zero_i, zero_j] = 0

        for i in range(imgHeight):
            for j in range(imgWidth):
                if edgeDetectedImage[i, j] == weak:
                    # check if one of the neighbours is strong (=255 by default)
                    try:
                        if ((edgeDetectedImage[i + 1, j] == strong) or (edgeDetectedImage[i - 1, j] == strong)
                             or (edgeDetectedImage[i, j + 1] == strong) or (edgeDetectedImage[i, j - 1] == strong)
                             or (edgeDetectedImage[i+1, j + 1] == strong) or (edgeDetectedImage[i-1, j - 1] == strong)):
                            edgeDetectedImage[i, j] = strong
                        else:
                            edgeDetectedImage[i, j] = 0
                    except IndexError as e:
                        # exception when index i, j goes out of the boundary of edgeDetectedImage
                        pass

        edgeDetectedImage = edgeDetectedImage.astype(np.uint8)
        self.finalDisplay(edgeDetectedImage)

    
    @pyqtSlot()
    def undoLast(self):
        # combine V-channel (self.image) with H,S channel
        self.hsvImage[:,:,2] = self.image
        # convert HSV to BGR since openCV default color format is BGR
        temp = cv2.cvtColor(self.hsvImage, cv2.COLOR_HSV2RGB)
        # assign the V-channel of previous image to present image after converting it to HSV
        self.image = cv2.cvtColor(self.prevImage, cv2.COLOR_RGB2HSV)[:,:,2]
        # convert (previous image variable to be displayed) cv image to QImage format for displaying using QPixmap
        qImage = self.convertCvToQImage(self.prevImage)
        self.label_result.setPixmap(QPixmap(qImage))
        # assign present image to previous image variable 
        self.prevImage = temp

    
    @pyqtSlot()
    def undoAll(self):
        # combine V-channel (self.image) with H,S channel
        self.hsvImage[:,:,2] = self.image
        # convert HSV to BGR since openCV default color format is BGR
        self.prevImage = cv2.cvtColor(self.hsvImage, cv2.COLOR_HSV2RGB)
        # convert original image loaded into HSV format for displaying 
        self.image = cv2.cvtColor(self.origImage, cv2.COLOR_RGB2HSV)[:,:,2]
        # convert (original image loaded to be displayed) cv image to QImage format for displaying using QPixmap
        qImage = self.convertCvToQImage(self.origImage)
        self.label_result.setPixmap(QPixmap(qImage))

    
    @pyqtSlot()
    def saveImage(self):
        fileSaveOptions = QFileDialog.Options()
        fileSaveOptions |= QFileDialog.DontUseNativeDialog
        # open a dialog box for choosing destination location for saving image
        fileName, _ = QFileDialog.getSaveFileName(self,"Save Image File","","All Files (*);;jpg Files(*.jpg);;png Files(*.png)", options=fileSaveOptions)
        # combine V-channel with H,S channel -> convert HSV to BGR since openCV default color format is BGR
        self.hsvImage[:,:,2] = self.image
        # save image at location specified by fileName
        cv2.imwrite(fileName, cv2.cvtColor(self.hsvImage, cv2.COLOR_HSV2BGR))

    
    @pyqtSlot()
    def myExit(self):
        self.close()                 # in-built function of QWidget class


if __name__ == '__main__':
    myApp = QApplication(sys.argv)    # defines pyqt application object 
    ex = MyImageEditor()
    sys.exit(myApp.exec_())