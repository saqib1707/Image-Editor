import sys,cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QMessageBox, QInputDialog, QFileDialog, QHBoxLayout, QGroupBox, QDialog, QVBoxLayout, QGridLayout,QLineEdit
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtGui import QPixmap, qRgb, QImage

gray_color_table = [qRgb(i, i, i) for i in range(256)]

def genGaussianKernel(winsize, sigma):
    """
        generates a gaussian kernel taking window size = winsize
        and standard deviation = sigma as input/control parameters. 

        Returns the gaussian kernel
    """
    kernel = np.zeros((winsize, winsize))        # generate a zero numpy kernel
    for i in range(winsize):
        for j in range(winsize):
            temp = pow(i-winsize//2,2)+pow(j-winsize//2,2)
            kernel[i,j] = np.exp(-1*temp/(2*pow(sigma,2)))
    kernel = kernel/(2*np.pi*pow(sigma,2))
    norm_factor = np.sum(kernel)               # finding the sum of the kernel generated
    kernel = kernel/norm_factor                # dividing by total sum to make the kernel matrix unit-sum
    return kernel

def boxKernel(winsize):
    """
        returns  a box kernel with window size = winsize
    """
    kernel = np.ones((winsize, winsize))/(winsize*winsize)
    return kernel

def sharpeningKernel(winsize):
    """
        returns a sharpening kernel with the control parameter as the 
        window size = winsize. 
    """
    kernel = -1*np.ones((winsize,winsize))/(winsize*winsize)
    kernel[winsize//2][winsize//2]*=-(winsize*winsize-1)
    return kernel

class MyImageEditor(QWidget):
    def __init__(self):
        super().__init__()
        self.title = '150070031-Image-Editor'
        """
        (the next two lines) x-y co-ordinate on the screen where the editor's top left 
        corner will lie. 
        """
        self.left = 100
        self.top = 100 
        self.width = 840         # the width and height of the editor window
        self.height = 480
        self.initUserInterface()
 
    def initUserInterface(self):
        self.setWindowTitle(self.title)    # sets window title as defined above
        self.setGeometry(self.left, self.top, self.width, self.height)  # sets the geometry of the window

        self.number_horizontal_box_layouts = 3
        self.createGridLayout()
        windowLayout = QHBoxLayout()

        for i in range(self.number_horizontal_box_layouts):
            windowLayout.addWidget(self.horizontalGroupBox[i])
        # windowLayout.addWidget(self.horizontalGroupBox0)
        # windowLayout.addWidget(self.horizontalGroupBox1)
        # windowLayout.addWidget(self.horizontalGroupBox2)
        self.setLayout(windowLayout)
        self.show()                                    # displays the window on screen

    def finalDisplay(self, result):
        self.hsvImage[:,:,2] = self.image
        self.prevImage = cv2.cvtColor(self.hsvImage, cv2.COLOR_HSV2RGB)
        self.hsvImage[:,:,2] = result
        qImage = self.toQImage(cv2.cvtColor(self.hsvImage, cv2.COLOR_HSV2RGB))
        self.label_result.setPixmap(QPixmap(qImage))
        self.image = result

    def createGridLayout(self):
        # self.horizontalGroupBox0 = QGroupBox()
        # self.horizontalGroupBox1 = QGroupBox()
        # self.horizontalGroupBox2 = QGroupBox()

        self.layout = []
        for i in range(3):
            self.layout.append(QGridLayout())
        # self.layout0 = QGridLayout()
        # self.layout1 = QGridLayout()
        # self.layout2 = QGridLayout()
        # layout.setColumnStretch(1, 3)
        # layout.setColumnStretch(2, 4)
 
        nButtons = 9                       # number of buttons
        buttonLabel = ['Load Image','Histogram Equalize','Gamma Correction','Log Transform','Gaussian Blur','Sharpening Blur','Undo','Undo All','Save']
        onButtonClick = [self.on_click,self.histEqualize,self.gammaCorrection,self.logTransform,self.blurImage,self.sharpImage,self.undoLast,self.undoAll,self.saveImage]
        buttons = []                   # an empty list created for storing button variables
        for i in range(nButtons):
            buttons.append(QPushButton(buttonLabel[i], self))    # Pushbutton object created with the first argument as the button label
                                                                # and appended in the button variable
            buttons[i].clicked.connect(onButtonClick[i])            # Pushbutton attached with an event listener which calls function on button click
            self.layout[2].addWidget(buttons[i],i,0)              # Pushbutton added to the layout at position (i,0)

        # load_btn = QPushButton('Load Image', self)         
        # load_btn.clicked.connect(self.on_click)           
        # self.layout2.addWidget(load_btn,0,0)              

        # hist_eq_btn = QPushButton('Histogram Equalize', self)
        # hist_eq_btn.clicked.connect(self.histEqualize)
        # self.layout2.addWidget(hist_eq_btn,1,0) 

        # gamma_corr_btn = QPushButton('Gamma Correction', self)
        # gamma_corr_btn.clicked.connect(self.gammaCorrection)
        # self.layout2.addWidget(gamma_corr_btn,2,0)

        # log_trans_btn = QPushButton('Log Transform', self)
        # log_trans_btn.clicked.connect(self.logTransform)
        # self.layout2.addWidget(log_trans_btn,3,0) 

        # blur_btn = QPushButton('Gaussian Blur', self)
        # blur_btn.clicked.connect(self.blurImage)
        # self.layout2.addWidget(blur_btn,4,0) 

        # sharp_btn = QPushButton('Sharpening Blur', self)
        # sharp_btn.clicked.connect(self.sharpImage)
        # self.layout2.addWidget(sharp_btn,5,0) 

        # undo_btn = QPushButton('Undo', self)
        # undo_btn.clicked.connect(self.undoLast)
        # self.layout2.addWidget(undo_btn,6,0) 

        # undoAll_btn = QPushButton('Undo All', self)
        # undoAll_btn.clicked.connect(self.undoAll)
        # self.layout2.addWidget(undoAll_btn,7,0) 

        # save_btn = QPushButton('Save', self)
        # save_btn.clicked.connect(self.saveImage)
        # self.layout2.addWidget(save_btn,8,0) 

        horizontalGroupBox_Label = ["Original Image Grid", "Modified Image Grid", "Buttons Grid"]
        self.horizontalGroupBox = []
        for i in range(self.number_horizontal_box_layouts):
            self.horizontalGroupBox.append(QGroupBox(horizontalGroupBox_Label[i]))
            self.horizontalGroupBox[i].setLayout(self.layout[i])
        # self.horizontalGroupBox0.setLayout(self.layout0)
        # self.horizontalGroupBox1.setLayout(self.layout1)
        # self.horizontalGroupBox2.setLayout(self.layout2)

    def openFileNameDialog(self):    
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"Load Image File","","All Files (*);;jpg (*.jpg);; png (*.png)", options=options)
        if fileName:
            self.origImage = cv2.resize(cv2.imread(fileName), (256, 256))
            self.origImage = cv2.cvtColor(self.origImage, cv2.COLOR_BGR2RGB)
            self.prevImage = self.origImage
            self.hsvImage = cv2.cvtColor(self.origImage, cv2.COLOR_RGB2HSV)
            self.image = self.hsvImage[:,:,2].astype(np.uint8)
            qImage = self.toQImage(self.origImage)
            self.label = QLabel(self)
            self.label.setPixmap(QPixmap(qImage))
            self.label_result = QLabel(self)
            self.label_result.setPixmap(QPixmap(qImage))
            self.layout[0].addWidget(self.label)
            self.layout[1].addWidget(self.label_result)

    def convertQImageToMat(self, incomingImage):
        '''  Converts a QImage into an opencv MAT format '''
        incomingImage = incomingImage.convertToFormat(4)
        width = incomingImage.width()
        height = incomingImage.height()
        ptr = incomingImage.bits()
        ptr.setsize(incomingImage.byteCount())
        arr = np.array(ptr).reshape(height, width, 4)  #  Copies the data
        # return arr[:,:,0:3]
        return arr

    def convertCvToQImage(self, incomingImage):
        if len(incomingImage.shape)>2:
            height, width, nChannel = incomingImage.shape
            byteValue = nChannel * width
        else:
            height, width = incomingImage.shape
            byteValue = width
        qImage = QImage(incomingImage.data, width, height, byteValue, QImage.Format_RGB888)
        return qImage

    def toQImage(self, im, copy=False):
        if im is None:
            return QImage()
        if im.dtype == np.uint8:
            if len(im.shape) == 2:
                qim = QImage(im.data, im.shape[1], im.shape[0], im.strides[0], QImage.Format_Indexed8)
                qim.setColorTable(gray_color_table)
                return qim.copy() if copy else qim
            elif len(im.shape) == 3:
                if im.shape[2] == 3:
                    qim = QImage(im.data, im.shape[1], im.shape[0], im.strides[0], QImage.Format_RGB888);
                    return qim.copy() if copy else qim
                elif im.shape[2] == 4:
                    qim = QImage(im.data, im.shape[1], im.shape[0], im.strides[0], QImage.Format_ARGB32);
                    return qim.copy() if copy else qim

    @pyqtSlot()
    def on_click(self):
        self.openFileNameDialog()

    @pyqtSlot()
    def histEqualize(self):
        L = 256
        imgrow,imgcol = self.image.shape;
        result = np.zeros(self.image.shape);
        orig_img_hist = np.zeros((L,1));
        mod_img_hist = np.zeros((L,1));
        for i in range(L):
            orig_img_hist[i,0] = np.sum(self.image == i)
        cdf = np.cumsum(orig_img_hist)
        temp = ((L-1)/(imgrow*imgcol))
        for i in range(L):
            result[np.where(self.image == i)] = np.round(temp*cdf[i])
        for i in range(L):
            mod_img_hist[i,0] = np.sum(result == i)
        result = result.astype(np.uint8)
        self.finalDisplay(result)

    @pyqtSlot()
    def gammaCorrection(self):
        c = 1.0
        gamma, okPressed = QInputDialog.getDouble(self, "Get Gamma value","Value:")
        if okPressed:
            out = cv2.normalize(self.image.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
            result = np.rint(c*pow(out,gamma)*255.0)
            result = result.astype(np.uint8)
            self.finalDisplay(result)

    @pyqtSlot()
    def logTransform(self):
        c = 1.0
        out = cv2.normalize(self.image.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
        result = np.rint(c*np.log(1+out)*255.0)
        result = result.astype(np.uint8)     # for converting to an opencv image datatype
        # cv2.imshow('Log Transform Image', result)
        self.finalDisplay(result)

    @pyqtSlot()
    def blurImage(self):
        winsize, ok1Pressed = QInputDialog.getInt(self, "Kernel Window Size","Value:")
        sigma, ok2pressed = QInputDialog.getDouble(self, "Standard Deviation","Value:")
        if (ok1Pressed and ok2pressed):
            if(winsize%2!=0):
                imgrow, imgcol = self.image.shape
                mykernel = genGaussianKernel(winsize, sigma)
                # mykernel = boxKernel(winsize)
                paddedImage = np.zeros((imgrow+(winsize//2)*2, imgcol+(winsize//2)*2), dtype=np.uint8)
                paddedImage[winsize//2:imgrow+winsize//2, winsize//2:imgcol+winsize//2] = self.image
                # cv2.imshow('paddedImage', paddedImage)
                filtImage = np.zeros(self.image.shape,dtype=np.uint8)
                for i in range(imgrow):
                    for j in range(imgcol):
                        filtImage[i,j] = np.round(np.sum(paddedImage[i:i+winsize, j:j+winsize]*mykernel))
                filtImage = filtImage.astype(np.uint8)
                self.finalDisplay(filtImage)
            else:
                buttonReply = QMessageBox.question(self, 'Warning Message', "Window Size has to be an odd number", QMessageBox.Ok, QMessageBox.Ok)

    @pyqtSlot()
    def sharpImage(self):
        winsize, ok1Pressed = QInputDialog.getInt(self, "Kernel Window Size","Value:")
        if(ok1Pressed):
            if(winsize%2!=0):
                imgrow, imgcol = self.image.shape
                mykernel = sharpeningKernel(winsize)
                paddedImage = np.zeros((imgrow+(winsize//2)*2, imgcol+(winsize//2)*2), dtype=np.uint8)
                paddedImage[winsize//2:imgrow+winsize//2, winsize//2:imgcol+winsize//2] = self.image
                filtImage = np.zeros(self.image.shape,dtype=np.uint8)
                for i in range(imgrow):
                    for j in range(imgcol):
                        filtImage[i,j] = np.round(np.sum(paddedImage[i:i+winsize, j:j+winsize]*mykernel))
                filtImage = filtImage.astype(np.uint8)
                self.finalDisplay(filtImage)
            else:
                buttonReply = QMessageBox.question(self, 'Warning Message', "Window Size has to be an odd number", QMessageBox.Ok, QMessageBox.Ok)

    @pyqtSlot()
    def undoLast(self):
        self.hsvImage[:,:,2] = self.image
        temp = cv2.cvtColor(self.hsvImage, cv2.COLOR_HSV2RGB)
        self.image = cv2.cvtColor(self.prevImage,cv2.COLOR_RGB2HSV)[:,:,2]
        qImage = self.toQImage(self.prevImage)
        self.label_result.setPixmap(QPixmap(qImage))
        self.prevImage = temp

    @pyqtSlot()
    def undoAll(self):
        self.hsvImage[:,:,2] = self.image
        self.prevImage = cv2.cvtColor(self.hsvImage, cv2.COLOR_HSV2RGB)
        self.image = cv2.cvtColor(self.origImage, cv2.COLOR_RGB2HSV)[:,:,2]
        qImage = self.toQImage(self.origImage)
        self.label_result.setPixmap(QPixmap(qImage))

    @pyqtSlot()
    def saveImage(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getSaveFileName(self,"Save Image File","","All Files (*);;jpg Files(*.jpg);;png Files(*.png)", options=options)
        cv2.imwrite(fileName, self.image)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MyImageEditor()
    sys.exit(app.exec_())