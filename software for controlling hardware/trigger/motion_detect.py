import numpy as np
import time
import cv2

from hw.hardware import Hardware

class MotionDetect:
    """
    Realizes differential motion detection with reaction to the falling edge of motion.
    """

    MOTION_THRESHOLD = 70000
    MODEL_THRESHOLD = 0.19

    def __init__(self):
        self.model_img = None
        self.last_img = None
        self.last = False

    def acknowledge(self):
        """Acknowledges new image and triggers if applicable"""

        # Get the image
        img = Hardware.cam(iso = 200, shutter = (1 / 120), type = Hardware.CAM_IMG_SMALL)
        mimg = self.__modif(img)

        # No model, let's make one and return true to trigger immediately
        if self.model_img is None:
            self.model_img = mimg 
            self.last_img = mimg
            self.last = False
            return True

        # Simple dynamic scene algorithm without environment model
        d = np.absolute(mimg - self.last_img).sum()
        ratio = d / self.last_img.sum()
        print(f"Motion detector ratio: {ratio}")

        self.last_img = mimg    
        if ratio > self.MODEL_THRESHOLD:
            self.last = True 
            return False
        elif self.last == True:
            self.last = False
            return True 
        else:
            self.last = False
            return False           
        

        
        # # Model updating algorithm
        # # http://midas.uamt.feec.vutbr.cz/POV/pov_cz.php
        # d1 = np.absolute(mimg - self.last_img).sum()
        # d2 = np.absolute(mimg - self.model_img).sum()
        # print('D1: {0} D2: {1}'.format(d1, d2))
        # if d1 < self.MOTION_THRESHOLD:
        #     # The scene is static, will we update the model?
        #     if d2 < self.MODEL_THRESHOLD:
        #         self.model_img = (self.model_img * 0.9 + 0.1 * mimg).astype(np.int16)

        #     self.last = False
        #     self.last_img = mimg
        #     return False 
        # else:
        #     # Dynamic scene, Motion detection
        #     self.last_img = mimg
        #     if d2 > self.MODEL_THRESHOLD:
        #         self.last = True 
        #         return False
        #     elif self.last == True:
        #         time.sleep(0.3)
        #         self.last = False
        #         self.model_img = mimg # Must be here to allow for demonstrator's board content changes
        #         return True 
        #     else:
        #         self.last = False
        #         return False                


    def dump_model(self, name):
        """Saves model image to a file defined by the name"""
        #cv2.imwrite('./img/{0}.jpg'.format(name), self.model_img)
        cv2.imwrite('./img/{0}.jpg'.format(name), self.last_img)

    def __modif(self, img):
        """Grayscales the image for easier operation and equalizes its values"""

        #clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #gray = clahe.apply(gray)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        return gray.astype(np.int16)

    