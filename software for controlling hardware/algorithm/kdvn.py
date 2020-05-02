import time 
import cv2
import argparse
import glob
import math
import os
import pickle
import numpy as np 

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

from algorithm.algorithm_interface import AlgorithmInterface
from hw.hardware import Hardware

class KDVN(AlgorithmInterface):

    mode = 'RGB'

    # Statically stored classifiers
    classifiers = {}

    # File locations of datasets for each material
    materialGlobs = {
        'Copper_cz': 'dataset/copper_cz/*',
        'Brass_cz': 'dataset/brass_cz/*',
        'Silver_cz': 'dataset/silver_cz/*',
        'Czech50': 'dataset/czech_50/*',
        'Copper_eu': 'dataset/copper_eu/*',
        'Brass_eu': 'dataset/brass_eu/*',
        'Euro1': 'dataset/euro_1/*',
        'Euro2': 'dataset/euro_2/*'  
    }

    # Coin material assignments
    coinMaterials = {
        ' 1Kc': 'Silver_cz', 
        ' 2Kc': 'Silver_cz', 
        ' 5Kc': 'Silver_cz',
        '10Kc': 'Copper_cz',
        '20Kc': 'Brass_cz',
        '50Kc': 'Czech50',
        '  1c': 'Copper_eu',
        '  2c': 'Copper_eu',
        '  5c': 'Copper_eu',
        ' 10c': 'Brass_eu',
        ' 20c': 'Brass_eu',
        ' 50c': 'Brass_eu',
        '1EUR': 'Euro1',
        '2EUR': 'Euro2'
    }

    # https://www.eprehledy.cz/ceske-bankovky-a-mince-autor-vaha-rozmery-motivy.php
    # https://ec.europa.eu/info/business-economy-euro/euro-area/euro-coins-and-notes/euro-coins/common-sides-euro-coins_en
    coinSizes = {
        ' 1Kc': 20.0, 
        ' 2Kc': 21.5, 
        ' 5Kc': 23.0,
        '10Kc': 24.5,
        '20Kc': 26.0,
        '50Kc': 27.5,
        '  1c': 16.25,
        '  2c': 18.75,
        '  5c': 21.25,
        ' 10c': 19.75,
        ' 20c': 22.25,
        ' 50c': 24.25,
        '1EUR': 23.25,
        '2EUR': 25.75
    }
    coinSizeScale=8.97  # Depends on camera resolution, adjust if necessary

    # Which coins are concentric?
    concentricCoins = {
        ' 1Kc': False, 
        ' 2Kc': False, 
        ' 5Kc': False,
        '10Kc': False,
        '20Kc': False,
        '50Kc': True,
        '  1c': False,
        '  2c': False,
        '  5c': False,
        ' 10c': False,
        ' 20c': False,
        ' 50c': False,
        '1EUR': True,
        '2EUR': True
    }

    # Which coins are noncircular
    noncircularCoins = {
        ' 1Kc': False, 
        ' 2Kc': True, 
        ' 5Kc': False,
        '10Kc': False,
        '20Kc': True,
        '50Kc': False,
        '  1c': False,
        '  2c': False,
        '  5c': False,
        ' 10c': False,
        ' 20c': False,
        ' 50c': False,
        '1EUR': False,
        '2EUR': False
    }
    noncircularThreshold = 0.96
    noncircularExtend = 0

    fixerThreshold = 220
    demonstratorBorderCut=(0,100) # To cut off demonstrator borders when detecting coins (Y,X dimensions)
    cnnShape = (200,200)

    iteration = None
    currency = None
    cnn = False

##################################################################
##################################################################
# Main algorithm method

    def __init__(self, cnn: bool):
        self.cnn = cnn

    def alg(self, iterations: int, currency: str) -> dict:
        self.iteration = iterations
        self.currency = currency

        # Capturing and preprocessing
        (master, bottom, fixer) = self.__capture()

        # Segmentation and descriptors
        circles = self.__hough(bottom)

        # Description
        dataset = self.__extractDescriptors(master, bottom, fixer, circles)

        # Classification
        findings = self.__classify(dataset)

        # Output
        return self.__output(findings, master)     


##################################################################
##################################################################
# Vision chain
    

    def __capture(self):
        """Captures images for algorithm"""

        Hardware.leds(False, False, True)
        time.sleep(0.1)
        imgBottom = Hardware.cam(iso = 50, shutter = (1/400))
        imgFixer = Hardware.cam(iso = 50, shutter = (1/80))
        (w,h,_) = imgBottom.shape
        imgBottom = imgBottom[self.demonstratorBorderCut[0]:(w-self.demonstratorBorderCut[0]), self.demonstratorBorderCut[1]:(h-self.demonstratorBorderCut[1]) , :] 
        imgFixer = imgFixer[self.demonstratorBorderCut[0]:(w-self.demonstratorBorderCut[0]), self.demonstratorBorderCut[1]:(h-self.demonstratorBorderCut[1]) , :]
        cv2.imwrite('img/bottom.jpg', imgBottom)
        cv2.imwrite('img/fixer.jpg', imgFixer)

        Hardware.leds(True, True, False)
        time.sleep(0.1)
        imgMaster = Hardware.cam(iso = 200, shutter = (1/200))
        imgMaster = imgMaster[self.demonstratorBorderCut[0]:(w-self.demonstratorBorderCut[0]), self.demonstratorBorderCut[1]:(h-self.demonstratorBorderCut[1]) , :]
        cv2.imwrite('img/master.jpg', imgMaster)

        Hardware.leds(False, False, False)

        # Adjust fixer for yellowish look
        res = np.float32(imgFixer[:,:,0]) * 1.25
        imgFixer[:,:,0] = np.clip(res, 0, 255).astype(np.uint8)
        imgFixer = (imgFixer * 0.9).astype(np.uint8)

        return (imgMaster, imgBottom, imgFixer)        


    def __hough(self, src):
        """Preprocesses the image and uses Hough transform to get circles from the image"""
        image = src.copy()

        #clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #gray = clahe.apply(gray)
        #gray = cv2.GaussianBlur(gray, (7, 7), 0)

        return cv2.HoughCircles(
            gray, 
            cv2.HOUGH_GRADIENT, 
            dp=2.5, 
            minDist= int(16*self.coinSizeScale), 
            param1 = 200, 
            param2 = 100, 
            minRadius = int((16 / 2) * self.coinSizeScale), 
            maxRadius = int((28 / 2) * self.coinSizeScale)
            # Do not forget to divide radii by 2 as sizes are diameters
        )

    def __extractDescriptors(self, image, bottom, fixer, circles):
        """Extracts descriptors from a preprocessed and segmentated image"""
        res = []
        
        # Prepare stencil for concentricity detection
        gray = cv2.cvtColor(bottom, cv2.COLOR_BGR2GRAY)
        (_,stencil) = cv2.threshold(gray,100,255,cv2.THRESH_BINARY_INV)

        # Prepare contrasted gray for concentricity detection
        gray = (cv2.cvtColor(image, cv2.COLOR_BGR2HSV))[:,:,1]
        concentricImg = gray
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        concentricImg = clahe.apply(concentricImg)
        concentricImg = np.clip(concentricImg * 1.2, 0, 255).astype(np.uint8)

        if circles is not None:

            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:

                # Extract regions of interest
                roi = self.__getRoi(r, x, y, image, gray, fixer)
                stencilRoi = stencil[y - r - self.noncircularExtend:y + r + self.noncircularExtend, x - r - self.noncircularExtend:x + r + self.noncircularExtend]
                concentricRoi = concentricImg[y - r:y + r, x - r:x + r]

                cv2.imwrite('img/roi_{0}_{1}.jpg'.format(self.iteration, r), roi)
                # Make predictions to form descriptor vector
                material = self.__predictMaterial(roi) 
                concentric = self.__predictConcentric(concentricRoi, r)
                noncircular = self.__predictNoncircular(stencilRoi, r + self.noncircularExtend)

                # Append the vector 
                res.append((x, y, r, material, concentric, noncircular))
                print(res[-1])

        return res



    def __classify(self, dataset):
        """Classifies the coins"""

        # Start by sorting and adjusting the coin sizes
        sizes = {k: (v * self.coinSizeScale) for k, v in sorted(self.coinSizes.items(), key=lambda item: item[1])}

        # Loop over each descriptor vector
        findings = []
        print('SCORES------------------')
        for tpl in dataset:
            (x,y,r,material, concentric, noncircular) = tpl

            # Obtain scores for each coin per descriptor
            sizeScore = self.__score_size(r, sizes)
            materialScore = self.__score_material(material, self.coinMaterials, 0.085)
            noncircularScore = self.__score_noncircular(noncircular, 0.04)
            concentricScore = self.__score_concentric(concentric, 0.04)
            currencyScore = self.__score_currency()
            
            # Apply a known metric (addition) to the different scores
            scores = {}
            for key in self.coinSizes:
                scores[key] = sizeScore[key] + materialScore[key] + noncircularScore[key] + concentricScore[key] + currencyScore[key]

            # Sort by score
            scores = {k: v for k, v in sorted(scores.items(), key=lambda item: item[1])}
            keys = list(scores.keys())
            
            # Append finding
            findings.append((x,y,r,material,(keys[0], keys[1])))

        return findings

    def __output(self, findings, image):
        """Creates algorithm output"""

        coins = {
            ' 1Kc': 0, 
            ' 2Kc': 0, 
            ' 5Kc': 0,
            '10Kc': 0,
            '20Kc': 0,
            '50Kc': 0,
            '  1c': 0,
            '  2c': 0,
            '  5c': 0,
            ' 10c': 0,
            ' 20c': 0,
            ' 50c': 0,
            '1EUR': 0,
            '2EUR': 0
        }

        # Draw the findings
        annotated = image.copy()
        for finding in findings:
            (x,y,s, material, keys) = finding 
            x = int(x)
            y = int(y)
            s = int(s)

            print(finding)
            cv2.circle(annotated, (x,y), s, (0, 255, 0), 3)
            cv2.putText(annotated, material, (x - 40, y - 20), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2, lineType=cv2.LINE_AA)
            cv2.putText(annotated, keys[0], (x - 60, y + 22), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3, lineType=cv2.LINE_AA)

        # Create final array
        for finding in findings:
            (_,_,_,_,keys) = finding 
            coins[keys[0]] += 1

        return {
            'found': len(findings) > 0,
            'image_annotated': cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
            'image_original': cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            'coins': coins
        }


##################################################################
##################################################################
# Scoring methods

    def __score_size(self, radius, sizes):
        """Scores coin types by their difference from the passed radius"""

        # Assign score
        score = {}
        m = 0
        for coin in sizes:
            diff = abs(sizes[coin] - (radius * 2))
            if diff > m:
                m = diff

            score[coin] = diff

        # Normalize to have values from 0 to 1
        for coin in score:
            score[coin] = score[coin] / m

        return score

    def __score_material(self, material, materials, weight = 0.3):
        """Scores coin types by their material type"""

        score = {}
        for coin in materials:
            if material == materials[coin]:
                score[coin] = 0
            else:
                score[coin] = weight 

        return score

    
    def __score_concentric(self, concentric, weight = 0.3):
        """Scores coin types by their material type"""

        score = {}
        for coin in self.concentricCoins:
            if concentric and not self.concentricCoins[coin]:
                score[coin] = weight 
            else:
                score[coin] = 0 

        return score

        
    def __score_noncircular(self, noncircular, weight = 0.3):
        """Scores coin types by their material type"""

        score = {}
        for coin in self.concentricCoins:
            if noncircular and not self.noncircularCoins[coin]:
                score[coin] = weight 
            else:
                score[coin] = 0 

        return score

    def __score_currency(self):
        """Scores based on currency to eliminate unwanted currencies"""

        score = {}
        for coin in self.concentricCoins:
            if self.currency == 'CZK' and 'Kc' not in coin:
                score[coin] = 100
            elif self.currency == 'EUR' and 'Kc' in coin:
                score[coin] = 100
            else: 
                score[coin] = 0
        
        return score


##################################################################
##################################################################
# Predictions

    def __predictMaterial(self, roi):
        """Predicts material type for a chosen region of interest"""

        index = 0
        if self.cnn:
            # CNN scaled roi

            if(roi.shape[0] == 0 or roi.shape[1] == 0):
                index = 0
            else:
                img = cv2.resize(roi, self.cnnShape, interpolation=cv2.INTER_AREA)
                img = img_to_array(img)
                img = np.expand_dims(img, axis=0)
                arr = self.__load_classifier().predict_classes(img)
                index = arr[0]

        else: 
            # MLP histogram
            histogram = self.__calculate_histogram(roi)
            arr = self.__load_classifier().predict([histogram])
            index = arr[0]


        if self.currency == 'EUR':
            index += 4

        return list(self.materialGlobs.keys())[index]

    def __predictConcentric(self, equalRoi, radius):
        """Returns boolean indicating whether the image contains another circle"""
        
        cv2.imwrite('img/con_{0}_{1}.jpg'.format(self.iteration, radius), equalRoi)
        maxr = int(radius * 0.85) # empirical value
        minr = int(radius * 0.55) # empirical value 

        if(equalRoi.shape[0] == 0 or equalRoi.shape[1] == 0):
            return False

        res = cv2.HoughCircles(
            equalRoi, 
            cv2.HOUGH_GRADIENT, 
            dp=2, 
            minDist= 1, # Can be more, will be filtered out later
            param1 = 110,
            param2 = 160, 
            minRadius = minr,
            maxRadius = maxr
        )

        if res is None:
            return False 

        # Check if any of the findings lie near image center
        for found in res[0,:]:
            if (found[0] - radius)**2 + (found[1] - radius)**2 < (radius * 0.1)**2:
                return True

        return False

    def __predictNoncircular(self, stencilRoi, radius):
        """Returns boolean indicating whether the coin image is circular or not"""
        cv2.imwrite('img/non_{0}_{1}.jpg'.format(self.iteration, radius), stencilRoi)

        # Calculate ratio of white pixels to all pixels
        # Count only those which lie within radius (circle)
        total = 0
        sm = 0
        r2 = radius * radius 
        for x in range(-radius,radius):
            x2 = x*x 
            for y in range(-radius,radius):
                y2 = y*y
                if (x2+y2) <= r2 and (radius + x) < stencilRoi.shape[1] and (radius + y) < stencilRoi.shape[0] and (radius + y) >= 0 and (radius + x) >= 0:
                    total += 255
                    sm += stencilRoi[radius + y,radius + x]

        if total <= 0:
            return False

        ratio = sm / total
        return ratio < self.noncircularThreshold

##################################################################
##################################################################
# Helpers


    def __getRoi(self, r, x, y, image, gray, fixer):
        """Resolves ROI for final detection"""

        iroi = (image.copy())[y - r:y + r, x - r:x + r]
        if(iroi.shape[0] == 0 or iroi.shape[1] == 0):
                return iroi

        groi = gray[y - r:y + r, x - r:x + r]
        if np.average(groi) <= self.fixerThreshold:
            # Standard roi taken from master image
            return iroi
        else:

            # Image is too bright, let's use our puzzle method 
            cx = int(iroi.shape[1] / 2)
            ty = iroi.shape[0]

            # Absolutely optimized
            froi = fixer[y - r:y + r, x - r:x + r]
            for y in range(0,r):
                dx = int(math.sqrt(2*r*y - y*y))
                if dx > 0:
                    iroi[y, (cx-dx):(cx+dx)] = froi[y, (cx-dx):(cx+dx)]
                    iroi[ty - y - 1, (cx-dx):(cx+dx)] = froi[ty - y - 1, (cx-dx):(cx+dx)]

            return iroi

    def __load_classifier(self):
        """Loads trained material classifier from a save file"""
        
        index = f"{self.currency}-{self.cnn}"
        if index not in KDVN.classifiers:
            # Just load trained classfier  
             
            if not self.cnn: 
                path = os.path.dirname(os.path.realpath(__file__)) + f"/classifier/classifier_{self.mode}_{self.currency}.sav"
                KDVN.classifiers[index] = pickle.load(open(path, 'rb'))
            else:
                path = os.path.dirname(os.path.realpath(__file__)) + f"/classifier/classifier_CNN_CZK.hdf5"
                KDVN.classifiers[index] = load_model(path)

        return KDVN.classifiers[index]

    
    def __calculate_histogram_from_file(self, file):
        """Calculates histogram from a file, for classification"""
        image = cv2.imread(file)
        return self.__calculate_histogram(image)

    def __calculate_histogram(self, image):
        """Calculates equalized histogram for classification"""

        # create mask
        mask = np.zeros(image.shape[:2], dtype="uint8")
        (width, height) = (int(image.shape[1] / 2), int(image.shape[0] / 2)) # get center coordinates

        # cv2.circle(img, center, radius, color, thickness, lineType, shift)
        cv2.circle(mask, (width, height), 60, 255, -1) # thickness of -1 = full color in circle

        # calcHist expects a list of images, color channels, mask, bins, ranges
        if self.mode == 'HSV':
            image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            histogram = cv2.calcHist([image_hsv], [0, 1, 2], mask, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        else:
            histogram = cv2.calcHist([image], [0, 1, 2], mask, [8, 8, 8], [0, 256, 0, 256, 0, 256])

        # return normalized flattened histogram which will be used in data to learn for classifier
        return cv2.normalize(histogram, histogram).flatten()

    




