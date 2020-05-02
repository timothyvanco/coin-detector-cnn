
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

import math
import numpy as np
import glob
import cv2

import pickle  # save & load classifier


def calculate_histogram(image):
    # create  mask
    mask = np.zeros(image.shape[:2], dtype="uint8")
    (width, height) = (int(image.shape[1] / 2), int(image.shape[0] / 2))  # get center coordinates

    # cv2.circle(img, center, radius, color, thickness, lineType, shift)
    cv2.circle(mask, (width, height), 60, 255, -1)  # thickness of -1 = full color in circle

    # calcHist expects a list of images, color channels, mask, bins, ranges
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    histogram = cv2.calcHist([image_hsv], [0, 1, 2], mask, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    # histogram = cv2.calcHist([image], [0, 1, 2], mask, [8, 8, 8], [0, 256, 0, 256, 0, 256])

    # return normalized flattened histogram which will be used in data to learn for classifier
    return cv2.normalize(histogram, histogram).flatten()


# load picture from file - calculate histogram
def calculate_histogram_from_file(file):
    image = cv2.imread(file)
    return calculate_histogram(image)


def predictMaterial(roi):
    # calculate feature vector for region of interest
    histogram = calculate_histogram(roi)

    # predict material type based on histogram
    material_type = classifier.predict([histogram])

    # return predicted material type
    return Material[int(material_type)]


input_image = 'input_eu_cz_3.jpg'
image = cv2.imread(input_image)

# create copy of an image, if we will apply some destructive functions
outputImage = image.copy()

# convert image to grayscale for detecting contours
grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# improve contrast not just by adaptive histogram equalization, but with CLAHE
# CLAHE - contrast limiting adaptive histogram equalization
# clipLimit - Threshold for contrast limiting
# tileGridSize - size of grid for hist equalization, input image divided into equally sized rectangular tiles
# - this define number of tiles in a row and column
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
grayImage = clahe.apply(grayImage)

# grayImage2= cv2.equalizeHist(grayImage)
# res = np.hstack((grayImage2,grayImage)) #stacking images side-by-side
# plt.rcParams["figure.figsize"] = (16, 9)
# plt.imshow(res, cmap='gray')
# cv2.imwrite('glob_vs_clahe.png', res)

plt.rcParams["figure.figsize"] = (16, 9)
plt.imshow(grayImage, cmap='gray')

# create Enum class
class Enum(tuple):
    __getattr__ = tuple.index


# enumerate material types for use in classifier
Material = Enum(('Copper_cz', 'Brass_cz', 'Silver_cz', 'Czech50', 'Copper_eu', 'Brass_eu', 'Euro1', 'Euro2'))

# locate sample image files
# CZECH coins
images_copper_cz = glob.glob("images/copper_cz/*")
images_brass_cz = glob.glob("images/brass_cz/*")
images_silver_cz = glob.glob("images/silver_cz/*")
images_czech_50 = glob.glob("images/czech50/*")

# EU coins
images_copper_eu = glob.glob("images/copper_eu/*")
images_brass_eu = glob.glob("images/brass_eu/*")
images_euro1 = glob.glob("images/euro1/*")
images_euro2 = glob.glob("images/euro2/*")

# training data & labels for classifier
data = []
labels = []

# prepare training data and labels
# data - histogram of coin
# label - material of coin
# CZ
for i in images_copper_cz:
    data.append(calculate_histogram_from_file(i))
    labels.append(Material.Copper_cz)

for i in images_brass_cz:
    data.append(calculate_histogram_from_file(i))
    labels.append(Material.Brass_cz)

for i in images_silver_cz:
    data.append(calculate_histogram_from_file(i))
    labels.append(Material.Silver_cz)

for i in images_czech_50:
    data.append(calculate_histogram_from_file(i))
    labels.append(Material.Czech50)

# EU
for i in images_copper_eu:
    data.append(calculate_histogram_from_file(i))
    labels.append(Material.Copper_eu)

for i in images_brass_eu:
    data.append(calculate_histogram_from_file(i))
    labels.append(Material.Brass_eu)

for i in images_euro1:
    data.append(calculate_histogram_from_file(i))
    labels.append(Material.Euro1)

for i in images_euro2:
    data.append(calculate_histogram_from_file(i))
    labels.append(Material.Euro2)

# CLASSIFIER
# Multi Layer Perceptron Classifier = MLPClassifier
# hidden_layer_sizes = (default 100)
# activation = (default relu)
# solver = (default adam)
# .... many other things
classifier = MLPClassifier(hidden_layer_sizes=5, activation='tanh', solver='lbfgs', alpha=0.05,
                           learning_rate='adaptive',
                           max_iter=200)  # lbfgs - optimizer in the family of quasi-Newton methods

# split samples into training and testing data, 80% for training & 20% for testing
data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size=0.2)

# train and score classifier
classifier.fit(data_train, labels_train)
score = int(classifier.score(data_test, labels_test) * 100)  # in percentage
print("[INFO] Classifier mean accuracy is: {} %".format(score))

# save the model to disk
filename = 'classifier_HSV.sav'
pickle.dump(classifier, open(filename, 'wb'))

# load it from disk
# loaded_classifier = pickle.load(open(filename, 'rb'))
# score = int(classifier.score(data_test, labels_test) * 100)
# print("[INFO] Classifier mean accuracy is: {} %".format(score))

coinSizeScale = 8.97  # Depends on camera resolution, adjust if necessary

# blur image using Gaussian blurring, where pixels closer to the center contribute more weight to the average
# GaussianBlur(image, kernel size, sigma - 0 for autodetect)
blurred = cv2.GaussianBlur(grayImage, (7, 7), 0)

# plt.rcParams["figure.figsize"] = (16, 9)
# plt.imshow(blurred, cmap='gray')

res2 = np.hstack((grayImage, blurred))  # stacking images side-by-side
plt.rcParams["figure.figsize"] = (16, 9)
plt.imshow(res2, cmap='gray')
cv2.imwrite('gray_vs_blur.png', res2)


# HoughCircles for detecting coins - return: x, y, r for each detected circle
# src_image = Input image (grayscale)
# CV_HOUGH_GRADIENT = defines detection method
# dp = 2.2 : inverse ratio of resolution
# min_dist = 140 - minimum distance between detected centers
# param_1 = 200 - upper threshold for the internal canny edge detector
# param_2 = 100 - threshold for center detection
# min_radius = 50 - minimum radius to be detected
# max_radius = 120 - maximum radius to be detected
circles = cv2.HoughCircles(blurred,
                           cv2.HOUGH_GRADIENT,
                           dp=2.2,
                           minDist=int(16 * coinSizeScale),
                           param1=200,
                           param2=100,
                           minRadius=int((16 / 2) * coinSizeScale),
                           maxRadius=int((28 / 2) * coinSizeScale))

diameter = []
materials = []
coordinates = []

# CLAHE on RGB picture
bgr = image
lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
lab_planes = cv2.split(lab)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(12, 12))

lab_planes[0] = clahe.apply(lab_planes[0])
lab = cv2.merge(lab_planes)
image_lab = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
blurred_rgb = cv2.GaussianBlur(image_lab, (9, 9), 0)
blurred_rgb = cv2.GaussianBlur(image_lab, (9, 9), 0)

stack = np.hstack((image, image_lab, blurred_rgb))

plt.rcParams["figure.figsize"] = (16, 9)
plt.imshow(stack)
# cv2.imshow("stack_clahe_rgb", stack)
cv2.imwrite("stack_clahe_rgb.jpg", stack)
cv2.waitKey(0)

count = 0
# If some coins were detected
if circles is not None:
    # append radius to list of diameters
    for (x, y, r) in circles[0, :]:
        diameter.append(r)

    # convert coordinates and diameter to integers
    circles = np.round(circles[0, :]).astype("int")

    # loop over coordinates and diameter of circles
    for (x, y, d) in circles:
        count += + 1
        coordinates.append((x, y))  # add coordinates to list
        ROI = blurred_rgb[y - d:y + d, x - d:x + d]  # extract region of interest

        # try recognition of material type and add result to list
        material = predictMaterial(ROI)
        materials.append(material)

        # draw contour and results in the output image
        cv2.circle(outputImage, (x, y), d, (0, 255, 0), 2)
        cv2.putText(outputImage, material, (x - 40, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), thickness=2,
                    lineType=cv2.LINE_AA)



# CZECH coins
# Czech 50
czech50 = 27.50
abs_czech50 = 1.00

# Brass 20
czech20 = 26.00
abs_czech20 = 1.70

# Copper 10
czech10 = 24.50
abs_czech10 = 1.00

# Silver 5
czech5 = 23.00
abs_czech5 = 1.00

# Silver 2
czech2 = 21.50
abs_czech2 = 1.00

# Silver 1
czech1 = 20.00
abs_czech1 = 1.00

# EU coins
# eu 2
euro2 = 25.75
abs_euro2 = 1.00

# eu 1
euro1 = 23.25
abs_euro1 = 1.50

# brass 50
cent50 = 24.25
abs_cent50 = 1.00

# brass 20
cent20 = 22.25
abs_cent20 = 1.00

# brass 10
cent10 = 19.75
abs_cent10 = 1.00

# copper 5
cent5 = 21.25
abs_cent5 = 1.00

# copper 2
cent2 = 18.75
abs_cent2 = 1.00

# copper 1
cent1 = 16.25
abs_cent1 = 1.50

# get biggest diameter
biggestDiameter = max(diameter)
index_diameter = diameter.index(biggestDiameter)

# scale everything according to maximum diameter
if materials[index_diameter] == "Czech50":
    diameter = [x / biggestDiameter * czech50 for x in diameter]
    scaledTo = "Scaled to 50 Kc"

elif materials[index_diameter] == "Brass_cz":
    diameter = [x / biggestDiameter * czech20 for x in diameter]
    scaledTo = "Scaled to 20 Kc"

elif materials[index_diameter] == "Euro2":
    diameter = [x / biggestDiameter * euro2 for x in diameter]
    scaledTo = "Scaled to 2 eura"

elif materials[index_diameter] == "Copper_cz":
    diameter = [x / biggestDiameter * czech10 for x in diameter]
    scaledTo = "Scaled to 10 Kc"

elif materials[index_diameter] == "Brass_eu":
    diameter = [x / biggestDiameter * cent50 for x in diameter]
    scaledTo = "Scaled to 50 cent"

elif materials[index_diameter] == "Euro1":
    diameter = [x / biggestDiameter * euro1 for x in diameter]
    scaledTo = "Scaled to 1 euro"

elif materials[index_diameter] == "Silver_cz":
    diameter = [x / biggestDiameter * czech5 for x in diameter]
    scaledTo = "Scaled to 5 Kc"

else:
    scaledTo = "unable to scale"

# sum up the value of coins in front of camera
i = 0
total_value = 0
total_value_czech = 0
total_value_eu = 0

while i < len(diameter):
    d = diameter[i]
    material = materials[i]
    (x, y) = coordinates[i]
    text = "Unknown"

    # compare to known diameters with some margin for error
    # is close compare value with other and with tolerance
    if math.isclose(d, czech50, abs_tol=abs_czech50) and material == "Czech50":
        text = "50 Kc"
        total_value_czech += 50

    elif math.isclose(d, czech20, abs_tol=abs_czech20) and material == "Brass_cz":
        text = "20 Kc"
        total_value_czech += 20

    elif math.isclose(d, czech10, abs_tol=abs_czech10) and material == "Copper_cz":
        text = "10 Kc"
        total_value_czech += 10

    elif math.isclose(d, czech5, abs_tol=abs_czech5) and material == "Silver_cz":
        text = "5 Kc"
        total_value_czech += 5

    elif math.isclose(d, czech2, abs_tol=abs_czech2) and material == "Silver_cz":
        text = "2 Kc"
        total_value_czech += 2

    elif math.isclose(d, czech1, abs_tol=abs_czech1) and material == "Silver_cz":
        text = "1 Kc"
        total_value_czech += 1

    # eura
    elif math.isclose(d, euro2, abs_tol=abs_euro2) and material == "Euro2":
        text = "2 Eura"
        total_value_eu += 200

    elif math.isclose(d, euro1, abs_tol=abs_euro1) and material == "Euro1":
        text = "1 Euro"
        total_value_eu += 100

    elif math.isclose(d, cent50, abs_tol=abs_cent50) and material == "Brass_eu":
        text = "50 cent"
        total_value_eu += 50

    elif math.isclose(d, cent20, abs_tol=abs_cent20) and material == "Brass_eu":
        text = "20 cent"
        total_value_eu += 20

    elif math.isclose(d, cent10, abs_tol=abs_cent10) and material == "Brass_eu":
        text = "10 cent"
        total_value_eu += 10

    elif math.isclose(d, cent5, abs_tol=abs_cent5) and material == "Copper_eu":
        text = "5 cent"
        total_value_eu += 5

    elif math.isclose(d, cent2, abs_tol=abs_cent2) and material == "Copper_eu":
        text = "2 cent"
        total_value_eu += 2

    elif math.isclose(d, cent1, abs_tol=abs_cent1) and material == "Copper_eu":
        text = "1 cent"
        total_value_eu += 1

    # write result on output image
    cv2.putText(outputImage, text, (x - 40, y + 22), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), thickness=2,
                lineType=cv2.LINE_AA)
    i += 1

# hodnota penazi premena na Ceske:
total_value_eu = total_value_eu / 100
total_value = total_value_czech + (total_value_eu * 27.5)


# resize output image while retaining aspect ratio
# d = 768 / outputImage.shape[1]
# dimension = (768, int(outputImage.shape[0] * d))
# image = cv2.resize(image, dimension, interpolation=cv2.INTER_AREA)
# outputImage = cv2.resize(outputImage, dimension, interpolation=cv2.INTER_AREA)

# write summary on output image
# cv2.putText(image, text, org, font, fontScale, color[, thickness[, lineType[, bottomLeftOrigin]]])
cv2.putText(outputImage, scaledTo, (5, outputImage.shape[0] - 140), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255),
            lineType=cv2.LINE_AA)
cv2.putText(outputImage,
            "Coins detected: {}, money value: {:2}Kc, {:0.2f} eur".format(count, total_value_czech, total_value_eu),
            (5, outputImage.shape[0] - 90), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), lineType=cv2.LINE_AA)
cv2.putText(outputImage, "Total value in Czech: {:2}".format(total_value), (5, outputImage.shape[0] - 50),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), lineType=cv2.LINE_AA)
cv2.putText(outputImage, "Classifier mean accuracy: {}%".format(score), (5, outputImage.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), lineType=cv2.LINE_AA)

print("Coins detected: {}, money value: {:2}Kc, {:0.2f} eur".format(count, total_value_czech, total_value_eu))
print("Total value in Czech: {:2}".format(total_value))
print("Classifier mean accuracy: {}%".format(score))

# show output and wait for key to terminate program
# hstack = np.hstack([image, outputImage])

plt.rcParams["figure.figsize"] = (16, 9)
plt.imshow(outputImage)

# cv2.imshow("Output", outputImage)
cv2.imwrite("output_image_eu_cz.jpg", outputImage)
cv2.waitKey(0)


