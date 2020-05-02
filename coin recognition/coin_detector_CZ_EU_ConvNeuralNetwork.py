
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import img_to_array

from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from skimage import feature  # histogram of orientated gradients (HOG) from feature

import math
import numpy as np
import glob
import cv2
import os

import pickle  # save & load classifier


class MiniVGGNet:
    @staticmethod
    def build(width, height, depth, classes):
        # initialize the model along with the input shape to be
        # "channels last" and the channels dimension itself
        model = Sequential()
        inputShape = (height, width, depth)

        # if we are using "channels first", update the input shape
        # and channels dimension
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)

        # first CONV => RELU => CONV => RELU => POOL layer set
        model.add(Conv2D(32, (3, 3), padding="same", input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))

        model.add(Conv2D(32, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # second CONV => RELU => CONV => RELU => POOL layer set
        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # first (and only) set of FC => RELU layers
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        # softmax classifier
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        # return the constructed network architecture
        return model


class LeNet:
    @staticmethod
    def build(width, height, depth, classes):
        model = Sequential()
        inputShape = (height, width, depth)

        # if using "channels first", update input shape
        if K.image_data_format() == "channel_first":
            inputShape = (depth, height, width)

        # first set of CONV => RELU => POOL layers
        model.add(Conv2D(20, (5, 5), padding="same", input_shape=inputShape))  # 20 filters, each of size 5x5
        model.add(Activation("relu"))  # ReLU activation function
        model.add(MaxPooling2D(pool_size=(2, 2),
                               strides=(2, 2)))  # 2x2 pooling with 2x2 stride - decreasing input volume size by 75%

        # second set of CONV => RELU => POOL layers
        model.add(Conv2D(50, (5, 5), padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Flatten())  # input volume is flattened
        model.add(Dense(500))  # fully-connected layer with 500 nodes
        model.add(Activation("relu"))

        model.add(Dense(classes))
        model.add(Activation("softmax"))

        # return constructed network architecture
        return model



def quantify_image(image):
    # compute HOG feature vectur for input image
    features = feature.hog(image, orientations=9,
                           pixels_per_cell=(10, 10), cells_per_block=(2, 2),
                           transform_sqrt=True, block_norm="L1")
    return features


# take path to input directory, initialize list of data (images) and class labels
def load_image(image):
    image = cv2.imread(image)  # load input image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # convert to grayscale
    image = cv2.resize(image, (200, 200))  # resize to 200x200 pixels
    image = img_to_array(image)  # convert image into array
    return image  # return resized image as input to data


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

print("[INFO] loading data...")

# training data & labels for classifier
data = []
labels = []

# prepare training data and labels
# data - histogram of coin
# label - material of coin
# CZ
for i in images_copper_cz:
    data.append(load_image(i))
    labels.append(Material.Copper_cz)

for i in images_brass_cz:
    data.append(load_image(i))
    labels.append(Material.Brass_cz)

for i in images_silver_cz:
    data.append(load_image(i))
    labels.append(Material.Silver_cz)

for i in images_czech_50:
    data.append(load_image(i))
    labels.append(Material.Czech50)

# EU
for i in images_copper_eu:
    data.append(load_image(i))
    labels.append(Material.Copper_eu)

for i in images_brass_eu:
    data.append(load_image(i))
    labels.append(Material.Brass_eu)

for i in images_euro1:
    data.append(load_image(i))
    labels.append(Material.Euro1)

for i in images_euro2:
    data.append(load_image(i))
    labels.append(Material.Euro2)

print("[INFO] finish loading data...")
print()
print("DATA length: {}".format(len(data)))
print("LABELS length: {}".format(len(labels)))


data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

print("[INFO] split dataset...")
# split samples into training and testing data, 80% for training & 20% for testing
data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size=0.2)

print("[INFO] data_train len: {}".format(len(data_train)))
print("[INFO] data_test len: {}".format(len(data_test)))
print("[INFO] labels_train len: {}".format(len(labels_train)))
print("[INFO] labels_test len: {}".format(len(labels_test)))

lb = LabelBinarizer().fit(labels_train)
labels_train = lb.transform(labels_train)
labels_test = lb.transform(labels_test)

# initialize label names for coin dataset
labelNames = ["Copper_cz", "Brass_cz", "Silver_cz", "Czech50", "Copper_eu", "Brass_eu", "Euro1", "Euro2"]

# initialize optimizer and model
print("[INFO] compiling model...")
# SGD with learning rate = 0.01, Nestrov accelerated gradient - True
# decay - slowly reduce learning rate over time - helpful in reducing overfitting -> higher accuracy, lr/40 - 40 epochs
# optimizer = SGD(lr=0.01, decay=0.01/40, momentum=0.9, nesterov=True)
model = LeNet.build(width=200, height=200, depth=1, classes=8)
optimizer = SGD(lr=0.01)
model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
print("[INFO] model compiled")


# train network
print("[INFO] training network...")
H = model.fit(data_train, labels_train, validation_data=(data_test, labels_test), batch_size=5, epochs=40, verbose=1)
print("[INFO] training finish")


labelNames = ["Copper_cz", "Brass_cz", "Silver_cz", "Czech50", "Copper_eu", "Brass_eu", "Euro1", "Euro2"]

# evaluate network
print("[INFO] evaluating network...")
predictions = model.predict(data_test, batch_size=10)
print(classification_report(labels_test.argmax(axis=1), predictions.argmax(axis=1), target_names=labelNames))

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 40), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 40), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 40), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, 40), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy on coin detector")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig('training_acc.png')


# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")


input_image = 'images/input_images/input_cz_1.jpg'
image = cv2.imread(input_image)

plt.rcParams["figure.figsize"] = (16, 9)
plt.imshow(image)


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
materials2 = []
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

blurred_gray = cv2.cvtColor(blurred_rgb, cv2.COLOR_BGR2GRAY)

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
        ROI = blurred_gray[y - d:y + d, x - d:x + d]  # extract region of interest

        ROI = cv2.resize(ROI, (200, 200))  # resize to 200x200 pixels
        ROI = img_to_array(ROI)  # convert image into array
        ROI = np.expand_dims(img_to_array(ROI), axis=0) / 255.0
        # try recognition of material type and add result to list
        material = model.predict(ROI).argmax(axis=1)[0] + 1
        materials.append(material)
        materials2.append(str(material))

        # draw contour and results in the output image
        cv2.circle(outputImage, (x, y), d, (0, 255, 0), 2)
        # cv2.putText(outputImage, material, (x-40, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
        cv2.putText(outputImage, str(material), (x - 40, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), thickness=2,
                    lineType=cv2.LINE_AA)



# resize output image while retaining aspect ratio
# d = 768 / outputImage.shape[1]
# dimension = (768, int(outputImage.shape[0] * d))
# image = cv2.resize(image, dimension, interpolation=cv2.INTER_AREA)
# outputImage = cv2.resize(outputImage, dimension, interpolation=cv2.INTER_AREA)

# write summary on output image
# cv2.putText(image, text, org, font, fontScale, color[, thickness[, lineType[, bottomLeftOrigin]]])
# cv2.putText(outputImage, scaledTo, (5, outputImage.shape[0] - 140), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), lineType=cv2.LINE_AA)
# cv2.putText(outputImage, "Coins detected: {}, money value: {:2}Kc, {:0.2f} eur".format(count, total_value_czech, total_value_eu), (5, outputImage.shape[0] - 90), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), lineType=cv2.LINE_AA)
# cv2.putText(outputImage, "Total value in Czech: {:2}".format(total_value), (5, outputImage.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), lineType=cv2.LINE_AA)
# cv2.putText(outputImage, "Classifier mean accuracy: {}%".format(score), (5, outputImage.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), lineType=cv2.LINE_AA)

# print("Coins detected: {}, money value: {:2}Kc, {:0.2f} eur".format(count, total_value_czech, total_value_eu))
# print("Total value in Czech: {:2}".format(total_value))
# print("Classifier mean accuracy: {}%".format(score))

# show output and wait for key to terminate program
# hstack = np.hstack([image, outputImage])

plt.rcParams["figure.figsize"] = (16, 9)
plt.imshow(outputImage)

# cv2.imshow("Output", outputImage)
cv2.imwrite("output_image_eu_cz.jpg", outputImage)
cv2.waitKey(0)


