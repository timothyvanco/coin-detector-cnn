# import the necessary packages
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense, MaxPool2D, Dropout
from tensorflow.keras import backend as K

class VGG_CNN:
	@staticmethod
	def build(width, height, depth, classes):
		# initialize the model along with the input shape to be
		# "channels last"
		model = Sequential()
		inputShape = (height, width, depth)

		# if we are using "channels first", update the input shape
		if K.image_data_format() == "channels_first":
			inputShape = (depth, height, width)


		model.add(Conv2D(input_shape=inputShape, filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
		model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
		model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
		model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
		model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
		model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
		model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
		model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
		model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
		model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
		model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
		model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
		model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
		model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
		model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
		model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
		model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
		model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

		model.add(Flatten())

		model.add(Dense(units=4096, activation="relu"))
		model.add(Dense(units=4096, activation="relu"))
		model.add(Dense(units=classes, activation="softmax"))

		return model