# import the necessary packages
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense, MaxPooling2D, Dropout
from tensorflow.keras import backend as K

class CNN:
	@staticmethod
	def build(width, height, depth, classes):
		# initialize the model along with the input shape to be
		# "channels last"
		model = Sequential()
		inputShape = (height, width, depth)

		# if we are using "channels first", update the input shape
		if K.image_data_format() == "channels_first":
			inputShape = (depth, height, width)

		# 1 block
		model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform',
						 padding='same', input_shape=inputShape))
		model.add(MaxPooling2D((2, 2)))
		model.add(Dropout(0.2))

		# 2 block
		model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
		model.add(MaxPooling2D((2, 2)))
		model.add(Dropout(0.2))

		# 3 block
		model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
		model.add(MaxPooling2D((2, 2)))
		model.add(Dropout(0.2))


		# softmax classifier
		model.add(Flatten())
		model.add(Dense(classes))
		model.add(Activation("softmax"))

		# return the constructed network architecture
		return model