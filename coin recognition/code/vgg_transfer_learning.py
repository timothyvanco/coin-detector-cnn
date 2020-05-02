# import the necessary packages
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense, MaxPooling2D, Dropout
from tensorflow.keras.applications.vgg16 import VGG16

class VGG_CNN:
	@staticmethod
	def build():
		# load model
		model = VGG16(include_top=False, input_shape=(224, 224, 3))

		# mark loaded layers as not trainable
		for layer in model.layers:
			layer.trainable = False

		# add new classifier layers
		flat1 = Flatten()(model.layers[-1].output)
		class1 = Dense(128, activation='relu', kernel_initializer='he_uniform')(flat1)
		output = Dense(4, activation='softmax')(class1)

		# define new model
		model = Model(inputs=model.inputs, outputs=output)
		return model
