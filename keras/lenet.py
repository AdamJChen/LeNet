from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Flatten
from keras.layers.core import Dense

class LeNet:
	"""Builds the LeNet model using keras"""

	@staticmethod
	def build(width, height, depth, classes):
		"""constructs the graph for the LeNet model"""
		
		model = Sequential()

		# first convolutional and max pool layer
		model.add(Conv2D(filters  =20, kernel_size = 5, padding ="same",
						 activation = 'relu', data_format = "channels_last",
						 input_shape = (width, height, depth))) # needed if conv2D is first layer
		model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))

		# second convolutional and max pool layer
		model.add(Conv2D(filters = 50, kernel_size = 5, padding = "same",
						 activation = 'relu'))
		model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))

		# first feed forward layer
		model.add(Flatten())
		model.add(Dense(units = 500, activation = 'relu'))

		# softmax classifier
		model.add(Dense(units =classes, activation = 'relu'))

		return model