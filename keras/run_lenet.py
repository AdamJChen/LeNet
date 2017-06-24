from lenet import LeNet
from keras.datasets import cifar10
from keras.optimizers import Adam
from keras.utils import np_utils
import numpy as np


# network parameters
n_classes = 10
image_width = 32
image_height = 32
image_depth = 3

# parameters
batch_size = 100
n_epochs = 20
learning_rate = 0.01


# get CIFAR10_data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# standardize data
def std_img(data): return (data - np.mean(data))/255.0
X_train = std_img(X_train)
X_test = std_img(X_test)

# convert labels to one-hot vectors
y_train = np_utils.to_categorical(y_train, n_classes)
y_test = np_utils.to_categorical(y_test, n_classes)


# initialize the optimizer and model
optimizer= Adam(lr= learning_rate)
model = LeNet.build(image_width, image_height, image_depth, n_classes)
model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

# train model
model.fit(X_train, y_train, batch_size = batch_size, epochs = n_epochs, verbose = 1)

# test model
(loss, accuracy) = model.evaluate(X_test, y_test, batch_size = batch_size, verbose=1)
print 'accuracy: %.2f%%' % (accuracy * 100)
