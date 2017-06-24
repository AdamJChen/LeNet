import numpy as np
import tensorflow as tf
from keras.datasets import cifar10
from networks import Conv_net
from networks import Feed_forward
from networks import Utes



# get CIFAR10 data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# format data
def std_img(data): return (data - np.mean(data)/255.0)
X_train = std_img(X_train)
X_test = std_img(X_test)
y_train = y_train[:,0] #remove a dimension for one hot encoding
y_test = y_test[:,0]


# network parameters
image_width = X_train.shape[1]
image_height = X_train.shape[2]
image_depth = X_train.shape[3]
shapes = {'c1' : [5, 5, 3, 20], 'c2' : [5, 5, 20, 50]}
strides = {'c' : [1, 1 ,1 ,1], 'p' : [1, 2, 2, 1]}
pool_win_size = [1, 2, 2, 1]
n_cells = 500
n_classes = 10

# parameters
learning_rate = 0.001
batch_size = 100
test_size = X_test.shape[0]
n_epochs = 20


# LeNet Model
def leNet(X, n_examples):
	"""Defines the LeNet architecture for image recognition. 
	   Returns logits for prediction or training"""
	with tf.variable_scope('layer1') as scope:
		conv1 = Conv_net.add_conv(X = X, shape = shapes['c1'], strides = strides['c'])
		pool1 = Conv_net.add_max_pool(X = conv1, window_size = pool_win_size, strides = strides['p'])

	with tf.variable_scope('layer2') as scope:
		conv2 = Conv_net.add_conv(X = pool1, shape = shapes['c2'], strides = strides['c'])
		pool2 = Conv_net.add_max_pool(X = conv2, window_size = pool_win_size , strides = strides['p'])

	# flattens volume into plane for feed forward layers
		flat = tf.reshape(pool2, (n_examples, -1)) 

	with tf.variable_scope('layer3') as scope:
		fc = Feed_forward.add_fc( X = flat, n_in = 3200, n_out = n_cells, act_func = True) # n_in = 8*8*50

	with tf.variable_scope('layer4') as scope:
		return Feed_forward.add_fc( X = fc, n_in = n_cells, n_out = n_classes, act_func = False)


# graph construction
graph = tf.Graph()

with graph.as_default():
	# data operations
	X = tf.placeholder(dtype = tf.float32, shape = (None,image_width, image_height, image_depth))
	y = tf.placeholder(dtype = tf.int32, shape=(None,))
	n_examples = tf.placeholder(tf.int32, shape = ()) # Distinguises between test or train data at fc layer

	# model
	model = leNet(X, n_examples)
	
	# training
	element_wise_cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = model, labels = y)
	cost = tf.reduce_mean(element_wise_cost)
	optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
	train = optimizer.minimize(cost)

	# predictions and accuracy
	predictions = Utes.predictions(logits = model)
	accuracy = Utes.accuracy(labels = y, predictions = predictions, n_classes = n_classes)


# graph  execution
with tf.Session(graph = graph) as sess:
	
	# initialize all variables
	sess.run(tf.global_variables_initializer())

	for epoch in range(n_epochs):
		for batch in range(X_train.shape[0]/ batch_size):
			# define offset and end of batch
			offset = (batch * batch_size)
			end_batch = offset + batch_size
			
			# get batch for SGD
			batch_data = X_train[offset: end_batch,:, :, :]
			batch_labels = y_train[offset: end_batch]
			
			# feed batch and train
			train_feed_dict = {X: batch_data, y: batch_labels, n_examples : batch_size}
			_, cost_value, train_accuracy = sess.run([train, cost, accuracy], feed_dict =train_feed_dict)

			if batch % 100 == 0:
				print 'batch : %d/400 cost: %.2f accuracy: %.2f%%' % (batch, cost_value, train_accuracy * 100)

		# feed test set and calculate test accuracy
		test_feed_dict = {X: X_test, y: y_test, n_examples : test_size}
		test_accuracy = sess.run(accuracy, feed_dict =test_feed_dict)
		print 'test accuracy: %.2f%%' % (test_accuracy * 100)
