'''
Created on 13 ago 2016
@author: vincenzo
'''

import tensorflow as tf	
import numpy as np
import matplotlib.pyplot as plt

##########################################################
''' tools definition'''

def init_weights(shape):
	return tf.Variable(tf.random_uniform(shape, -0.1, 0.1))

def mlp_output(X, W_h, W_o, b_h, b_o):
	ak = tf.matmul(X, W_h) + b_h
	O = tf.nn.relu(ak) #output layer 1
	
	a2 = tf.matmul(O, W_o) + b_o
	o2 = tf.nn.softmax(a2)  #output layer2
	return o2

#########################################################
''' data preparation '''

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x_dim = 784
y_dim = 10

x = tf.placeholder(tf.float32, [None, x_dim])

#target output
y_ = tf.placeholder(tf.float32, [None, y_dim])

########################################################
''' model creation '''

h_layer_dim = 10
epochs = 1000
LEARNING_RATE = 10**-4

W1 = init_weights([x_dim, h_layer_dim])
b1 = init_weights([h_layer_dim])

W2 = init_weights([h_layer_dim, y_dim])
b2 = init_weights([y_dim])

#predicted output
y = mlp_output(x, W1, W2, b1, b2)

# Loss
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))

# regularization term
regularization = tf.reduce_sum(tf.square(W1), [0, 1]) + tf.reduce_sum(tf.square(W2), [0, 1])

train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy + 10**-4 * regularization)

########################################################
''' model running and evaluation'''

sess = tf.Session()
sess.run(tf.global_variables_initializer())

errors_train=[]
errors_test=[]
errors_val=[]

# Early stopping setup, to check on validation set
prec_err = 10**6 # just a very big vaLue
val_count = 0
val_max_steps = 6

BATCH_SIZE = np.shape(mnist.train.images)[0]
MINI_BATCH_SIZE = 1000

i = 1
while i <= epochs and val_count < val_max_steps:

	for j in range(BATCH_SIZE/MINI_BATCH_SIZE): 
		batch_xs, batch_ys = mnist.train.next_batch(MINI_BATCH_SIZE)
		sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

	curr_err = sess.run(cross_entropy, feed_dict={x: mnist.validation.images, y_: mnist.validation.labels})
	if curr_err >= prec_err*0.9999:
		val_count = val_count + 1
	else:
		val_count = 0
	prec_err = curr_err
	
	if i % 1 == 0:
		errors_val.append(curr_err)
		c_test = sess.run(cross_entropy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
		errors_test.append(c_test)
		c_train = sess.run(cross_entropy, feed_dict={x: mnist.train.images, y_: mnist.train.labels})
		errors_train.append(c_train)
		print "\n\nEPOCH: ",i, "/", epochs,"\n  TRAIN ERR: ", c_train, "\n  VALIDATION ERR: ", curr_err, "\n  TEST ERR: ", c_test,
		print "\n(Early stopping criterion: ", val_count, "/", val_max_steps, ")"
	i = i+1

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

aa = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
print "Accuracy: ", aa


"Plot errors"
E = range(np.shape(errors_train)[0])
E = np.asanyarray(E)*1
line_train, = plt.plot(E, errors_train)
line_test, = plt.plot(E, errors_test)
line_val, = plt.plot(E, errors_val)
plt.legend([line_train, line_val, line_test], ['Training', 'Validation', 'Test'])
plt.ylabel('Cross-Entropy')
plt.xlabel('Epochs')
plt.show()

