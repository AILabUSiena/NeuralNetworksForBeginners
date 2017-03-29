'''
@author: Dario Zanca
@summary: 2-layers neural network to solve MNIST
'''

import tensorflow as tf	
import numpy as np

########################################################
''' data preparation '''

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x_dim = 784
y_dim = 10

x_ = tf.placeholder(tf.float32, [None, x_dim])
y_ = tf.placeholder(tf.float32, [None, y_dim])

########################################################
''' model definition '''

h_layer_dim = 10
epochs = 100

W1 = tf.Variable(tf.random_uniform([x_dim, h_layer_dim],-0.1, 0.1))
b1 = tf.Variable(tf.random_uniform([h_layer_dim],-0.1, 0.1))

h1 = tf.nn.relu(tf.matmul(x_,W1)+b1) # hidden layer

W2 = tf.Variable(tf.random_uniform([h_layer_dim, y_dim],-0.1, 0.1))
b2 = tf.Variable(tf.random_uniform([y_dim],-0.1, 0.1))

y = tf.nn.relu(tf.matmul(h1,W2)+b2) # prediction

MSE = tf.losses.mean_squared_error(y_, y) # Mean squared error

train_step = tf.train.AdamOptimizer(0.01).minimize(MSE)

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

########################################################
''' model running and evaluation'''

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(epochs): 
	sess.run(train_step, feed_dict={x_: mnist.train.images, y_: mnist.train.labels})
	print "\nEpoch: ", i+1 ,"/", epochs, " -- MSE =", sess.run(MSE, feed_dict={x_: mnist.train.images, y_: mnist.train.labels})
	
# final result
print "\nAccuracy on test: ", sess.run(accuracy, feed_dict={x_: mnist.test.images, y_: mnist.test.labels})
