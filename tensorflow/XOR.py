'''

@author: Dario Zanca
@date: 03-Nov-2016
@summary: XOR with tensorflow

'''
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

"Dataset definition"

INPUT = np.array([[0,0],[0,1],[1,0],[1,1]])
TARGET = np.array([[0],[1],[1],[0]])

HU = 3 # number of hidden units
Epochs = 10001

"Define symbolic variables"

x_ = tf.placeholder(tf.float32, shape=[None,2]) # for the input
y_ = tf.placeholder(tf.float32, shape=[None,1]) # for the target

"Definition of the Model"

# First layer
W1 = tf.Variable(tf.random_uniform([2,HU], -1.0, 1.0))
b1 = tf.Variable(tf.zeros([HU]))

O = tf.sigmoid(tf.matmul(x_, W1) + b1)

# Second layer
W2 = tf.Variable(tf.random_uniform([HU,1], -1.0, 1.0))
b2 = tf.Variable(tf.zeros([1]))

y = tf.sigmoid(tf.matmul(O, W2) + b2)

"Definition of the cost function and optimizer"

cost = tf.reduce_mean(tf.square(y_ - y))

train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cost)

"Start Session"

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

"Training"
for i in range(Epochs):
        sess.run(train_step, feed_dict={x_: INPUT, y_: TARGET})
        if i % 1000 == 0:
            print('Epoch:', i, ' -- Cost:', sess.run(cost, feed_dict={x_: INPUT, y_: TARGET}))
            #print('Output ', sess.run(y, feed_dict={x_: INPUT, y_: TARGET}))
            #print('W1 ', sess.run(W1))
            #print('b1 ', sess.run(b1))
            #print('W2 ', sess.run(W2))
            #print('b2 ', sess.run(b2))
            
        

"Test the trained model"

correct_prediction = abs(y_ - y) < 0.5 
cast = tf.cast(correct_prediction, "float")
accuracy = tf.reduce_mean(cast)


yy, cc, aa = sess.run([y, cast, accuracy],feed_dict={x_: INPUT, y_: TARGET})
print "\n\n\n Final Accuracy: ", aa

"Draw separation surfaces"
plt.figure()
# Plotting dataset
c1 = plt.scatter([1,0], [0,1], marker='s', color='gray', s=100)
c0 = plt.scatter([1,0], [1,0], marker='^', color='gray', s=100)
# Generating points in [-1,2]x[-1,2]
DATA_x = (np.random.rand(10**6,2)*3)-1
DATA_y = sess.run(y,feed_dict={x_: DATA_x})
# Selecting borderline predictions
ind = np.where(np.logical_and(0.49 < DATA_y, DATA_y< 0.51))[0]
DATA_ind = DATA_x[ind]
# Plotting separation surfaces
ss = plt.scatter(DATA_ind[:,0], DATA_ind[:,1], marker='_', color='black', s=5)
# Some figure's settings
plt.legend((c1, c0, ss), ('Class 1', 'Class 0', 'Separation surfaces'), scatterpoints=1)
plt.xlabel('Input x1')
plt.ylabel('Input x2')
plt.axis([-1,2,-1,2])
plt.show()

