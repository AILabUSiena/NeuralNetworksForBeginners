import tensorflow as tf	
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

#____________tools_______________________

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

# strides: 1, padding: 0
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')#strides shape???

# max pooling over 2x2 blocks
def max_pool_3x3(x):
  return tf.nn.max_pool(x, ksize=[1, 3, 3, 1], # ksize is the size of the sliding window (for each dimension)
                        strides=[1, 3, 3, 1], padding='SAME')#strides shapes indicate how the sliding window moves

#_____________ model_____________________

#convolution layer1 - compute 32 features for each 5x5 patch
INPUT_C1 = 1
OUTPUT_C1 = 12
W_conv1 = weight_variable([5, 5, INPUT_C1, OUTPUT_C1]) #shape:[patch_size_x, patch_size_y, input_channels, output_channels]
b_conv1 = bias_variable([OUTPUT_C1])

# reshape the tensor x (dataset of images (None x 728)) to a 4D tensor
#2nd and 3rd arguments are the image width and height
#4th argument corresponds to the number of color channels
#1st arguments: the dimensione has to be computed,
# in our case is the size of the batch
x_image = tf.reshape(x, [-1,28,28,1]) 

#convolution step
h_conv1 = tf.nn.relu( conv2d(x_image, W_conv1) + b_conv1 )

#max pooling step
h_pool1 = max_pool_3x3(h_conv1)

#convolution layer2 - 
INPUT_C2 = OUTPUT_C1
OUTPUT_C2 = 16
W_conv2 = weight_variable([5, 5, INPUT_C2, OUTPUT_C2])
b_conv2 = bias_variable([OUTPUT_C2])

#convolution step
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

#max pooling step
h_pool2 = max_pool_3x3(h_conv2)

'''
We have a 7x7 image. Reshape it and fed in a fully connected network.
'''
#create the layer and baias
FS = 4 # final size, it is possible to compute it! (I left you as exercize
W_fc1 = weight_variable([ FS * FS * OUTPUT_C2, 1024])
b_fc1 = bias_variable([1024])

#reshape images
h_pool2_flat = tf.reshape(h_pool2, [-1, FS * FS * OUTPUT_C2])

#forward step
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

#output layer
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

#forward step
y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2

''' 
Here follows the code to train and evaluate the model
'''
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#sess = tf.Session()
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

# Early stopping setup, to check on validation set
prec_err = 10**6 # just a very big vaLue
val_count = 0
val_max_steps = 6

# Training specs
epochs = 100
BATCH_SIZE = 1000
num_of_batches = 60000/BATCH_SIZE

i=1
while i <= epochs and val_count < val_max_steps:

    print 'Epoch:', i, '(Early stopping criterion: ', val_count, '/', val_max_steps, ')'

    for j in range(num_of_batches):
        # training step
        batch = mnist.train.next_batch(BATCH_SIZE)
        sess.run(train_step, feed_dict={x: batch[0], y_: batch[1]})

    # visualize accuracy each 10 epochs
    if i == 1 or i%10 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: mnist.train.images, y_: mnist.train.labels})    
        test_accuracy = accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels})
        print("\nAccuracy at epoch %d: train accuracy %g, test accuracy %g\n"%(i, train_accuracy, test_accuracy))

    # validation check
    curr_err = sess.run(cross_entropy, feed_dict={x: mnist.validation.images, y_: mnist.validation.labels})
    if curr_err >= prec_err*0.9999:
        val_count = val_count + 1
    else:
        val_count = 0
    prec_err = curr_err

    i+=1


print("\n\nResult:\nTest accuracy %g" % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))


"Visualize filters"

FILTERS = W_conv1.eval()

fig = plt.figure()

for i in range(np.shape(FILTERS)[3]):
    ax = fig.add_subplot(2, 6, i+1)
    ax.matshow(FILTERS[:,:,0,i], cmap='gray')
plt.show()
