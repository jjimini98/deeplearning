import tensorflow.contrib.layers as lays
import numpy as np
from skimage import transform # conda install scikit-image
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
def autoencoder(inputs):
# encoder
# 32 x 32 x 1 -> 16 x 16 x 32
# 16 x 16 x 32 -> 8 x 8 x 16
# 8 x 8 x 16 -> 2 x 2 x 8
    net = lays.conv2d(inputs, 32, [5, 5], stride=2, padding='SAME')
    net = lays.conv2d(net, 16, [5, 5], stride=2, padding='SAME')
    net = lays.conv2d(net, 8, [5, 5], stride=4, padding='SAME')
# decoder
# 2 x 2 x 8 -> 8 x 8 x 16
# 8 x 8 x 16 -> 16 x 16 x 32
# 16 x 16 x 32 -> 32 x 32 x 1
    net = lays.conv2d_transpose(net, 16, [5, 5], stride=4, padding='SAME')
    net = lays.conv2d_transpose(net, 32, [5, 5], stride=2, padding='SAME')
    net = lays.conv2d_transpose(net, 1, [5, 5], stride=2, padding='SAME', activation_fn=tf.nn.tanh)
    return net


def resize_batch(imgs):
# 28 X 28 -> 32 X 32
    imgs = imgs.reshape((-1, 28, 28, 1))
    resized_imgs = np.zeros((imgs.shape[0], 32, 32, 1))
    for i in range(imgs.shape[0]):
        resized_imgs[i, ..., 0] = transform.resize(imgs[i, ..., 0], (32, 32))
    return resized_imgs


# Introduce Gaussian Noise
def noisy(image):
    row, col= image.shape
    mean = 0
    var = 0.1
    sigma = var**0.5
    gauss = np.random.normal(mean,sigma,(row, col))
    gauss = gauss.reshape(row, col)
    noisy = image + gauss
    return noisy
ae_inputs = tf.placeholder(tf.float32, (None, 32, 32, 1)) # input to the network (MNIST images)
ae_inputs_noise = tf.placeholder(tf.float32, (None, 32, 32, 1))
ae_outputs = autoencoder(ae_inputs_noise) # create the Autoencoder network
# calculate the loss and optimize the network
loss = tf.reduce_mean(tf.square(ae_outputs - ae_inputs)) # claculate the mean square error loss
train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
# initialize the network
init = tf.global_variables_initializer()
batch_size = 500 # Number of samples in each batch
epoch_num = 5 # Number of epochs to train the network
lr = 0.001 # Learning rate
# read MNIST dataset
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
# calculate the number of batches per epoch
batch_per_ep = mnist.train.num_examples // batch_size

with tf.Session() as sess:
    sess.run(init)
    for ep in range(epoch_num): # epochs loop
        for batch_n in range(batch_per_ep): # batches loop
            batch_img, batch_label = mnist.train.next_batch(batch_size) # read a batch
            batch_img = batch_img.reshape((-1, 28, 28, 1)) # reshape each sample to an (28, 28) image
            batch_img = resize_batch(batch_img) # reshape the images to (32, 32)
            image_arr = []
            for i in range(len(batch_img)):
                img = batch_img[i,:,:,0]
                img = noisy(img)
                image_arr.append(img)
            image_arr = np.array(image_arr)
            image_arr = image_arr.reshape(-1, 32, 32, 1)
            batch_img = image_arr
            _, c = sess.run([train_op, loss], feed_dict={ae_inputs: batch_img, ae_inputs_noise: image_arr})
            print('[Epoch: {}, Batch: {}] cost = {:.5f}'.format((ep + 1), (batch_n+1), c))

    # test the trained network
    batch_img, batch_label = mnist.test.next_batch(50)
    batch_img = resize_batch(batch_img)
    image_arr = []
    for i in range(50):
        img = batch_img[i, :, :, 0]
        img = noisy(img)
        image_arr.append(img)
    image_arr = np.array(image_arr)
    image_arr = image_arr.reshape(-1, 32, 32, 1)
    batch_img = image_arr

    recon_img = sess.run([ae_outputs], feed_dict={ae_inputs_noise: batch_img})[0]


    # plot the reconstructed images and their ground truths (inputs)
    plt.figure(1)
    plt.title('Input Images with Gaussian Noise')
    for i in range(50):
        plt.subplot(5, 10, i+1)
        plt.imshow(batch_img[i, ..., 0], cmap='gray')
    plt.figure(2)
    plt.title('Reconstructed Images')
    for i in range(50):
        plt.subplot(5, 10, i+1)
        plt.imshow(recon_img[i, ..., 0], cmap='gray')
    plt.show()