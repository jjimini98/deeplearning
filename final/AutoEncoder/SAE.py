import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from functools import partial

tf.reset_default_graph()

(train_x, train_y), (test_x, test_y) = tf.keras.datasets.mnist.load_data()
train_x = train_x.astype(np.float32).reshape(-1, 28*28) / 255.0 # 0.0~1.0 scaling
test_x = test_x.astype(np.float32).reshape(-1, 28*28) / 255.0
train_y = train_y.astype(np.int32)
test_y = test_y.astype(np.int32)
valid_x, train_x = train_x[:5000], train_x[5000:]
valid_y, train_y = train_y[:5000], train_y[5000:]

# Mini-batch
def shuffle_batch(features, labels, batch_size):
    rnd_idx = np.random.permutation(len(features))
    n_batches = len(features) // batch_size
    for batch_idx in np.array_split(rnd_idx, n_batches):
        batch_x, batch_y = features[batch_idx], labels[batch_idx]
        yield batch_x, batch_y

n_inputs = 28 * 28
n_hidden1 = 300 # encoder
n_hidden2 = 150 # coding units
n_hidden3 = n_hidden1 # decoder
n_outputs = n_inputs # reconstruction
learning_rate = 0.01
l2_reg = 0.0001
n_epochs = 5
batch_size = 150
n_batches = len(train_x) // batch_size

he_init = tf.keras.initializers.he_normal()
l2_regularizer = tf.contrib.layers.l2_regularizer(scale=l2_reg)
dense_layer = partial(tf.layers.dense, activation=tf.nn.elu, kernel_initializer=he_init, kernel_regularizer=l2_regularizer) # regularizer : 과적합 방지

# stacked autoencoder
inputs = tf.placeholder(tf.float32, shape=[None, n_inputs])
hidden1 = dense_layer(inputs, n_hidden1)
hidden2 = dense_layer(hidden1, n_hidden2)
hidden3 = dense_layer(hidden2, n_hidden3)
outputs = dense_layer(hidden3, n_outputs, activation=None)
reconstruction_loss = tf.reduce_mean(tf.square(outputs - inputs))
reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
loss = tf.add_n([reconstruction_loss] + reg_losses) # to avoid overfitting
train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    n_batches = len(train_x) // batch_size
    for epoch in range(n_epochs):
        for iteration in range(n_batches):
            batch_x, batch_y = next(shuffle_batch(train_x, train_y, batch_size))
            _, input_x, output_x = sess.run([train_op, inputs, outputs], feed_dict={inputs: batch_x})
        loss_train = reconstruction_loss.eval(feed_dict={inputs: batch_x})
        print('epoch : {}, Train MSE : {:.5f}'.format(epoch, loss_train))


plt.figure(figsize=(5, 5))
image = np.reshape(input_x[1], [28, 28])
plt.imshow(image, cmap='Greys')
plt.show()
plt.figure(figsize=(5, 5))
image = np.reshape(output_x[1], [28, 28])
plt.imshow(image, cmap='Greys')
plt.show()