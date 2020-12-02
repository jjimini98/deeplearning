import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from functools import partial

tf.reset_default_graph()

(train_x, train_y), (test_x, test_y) = tf.keras.datasets.mnist.load_data()
train_x = train_x.astype(np.float32).reshape(-1, 28*28) / 255.0
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
n_hidden1 = 500
n_hidden2 = 500
n_hidden3 = 20 # coding unit
n_hidden4 = n_hidden2
n_hidden5 = n_hidden1
n_outputs = n_inputs
learning_rate = 0.001
n_digits = 60
n_epochs = 50
batch_size = 150

learning_rate = 0.001
n_digits = 60
n_epochs = 50
batch_size = 150
initializer = tf.variance_scaling_initializer()
dense_layer = partial(tf.layers.dense,activation=tf.nn.elu,kernel_initializer=initializer)

inputs = tf.placeholder(tf.float32, [None, n_inputs])
hidden1 = dense_layer(inputs, n_hidden1)
hidden2 = dense_layer(hidden1, n_hidden2)
hidden3_mean  = dense_layer(hidden2, n_hidden3, activation= None)
hidden3_sigma = dense_layer(hidden2, n_hidden3, activation = None)
noise = tf.random_normal(tf.shape(hidden3_sigma), dtype = tf.float32)
hidden3 = hidden3_mean + hidden3_sigma * noise
hidden4 = dense_layer(hidden3, n_hidden4)
hidden5 = dense_layer(hidden4, n_hidden5)
logits = dense_layer(hidden5, n_outputs, activation = None)
outputs = tf.sigmoid(logits)

eps = 1e-10
xentropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=inputs, logits=logits)
reconstruction_loss = tf.reduce_mean(xentropy)
latent_loss = 0.5 * tf.reduce_sum(tf.square(hidden3_sigma) + tf.square(hidden3_mean) - 1 - tf.log(eps + tf.square(hidden3_sigma)))
loss = reconstruction_loss + latent_loss
train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    n_batches = len(train_x) // batch_size
    for epoch in range(n_epochs):
        for iteration in range(n_batches):
            print("\r{}%".format(100 * iteration // n_batches), end="")
            batch_x, batch_y = next(shuffle_batch(train_x, train_y, batch_size))
            sess.run(train_op, feed_dict={inputs: batch_x})
    recon_loss_val, latent_loss_val, loss_val = sess.run([reconstruction_loss, latent_loss, loss], feed_dict={inputs: batch_x})
    print('\repoch : {}, Train MSE : {:.5f},'.format(epoch, recon_loss_val),'latent_loss : {:.5f}, total_loss : {:.5f}'.format(latent_loss_val, loss_val))


    codings_rnd = np.random.normal(size=[n_digits, n_hidden3])
    outputs_val = outputs.eval(feed_dict={hidden3: codings_rnd})

def plot_image(image, shape=[28, 28]):
    plt.imshow(image.reshape(shape), cmap="Greys", interpolation="nearest")
    plt.axis("off")


plt.figure(figsize=(8,50))

for iteration in range(n_digits):
    plt.subplot(n_digits, 10, iteration + 1)
    plot_image(outputs_val[iteration])