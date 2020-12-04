import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

reset_graph()

t_min , t_max = 0,30
resolution = 0.1
n_steps = 20
t_instance = np.linspace(12.2 , 12.2+ resolution * (n_steps+1), n_steps+1)

def time_series(t):
    return t*np.sin(t) / 3+2*np.sin(t*5)

def next_batch(batch_size, n_steps):
    t0 = np.random.rand(batch_size,1) * (t_max - t_min - n_steps * resolution)
    Ts = t0 + np.arange(0., n_steps+1) * resolution
    ys = time_series(Ts)
    return ys[: , :-1].reshape(-1,n_steps,1), ys[:, 1:].reshape(-1,n_steps,1)

n_steps = 20
n_neurons = 100
n_inputs = 1
n_outputs = 1
n_layers = 3

X= tf.placeholder(tf.float32, [None,n_steps,n_inputs])
Y= tf.placeholder(tf.float32,[None,n_steps,n_outputs])

# without dropout rate
layers = [tf.nn.rnn_cell.BasicRNNCell(num_units=n_neurons) for layer in range(n_layers)]
multi_layer_cell = tf.nn.rnn_cell.MultiRNNCell(layers)

# with dropout rate
keep_prob = tf.placeholder_with_default(1.0, shape=())
layers_drop = [tf.nn.rnn_cell.DropoutWrapper(layer, input_keep_prob=keep_prob, state_keep_prob=keep_prob) for layer in layers]
multi_layer_cell = tf.nn.rnn_cell.MultiRNNCell(layers_drop)
rnn_outputs, states = tf.nn.dynamic_rnn(multi_layer_cell, X, dtype=tf.float32)

# for one output
stacked_rnn_outputs = tf.reshape(rnn_outputs, [-1, n_neurons])
stacked_outputs = tf.layers.dense(stacked_rnn_outputs, n_outputs)
predictions = tf.reshape(stacked_outputs, [-1, n_steps, n_outputs])

# loss
mse = tf.losses.mean_squared_error(labels=Y, predictions=predictions)

stacked_rnn_outputs = tf.reshape(rnn_outputs,[-1,n_neurons])

learning_rate = 0.01
train_keep_prob = 0.3 # 1.0
n_iterations = 1500
batch_size = 50
# optimizer
train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(mse)
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for iteration in range(n_iterations):
        batch_x, batch_y = next_batch(batch_size, n_steps)
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
        #sess.run(train_op, feed_dict={X: batch_x, y: batch_y, keep_prob: train_keep_prob})
        if iteration % 300 == 0:
            loss = mse.eval(feed_dict={X: batch_x, Y: batch_y})
            print('step: {:03d}, MSE: {:.4f}'.format(iteration, loss))

    X_new = time_series(np.array(t_instance[:-1].reshape(-1, n_steps, n_inputs)))
    y_pred = sess.run(predictions, feed_dict={X: X_new})

plt.title("Testing the Model", fontsize=14)
plt.plot(t_instance[:-1], time_series(t_instance[:-1]), "bo", markersize=10, label="instance")
plt.plot(t_instance[1:], time_series(t_instance[1:]), "w*", markersize=10, label="target", color='yellow')
plt.plot(t_instance[1:], y_pred[0,:,0], "r.", markersize=10, label="prediction")
plt.legend(loc="upper left")
plt.xlabel("Time")
plt.show()



