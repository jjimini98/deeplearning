
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

tf.reset_default_graph()
tf.set_random_seed(777)

def MinMaxScaler(data):
    numerator = data-np.min(data,0)
    denominator = np.max(data,0) - np.min(data,0)
    return numerator / (denominator + 1e-7)

seq_length = 7
data_dim = 5
hidden_dim = 5
output_dim = 1
learning_rate = 0.01
iterations = 1000 # 1000번 학습하니까 거의 비슷해진따~!! 근데 만 번 학습하면 더 이상하다?

xy = np.loadtxt('data-02-stock_daily.csv',delimiter= ',')
xy = xy[::-1]
xy = MinMaxScaler(xy)
x = xy
y = xy[:,[-1]]

dataX = []
dataY = []

for i in range(0,len(y)-seq_length):
    _x = x[i:i+seq_length]
    _y = y[i+seq_length]
    print(_x,"->",_y)
    dataX.append(_x)
    dataY.append(_y)

train_size = int(len(dataY)*0.7)
test_size = len(dataY) - train_size

trainX , testX = np.array(dataX[0:train_size]),np.array(dataX[train_size:len(dataX)])
trainY , testY = np.array(dataY[0:train_size]), np.array(dataY[train_size:len(dataY)])

X = tf.placeholder (tf.float32, [None, seq_length, data_dim])
Y = tf.placeholder(tf.float32, [None, 1])
# build a LSTM network
cell = tf.contrib.rnn.BasicRNNCell(num_units=hidden_dim, activation=tf.tanh)
#cell = tf.contrib.rnn.GRUCell(num_units=hidden_dim, activation=tf.tanh)
#cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_dim, activation=tf.tanh)
outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
Y_pred = tf.contrib.layers.fully_connected(outputs[:, -1], output_dim, activation_fn=None)
loss = tf.reduce_sum(tf.square(Y_pred - Y)) # sum of the squares
# optimizer
optimizer = tf.train.AdamOptimizer(learning_rate)
train = optimizer.minimize(loss)
# RMSE
targets = tf.placeholder(tf.float32, [None, 1])
predictions = tf.placeholder(tf.float32, [None, 1])
rmse = tf.sqrt(tf.reduce_mean(tf.square(targets - predictions)))
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
# Training step
    for i in range(iterations):
        _, step_loss = sess.run([train, loss], feed_dict={X: trainX, Y: trainY})
        print("[step: {}] loss: {}".format(i, step_loss))
# Test step
    test_predict = sess.run(Y_pred, feed_dict={X: testX})
    rmse_val = sess.run(rmse, feed_dict={targets: testY, predictions: test_predict})
    print("RMSE: {}".format(rmse_val))
# Plot predictions
    plt.plot(testY)
    plt.plot(test_predict)
    plt.xlabel("Time Period")
    plt.ylabel("Stock Price")
    plt.show()