import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

tf.reset_default_graph()
tf.set_random_seed(777)

def MinMaxScaler(data):
    numerator = data - np.min(data,0)
    denominator = np.max(data,0) - np.min(data,0)
    return numerator / (denominator + 1e-7)

seq_length = 7 #일주일치 예측하는거라서
data_dim = 5
hidden_dim = 10
output_dim = 1
learning_rate = 0.01
iterations = 50

xy = np.loadtxt('data-02-stock_daily.csv',delimiter= ',')
xy = xy[::-1]
xy = MinMaxScaler(xy)
x = xy
y = xy[:,[-1]] #close 칼럼에 해당 + 값이 scaling 됨
dataX = []
dataY = []

for i in range(0,len(y)-seq_length):
    _x = x[i:i+seq_length]
    _y = y[i+seq_length]
    # print(_x,"->",_y)
    dataX.append(_x)
    dataY.append(_y)


train_size = int(len(dataY)*0.7)
test_size = len(dataY) - train_size


trainX , testX = np.array(dataX[0:train_size]), np.array(dataX[train_size:len(dataX)])
trainY , testY = np.array(dataY[0:train_size]) , np.array(dataY[train_size:len(dataY)])
print(trainY[0])

X = tf.placeholder(tf.float32, [None, seq_length, data_dim])
Y = tf.placeholder(tf.float32, [None, 1])

cell = tf.contrib.rnn.BasicRNNCell(num_units= hidden_dim, activation = tf.tanh)
outputs,states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)

Y_pred = tf.contrib.layers.fully_connected(outputs[:,-1],output_dim,activation_fn=None)

loss = tf.reduce_sum(tf.square(Y_pred-Y))

optimizer = tf.train.AdamOptimizer(learning_rate)
train = optimizer.minimize(loss)

targets = tf.placeholder(tf.float32, [None,1])
prediction = tf.placeholder(tf.float32,[None,1])
rmse = tf.sqrt(tf.reduce_mean(tf.square(targets- prediction)))


with tf.Session() as sess :
    sess.run(tf.global_variables_initializer())

    for i in range(iterations):
        _, step_loss = sess.run([train,loss], feed_dict={X:trainX, Y:trainY})
        print("[step : {} ] loss : {}".format(i,step_loss))

        test_predict = sess.run(Y_pred, feed_dict={X:testX})
        rmse_val = sess.run(rmse, feed_dict={targets : testY, prediction: test_predict})
        print("RMSE: {}".format(rmse_val))

        plt.plot(testY)
        plt.plot(test_predict)
        plt.xlabel("Time Period")
        plt.ylabel("Stock Price")

        plt.show()

