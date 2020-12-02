import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

tf.reset_default_graph()
tf.set_random_seed(777)

def MinMaxScaler(data):
    numerator = data- np.min(data,0)
    denominator = np.max(data,0)- np.min(data,0)
    return numerator / (denominator + 1e-7 )

seq_length = 7
data_dim = 5
hidden_dim = 5
# 이틀을 예측해야하므로 output_dim은 2이다
output_dim = 2
# learning rate도 조절해봄. 0.01 -> 0.03
learning_rate = 0.03
iteration = 100

xy = np.loadtxt('./data-02-stock_daily.csv',delimiter=',')  #(732,5)

xy = xy[::-1]
xy = MinMaxScaler(xy)
x = xy
y = xy[:,[-1]]

dataX = []
dataY = []

for i in range(0,len(y)-seq_length):
    _x = x[i:i+seq_length]
    _y = y[i+seq_length-2:i+seq_length]
    print(_x, "->", _y)
    dataX.append(_x)
    dataY.append(_y)

train_size = int(len(dataY)*0.7)
test_size = len(dataY) - train_size

trainX, testX = np.array(dataX[0:train_size]), np.array(dataX[train_size:len(dataX)])
trainY , testY = np.array(dataY[0:train_size]) , np.array(dataY[train_size:len(dataY)])

# trainY 와 testY의 차원이 맞지 않아 에러가 발생
# trainY의 차원을 변경한다 (507,2,1) -> (507,2)
# testY의 차원을 변경한다 (218,2,1) -> (507,2)
trainY = trainY.reshape(507,2)
testY = testY.reshape(218,2)

X = tf.placeholder(tf.float32, [None,seq_length,data_dim])
# output_dim에 맞게 Y의 크기 수정
Y = tf.placeholder(tf.float32, [None,2])
cell = tf.contrib.rnn.BasicRNNCell(num_units = hidden_dim, activation = tf.tanh)

outputs , states = tf.nn.dynamic_rnn(cell, X, dtype = tf.float32)

Y_pred = tf.contrib.layers.fully_connected(outputs[:, -1], output_dim, activation_fn = None)
loss = tf.reduce_sum(tf.square(Y_pred-Y))

optimizer = tf.train.AdamOptimizer(learning_rate)
train = optimizer.minimize(loss)

# output_dim에 맞게 target과 prediction 수정
targets = tf.placeholder(tf.float32, [None,2])
prediction = tf.placeholder(tf.float32,[None,2])

rmse = tf.sqrt(tf.reduce_mean(tf.square(targets-prediction)))


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(iteration):
        _,step_loss = sess.run([train,loss],feed_dict={X:trainX, Y:trainY})
        print("[step : {} ] loss : {}".format(i,step_loss))

        test_predict = sess.run(Y_pred, feed_dict= {X:testX})
        rmse_val = sess.run(rmse, feed_dict={targets:testY, prediction : test_predict})
        print("RMSE:{}".format(rmse_val))


        plt.plot(testY)
        plt.plot(test_predict)
        plt.xlabel("Time label")
        plt.ylabel("stock price")
        plt.show()
