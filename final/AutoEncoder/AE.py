import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

np.random.seed(4)
m = 200
w1, w2 = 0.1, 0.3
noise = 0.1

# random 하게 m개의 데이터를 뽑는다.  숫자의 의미는 모르겠삼
angles = np.random.rand(m) * 3 * np.pi / 2 - 0.5
data = np.empty((m,3))

data[:, 0] = np.cos(angles) + np.sin(angles)/2 + noise * np.random.randn(m) / 2
data[:, 1] = np.sin(angles) * 0.7 + noise * np.random.randn(m)/2
data[:, 2] = data[:,0] * w1 + data[:,1]* w2 + noise * np.random.randn(m)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
x_train = scaler.fit_transform(data[:100])
x_test = scaler.transform(data[100:])

tf.reset_default_graph()

n_inputs = 3
n_hidden = 2
n_outputs = n_inputs

X = tf.placeholder(tf.float32, shape= [None,n_inputs])
hidden = tf.layers.dense(X,n_hidden)
outputs = tf.layers.dense(hidden, n_outputs)

learning_rate = 0.01
n_iterations = 1000
pca = hidden

reconstruction_loss = tf.reduce_mean(tf.square(outputs-X))
train_op = tf.train.AdamOptimizer(learning_rate).minimize(reconstruction_loss)

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for iteration in range(n_iterations):
        train_op.run(feed_dict={X:x_train})
    pca_val = pca.eval(feed_dict={X:x_test})


fig = plt.figure(figsize=(8,6))
ax = fig.gca(projection="3d")
ax.scatter(x_test[:,0], x_test[:,1], x_test[:,2])
ax.set_xlabel("$x_1$", fontsize = 18)
ax.set_ylabel("$x_2$", fontsize = 18)
ax.set_zlabel("$x_3$", fontsize = 18)
plt.show()


fig = plt.figure(figsize=(5,4))
plt.plot(pca_val[:,0],pca_val[:,1],"b.")
plt.xlabel("$z_1$",fontsize = 18)
plt.ylabel("$z_2$",fontsize= 18)
plt.show()