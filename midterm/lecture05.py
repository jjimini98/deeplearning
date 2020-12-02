import tensorflow as tf
import random
tf.set_random_seed(777)

x_data = [[1, 2, 1],[1, 3, 2],[1, 3, 4],[1, 5, 5],[1, 7, 5],[1, 2, 5],[1, 6, 6],[1, 7, 7]]
y_data = [[0, 0, 1],[0, 0, 1],[0, 0, 1],[0, 1, 0],[0, 1, 0],[0, 1, 0],[1, 0, 0],[1, 0, 0]]

x_test = [[2, 1, 1],[3, 1, 2],[3, 3, 4]]
y_test = [[0, 0, 1],[0, 0, 1],[0, 0, 1]]

X = tf.placeholder(tf.float32, shape= [None,3])
Y = tf.placeholder(tf.float32, shape= [None,3])

W = tf.Variable(tf.random_normal([3,3]))
b = tf.Variable(tf.random_normal([3]))

hypothesis = tf.nn.softmax(tf.matmul(X,W)+b)

cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(hypothesis),axis =1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

prediction = tf.argmax(hypothesis,1)
is_correct = tf.equal(prediction,tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(is_correct,tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
        cost_val,w_val,_ = sess.run([cost,W,optimizer],feed_dict={X:x_data,Y:y_data})
        print(step,cost_val,w_val)

    print("Prediction:" , sess.run(prediction,feed_dict={X:x_test}))
    print("Accuracy : ", sess.run(accuracy,feed_dict={X:x_test, Y:y_test}))


import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)
nb_classes = 10
X =tf.placeholder (tf.float32 , [None, 784])
Y = tf.placeholder ( tf.float32, [None, nb_classes])
W = tf.Variable(tf.random_normal([784,nb_classes]),name= 'weight')
b = tf.Variable (tf.random_normal([nb_classes]),name='bias')

logits = tf.matmul(X,W)+b
hypothesis = tf.nn.softmax(logits)

cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(hypothesis),axis =1)) # axis 가 뭐지
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

is_correct = tf.equal(tf.argmax(Y,1),tf.argmax(hypothesis,1))
accuracy = tf.reduce_mean(tf.cast(is_correct , tf.float32))

nums_epoch = 15 #전체를 15번 반복하겠다는 의미
batch_size = 100 # 전체 훈련 데이터셋을 100개로 나누어서 한 번에 7개씩(?) 학습하겠다는 의미
number_iteration = int(mnist.train.num_example / batch_size)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(nums_epoch):
        avg_cost = 0
        for i in range(number_iteration):
            batch_xs , batch_ys = mnist.train.next_batch(batch_size)
            _, cost_val = sess.run([train,cost],feed_dict = {X:batch_xs, Y:batch_ys})

            avg_cost += cost_val/ number_iteration
            print("Epoch: {:4d}, cost: {:.9f}". format(epoch+1, avg_cost))
        print("Learning Finished")

        print("Accuracy : ",  accuracy , eval(session=sess , feed_dict = {X : mnist.test.images, Y: mnist.test.labels}),)
        r = random.randint(0, mnist.test.num_examples-1)
        print("Label : ",sess.run(tf.argmax(mnist.test.labels[r:r+1],1)))
        print("Prediction:",sess.run(tf.argmax(hypothesis,1),feed_dict={X:mnist.test.images[r:r+1]}))


        plt.imshow(
            mnist.test.images[r:r+1].reshape(28,28),
            cmap = 'Greys',
            interpolation='nearest',)
        plt.show()


