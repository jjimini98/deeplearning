import tensorflow as tf
tf.reset_default_graph()
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./mnist/data/", one_hot= True)

learning_rate = 0.001
total_epoch = 30
batch_size = 128

n_input = 28
n_step = 28
n_hidden = 128
n_class = 10

X = tf.placeholder(tf.float32, [None, n_step, n_input])
Y = tf.placeholder(tf.float32, [None, n_class])

W = tf.Variable(tf.random_normal([n_hidden, n_class]))
b = tf.Variable(tf.random_normal([n_class]))

cell = tf.nn.rnn_cell.BasicRNNCell(n_hidden)
outputs , states = tf.nn.dynamic_rnn(cell, X , dtype = tf.float32)

model = tf.matmul(states, W )+b

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = model, labels = Y))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

with tf.Session() as sess :
    sess.run(tf.global_variables_initializer)

    total_batch = int(mnist.train.num_examples/batch_size)

    for epoch in range(total_epoch):
        total_cost = 0

        for i in range(total_batch):
            batch_xs , batch_ys = mnist.train.next_batch ( batch_size)
            batch_xs = batch_xs.reshape((batch_size,n_step,n_input))






