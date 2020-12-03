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


X= tf.placeholder(tf.float32, [None,n_steps,n_inputs])
Y= tf.placeholder(tf.float32,[None,n_steps,n_outputs])

cell = tf.contrib.rnn.OutputProjectionWrapper(
    tf.nn.rnn_cell.BasicRNNCell(num_units= n_neurons , activation=tf.nn.relu),
    output_size = n_outputs)
prediction, states  = tf.nn.dynamic_rnn(cell,X,dtype=tf.float32)

learning_rate = 0.001
n_iterations =600
batch_size = 50

mse = tf.losses.mean_squared_error(labels=Y, predictions=prediction)
train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(mse)

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for iteration in range(n_iterations):
        batch_x,batch_y = next_batch(batch_size,n_steps)
        sess.run(train_op, feed_dict={X:batch_x,Y:batch_y})
        if iteration % 100 ==0:
            loss = mse.eval(feed_dict = {X:batch_x, Y:batch_y})
            print('step: {:03d}, MSE: {:.4f}'.format(iteration,loss))

    X_new = time_series(np.array(t_instance[:-1].reshape(-1,n_steps,n_inputs)))
    Y_pred = sess.run(prediction, feed_dict={X:X_new})

print('Y_pred:{} \n {}'.format(Y_pred.shape,Y_pred))

plt.title("Testing the model",fontsize = 14)
plt.plot(t_instance[:-1],time_series(t_instance[:-1]),"bo",markersize=10, label = "instance")
plt.plot(t_instance[1:],time_series(t_instance[1:]),"w*",markersize=10, label = "target", color="yellow")
plt.plot(t_instance[1:],Y_pred[0,:,0],"r.",markersize=10,label="prediction")
plt.legend(loc = "upper left")
plt.xlabel("Time")

plt.show()
