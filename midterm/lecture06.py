# import tensorflow as tf

# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets('/tmp/data/',one_hot=True)
# def build_CNN_clasifier(x):
#      x_image = tf.reshape (x, [-1,28,28,1])
#
#      #layer1
#      w_conv1 = tf.Variable(tf.truncated_normal(shape = [5,5,1,32],stddev= 5e-2))
#      b_conv1 = tf.Variable(tf.constant(0.1,shape=[32]))
#      h_conv1 = tf.nn.relu(tf.nn.conv2d(x_image,w_conv1,stride=[1,1,1,1,],padding='SAME')+b_conv1)
#      h_pool1 = tf.nn.max_pool(h_conv1,ksize=[1,2,2,1],strides = [1,2,2,1],padding='SAME')
#
#      #layer2
 #    w_conv2 = tf.Variable(tf.truncated_normal(shape=[5,5,32,64],stddev = 5e-2))
 #    b_conv2 = tf.Variable(tf.constant(0.1,shape=[64]))
     # h_conv2 = tf.nn.relu(tf.nn.conv2d(h_conv1,w_conv2,strides=[1,1,1,1],padding='SAME')+b_conv2)
     #
     # h_pool2 = tf.nn.max_pool(h_conv2,ksize=[1,2,2,1],strides= [1,2,2,1],padding='SAME')
     #
     # #fully-connected layer
     # w_fc_1 = tf.Variable(tf.truncated_normal(shape=[7*7*64,1024],stddev=5e-2))
     # b_fc_1 = tf.Variable(tf.constant(0.1,shape=[1024]))
     # h_pool2_flat= tf.reshape(h_pool2,[-1,7*7*64])
     # h_fc_1 = tf.nn.relu(tf.matmul(h_pool2_flat,w_fc_1)+b_fc_1)
     #
     #
     #
     #
     # with tf.Session() as sess:
     #     sess.run(x_image, feed_dict={x:mnist})
     #     print(x_image)
     #     print(x_image.shape)


import numpy as np

def conv1d(x, w, p=0, s=1):
     w_rot = np.array(w[::-1])

     x_padded = np.array(x)
     if p > 0:
          zero_pad = np.zeros(shape=p)
          x_padded = np.concatenate([zero_pad, x_padded, zero_pad])
     res = []
     for i in range(0, int((len(x)+2*p-len(w))/s)+1):
          j = s*i;
          res.append(np.sum(x_padded[j:j+w_rot.shape[0]] * w_rot))

     return np.array(res)
## Testing:
x = [1, 0, 2, 3, 0, 1, 1]
w = [2, 1, 3]
print('Conv1d Implementation: ', conv1d(x, w, p=0, s=1))
print('Numpy Results: ', np.convolve(x, w, mode='valid'))






import tensorflow as tf
i = tf.constant([1, 0, 2, 3, 0, 1, 1], dtype=tf.float32, name='i')
k = tf.constant([2, 1, 3], dtype=tf.float32, name='k')
print(i, '\n', k, '\n')
data = tf.reshape(i, [1, int(i.shape[0]), 1], name='data')
kernel = tf.reshape(k, [int(k.shape[0]), 1, 1], name='kernel')
print(data, '\n', kernel, '\n')
res = tf.squeeze(tf.nn.conv1d(data, kernel, 1, 'VALID'))
#res = tf.squeeze(tf.nn.conv1d(data, kernel, 1, 'SAME'))
#res = tf.squeeze(tf.nn.conv1d(data, kernel, 2, 'SAME’))
#res = tf.nn.conv1d(data, kernel, 2, 'SAME')
with tf.Session() as sess:
     print(sess.run(res))
     print(sess.run(data))