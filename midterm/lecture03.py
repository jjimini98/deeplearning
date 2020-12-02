import tensorflow as tf

#p.23
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

adder_node = a+b

sess = tf.Session()
#print(sess.run(adder_node,feed_dict={a:[1,2],b:[4,5]}))
sess.close()


#p.24
hello = tf.constant("hello tensorflow!")

sess = tf.Session()
# print(sess.run(hello))
sess.close()

#p.25-27

X = [1,2,3]
Y = [1,2,3] # label 에 해당한다

w = tf.Variable ( tf.random_normal([1]), name= "weight")
b = tf.Variable ( tf.random_normal([1]), name = 'bias')

hypothesis = w*X + b

cost = tf.reduce_mean(tf.square(hypothesis-Y)) # 예측한값-실제값을 제곱한 값들의 평균이 cost

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost) #이 문장이 이해가 안감

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001):
    sess.run(train)
    if step%20==0:
        continue
#        print(step,sess.run(cost),sess.run(w),sess.run(b))

sess.close()

#p.29
import tensorflow as tf
import matplotlib.pyplot as plt

X = [1,2,3]
Y = [1,2,3] # label 에 해당한다

w = tf.placeholder(tf.float32)
hypothesis = X*w

cost = tf.reduce_mean(tf.square(hypothesis-Y))

with tf.Session() as sess:
    w_val =[]
    cost_val = []

    for i in range(-30,50):
        feed_w = i*0.1
        curr_cost , curr_w = sess.run([cost,w], feed_dict={w:feed_w})
#        print(curr_cost)
#        print(curr_w)
        w_val.append(curr_w)
        cost_val.append(curr_cost)

#    plt.plot(w_val,cost_val)
#    plt.show()

#  p. 30 - 31
import tensorflow as tf

x= [1,2,3]
y= [1,2,3]

w = tf.Variable (-3.0)

hypothesis = x*w

cost = tf.reduce_mean ( tf.square(hypothesis- y ))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train = optimizer.minimize(cost)

with tf.Session() as sess :
    sess.run(tf.global_variables_initializer())

    for step in range(11):
#        print(step,sess.run(w))
        sess.run(train)


#p.32

import tensorflow as tf

x_data = [1,2,3]
y_data = [1,2,3]

w = tf.Variable(tf.random_normal([1]),name='weigt')

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

hypothesis = x*w

cost = tf.reduce_mean(tf.square(hypothesis-y))

learning_rate = 0.1
gradient  = tf.reduce_mean((w*x-y)*x)
descent = w - learning_rate *gradient
update = w.assign(descent)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(21):
        sess.run(update,feed_dict={x:x_data, y: y_data})
#        print(step, sess.run(cost, feed_dict={x:x_data, y:y_data}),sess.run(w))

#p.35

import tensorflow as tf

x_data = [[73.,80.,75.],[93.,88.,93.],[89.,91.,90.],[96.,98.,100.],[73.,66.,70]]
y_data = [[152.],[185.],[180.],[196.],[142.]]

X = tf.placeholder (tf.float32,shape=[None,3])
Y = tf.placeholder (tf.float32,shape=[None,1])

W = tf.Variable(tf.random_normal([3,1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.matmul(X,W) + b #matmul 을 사용하는 이유가 무엇일까 ? 곱할 때 순서가 중요함. X*W 랑 W*X랑 다름

cost = tf.reduce_mean(tf.square(hypothesis-Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

with tf.Session() as sess :

    sess.run(tf.global_variables_initializer())

    for step in range(100001):
        cost_val,hy_val,_ = sess.run(
            [cost,hypothesis,train], feed_dict={X:x_data, Y:y_data})
        if step%10 ==0:
            print(step,"Cost: " , cost_val, "\n Prediction : \n", hy_val)

#정답 : [[152.],[185.],[180.],[196.],[142.]]


