# p.6
# import  tensorflow as tf
#
# x_data = [[1,2],[2,3],[3,1],[4,3],[5,3],[6,2]]
# y_data = [[0],[0],[0],[1],[1],[1]]
#
# X = tf.placeholder (tf.float32, shape= [None,2]) # 열의 사이즈가 2인 행렬
# Y = tf.placeholder (tf.float32, shape=[None,1]) # 열의 사이즈가 1인 행렬
# W = tf.Variable (tf.random_normal([2,1]),name= 'weight') #  2by1 짜리 난수값을 뽑아냄 in 정규분포
# # b = tf.Variable (tf.random_normal([1]), name='bias') # 1 by 1 짜리 난수값을 뽑아냄 in 정규분포
#
# hypothesis = tf.sigmoid(tf.matmul(X,W)+b)
#
# #linear classification 이니까 cost function은 cross entropy error
# # 여기서 보면 y label에 해당하는 값이 0 또는 1로 값이 2개다. 따라서 binary cross entropy error 라고 할 수 있다.
# # Y인 확률
# cost =-tf.reduce_mean(Y*tf.log(hypothesis)+(1-Y)*tf.log(1-hypothesis))
#
# train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)
#
# predicted = tf.cast(hypothesis>0.5, dtype=tf.float32)
# accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted,Y), dtype=tf.float32))
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#
#     for step in range(10001):
#         cost_val, _= sess.run([cost,train],feed_dict={X:x_data, Y:y_data})
#    #     if step % 20 ==0:
#    #         print(step,cost_val)
#
#     h,c,a = sess.run([hypothesis,predicted,accuracy], feed_dict={X:x_data, Y:y_data})
#   #  print("\nHypothesis : " , h , "\nCorrect(Y) : ",c, "\nAccuracy : ",a)

# #p.7-8 Classification
#
# import tensorflow as tf
# import numpy as np
#
# xy = np.loadtxt("./data-03-diabetes.csv", delimiter= ',', dtype=np.float32) #(759,9)
# x_data = xy[:,0:-1] #(759,8) 마지막 한 행 제외하고 출력
# y_data = xy[:,[-1]] #(759,1)
#
# X = tf.placeholder(tf.float32, shape=[None,8])
# Y = tf.placeholder(tf.float32, shape= [None,1])
# W = tf.Variable(tf.random_normal([8,1]), name= 'weight')
# b = tf.Variable(tf.random_normal([1]), name= 'bias')
#
# hypothesis = tf.sigmoid(tf.matmul(X,W)+b)
#
# cost  = - tf.reduce_mean(Y*tf.log(hypothesis) + (1-Y)*tf.log(1-hypothesis))
# train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)
#
# predicted = tf.cast (hypothesis > 0.5 , dtype=tf.float32)
# accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted,Y),dtype=tf.float32))
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#
# #    for step in range(10001):
# #        cost_val, _ = sess.run([cost,train],feed_dict={X:x_data, Y: y_data})
# #        if step % 200==0:
# #            print(step,cost_val)
#
#     h, c, a = sess.run([hypothesis,predicted,accuracy], feed_dict= {X:x_data,  Y : y_data})
# #    print("\n hypothesis: ",h,"\n correct(y):",c, "\n accuracy",a )
#


#p.15-16 multi class logistic regression

import tensorflow as tf
import numpy as np

xy = np.loadtxt('data-04-zoo.csv', delimiter = ",", dtype=np.float32)
x_data = xy[: , 0:-1] #(101,16)
y_data = xy[:,[-1]] #(101,1)

nb_classes= 7 #동물의 종류?

X = tf.placeholder(tf.float32,[None,16]) #x_data의 column 의 크기랑 맞춘다.
Y = tf.placeholder(tf.int32, [None,1]) # y_data의 column의 크기랑 맞춘다.

Y_one_hot = tf.one_hot(Y,nb_classes)
Y_one_hot = tf.reshape(Y_one_hot,[-1,nb_classes]) #-1 을 하면 사이즈를 자동으로 조정
# ( 위에 Y 의 크기가 확실하게 정해지지 않았아서 -1을 하면 행 개수를 알아서 지정해준다는 의미같음)

W= tf.Variable(tf.random_normal([16,nb_classes]),name = "weight")
b = tf.Variable(tf.random_normal([nb_classes]),name ='bias')

logits = tf.matmul(X,W)+b

hypothesis = tf.nn.softmax(logits)

cost_x = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(hypothesis),axis =1 ))

cost_i = tf.nn.softmax_cross_entropy_with_logits(logits= logits,labels = Y_one_hot)
cost = tf.reduce_mean(cost_i)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)
optimizer_x = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize((cost_x))

prediction = tf.argmax(hypothesis,1)
correct_prediction = tf.equal(prediction,tf.argmax(Y_one_hot,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2000):
            sess.run([optimizer],feed_dict={X:x_data,Y:y_data})
            if step% 100 ==0 :
                loss, acc = sess.run([cost,accuracy],feed_dict={X:x_data,Y:y_data})
                print("Step: {:5}\tLoss:{:.3f}\tAcc:{:.2%}".format(step,loss,acc))

    pred = sess.run(prediction, feed_dict={X:x_data,Y:y_data})
    for p,y in zip(pred,y_data.flatten()):
            print("[{}] prediction: {} True Y : {}".format(p==int(y), p, int(y)))
