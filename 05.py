# 非线性回归测试
import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt

x_data = np.linspace(-0.5, 0.5, 200)[:, np.newaxis]
noise = np.random.normal(0, 0.02, x_data.shape)
y_data = np.square(x_data) + noise

x = tf.placeholder(tf.float32, [None, 1])
y = tf.placeholder(tf.float32, [None, 1])

# 定义神经网络中间层
Weights_L1 = tf.Variable(tf.random_normal([1, 10]))
biases_L1 = tf.Variable(tf.zeros([1, 10]))
wx_plus_L1 = tf.matmul(x, Weights_L1) + biases_L1
L1 = tf.nn.tanh(wx_plus_L1)

# 定义输出层
Weights_L2 = tf.Variable(tf.random_normal([10, 1]))
biases_L2 = tf.Variable(tf.zeros([1, 1]))
wx_plus_L2 = tf.matmul(L1, Weights_L2) + biases_L2


pred = tf.nn.tanh(wx_plus_L2)

#二次代价函数
loss = tf.reduce_mean(tf.square(y-pred))
#使用梯度下降法
train = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for _ in range(2000):
        sess.run(train,feed_dict={
            x:x_data,
            y:y_data
        })

    pred_value = sess.run(pred,feed_dict={
        x:x_data
    })
    plt.figure()
    plt.scatter(x_data,y_data)
    plt.plot(x_data,pred_value,'r-',lw=5)
    plt.show()