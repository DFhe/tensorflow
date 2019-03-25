# TensorFlow的简单实例
import tensorflow as tf
import numpy as np

# 生成100个随机点
x_data = np.random.rand(100)
y_data = x_data * 1.1 + 0.4

#训练数据
b = tf.Variable(0.)
k = tf.Variable(0.)
y = k * x_data + b

#二阶代价函数
loss = tf.reduce_mean(tf.square(y_data-y))
#定义一个梯度下降法
optimizer = tf.train.GradientDescentOptimizer(0.2)

#定义最小化待见函数
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for step in range(401):
        sess.run(train)
        if step%20==0:
            print(sess.run([k,b]))


