# 手写数字识别
#优化器使用
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

# 载入数据集
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# 每个批次的大小
batch_size = 50
# 计算一共有多少个批次
n_batch = mnist.train.num_examples // batch_size

# 定义两个placeholder
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])


# 创建一个简单的神经网络
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
predition = tf.nn.softmax(tf.matmul(x, W) + b)

# 二次代价函数
loss = tf.reduce_mean(tf.square(y - predition))

# 使用梯度下降法
#train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)
train_step = tf.train.AdamOptimizer(1e-2).minimize(loss)



init = tf.global_variables_initializer()

# 求准确率的方法
# tf.argmax(y,1)判断最大概率的标签（位置
correct_predtion = tf.equal(tf.argmax(y, 1), tf.argmax(predition, 1))

# 求准确率
# tf.cast类型转化
accuracy = tf.reduce_mean(tf.cast(correct_predtion, tf.float32))

# 进行训练
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(21):  # 整体循环21次训练
        for batch in range(n_batch):  # 整体执行一次图片循环
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={
                x: batch_xs,
                y: batch_ys
            })

        acc = sess.run(accuracy, feed_dict={
            x: mnist.test.images,
            y: mnist.test.labels
        })
        print("Iter " + str(epoch) + " ,Test Accuracy:" + str(acc))
