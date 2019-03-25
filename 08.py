# 使用TensorBoard
# 手写数字识别
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

# 载入数据集
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
# 命名空间

# 每个批次的大小
batch_size = 50
# 计算一共有多少个批次
n_batch = mnist.train.num_examples // batch_size
with tf.name_scope('input'):
    # 定义两个placeholder
    x = tf.placeholder(tf.float32, [None, 784], name='x-input')
    y = tf.placeholder(tf.float32, [None, 10], name='y-input')
with tf.name_scope('layer'):
    # 创建一个简单的神经网络
    with tf.name_scope('weights'):
        W = tf.Variable(tf.zeros([784, 10]),name="W")
    with tf.name_scope('biases'):
        b = tf.Variable(tf.zeros([10]),name='b')
    with tf.name_scope('wx_plus_b'):
        wx_plus_b = tf.matmul(x, W) + b
    with tf.name_scope('softmax'):
        predition = tf.nn.softmax(wx_plus_b)

with tf.name_scope('loss'):
    # 二次代价函数
    loss = tf.reduce_mean(tf.square(y - predition))
with tf.name_scope('train'):
    # 使用梯度下降法
    train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)


with tf.name_scope('accuracy'):
    # 求准确率的方法
    # tf.argmax(y,1)判断最大概率的标签（位置
    with tf.name_scope('correct_predtion'):
        correct_predtion = tf.equal(tf.argmax(y, 1), tf.argmax(predition, 1))

    # 求准确率
    # tf.cast类型转化
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_predtion, tf.float32))

# 进行训练
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    writer = tf.summary.FileWriter('logs/', sess.graph)
    for epoch in range(1):  # 整体循环21次训练
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
