# 卷积神经网络
# 查看参数
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

# 载入数据集
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# 每个批次的大小
batch_size = 100
# 计算一共有多少个批次
n_batch = mnist.train.num_examples // batch_size
print("n_batch:"+str(n_batch))


# 初始化权值
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


# 初始化偏值
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# 卷积层
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


# 池化层
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# 定义两个placeholder
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

# 改变x的格式转为4d的向量
x_image = tf.reshape(x, [-1, 28, 28, 1])

# 初始化第一个卷积层的权重和偏置
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
# 第一个进行卷积操作并池化
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)


# 初始化第2个卷积层的权重和偏置
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

# 第2个进行卷积操作并池化

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)


#初始化第一个全连接层的权值
W_fc1 = weight_variable([7*7*64,1024])
b_fc1  = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64])

#求第一个全连接层的输出
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)
#keep_prod代表神经元输出概率
keep_prod = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1,keep_prod)

#初始化第一=2个全连接层的权值
W_fc2 = weight_variable([1024,10])
b_fc2  = bias_variable([10])

#计算输出
predition = tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2)+b_fc2)

#交叉熵代价函数
correct_entropy =  tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=predition))

#开始优化
train_step = tf.train.AdamOptimizer(1e-4).minimize(correct_entropy)

#结果放在一个布尔列表中

correct_pred = tf.equal(tf.argmax(predition,1),tf.argmax(y,1))

#求准确率
accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))

# 进行训练

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(21):  # 整体循环21次训练
        for batch in range(n_batch):  # 整体执行一次图片循环
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={
                x: batch_xs,
                y: batch_ys,
                keep_prod:0.7
            })
            print("运行中："+str(batch))
        acc = sess.run(accuracy, feed_dict={
            x: mnist.test.images,
            y: mnist.test.labels,
            keep_prod:0.7
        })
        print("Iter " + str(epoch) + " ,Test Accuracy:" + str(acc))


