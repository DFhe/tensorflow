#创建会话使用
import tensorflow as tf
m1 = tf.constant([[3,3]])
m2 = tf.constant([[2],[3]])
pro = tf.matmul(m1,m2)
sess = tf.Session()
print(sess.run(pro))