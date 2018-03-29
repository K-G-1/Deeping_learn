# coding=utf-8

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('data/', one_hot=True)
print("MNIST ready")

sess = tf.InteractiveSession()

# 定义好初始化函数以便重复使用。给权重制造一些随机噪声来打破完全对称，使用截断的正态分布，标准差设为0.1，
# 同时因为使用relu，也给偏执增加一些小的正值(0.1)用来避免死亡节点(dead neurons)


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    # 参数分别指定了卷积核的尺寸、多少个channel、filter的个数即产生特征图的个数
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# 2x2最大池化，即将一个2x2的像素块降为1x1的像素。最大池化会保留原始像素块中灰度值最高的那一个像素，即保留最显著的特征。


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


n_input = 784  # 28*28的灰度图，像素个数784
n_output = 10  # 是10分类问题

# 在设计网络结构前，先定义输入的placeholder，x是特征，y是真实的label
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_output])
# 对图像做预处理，将1D的输入向量转为2D的图片结构，即1*784到28*28的结构,-1代表样本数量不固定，1代表颜色通道数量
x_image = tf.reshape(x, [-1, 28, 28, 1])

# 定义第一个卷积层，使用前面写好的函数进行参数初始化，包括weight和bias
W_conv1 = weight_variable([3, 3, 1, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# 定义第二个卷积层
W_conv2 = weight_variable([3, 3, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# fc1，将两次池化后的7*7共128个特征图转换为1D向量，隐含节点1024由自己定义
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# 为了减轻过拟合，使用Dropout层
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Dropout层输出连接一个Softmax层,得到最后的概率输出
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
pred = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)  # 前向传播的预测值，
print("CNN READY")

# 定义损失函数为交叉熵损失函数
cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred), reduction_indices=[1]))
# 优化器
optm = tf.train.AdamOptimizer(0.001).minimize(cost)
# 定义评测准确率的操作
# 对比预测值的索引和真实label的索引是否一样，一样返回True，不一样返回False
corr = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(corr, tf.float32))
# 初始化所有参数
tf.global_variables_initializer().run()
print("FUNCTIONS READY")

training_epochs = 1000  # 所有样本迭代1000次
batch_size = 100  # 每进行一次迭代选择100个样本
display_step = 1

for i in range(training_epochs):
    avg_cost = 0.
    total_batch = int(mnist.train.num_examples / batch_size)
    batch = mnist.train.next_batch(batch_size)
    optm.run(feed_dict={x: batch[0], y: batch[1], keep_prob: 0.7})
    avg_cost += sess.run(cost,
                         feed_dict={x: batch[0], y: batch[1], keep_prob: 1.0}) / total_batch
    if i % display_step == 0:  # 每10次训练，对准确率进行一次测试
        train_accuracy = accuracy.eval(
            feed_dict={x: batch[0], y: batch[1], keep_prob: 1.0})
        test_accuracy = accuracy.eval(
            feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0})
        print("step: %d  cost: %.9f  TRAIN ACCURACY: %.3f  TEST ACCURACY: %.3f" % (
            i, avg_cost, train_accuracy, test_accuracy))
print("DONE")
