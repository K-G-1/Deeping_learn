# coding=utf-8
import tensorflow as tf
import numpy as np
import input_data

mnist = input_data.read_data_sets('data/', one_hot=True)

n_hidden_1 = 256
n_input = 784
n_classes = 10

# INPUTS AND OUTPUTS
# 用placeholder先占地方，样本个数不确定为None
x = tf.placeholder(tf.float32, [None, n_input])
# 用placeholder先占地方，样本个数不确定为None
y = tf.placeholder(tf.float32, [None, n_classes])

# NETWORK PARAMETERS
weights = {
    'w1': tf.Variable(tf.random_normal([n_input, n_hidden_1], stddev=0.1)),
    'out': tf.Variable(tf.zeros([n_hidden_1, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.zeros([n_hidden_1])),
    'out': tf.Variable(tf.zeros([n_classes]))
}
print("NETWORK READY")


def multilayer_perceptron(_X, _weights, _biases):  # 前向传播，l1、l2每一层后面加relu激活函数
    layer_1 = tf.nn.relu(
        tf.add(tf.matmul(_X, _weights['w1']), _biases['b1']))  # 隐层
    # 返回输出层的结果，得到十个类别的得分值
    return (tf.matmul(layer_1, _weights['out']) + _biases['out'])


pred = multilayer_perceptron(x, weights, biases)  # 前向传播的预测值
# 交叉熵损失函数，参数分别为预测值pred和实际label值y，reduce_mean为求平均loss
cost = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred))
optm = tf.train.GradientDescentOptimizer(0.01).minimize(cost)  # 梯度下降优化器
# tf.equal()对比预测值的索引和实际label的索引是否一样，一样返回True，不一样返回False
corr = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
# 将pred即True或False转换为1或0,并对所有的判断结果求均值
accr = tf.reduce_mean(tf.cast(corr, tf.float32))

init = tf.global_variables_initializer()
print("FUNCTIONS READY")

# 上面神经网络结构定义好之后，下面定义一些超参数
training_epochs = 100  # 所有样本迭代100次
batch_size = 100  # 每进行一次迭代选择100个样本
display_step = 5
# LAUNCH THE GRAPH
sess = tf.Session()  # 定义一个Session
sess.run(init)  # 在sess里run一下初始化操作
# OPTIMIZE
for epoch in range(training_epochs):
    avg_cost = 0.
    total_batch = int(mnist.train.num_examples / batch_size)
    # Loop over all batches
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)  # 逐个batch的去取数据
        sess.run(optm, feed_dict={x: batch_xs, y: batch_ys})
        avg_cost += sess.run(cost,
                             feed_dict={x: batch_xs, y: batch_ys}) / total_batch
    # Display logs per epoch step
    if epoch % display_step == 0:
        train_acc = sess.run(accr, feed_dict={x: batch_xs, y: batch_ys})
        test_acc = sess.run(
            accr, feed_dict={x: mnist.test.images, y: mnist.test.labels})
        print("Epoch: %03d/%03d cost: %.9f TRAIN ACCURACY: %.3f TEST ACCURACY: %.3f"
              % (epoch, training_epochs, avg_cost, train_acc, test_acc))
print("DONE")
