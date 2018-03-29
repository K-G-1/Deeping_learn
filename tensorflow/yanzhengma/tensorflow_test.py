# coding=utf-8
#


import tensorflow as tf
from os import listdir
import numpy as np
import cv2


def img2vector(filename):
    # 创建1x1024零向量
    returnVect = np.zeros((1, 1024))
    # 打开文件
    img = cv2.imread(filename)

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edge = cv2.threshold(img_gray, 200, 255, cv2.THRESH_BINARY)
    # edge = cv2.threshold(img_gray, 200, 255,cv2.THRESH_BINARY);
    h, w = edge[1].shape

    returnVect = edge[1].reshape(h * w)

    for i in range(1024):
        if(returnVect[i] > 0):
            returnVect[i] = 1

    # print (len(returnVect))
    # print(returnVect)
    # 返回转换后的1x1024向量
    # input()
    return returnVect


def get_lables(num):
    lables = np.zeros(8)

    for i in range(8):
        lables[i] = 0
    lables[num - 2] = 1
    return lables


def train_image_data():
    filelen = 0
    count = 0
    # 测试集的Labels

    # 返回train_image目录下的文件名
    trainingFileList = listdir('train_image')
    # 返回文件夹下文件夹的个数
    m = len(trainingFileList)

    # 计算共有多少个样本
    for i in range(m):
        classNameStr = trainingFileList[i]
        fileNamelist = listdir('train_image/%s' % classNameStr)
        filelen = filelen + len(fileNamelist)

    # 初始化训练的Mat矩阵,测试集
    trainingMat = np.zeros((filelen, 1024))
    hwLabels = np.zeros((filelen, 8))
    # 从文件名中解析出训练集的类别
    for i in range(m):
        # 获得标签的名字
        classNameStr = trainingFileList[i]
        # 获得分类的数字
        classNumber = int(classNameStr)

        # 获得样本名与每个类别下的样本个数
        fileNamelist = listdir('train_image/%s' % classNameStr)
        filelens = len(fileNamelist)

        # # 将每一个文件的1x1024数据存储到trainingMat矩阵中
        for j in range(filelens):
            # 将获得的类别添加到hwLabels中
            hwLabels[count + j] = get_lables(classNumber)
            trainingMat[count + j, :] = img2vector(
                'train_image/%s/%s' % (classNameStr, fileNamelist[j]))
        # 因为每个类别下的样本个数不一样
        count = count + filelens
    return trainingMat, hwLabels


def test_image_data():
    # 返回test目录下的文件列表
    testFileList = listdir('test')

    # 测试数据的数量
    mTest = len(testFileList)
    hwLabels = np.zeros((mTest, 8))
    vectorUnderTest = np.zeros((mTest, 1024))
    # 从文件中解析出测试集的类别并进行分类测试
    for i in range(mTest):
        # 获得文件的名字
        fileNameStr = testFileList[i]
        # 获得分类的类别
        classNumber = int(fileNameStr.split('-')[0])
        hwLabels[i] = get_lables(classNumber)
        # 获得测试集的1x1024向量,用于训练
        vectorUnderTest[i] = img2vector('test/%s' % (fileNameStr))
    return vectorUnderTest, hwLabels


def compute_accuracy(v_xs, v_ys):
    # 全局变量
    global y, sess
    # 生成预测值，也就是概率，即每个数字的概率
    y_pre = sess.run(y, feed_dict={x: v_xs})
    # 对比预测的数据是否和真实值相等，对比位置是否相等，相等就对了
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    # 计算多少个对，多少个错
    # tf.cast(x,dtype)，将x数据转换为dtype类型
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={x: v_xs, y_: v_ys})
    # print(y_pre[2])
    return result


def main():
    # Import data
    # train_images, train_labels = train_image_data()
    # print (type(train_labels))
    # print (train_labels)
    # Create the model
    x = tf.placeholder(tf.float32, [None, 1024])
    # a 2-D tensor of floating-point numbers
    # None means that a dimension can be of any length
    W = tf.Variable(tf.zeros([1024, 8]))
    b = tf.Variable(tf.zeros([8]))
    y = tf.matmul(x, W) + b
    # It only takes one line to define it

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 8])

    # The raw formulation of cross-entropy,
    #
    #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
    #                                 reduction_indices=[1]))
    # tf.reduce_sum adds the elements in the second dimension of y,
    # due to the reduction_indices=[1] parameter.
    # tf.reduce_mean computes the mean over all the examples in the batch.
    #
    # can be numerically unstable.
    #
    # So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
    # outputs of 'y', and then average across the batch.

    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
    # apply your choice of optimization algorithm to modify the variables and reduce the loss.

    # sess = tf.InteractiveSession()
    # # launch the model in an InteractiveSession
    # tf.global_variables_initializer().run()
    # create an operation to initialize the variables
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for _ in range(100):
            batch_xs, batch_ys = train_image_data()
            # Each step of the loop,
            # we get a "batch" of one hundred random data points from our training set.
            sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

        test_images, test_labels = test_image_data()
        # 生成预测值，也就是概率，即每个数字的概率
        y_pre = sess.run(y, feed_dict={x: test_images})
        # 对比预测的数据是否和真实值相等，对比位置是否相等，相等就对了
        correct_prediction = tf.equal(
            tf.argmax(y_pre, 1), tf.argmax(test_labels, 1))
        # 计算多少个对，多少个错
        # tf.cast(x,dtype)，将x数据转换为dtype类型
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        result = sess.run(accuracy, feed_dict={
                          x: test_images, y_: test_labels})
        print (result)


if __name__ == '__main__':
    main()
