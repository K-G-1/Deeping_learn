#!/usr/bin/python
# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier as kNN
import pickle
from os import listdir
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


def KnnClassTest():
    # 测试集的Labels
    hwLabels = []
    # 返回trainingDigits目录下的文件名
    trainingFileList = listdir('train_image')
    # 返回文件夹下文件的个数
    m = len(trainingFileList)
    # 初始化训练的Mat矩阵,测试集
    trainingMat = np.zeros((m * 4, 1024))
    # 从文件名中解析出训练集的类别
    for i in range(m):
        # 获得文件的名字
        classNameStr = trainingFileList[i]
        fileNamelist = listdir('train_image/%s' % classNameStr)
        # 获得分类的数字
        classNumber = str(classNameStr)
        # 将获得的类别添加到hwLabels中
        filelen = len(fileNamelist)
        # # 将每一个文件的1x1024数据存储到trainingMat矩阵中
        for j in range(4):
            hwLabels.append(classNumber)
            trainingMat[i * 4 + j, :] = img2vector(
                'train_image/%s/%s' % (classNameStr, fileNamelist[j]))

    # print(trainingMat)
    # print(hwLabels)
    # 构建kNN分类器
    neigh = kNN(n_neighbors=3, algorithm='auto')
    # 拟合模型, trainingMat为训练矩阵,hwLabels为对应的标签
    neigh.fit(trainingMat, hwLabels)
    # 返回testDigits目录下的文件列表
    testFileList = listdir('test')
    # 错误检测计数
    errorCount = 0.0
    # 测试数据的数量
    mTest = len(testFileList)
    vectorUnderTest = np.zeros((1, 1024))
    # 从文件中解析出测试集的类别并进行分类测试
    for i in range(mTest):
        # 获得文件的名字
        fileNameStr = testFileList[i]
        # 获得分类的数字
        classNumber = str(fileNameStr.split('-')[0])
        # 获得测试集的1x1024向量,用于训练
        vectorUnderTest[0] = img2vector('test/%s' % (fileNameStr))
        # 获得预测结果
        # classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        classifierResult = neigh.predict(vectorUnderTest)
        print("分类返回结果为%s\t真实结果为%s" % (classifierResult, classNumber))
        if(classifierResult != classNumber):
            errorCount += 1.0
    print("总共错了%d个数据\n错误率为%f%%" % (errorCount, errorCount / mTest * 100))


def SvmClassTest():
    filelen = 0
    count = 0
    # 测试集的Labels
    hwLabels = []
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

    # 从文件名中解析出训练集的类别
    for i in range(m):
        # 获得标签的名字
        classNameStr = trainingFileList[i]
        # 获得分类的数字
        classNumber = str(classNameStr)

        # 获得样本名与每个类别下的样本个数
        fileNamelist = listdir('train_image/%s' % classNameStr)
        filelens = len(fileNamelist)

        # # 将每一个文件的1x1024数据存储到trainingMat矩阵中
        for j in range(filelens):
            # 将获得的类别添加到hwLabels中
            hwLabels.append(classNumber)
            trainingMat[count + j, :] = img2vector(
                'train_image/%s/%s' % (classNameStr, fileNamelist[j]))
        # 因为每个类别下的样本个数不一样
        count = count + filelens

    C = 1.0
    svc = svm.SVC(kernel='linear', C=0.5).fit(trainingMat, hwLabels)  # 线性核
    rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(
        trainingMat, hwLabels)  # 径向基核
    poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(
        trainingMat, hwLabels)  # 多项式核

    # 返回test目录下的文件列表
    testFileList = listdir('test_image')
    # 错误检测计数
    errorCount = 0.0
    # 测试数据的数量
    mTest = len(testFileList)
    vectorUnderTest = np.zeros((1, 1024))
    # 从文件中解析出测试集的类别并进行分类测试
    for i in range(mTest):
        # 获得文件的名字
        fileNameStr = testFileList[i]
        # 获得分类的类别
        classNumber = str(fileNameStr.split('-')[0])
        # 获得测试集的1x1024向量,用于训练
        vectorUnderTest[0] = img2vector('test_image/%s' % (fileNameStr))
        # 获得预测结果
        # classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        classifierResult = svc.predict(vectorUnderTest)
        print("分类返回结果为%s\t真实结果为%s" % (classifierResult, classNumber))
        if(classifierResult != classNumber):
            errorCount += 1.0
        # print (fileNameStr)
        img = cv2.imread('test_image/%s' % (fileNameStr))
        plt.imshow(img)
        plt.show()
        # input(">")
    print("总共错了%d个数据\n错误率为%f%%" % (errorCount, errorCount / mTest * 100))


if __name__ == '__main__':
    SvmClassTest()
    # KnnClassTest()
