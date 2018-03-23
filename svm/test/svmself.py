#!/usr/bin/python
#coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets

# 导入数据集
def loadDataSet(fileName):
    dataMat = []
    labelMat = []
    testMat = []
    test_labelMat = []
    fr = open(fileName)
    for line in fr.readlines():  # 逐行读取，滤除空格等
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])  # 添加数据
        labelMat.append(float(lineArr[2]))  # 添加标签
    testMat = dataMat[90:100]
    test_labelMat = labelMat[90:100]
    dataMat = dataMat[:90]
    labelMat = labelMat[:90]
    return dataMat, labelMat,testMat,test_labelMat

dataMat, labelMat ,testMat ,test_labelMat = loadDataSet('testSetRBF.txt')
# print(type(dataMat))
print (dataMat)
print (len(dataMat))
print (len(testMat))
# h = .02  # 网格中的步长

# 创建支持向量机实例，并拟合出数据
C = 1.0  # SVM正则化参数
svc = svm.SVC(kernel='linear', C=C).fit(dataMat, labelMat) # 线性核
rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(dataMat, labelMat) # 径向基核
poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(dataMat, labelMat) # 多项式核
lin_svc = svm.LinearSVC(C=C).fit(dataMat, labelMat) #线性核

x_min = 10
x_max = -10
y_min = 10
y_max = -10
# 创建网格，以绘制图像
for i in xrange(len(dataMat)) :
    if(x_min > dataMat[i][0]):
        x_min = dataMat[i][0]

    if(x_max < dataMat[i][0]):
        x_max = dataMat[i][0]

    if(y_min > dataMat[i][1]):
        y_min = dataMat[i][1]

    if(y_max < dataMat[i][1]):
        y_max = dataMat[i][1]
print(x_max,x_min)

h = .02
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# 图的标题
titles = ['SVC with linear kernel',
          'LinearSVC (linear kernel)',
          'SVC with RBF kernel',
          'SVC with polynomial (degree 3) kernel']

for i, clf in enumerate((svc, lin_svc, rbf_svc, poly_svc)):
    # 绘出决策边界，不同的区域分配不同的颜色
    plt.subplot(2, 2, i + 1) # 创建一个2行2列的图，并以第i个图为当前图
    plt.subplots_adjust(wspace=0.4, hspace=0.4) # 设置子图间隔

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]) #将xx和yy中的元素组成一对对坐标，作为支持向量机的输入，返回一个array

    # 把分类结果绘制出来
    Z = Z.reshape(xx.shape) #(220, 280)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8) #使用等高线的函数将不同的区域绘制出来


    # # 将训练数据以离散点的形式绘制出来
    for y in xrange(len(dataMat)) :
        plt.scatter(dataMat[y][0], dataMat[y][1], c= 'k' ,cmap=plt.cm.Paired)

    for y in xrange(len(testMat)) :
        plt.scatter(testMat[y][0], testMat[y][1], c= 'r' ,cmap=plt.cm.Paired)  
    # plt.scatter(dataMat[0][0], dataMat[0][1], cmap=plt.cm.Paired)
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.title(titles[i])

test1 = []
test2 = []
test3 = []
for i in range(10):
    test1.append(lin_svc.predict(np.c_[testMat[i][0], testMat[i][1]]))
    test2.append(rbf_svc.predict(np.c_[testMat[i][0], testMat[i][1]]))
    test3.append(poly_svc.predict(np.c_[testMat[i][0], testMat[i][1]]))
print (test1,'\r',test2,'\r',test3)
plt.show()