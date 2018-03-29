#!/usr/bin/python
#coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
import pickle
import csv

filename = './data/bmi.csv'
with open(filename) as f:
    reader = list(csv.reader(f))
    reader = reader[1:]

dataMat =[]
labelMat = []    
for dat in reader:
    dataMat.append([dat[0],dat[1]])

    labelMat.append(dat[2])
# print (dataMat)
C = 1.0  # SVM正则化参数
svc = svm.SVC(kernel='linear', C=C).fit(dataMat, labelMat) # 线性核

with open('weight.pkl', 'wb') as f:
    pickle.dump(svc, f)
    f.close()

Z = svc.predict(np.c_[175, 85])

print (Z)
