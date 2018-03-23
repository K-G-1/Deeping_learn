#!/usr/bin/python
#coding=utf-8
from sklearn import svm
import numpy as np
import pickle
import csv


with open('weight.pkl', 'rb') as f:
    svc = pickle.load(f)

Z = svc.predict(np.c_[175, 85])

print (Z)