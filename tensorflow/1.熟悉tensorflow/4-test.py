# coding:utf8
# 
import tensorflow as tf  


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)    
    return initial

W_conv1 = weight_variable([5, 5, 1, 32])     
result=tf.truncated_normal(shape=[5,5],mean=0,stddev=1) 

with tf.Session() as sess:  
    print(sess.run(W_conv1))