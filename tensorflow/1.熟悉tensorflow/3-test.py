#coding=utf-8

import tensorflow as tf

# 给定type，tf大部分只能处理float32数据
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)

# Tensorflow 1.0 修改版
# tf.mul---tf.multiply
# tf.sub---tf.subtract
# tf.neg---tf.negative
output = tf.multiply(input1, input2)

with tf.Session() as sess:
    # placeholder在sess.run()的时候传入值
    print(sess.run(output, feed_dict={input1: [7.], input2: [2.]}))