# coding:utf8
import tensorflow as tf
import numpy as np
sess = tf.InteractiveSession()
x = tf.constant([[11, 12, 13, 14],[21, 22, 23, 24],[31, 32, 33, 4],[41, 42, 43, 44]])
t = tf.convert_to_tensor(x)
print "tensor", "*" * 16
print t.eval()
print "split x_axix", "*" * 10
# split第一个参数指定的沿某轴进行分割，
# 第二个参数分成几个，tensor再这个方向上的维度值应该能被此参数整除
# 
a0 = tf.split(t, num_or_size_splits=2, axis=0)
print type(a0)
print a0[0].eval()
print a0[1].eval()
# print tf.split(0, 2, t)[0].eval()
# print "*" * 23
# print tf.split(0, 2, t)[1].eval()
# print "split y_axix", "*" * 10
# for i in range(4):
#     print tf.split(1, 4, t)[i].eval()
sess.close()

# import tensorflow as tf
state = tf.Variable(0.0,dtype=tf.float32)
one = tf.constant(1.0,dtype=tf.float32)
new_val = tf.add(state, one)
update = tf.assign(state, new_val) #返回tensor， 值为new_val
update2 = tf.assign(state, 10000)  #没有fetch，便没有执行
init = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)
    for _ in range(3):
        print sess.run(update)