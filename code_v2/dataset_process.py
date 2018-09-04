# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 16:59:02 2017

@author: ThinkCentre
"""
#测试数据生成
#之后作为数据预处理程序，构建适合于本网络的数据输入格式
import tensorflow as tf
import constant

xs = []
for i in range(constant.INPUT_PART_NUMBER):
    str1 = 'xs' + str(i) + '=tf.get_variable(\'xs'+str(i)+\
    '\',[1,constant.INPUT_PART_NODE],dtype=tf.float32,initializer = tf.truncated_normal_initializer(10,0.1),trainable=False)'
    exec(str1)
    str2 = 'xs.append(xs'+str(i)+')'
    exec(str2)
ys = tf.get_variable('ys',[1,1],dtype=tf.float32,initializer = tf.constant_initializer(1),trainable = False)
        #xs为一个输入数据列表，ys为所有包含数据的共同标签
#with tf.Session() as sess:
#    init_op = tf.global_variables_initializer()
#    sess.run(init_op)
#    #print(xs)
#   print(sess.run(xs))