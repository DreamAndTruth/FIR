# -*- coding: utf-8 -*-
'''
        *************************************************************
        数据读取为本地数据。
        存放位置为代码所在文件夹的上层目录中dataset文件夹
        
        
        需要生成一个迭代器，每次生成一组可训练数据（与input_part_number有关）
        （生成器）
        *************************************************************
'''
import tensorflow as tf
import constant
import numpy as np
import sys

#测试数据生成
#之后作为数据预处理程序，构建适合于本网络的数据输入格式

#xs = []
#for i in range(constant.INPUT_PART_NUMBER):
#    str1 = 'xs' + str(i) + '=tf.get_variable(\'xs'+str(i)+\
#    '\',[1,constant.INPUT_PART_NODE],dtype=tf.float32,initializer = tf.truncated_normal_initializer(1,0.1),trainable=False)'
#    exec(str1)
#    str2 = 'xs.append(xs'+str(i)+')'
#    exec(str2)
#ys = tf.get_variable('ys',[1,constant.OUTPUT_NODE],dtype=tf.float32,initializer = tf.constant_initializer(1.0),trainable = False)
#        #xs为一个输入数据列表，ys为所有包含数据的共同标签
#with tf.Session() as sess:
#    init_op = tf.global_variables_initializer()
#    sess.run(init_op)
#    #print(xs)
#   print(sess.run(xs))


'''
数据集不可训练，变量定义中，需要加入trainable=False

'''
def one_hot(loc):
    temp = np.zeros([1,20])
    temp[0,loc-1] = 1
    return temp
x = one_hot(5)
input_tensor = tf.get_variable('input_tensor',[1,25*84480],\
                           initializer = tf.truncated_normal_initializer(1,0.1),trainable=False)
#第一层输出:
#output_tensor = tf.get_variable('output_tensor',[270,1],initializer = tf.ones_initializer(),trainable=False)

#第二层输出：
#output_tensor = tf.get_variable('output_tensor',[100,1],initializer = tf.ones_initializer(),trainable=False)

#最后一层输出：
output_tensor = tf.get_variable('output_tensor',initializer = \
                                [0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,\
                                 0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],trainable=False)

#with tf.Session() as sess:
#    sess.run(tf.global_variables_initializer())
#    print(sess.run(output_tensor))