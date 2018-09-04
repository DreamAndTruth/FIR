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


#输入随机数，输出为常数的拟合
'''
def one_hot(loc):
    temp = np.zeros([1,20])
    temp[0,loc-1] = 1
    return temp
x = one_hot(5)
'''
rand = np.random.seed(1)
x = np.random.rand(1,50*10)
##input_tensor = tf.get_variable('input_tensor',[1,50*10],\
                           ##initializer = tf.truncated_normal_initializer(1,0.1),trainable=False)

#第一层输出:
#output_tensor = tf.get_variable('output_tensor',[270,1],initializer = tf.ones_initializer(),trainable=False)

#第二层输出：
#output_tensor = tf.get_variable('output_tensor',[100,1],initializer = tf.ones_initializer(),trainable=False)

#最后一层输出：
#output_tensor = tf.get_variable('output_tensor',initializer = \
#                                [1.0,1.0,0.0,2.0,5.0],trainable=False)
y = np.array([1.0,1.0,0.0,2.0,5.0])
y = np.reshape(y,[1,5])
size = np.ones([1,5])
input_tensor = np.outer(size,x)
output_tensor = np.outer(size,y)

input_tensor = tf.convert_to_tensor(input_tensor,dtype = tf.float32)
output_tensor = tf.convert_to_tensor(output_tensor,dtype = tf.float32)
'''
产生正弦信号数据，参数与LSTM预测参数相同（pre14.py）

'''
#def generate_data(seq):
#    X = []
#    y = []
#    # 序列的第i项和后面的TIMESTEPS-1项合在一起作为输入；第i + TIMESTEPS项作为输
#    # 出。即用sin函数前面的TIMESTEPS个点的信息，预测第i + TIMESTEPS个点的函数值。
#    for i in range(len(seq) - TIMESTEPS):
#        #因为y的元素是下一个时刻的预测值，所以range中的计算不需要加1，否则最后一组数据将没有target
#        X.append([seq[i: i + TIMESTEPS]])
#        #列表中的元素也是列表
#        y.append([seq[i + TIMESTEPS]])
#    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)  
#
## 用正弦函数生成训练和测试数据集合。
#test_start = (TRAINING_EXAMPLES + TIMESTEPS) * SAMPLE_GAP
##100.1
#test_end = test_start + (TESTING_EXAMPLES + TIMESTEPS) * SAMPLE_GAP
##100.1+10.1 = 110.2
#train_X, train_y = generate_data(np.sin(np.linspace(
#    0, test_start, TRAINING_EXAMPLES + TIMESTEPS, dtype=np.float32)))
##此处的linspace中num的计算不需要减一，原因和上面类似
#test_X, test_y = generate_data(np.sin(np.linspace(
#    test_start, test_end, TESTING_EXAMPLES + TIMESTEPS, dtype=np.float32)))





