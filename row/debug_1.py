# -*- coding: utf-8 -*-
"""
Created on Wed May 30 11:09:30 2018

@author: ThinkCentre
"""

#生成正弦信号，进行前向传播



import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

TIMESTEPS = 10                              # 循环神经网络的训练序列长度(单个样本的时间长度)
BATCH_SIZE = 32                             # batch大小。
TRAINING_EXAMPLES = 10000                   # 训练数据个数。
TESTING_EXAMPLES = 1000                     # 测试数据个数。
SAMPLE_GAP = 0.01                           # 采样间隔。 

def generate_data(seq):
    X = []
    y = []
    # 序列的第i项和后面的TIMESTEPS-1项合在一起作为输入；第i + TIMESTEPS项作为输
    # 出。即用sin函数前面的TIMESTEPS个点的信息，预测第i + TIMESTEPS个点的函数值。
    for i in range(len(seq) - TIMESTEPS):
        #因为y的元素是下一个时刻的预测值，所以range中的计算不需要加1，否则最后一组数据将没有target
        X.append([seq[i: i + TIMESTEPS]])
        #列表中的元素也是列表
        y.append([seq[i + TIMESTEPS]])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32) 

# 用正弦函数生成训练和测试数据集合。
test_start = (TRAINING_EXAMPLES + TIMESTEPS) * SAMPLE_GAP
#100.1
test_end = test_start + (TESTING_EXAMPLES + TIMESTEPS) * SAMPLE_GAP
#100.1+10.1 = 110.2

train_X, train_y = generate_data(np.sin(np.linspace(
    0, test_start, TRAINING_EXAMPLES + TIMESTEPS, dtype=np.float32)))
#返回train_X=[10000,1,10] train_y = [10000,1] np矩阵
#此处的linspace中num的计算不需要减一，原因和上面类似
test_X, test_y = generate_data(np.sin(np.linspace(
    test_start, test_end, TESTING_EXAMPLES + TIMESTEPS, dtype=np.float32)))

train_X_shape = np.shape(train_X)
train_X = np.reshape(train_X,[train_X_shape[0],train_X_shape[2]])
test_X_shape = np.shape(test_X)
test_X = np.reshape(test_X,[test_X_shape[0],test_X_shape[2]])

def generate_batch(batch_size,X,y):
    dataset1 = tf.data.Dataset.from_tensor_slices({'a':X ,'b': y})
#    dataset2 = tf.data.Dataset.from_tensor_slices(y)
#    dataset3 = tf.data.Dataset.zip((dataset1,dataset2))
    dataset = dataset1.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()
#可以将数据以字典的方式进行传递，或者进行zip处理
#如何进行数据随机化抽取处理
xx = generate_batch(1,train_X,train_y)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    x = sess.run(xx['a'])
    print(x)
    #以列表形式返回[x,y]