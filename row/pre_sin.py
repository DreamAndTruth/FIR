# -*- coding: utf-8 -*-
"""
Created on Thu May 17 03:25:35 2018

@author: DREAM
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import inference
import constant

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
#此处的linspace中num的计算不需要减一，原因和上面类似
test_X, test_y = generate_data(np.sin(np.linspace(
    test_start, test_end, TESTING_EXAMPLES + TIMESTEPS, dtype=np.float32)))



'''
在使用FIR对sin函数进行预测的时候，只需要更改此处的模型。
在fir_model（X,y，is_training）当中返回预测值，loss以及train_op
'''
def lstm_model(X, y, is_training):
    # 使用四层fir网络结构
    inputs = tf.reshape(X,[tf.shape(X)[0],tf.shape(X)[2]])
    layer2_input_ = inference.inference('layer1',inputs,constant.LAYER1_PART_NODE,\
                      constant.LAYER2_PART_NODE,constant.LAYER1_PART_NUMBER,constant.LAYER2_PART_NUMBER,\
                      constant.LAYER2_TIME_DELAY,None)
    layer2_input = tf.nn.relu(layer2_input_)

    layer3_input_ = inference.inference('layer2',layer2_input,constant.LAYER2_PART_NODE,constant.LAYER3_PART_NODE,\
                         constant.LAYER2_PART_NUMBER,constant.LAYER3_PART_NUMBER,constant.LAYER3_TIME_DELAY,\
                         None)
    layer3_input = tf.nn.relu(layer3_input_)

    out = inference.inference('layer3',layer3_input,constant.LAYER3_PART_NODE,constant.OUTPUT_NODE,\
                    constant.LAYER3_PART_NUMBER,constant.OUTPUT_PART_NUMBER,\
                    constant.LAYER4_TIME_DELAY,None)
    
    # 只在训练时计算损失函数和优化步骤。测试时直接返回预测结果。
    if not is_training:
        return out, None, None
        
    # 计算损失函数。
    loss = tf.losses.mean_squared_error(labels=y, predictions=out)

    # 创建模型优化器并得到优化步骤。
    train_op = tf.contrib.layers.optimize_loss(
        loss, tf.train.get_global_step(),
        optimizer="Adagrad", learning_rate=0.1)
    return out, loss, train_op
    
def run_eval(sess, test_X, test_y):
    # 将测试数据以数据集的方式提供给计算图。
    ds = tf.data.Dataset.from_tensor_slices((test_X, test_y))
    ds = ds.batch(1)
    X, y = ds.make_one_shot_iterator().get_next()
    
#    X_shape = tf.shape(X_)
#    
#    X = tf.reshape(X_,[X_shape[0],X_shape[2]])
    
    # 调用模型得到计算结果。这里不需要输入真实的y值。
    with tf.variable_scope("model", reuse=True):
        prediction, _, _ = lstm_model(X, [0.0], False)
    
    # 将预测结果存入一个数组。
    predictions = []
    labels = []
    for i in range(TESTING_EXAMPLES):
        p, l = sess.run([prediction, y])
        predictions.append(p)
        labels.append(l)

    # 计算rmse作为评价指标。
    predictions = np.array(predictions).squeeze()
    labels = np.array(labels).squeeze()
    rmse = np.sqrt(((predictions - labels) ** 2).mean(axis=0))
    print("Root Mean Square Error is: %f" % rmse)
    
    #对预测的sin函数曲线进行绘图。
    plt.figure()
    plt.plot(predictions, label='predictions')
    plt.plot(labels, label='real_sin')
    plt.legend()
    plt.show()

# 将训练数据以数据集的方式提供给计算图。
ds = tf.data.Dataset.from_tensor_slices((train_X, train_y))
ds = ds.repeat().shuffle(1000).batch(BATCH_SIZE)
X, y = ds.make_one_shot_iterator().get_next()

# 定义模型，得到预测结果、损失函数，和训练操作。
with tf.variable_scope("model"):
    _, loss, train_op = lstm_model(X, y, True)
    
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    # 测试在训练之前的模型效果。
    print ("Evaluate model before training.")
    run_eval(sess, test_X, test_y)
    
    # 训练模型。
    for i in range(constant.TRAINING_STEPS):
        _, l = sess.run([train_op, loss])
        if i % 1000 == 0:
            print("train step: " + str(i) + ", loss: " + str(l))
    
    # 使用训练好的模型对测试数据进行预测。
    print ("Evaluate model after training.")
    run_eval(sess, test_X, test_y)
    
