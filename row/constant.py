# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 17:09:51 2017

@author: ThinkCentre
"""
#常量定义
# In[3]:

#定义神经网络结构相关参数
#命名：层名称_常量作用相关描述

LAYER2_TIME_DELAY = 5
#第1层时间延迟长度（下标与Word相同）
LAYER3_TIME_DELAY = 4
#第2层时间延迟长度
LAYER4_TIME_DELAY = 3

OUTPUT_PART_NUMBER = 1

#存在权值共享：同一个动作，无论发生在什么时间段之内，均可以被检测出,且检测结果相同
#时间上的“位移不变性”——区别与联系空间上的位移不变性

#输入时，将INPUT_TIME_DELAY个数据组合起来，作为一条数据进行处理
#将上述组合数据进行Batch训练

#实现上的两种方式：
#1）对数据进行预处理，将INPUT_TIME_DELAY个时间序列相关数据进行组合，
#如：[x1,x2,x3],[x2,x3,x4]...
#若为图像数据，则构造三维矩阵进行处理
#2）在网络每一层上进行分块处理（现在的实现方式）
#将每个时间片段的数据构成一个模块

'''
在网络节点数目随着层数变化而变化的同时，每层的时间延迟也随之改变
网络架构由多种参数决定
'''

LAYER1_PART_NODE = 1
#每一时刻的输入数据构成一个模块
#数据为一维向量[INPUT_PART_NODE x 1]  ***数据的向量化表示
#每个模块表示一个时刻的向量化数据
#输入层的每一个模块中所包含的神经元数目
LAYER2_PART_NODE= 3
#第1层的每一个模块中所包含的神经元数目
LAYER3_PART_NODE = 2

OUTPUT_NODE = 1

#现在的每个模块的神经元数目将导致网络每层的神经元数目逐层递减
#应该加大中间层每个模块的神经元数目，使得网络呈现橄榄形

LAYER1_PART_NUMBER = LAYER2_TIME_DELAY +LAYER3_TIME_DELAY + LAYER4_TIME_DELAY - 2 
#输入层的模块数目（与时间延迟相对应）
#每层神经元数目为模块数目×模块中神经元的数目
LAYER2_PART_NUMBER = LAYER3_TIME_DELAY + LAYER4_TIME_DELAY - 1 
#第1层的模块数目
LAYER3_PART_NUMBER = LAYER4_TIME_DELAY 

# In[ ]:

BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.0001
#基础学习率——用于进行指数型衰减学习率
LEARNING_RATE_DECAY = 0.99
#学习率的衰减率——一般接近1
REGULARIZATION_RATE = 0.0001
#描述模型复杂度的正则化项在损失函数中的系数
TRAINING_STEPS = 10000
#总的训练迭代次数
MOVING_AVERAGE_DECAY = 0.99 
#滑动平均衰减率
MODEL_SAVE_PATH = './ModelSave'
#此文件夹需为已经存在的文件夹
MODEL_NAME = 'model.ckpt'

