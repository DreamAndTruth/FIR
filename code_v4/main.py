# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 21:04:43 2017

@author: ThinkCentre
"""
import tensorflow as tf
import dataset_process
import train

#载入数据  运行函数
train.train(dataset_process.xs)