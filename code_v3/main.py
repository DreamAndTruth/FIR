# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 21:04:43 2017

@author: ThinkCentre
"""
import tensorflow as tf
import inference
import dataset_process
import train
import constant

#载入数据  运行函数
y = inference.inference(dataset_process.xs,None)
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    
#    print(sess.run(inference.inference))
#    print(sess.run(inference.layer2_input_part))
#    print(sess.run(inference.layer3_input_part))
    print(sess.run(y))
    writer = tf.summary.FileWriter('E:\FIR\log',sess.graph)
writer.close()