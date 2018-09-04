# -*- coding: utf-8 -*-
"""
Created on Sat Oct  7 20:31:05 2017

@author: ThinkCentre
"""


import tensorflow as tf
import constant
import dataset_process

# In[4]:

#获取权值向量
def get_weight_variable(name,shape,regularizer):
    weights = tf.get_variable(name,shape,dtype=tf.float32,initializer = tf.truncated_normal_initializer(stddev = 0.1))
    #将weights初始化为标准差为0.1的服从正态分布的随机数，当随机出的数值偏离平均值两倍的标准差时，将重新随机
    if regularizer != None:
        tf.add_to_collection('losses',regularizer(weights))
        #当传入正则化类时，将weights的正则化结果张量传入losses集合当中进行管理
    return weights


# In[9]:

def inference(input_tensor,regularizer):
   
    with tf.variable_scope('layer1'):
      
        layer1_weights = []
       
        for i in range(constant.INPUT_TIME_DELAY):
            str1 = 'weights'+str(i)+\
            '=get_weight_variable(\'weights' + str(i) +\
                                  '\',[constant.INPUT_PART_NODE,constant.LAYER1_PART_NODE],regularizer)'
            exec(str1)
            str2 = 'layer1_weights.append(weights' + str(i) + ')'
            exec(str2)
           
        
        layer1_biases = tf.get_variable('layer1_biases',[1,constant.LAYER1_PART_NODE],\
                                        dtype=tf.float32,initializer = tf.constant_initializer(0.0))
       
#       
        layer1_input_part = tf.get_variable\
        ('layer1_input_part',[1,constant.LAYER1_PART_NODE],dtype=tf.float32,initializer = \
         tf.constant_initializer(0.0),trainable = False)

        layer1_output = []
      
        layer1_input_0 = tf.zeros([1,constant.LAYER1_PART_NODE])
        layer1_input_part = tf.Variable(tf.zeros([1,constant.LAYER1_PART_NODE]),name='layer1_input_part')
        for i in range(constant.INPUT_PART_NUMBER - constant.INPUT_TIME_DELAY + 1):
          
            layer1_input_part = tf.assign(layer1_input_part,layer1_input_0)
                  
            for j in range(constant.INPUT_TIME_DELAY):
                layer1_matmul = tf.matmul(input_tensor[i+j],layer1_weights[j])
                
                layer1_input_part = tf.assign_add(layer1_input_part,layer1_matmul)
              
            layer1_input_part = tf.assign_add(layer1_input_part,layer1_biases)
           
            str3 = 'layer1_output_part' + str(i) + ' = tf.get_variable(\'layer1_output_part' + str(i) +\
                                              '\',[1,constant.LAYER1_PART_NODE],initializer = \
                                              tf.constant_initializer(0.0),trainable = False)'
            exec(str3)
            
            str4 = 'layer1_output_part' + str(i) + ' = tf.assign(layer1_output_part'+str(i)+',tf.nn.relu(layer1_input_part))'
            
            exec(str4)
            str5 = 'layer1_output.append(layer1_output_part' + str(i) + ')'
            exec(str5)
        return layer1_output
out = inference(dataset_process.xs,None)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(out))
    writer = tf.summary.FileWriter('E:\practice\log',sess.graph)
writer.close()
