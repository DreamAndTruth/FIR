
import tensorflow as tf
import constant
import dataset_process
import numpy as np

# In[]:
#获取权值,偏置矩阵
def get_weights_variable(name,shape,regularizer):
    
    weights = tf.get_variable(name,shape,dtype=tf.float32,\
                              initializer = tf.truncated_normal_initializer\
                              (stddev = 0.1),trainable = True)
    #layer_i的权值矩阵，使用切片进行操作
    #此处时间延迟下标与Word相同
    #将weights初始化为标准差为0.1的服从正态分布的随机数，当随机出的数值偏离平均值两倍的标准差时，将重新随机
    if regularizer != None:
        tf.add_to_collection('losses',regularizer(weights))
        #当传入正则化类时，将weights的正则化结果张量传入losses集合当中进行管理
    return weights
    
def get_biases_variable(name,shape):
    biases = tf.get_variable(name,shape,dtype=tf.float32,\
                             initializer=tf.constant_initializer(0.0),trainable=False)
    
    #*****trainable
    return biases
# In[9]:
'''
            ******************************************************
            *前向传播网络：
            **第i层与第i+1层之间的权值，重新构建为一个二维矩阵：
            Weights = [W0,W1,W2...WD_i+1]
            D_i+1：第i+1层的时间延迟
            **将第i层的输入数据，重新构建为一个列向量：
            Data = [[x0],[x1]...[xN_i]]
            N_i：第i层的模块个数
            **将第i+1层的偏置，重新构建为一个列向量：
            Biases = [[bias1],[bias2]...[biasN_i]]
            ******************************************************
'''   
def inference(layer_name,\
              input_tensor,\
              now_part_node,\
              next_part_node,\
              now_part_number,\
              next_part_number,\
              next_time_delay,\
              regularizer):
    '''
            ******************************************************
            layer_name:当前层的名称 string
            input_tensor:当前层输入 列向量
            now_part_node:当前层每个模块的节点数目 scalar
            next_part_node:下一层每个模块的节点数目 scalar
            now_part_number:当前层模块数目 scalar
            next_part_number:下一层模块数目 sclar
            next_time_delay:下一层时间延迟 sclar
            regularizer:正则化项
            ******************************************************
    '''

    with tf.variable_scope(layer_name):
        weights = get_weights_variable('weights',[now_part_node*\
                                                  next_time_delay,\
                                                  next_part_node],\
                                                  regularizer)
        biases = get_biases_variable('biases',\
                                     [1,next_part_node*\
                                      next_part_number])
        for i in range(now_part_number - next_time_delay + 1):
            temp = tf.matmul(input_tensor[:,i*now_part_node:(i+next_time_delay)*\
                                          now_part_node],weights)
            if i == 0:
                if (now_part_number - next_time_delay + 1) == 1:
                    temp = tf.add(temp,biases)
                    return temp
                else:
                    next_layer_input = tf.Variable(tf.random_normal(tf.shape(temp),stddev=1,seed=1),trainable = False)
                    next_layer_input = tf.assign(next_layer_input,temp)
            else:
                next_layer_input = tf.concat([next_layer_input,temp],1)
        next_layer_input = tf.add(next_layer_input,biases)
    return next_layer_input
    
# In[]:
layer2_input_ = inference('layer1',dataset_process.input_tensor,8448,4000,25,16,10,None)
layer2_input = tf.nn.relu(layer2_input_)
layer3_input_ = inference('layer2',layer2_input,4000,1000,16,9,8,None)
layer3_input = tf.nn.relu(layer3_input_)
                
layer4_input_ = inference('layer3',layer3_input,1000,500,9,5,5,None)
layer4_input = tf.nn.relu(layer4_input_)
                
layer5_input_ = inference('layer4',layer4_input,500,100,5,3,3,None)
layer5_input = tf.nn.relu(layer5_input_)
                
output_ = inference('layer5',layer5_input,100,20,3,1,3,None)
output = tf.nn.relu(output_)
# In[]:

global_step = tf.Variable(0,trainable = False)

#loss = tf.nn.softmax_cross_entropy_with_logits(labels = output,\
#                                                     logits = dataset_process.output_tensor)
square = tf.square(output-dataset_process.output_tensor)
loss = tf.reduce_mean(square)

#cross_entory = -tf.reduce_mean(dataset_process.output_tensor*tf.log(tf.clip_by_value(out,1e-10,1e10)))
#需要在前一层使用softmax使得函数值落在[0,1]

learning_rate = constant.LEARNING_RATE_BASE
train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss,global_step = global_step)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    loss_val = 0.0
    for i in range(constant.TRAINING_STEPS):
        _,loss_val,step = sess.run([train_step,loss,global_step])
        if i%1 == 0:
            print('(%d,%g)'%(step,loss_val))
            #weights = sesss.run(weights)
    print(sess.run(output))
    print(sess.run(loss))
    writer = tf.summary.FileWriter('E:\practice\log',sess.graph)
    writer.close()
