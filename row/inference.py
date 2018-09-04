

# In[]:
import tensorflow as tf
import constant
#下次更改为config
import dataset_process
import numpy as np
# In[]:
#获取权值,偏置矩阵
def get_weights_variable(name,shape,regularizer):
    
    weights = tf.get_variable(name,shape,dtype=tf.float32,\
                              initializer = tf.truncated_normal_initializer\
                              (mean = 0.0,stddev = 0.1),trainable = True)
    #layer_i的权值矩阵，使用切片进行操作
    #此处时间延迟下标与Word相同
    #将weights初始化为标准差为0.1的服从正态分布的随机数，当随机出的数值偏离平均值两倍的标准差时，将重新随机
    if regularizer != None:
        tf.add_to_collection('losses',regularizer(weights))
        #当传入正则化类时，将weights的正则化结果张量传入losses集合当中进行管理
    return weights
    
def get_biases_variable(name,shape):
    biases = tf.get_variable(name,shape,dtype=tf.float32,\
                             initializer=tf.constant_initializer(0.0),trainable=True)
    return biases
# In[9]:
'''
******************************************************
前向传播网络：
第i层与第i+1层之间的权值，构建为一个二维矩阵(按列排列)：
Weights = [W1,W2...WD_i+1]T
D_i+1：第i+1层的时间延迟
将第i层的输出数据，重新构建为一个行向量：
Data = [x1...xN_i]
N_i：第i层的模块个数
将第i+1层的偏置，重新构建为一个行向量：
Biases = [bias1,bias2...biasN_i]
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
*fun：构建当前层到下一层的前向传播实例
*args:
    layer_name:变量空间名称
    input_tensor:输入数据 size:[batch_size,input_size]
    now_part_node:当前层模块节点数目
    next_part_node：下一层模块节点数目
    now_part_number：当前层模块数目
    next_part_number:下一层模块数目
    next_time_delay:下一层时间延迟数目
    regularizer：正则化项
*output:
    next_layer_input:[batch_size,*]
    '''
    with tf.variable_scope(layer_name):
        weights = get_weights_variable('weights',\
                                       [now_part_node*next_time_delay,\
                                        next_part_node],\
                                        regularizer)
        biases = get_biases_variable('biases',\
                                 [1,next_part_node*\
                                  next_part_number])
        #将下一层的所有偏置向量构成一个行向量进行计算
        for i in range(now_part_number - next_time_delay + 1):
            #input_tensor=[N_1*Layer1_PART_NUMBER,1]
            #layer1_input=[N_2*Layer2_Part_Number,1]
            #每次输数据为能够计算所有下层结果的时间长度数据所构成矩阵
            #now_part_number实际为输入时间序列长度
                temp = tf.matmul(input_tensor[:,i*now_part_node:(i+next_time_delay)\
                                              *now_part_node],weights)
                #:不可更改为沿着第1维进行切片，否则生成shape=(,a)
                
                #tf.convert_to_tensor输入可以为(np.array,lists,tensor)
                
                if i==0:
                    if (now_part_number - next_time_delay + 1) == 1:
                        #构建的前向传播为网络的最后一层
                        next_layer_input = temp
                        #return temp
                        #两种方案均可运行
                    else:
                        next_layer_input = tf.Variable(np.zeros([constant.BATCH_SIZE,next_part_node],dtype = np.float32),\
                                                                trainable = False)
                else:
                    next_layer_input = tf.concat([next_layer_input,temp],1)
#tf.assign() 和tf.concat() 操作会返回操作的结果，需要给等号左边写上需要赋值的变量
    return next_layer_input
# In[]:
    #定义生成每层网络的函数
#def fir_nn(layer_name,input_tensor,now_part_node,next_part_node,now_part_number,next_part_number,next_time_delay,regularizer):
#    
#    
#    next_layer_input = inference(layer_name,input_tensor,now_part_node,\
#                                 next_part_node,now_part_number,next_part_number,next_time_delay,regularizer)
#    
#    return next_layer_input   
#    
# In[]：构建四层网络
#layer2_input_ = inference('layer1',dataset_process.input_tensor,constant.LAYER1_PART_NODE,\
#                      constant.LAYER2_PART_NODE,constant.LAYER1_PART_NUMBER,constant.LAYER2_PART_NUMBER,\
#                      constant.LAYER2_TIME_DELAY,None)
##layer2_input = tf.nn.relu(layer2_input_)
#layer2_input = tf.nn.sigmoid(layer2_input_)
#
#layer3_input_ = inference('layer2',layer2_input,constant.LAYER2_PART_NODE,constant.LAYER3_PART_NODE,\
#                         constant.LAYER2_PART_NUMBER,constant.LAYER3_PART_NUMBER,constant.LAYER3_TIME_DELAY,\
#                         None)
##layer3_input = tf.nn.relu(layer3_input_)
#layer3_input = tf.nn.sigmoid(layer3_input_)
#
#out = inference('layer3',layer3_input,constant.LAYER3_PART_NODE,constant.OUTPUT_NODE,\
#                constant.LAYER3_PART_NUMBER,constant.OUTPUT_PART_NUMBER,\
#                constant.LAYER4_TIME_DELAY,None)
#                
#前向传播可以运行，得到[5,1]向量
# In[]:对于前向网络的简单训练

#global_step = tf.Variable(0,trainable = False)
#
#square = tf.square(out-dataset_process.output_tensor)
#loss = tf.reduce_mean(square)
#'''***************'''
#
##cross_entory = -tf.reduce_mean(dataset_process.output_tensor*tf.log(tf.clip_by_value(out,1e-10,1e10)))
##需要在前一层使用softmax使得函数值落在[0,1]
#
#learning_rate = constant.LEARNING_RATE_BASE
#train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss,global_step = global_step)
#
#
#with tf.Session() as sess:
#    sess.run(tf.global_variables_initializer())
#    loss_val = 0.0
#    for i in range(constant.TRAINING_STEPS):
#        _,loss_val,step = sess.run([train_step,loss,global_step])
#        if i%1000 == 0:
#            print('(%d,%g)'%(step,loss_val))
#            #weights = sesss.run(weights)
#    print(sess.run(out))
#    writer = tf.summary.FileWriter('.\log',sess.graph)
#    writer.close()
