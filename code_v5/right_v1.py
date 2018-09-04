
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
                             initializer=tf.constant_initializer(0.0),trainable=True)
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
def inference  (input_tensor,regularizer):

    with tf.variable_scope('layer1'):
        weights = get_weights_variable('weights',\
                                       [constant.LAYER2_PART_NODE,\
                                        constant.LAYER1_PART_NODE*\
                                        constant.LAYER2_TIME_DELAY],\
                                        regularizer)
        biases = get_biases_variable('biases',\
                                 [constant.LAYER2_PART_NODE*\
                                  constant.LAYER2_PART_NUMBER,1])
        
        for i in range(constant.LAYER1_PART_NUMBER - constant.LAYER2_TIME_DELAY + 1):
            #input_tensor=[N_1*Layer1_PART_NUMBER,1]
            #layer1_input=[N_2*Layer2_Part_Number,1]
            #每次输数据为能够计算所有下层结果的时间长度数据所构成矩阵
            #LAYER1_PART_NUMBER实际为输入时间序列长度
                temp = tf.matmul(weights,\
                                 input_tensor[i*constant.LAYER1_PART_NODE:(i+constant.LAYER2_TIME_DELAY)\
                                              *constant.LAYER1_PART_NODE,:])
                #:不可更改为沿着0维进行切片，否则生成shape=(a,)
                if i==0:
                    print(i)
                    layer2_input = tf.Variable(tf.random_normal(tf.shape(temp),stddev=1,seed=1))
                    layer2_input = tf.assign(layer2_input,temp)
                else:
                    print(i)
                    layer2_input = tf.concat([layer2_input,temp],0)
#tf.assign() 和tf.concat() 操作会返回操作的结果，需要给等号左边写上需要赋值的变量
    #layer2_input = tf.nn.relu(layer2_input)
    return layer2_input,weights

# In[]:
out,weights = inference(dataset_process.input_tensor,None)

global_step = tf.Variable(0,trainable = False)

loss = tf.reduce_mean(tf.square(out-dataset_process.output_tensor))

learning_rate = constant.LEARNING_RATE_BASE
train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss,global_step = global_step)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    loss_val = 0.0
    for i in range(constant.TRAINING_STEPS):
        _,loss_val,step = sess.run([train_step,loss,global_step])
        if i%1000 == 0:
            print('(%d,%g)'%(step,loss_val))
            #weights = sesss.run(weights)
            print(sess.run(weights[:,0]))
    print(sess.run(tf.nn.relu(out)))
    writer = tf.summary.FileWriter('E:\practice\log',sess.graph)
    writer.close()
