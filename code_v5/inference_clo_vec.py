
# coding: utf-8

# In[2]:

import tensorflow as tf
import constant
import dataset_process

# In[4]:
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
def inference  (input_tensor,regularizer):
    '''
            *****************************************************
            第i层与第i+1层之间的权值，与第i+1层的偏置：
            *将权值矩阵构建成三维矩阵形式，每层的权值为一个三维矩阵:
            第i层与第i+1层之间的权值，与第i+1层的偏置：
                size:[N_i+1,N_i,Layer_i_Time_Delay]
            *偏置构建为一个二维矩阵：
                size:[N_i+1,Layer_i+1_Part_Number] 
                偏置矩阵是否过多？？？
            *数据构建未知？？？
            
            *****************************************************
    '''
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
                                 input_tensor[i,(i+constant.LAYER2_TIME_DELAY)\
                                              *constant.LAYER1_PART_NODE,:])
                #:不可更改为沿着0维进行切片，否则生成shape=(a,)
                if i==0:
                    layer2_input=tf.Variable(shape(temp))
                    copy = tf.assign(layer2_input,temp)
                else:
                    concat = tf.concat([layer2_input,temp],0)
        
                
        
        layer1_output = []
        #构建第一层输出的列表
        layer1_input_zeros = tf.zeros([1,constant.LAYER1_PART_NODE])
        layer1_input_part = tf.Variable(tf.zeros([1,constant.LAYER1_PART_NODE]),name='layer1_input_part')
        for i in range(constant.INPUT_PART_NUMBER - constant.INPUT_TIME_DELAY + 1):
            #下层的计算需要所有上一层的计算结果，需要将一个——’Batch‘—此处不是真正的batch—中的所有数据进行输入计算
            
            layer1_input_part = tf.assign(layer1_input_part,layer1_input_zeros)
                    #每轮循环前对layer1_input进行初始化
                    
            for j in range(constant.INPUT_TIME_DELAY):
                layer1_input_matmul = tf.matmul(input_tensor[i+j],layer1_weights[j])
                layer1_input_part = tf.assign_add(layer1_input_part,layer1_input_matmul)
                #此处体现同一层网络上的时间权值共享特性
                #j取值范围从0-4,0-TIME_DELAY_
                
                #*******偏置共享，所以每层的偏置向量只有一个
            layer1_input_part = tf.assign_add(layer1_input_part,layer1_biases)
            #layer1_input_part = layer1_input_part + layer1_biases
            str3 = 'layer1_output_part' + str(i) + ' = tf.get_variable(\'layer1_output_part' + str(i) +\
                                              '\',[1,constant.LAYER1_PART_NODE],initializer = tf.constant_initializer(0.0),trainable = False)'
            exec(str3)
            #对于变量的可训练性进行控制
            str4 = 'layer1_output_part' + str(i) + ' = tf.nn.relu(layer1_input_part)'
            #i——也是此次计算结果在下一层中所在的index
            exec(str4)
            str5 = 'layer1_output.append(layer1_output_part' + str(i) + ')'
            exec(str5)
        #—————————————————注释与旧代码—————————————————————————————
        #for i in range(INPUT_TIME_DELAY):
            #str2 = 'layer1_input = layer1_input+tf.matmul(input_part['+str(i)+'],weights'+str(i)+')'
            # i 取值范围从0-4,0-TIME_DELAY_
            #exec(str2)
       #layer1_input = layer1_input + biases
        #str3 = 'layer1_output_part' + str(next_layer_index) + ' = tf.nn.relu(layer1_input)'
        #exec(str3)
        #——————————————————————————————————————————————
        
        #layer1_output_part-----[0,1,2,3,4.....]--------加入序列------ = tf.nn.relu(layer1_input)
        #无法从input_tensor中的第一个名称获取序列数，
        #参数传递进来后成为实际参数，名称没有意义
        #使用exec函数
        
        #example : layer1_part = tf.nn.relu(tf.matmul(input_part1,weights1) + tf.matmul(input_part2,weights2) \
        # + tf.matmul(input_part3,weights3) + tf.matmul(input_part4,weights4) \
        # + tf.matmul(input_part5,weights5) + biases1)
        #计算LAYER1中的第n个模块的输出值
        #n的取值取决于输入张量的part下标
        
        #使用循环对输入时间相关序列进行处理，每INPUT_TIME_DELAY个进行输入（具有交叉项）
        #对输入进行循环变化i1=i2,i2=i3.....i5=i_new
        for i in range(5):
            layer1_weights.pop()
    #————————————————————————隐藏层（1）—————————————————————————————
    
    with tf.variable_scope('layer2'):
        #此处生成的所有变量均存在与集合layer1当中
        layer2_weights = []
        #生成一个空列表
        for i in range(constant.LAYER1_TIME_DELAY):
            str1 = 'weights'+str(i)+'=get_weight_variable(\'weights' + str(i)+\
                                 '\',[constant.LAYER1_PART_NODE,constant.LAYER2_PART_NODE],regularizer)'
            exec(str1)
            str2 = 'layer2_weights.append(weights' + str(i) + ')'
            exec(str2)
            
        layer2_biases = tf.get_variable('layer2_biases',[1,constant.LAYER2_PART_NODE],\
                                        initializer = tf.constant_initializer(0.0))
            
            
#        layer2_biases = []
#        for i in range(constant.LAYER1_TIME_DELAY):
#            str1 = 'biases' + str(i) +' = tf.get_variable(\'biases' + str(i) + '\',[constant.LAYER2_PART_NODE],initializer = tf.constant_initializer(0.0))'
#            exec(str1)
#            str2 = 'layer2_biases.append(' + 'biases' + str(i) + ')'
#            exec(str2)
            
        layer2_input_part = tf.get_variable('layer2_input_part',[1,constant.LAYER2_PART_NODE],\
                                            initializer = tf.constant_initializer(0.0),trainable = False)
        layer2_output = []

        layer2_input_zeros = tf.zeros([1,constant.LAYER2_PART_NODE])    
        for i in range(constant.LAYER1_PART_NUMBER - constant.LAYER1_TIME_DELAY + 1):
            #下层的计算需要所有上一层的计算结果，需要将一个——’Batch‘—此处不是真正的batch—中的所有数据进行输入计算
            layer2_input_part = tf.assign(layer2_input_part,layer2_input_zeros)
            
            for j in range(constant.LAYER1_TIME_DELAY):
                layer2_input_matmul = tf.matmul(layer1_output[i+j],layer2_weights[j])
                
                layer2_input_part = tf.assign_add(layer2_input_part,layer2_input_matmul)
                
            layer2_input_part = tf.assign_add(layer2_input_part,layer2_biases)
            #__________________________需要循环生成
            str3 = 'layer2_output_part' + str(i) + ' = tf.get_variable(\'layer2_output_part' + str(i) + \
            '\',[1,constant.LAYER2_PART_NODE],initializer = tf.constant_initializer(0.0),trainable = False)'
            exec(str3)
            
            str4 = 'layer2_output_part' + str(i) + ' = tf.nn.relu(layer2_input_part)'
            exec(str4)
            str5 = 'layer2_output.append(layer2_output_part' + str(i) + ')'
            exec(str5)
        for j in range(5):
            layer2_weights.pop()
    
    #————————————————————————————（输出层）——————————————————————————
    
    with tf.variable_scope('layer3'):
        layer3_weights = []
        for i in range(constant.LAYER2_TIME_DELAY):
            str1 = 'weights'+str(i)+'=get_weight_variable(\'weights' + str(i)+\
                                 '\',[constant.LAYER2_PART_NODE,constant.OUTPUT_NODE],regularizer)'
            exec(str1)
            str2 = 'layer3_weights.append(weights' + str(i) + ')'
            exec(str2)
            
        layer3_biases= tf.get_variable('layer3_biases' ,[1,constant.OUTPUT_NODE],initializer = tf.constant_initializer(0.0))
        layer3_input_part = tf.get_variable('layer3_input_part',[1,constant.OUTPUT_NODE],initializer = tf.constant_initializer(0.0),trainable = False)
        
        #此处的外层循环可以去掉——输出层没有时间延迟
        layer3_input_zeros = tf.zeros([1,constant.OUTPUT_NODE])
        layer3_input_part = tf.assign(layer3_input_part,layer3_input_zeros)
        for i in range(constant.LAYER2_TIME_DELAY):
            layer3_input_matmul = tf.matmul(layer2_output[i],layer3_weights[i])
            layer3_input_part = tf.assign_add(layer3_input_part,layer3_input_matmul)
            
        layer3_input_part = layer3_input_part + layer3_biases
        output_tensor = tf.get_variable('output_tensor',[1,constant.OUTPUT_NODE],initializer = tf.constant_initializer(0.0),trainable = False)
        output_tensor = tf.nn.relu(layer3_input_part)
        
        for i in range(5):
            layer3_weights.pop()
    return output_tensor
#载入数据  运行函数
   

# In[9]:
#out = inference(dataset_process.xs,None)
#    
#with tf.Session() as sess:
#    writer = tf.summary.FileWriter('E:\FIR\log',sess.graph)
#writer.close()

#output = inference(dataset_process.xs,None)
#with tf.Session() as sess:
#    sess.run(tf.global_variables_initializer())
#    print(sess.run(output))
#    writer = tf.summary.FileWriter('F:FIR\log',sess.graph)
#writer.close()