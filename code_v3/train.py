
# coding: utf-8

# In[ ]:
import inference
import tensorflow as tf
import os
from tensorflow.examples.tutorials.mnist import input_data
import inference
import dataset_process
import constant
#加载inference.py中的函数和常量


# In[ ]:
# 常量定义及其引用将置于constant中

BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8
#基础学习率——用于进行指数型衰减学习率
LEARNING_RATE_DECAY = 0.99
#学习率的衰减率——一般接近1
REGULARIZATION_RATE = 0.0001
#描述模型复杂度的正则化项在损失函数中的系数
TRAINING_STEPS = 30000
#总的训练迭代次数
MOVING_AVERAGE_DECAY = 0.99 
#滑动平均衰减率
MODEL_SAVE_PATH = './ModelSave'
#此文件夹需为已经存在的文件夹
MODEL_NAME = 'model.ckpt'


# In[ ]:

def train():
    
    #数据集将随dataset_process导入
    input_tensor = tf.placeholder(tf.float32,[None,constant.INPUT_PART_NODE],name = 'input_tensor')
    label = tf.placeholder(tf.float32,[None,constant.OUTPUT_NODE],name = 'label')
    #定义输入,输出placeholder
    
    #——————————————————————生成测试数据—————————————此处为debug阶段使用（可删除）———————
    for i in range(constant.INPUT_PART_NUMBER):
        str1 = 'xs' + str(i) + '=tf.get_variable(\'xs'+str(i)+\
        '\',[INPUT_PART_NODE],initializer = tf.truncated_normal_initializer(0.1),trainable=False)'
        exec(str1)
        str2 = 'xs.append(xs'+str(i)+')'
        exec(str2)
    ys = tf.get_variable('ys',[1],initializer = tf.constant_initializer(1),trainable = False)
    
    #————————————————————————xs为一个输入数据列表，ys为所有包含数据的共同标签
    
    
    regularizer = tf.contrib.layers.l2_regularizer(constant.REGULARIZATION_RATE)
    #定义正则化类，此处使用l2正则化
    
    #————————————————————数据处理和输入————————————————————————————
    #在进行前向传播前需要对数据进行重组
    #input_tensor = [x1,x2,...x13]
    
    prediction = inference.inference(input_tensor,regularizer)
    #使用inference中的函数，计算前向传播结果
    
    #———————————————————计算滑动平均————————————不理解滑动平均对训练结果的优化———————————————————
    global_step = tf.Variable(0,trainable = False)
    #使得global_step不可训练
    variables_averages = tf.train.ExponentialMovingAverage(constant.MOVING_AVERAGE_DECAY,global_step)
    #定义滑动平均类
    variables_averages_op = variables_averages.apply(tf.trainable_variables())
    #对所有可进行训练的变量应用滑动平均
    
    #————————————————————————计算损失函数———————————————————————————
    
    #cross_entroy = tf.nn.sparse_softmax_cross_entroy_with_logits(y,tf.argmax(y_,1）
    ————————————————————————————————————————————————————————————————————————————————
    cross_entroy = tf.nn.sparse_softmax_cross_entroy_with_logits(prediction,label)
    #此函数有对于交叉熵计算的加速——不太理解，可以自己计算损失函数
    #计算交叉熵 此函数包括对输出进行softmax操作
    #所以，在输出层之后可去掉softmax层
    
    cross_entroy_mean = tf.reduce_mean(cross_entroy)
    loss_regularization = tf.add_n(tf.get_collection('losses'))
    loss = cross_entroy_mean + loss_regularization
    #损失函数为平均交叉熵与正则化项之和，在集合losses中获取所有参数的正则化损失
    
    learning_rate = tf.train.exponential_decay(constant.LEARNING_RATE_BASE,\
                                               global_step,dataset_process.DATASTES_NUMBER / constant.BATCH_SIZE,\
                                               constant.LEARNING_RATE_DECAY)
    #生层指数衰减型学习率
    #num_datastes / BATCH_SIZE 过完所有的训练数据所需要的迭代次数
    #使得学习率在同一次训练所有数据的时候保持不变，学习率呈阶梯状变化
    
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step = global_step)
    #global_step会自动更新
    train_op = tf.group(train_step,variables_averages_op)
    #在每次对参数进行训练之后，再对所有参数进行滑动平均操作，整合在一块进行操作
    
    
    #————————————————————————持久化模型训练结果————————————————————
    saver = tf.train.Saver()
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        #初始化所有变量
        for i in range(TRAINING_STEPS):
            #INPUT_PART_NUMBER条数据构成一个输入列表
            #此时构建的前向传播网络为单条数据进行训练
            #-------填充新BATCH_SIZE数据-------
            _,loss_value,step=sess.run([train_op,loss,global_step],feed_dict={x:xs, y_:ys})
            if i % 1000 == 0:
                print('After %d training step(s), loss on training batch is %g.' %(step,loss_value))
                
                saver.save(sess,os.path.join(MODEL_SAVE_PATH,MODEL_NAME),global_step = global_step )


# In[ ]:

def main(argv = None):
    #datasets = input_data.read_data_sets('./MNIST_data',one_hot = True)
    #train(datasets)
    train
if __name__ == '__main__':
    tf.app.run()
    #主程序入口，会自动调用上面定义的main（）函数
