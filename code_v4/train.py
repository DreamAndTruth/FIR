
# coding: utf-8

# In[ ]:
import tensorflow as tf
import os
#from tensorflow.examples.tutorials.mnist import input_data
import inference
import dataset_process
import constant
#加载inference.py中的函数和常量



# In[ ]:

def train(dataset):
    
    #数据集将随dataset_process导入
    
#    input_tensor = tf.placeholder(tf.float32,[1,constant.INPUT_PART_NODE],name = 'input_tensor')
#    label = tf.placeholder(tf.float32,[1,constant.OUTPUT_NODE],name = 'label')
    
    #定义输入,输出placeholder
    
    #——————————————————————生成测试数据—————————————此处为debug阶段使用（可删除）———————
#    for i in range(constant.INPUT_PART_NUMBER):
#        str1 = 'xs' + str(i) + '=tf.get_variable(\'xs'+str(i)+\
#        '\',[INPUT_PART_NODE],initializer = tf.truncated_normal_initializer(0.1),trainable=False)'
#        exec(str1)
#        str2 = 'xs.append(xs'+str(i)+')'
#        exec(str2)
#    ys = tf.get_variable('ys',[1],initializer = tf.constant_initializer(1),trainable = False)
    
    #————————————————————————xs为一个输入数据列表，ys为所有包含数据的共同标签
    
    
    #regularizer = tf.contrib.layers.l2_regularizer(constant.REGULARIZATION_RATE)
    #定义正则化类，此处使用l2正则化
    
    #————————————————————数据处理和输入————————————————————————————
    #在进行前向传播前需要对数据进行重组
    #input_tensor = [x1,x2,...x13]
    prediction = inference.inference(dataset,None)
    #可改变参数  None——> regularizer
    #使用inference中的函数，计算前向传播结果
    
    #———————————————————计算滑动平均————————————不理解滑动平均对训练结果的优化———————————————————
    global_step = tf.Variable(0,trainable = False)
    #使得global_step不可训练
    #variables_averages = tf.train.ExponentialMovingAverage(constant.MOVING_AVERAGE_DECAY,global_step)
    #定义滑动平均类
    #variables_averages_op = variables_averages.apply(tf.trainable_variables())
    #对所有可进行训练的变量应用滑动平均
    
    #————————————————————————计算损失函数———————————————————————————
    
    #cross_entroy = tf.nn.sparse_softmax_cross_entroy_with_logits(y,tf.argmax(y_,1）
    #—————————————————————————————————?????????———————————————————————————————————————————————
    
    #cross_entroy = tf.nn.sparse_softmax_cross_entroy_with_logits(prediction,label)
    
    #此函数有对于交叉熵计算的加速——不太理解，可以自己计算损失函数
    #计算交叉熵 此函数包括对输出进行softmax操作
    #所以，在输出层之后可去掉softmax层
    
    cross_entroy = tf.nn.softmax_cross_entropy_with_logits(logits = prediction,labels = dataset_process.ys,name = 'loss')
    cross_entroy = tf.square(prediction - dataset_process.ys)
    loss = tf.reduce_mean(cross_entroy)
    #loss_regularization = tf.add_n(tf.get_collection('losses'))
     #+ loss_regularization
    #损失函数为平均交叉熵与正则化项之和，在集合losses中获取所有参数的正则化损失
    
    learning_rate = constant.LEARNING_RATE_BASE
#    tf.train.exponential_decay(constant.LEARNING_RATE_BASE,\
#                                              global_step,dataset_process.DATASTES_NUMBER / constant.BATCH_SIZE,\
#                                               constant.LEARNING_RATE_DECAY)
    #生层指数衰减型学习率
    #num_datastes / BATCH_SIZE 过完所有的训练数据所需要的迭代次数
    #使得学习率在同一次训练所有数据的时候保持不变，学习率呈阶梯状变化

    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss,global_step = global_step)
    #global_step会自动更新
    #train_op = tf.group(train_step,variables_averages_op)
    #在每次对参数进行训练之后，再对所有参数进行滑动平均操作，整合在一块进行操作
    
    
    #————————————————————————持久化模型训练结果————————————————————
    #saver = tf.train.Saver()
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        #初始化所有变量
        for i in range(constant.TRAINING_STEPS):
            
            #INPUT_PART_NUMBER条数据构成一个输入列表
            #此时构建的前向传播网络为单条数据进行训练
            #-------填充新BATCH_SIZE数据-------
           # _,loss_value,step=sess.run([train_step,loss,global_step],feed_dict={input_tensor:dataset_process.xs, \
#                                       label:dataset_peocess.ys})
            _,loss_value,step= sess.run([train_step,loss,global_step])
            if i % 1000 == 0:
                print('(%d , %g)' %(step,loss_value))
        final_out = sess.run(prediction)
        print(final_out)
        writer = tf.summary.FileWriter('E:\FIR\log',sess.graph)
        writer.close()
                #print(sess.run(prediction))
                #saver.save(sess,os.path.join(MODEL_SAVE_PATH,MODEL_NAME),global_step = global_step )


# In[ ]:

#def main(argv = None):
#    #datasets = input_data.read_data_sets('./MNIST_data',one_hot = True)
#    #train(datasets)
#    train
#if __name__ == '__main__':
#    tf.app.run()
#    #主程序入口，会自动调用上面定义的main（）函数