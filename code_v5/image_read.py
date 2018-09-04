#coding=utf-8
"""
Created on Thu Jan 18 21:16:25 2018

@author: ThinkCentre
"""



import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as im

import pandas as pd
import skimage.io as io

data_dir = '.data\\1\\001'

#data_dir可为多个路径或格式由冒号隔开表示，以读取多个路径下文件
#data_dir1:data_dir2:.....:data_dirN
string = data_dir+'\\*.png'
coll = io.ImageCollection(string)
#第二个参数默认为None，即图片读取
print(len(coll))
print(coll[0])
print(np.shape(coll[0]))
#io.imshow(coll[0])

mat = io.concatenate_images(coll)
#构成(75,240,352)的三维矩阵

#for i in range(len(coll)):
print(mat.shape)    
#np.savetxt('test.csv',mat,delimiter = ',')
io.imshow(mat[8,:,:])

mat2 = np.reshape(mat,[75,240*352])
#构成(75,84480)矩阵

mat3 = np.reshape(mat2,[75*240*352,1])
#构成向量数据（按列排列数据）





'''
im读取图片
import matplotlib.image as im
'''

#PATH = '.\\test_1.png'
#文件路径分隔符应写为双斜线
#pic = im.imread(PATH)
#plt.imshow(pic)
#print(np.shape(pic))








'''tf读取图片具有编码错误'''
#image = tf.gfile.FastGFile(PATH,'r').read()
#with tf.Session() as sess: 
#    img_data = tf.image.decode_png(image)
#    print(img_data.eval())
#    
#    plt.imshow(img_data.eval())
#    plt.show()
#    
#    
#    img_data = tf.image.convert_image_dtype(img_data,dtype = tf.float32)
#    
#    
#    encoded_img = tf.image.encode_png(img_data)
#    with tf.gfile.GFile(PATH,'wb') as f:
#        f.write(encoded_img.eval())