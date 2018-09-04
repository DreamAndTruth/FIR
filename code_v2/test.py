# -*- coding: utf-8 -*-
"""
Created on Sat Sep 16 20:39:09 2017

@author: ThinkCentre
"""

import tensorflow as tf

x = tf.get_variable('x',[1,2],initializer = tf.constant_initializer(1.2)
y = tf.ones([1,2])

tf.assign_add(x,x+y)
