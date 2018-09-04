# -*- coding: utf-8 -*-
"""
        *************************************************************
        **检测行人的行走角度0——180度  间隔18度 总共十一个角度
        
        数据读取为本地数据。
        存放位置为代码所在文件夹的上层目录中dataset文件夹
        
        
        需要生成一个迭代器，每次生成一组可训练数据（与input_part_number有关）
        （生成器）
        *************************************************************
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import Counter
import random
import os
import sys
sys.path.append('..')
#在路径上加入上层目录表示符

import zipfile

import numpy as np
from six.moves import urllib
import tensorflow as tf

import utils

# Parameters for downloading data
DOWNLOAD_URL = 'http://mattmahoney.net/dc/'
EXPECTED_BYTES = 31344016 
#以字节为单位
DATA_FOLDER = 'data/'
#在程序所在文件夹下寻找 'data/' 子文件夹
FILE_NAME = 'text8.zip'

#文件下载
def download(file_name, expected_bytes):
    """ Download the dataset text8 if it's not already downloaded """
    file_path = DATA_FOLDER + file_name
    if os.path.exists(file_path):
        print("Dataset ready")
        return file_path
    file_name, _ = urllib.request.urlretrieve(DOWNLOAD_URL + file_name, file_path)
    file_stat = os.stat(file_path)
    if file_stat.st_size == expected_bytes:
        print('Successfully downloaded the file', file_name)
    else:
        raise Exception('File ' + file_name +
                        ' might be corrupted. You should try downloading it with a browser.')
    return file_path

def read_data(file_path):
    """ Read data into a list of tokens 
    There should be 17,005,207 tokens
    """
    with zipfile.ZipFile(file_path) as f:
        words = tf.compat.as_str(f.read(f.namelist()[0])).split() 
        #读取压缩包中的第一个文件
        # tf.compat.as_str() converts the input into the string
    return words
    
    #f.namelist()是ZipFile对象的基本成员函数
    #f.namelist() => 由names构成的list（Zip解压出来的文件，未必只有一个）
    
def build_vocab(words, vocab_size):
    """ Build vocabulary of VOCAB_SIZE most frequent words """
    dictionary = dict()
    count = [('UNK', -1)]
    #列表中的第一个元素为元组('UNK', -1)
    count.extend(Counter(words).most_common(vocab_size - 1))
    #将频率最高的前vocab_size-1个词以（Word，Frequency）元组形式进行存储
    index = 0
    utils.make_dir('processed')
    with open('processed/vocab_1000.tsv', "w") as f:
        #将前1000词频的词以tsv格式进行存储(text mode)（实际为前999，第一个为UNK）
        for word, _ in count:
            dictionary[word] = index
            #构建键值对（word,index）
            if index < 1000:
                f.write(word + "\n")
            index += 1
    index_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    #返回一个zip对象,构建（index，word）词典
    return dictionary, index_dictionary

def convert_words_to_index(words, dictionary):
    """ Replace each word in the dataset with its index in the dictionary """
    return [dictionary[word] if word in dictionary else 0 for word in words]
    #返回由每个index构成的list

    #yield生成器原理？？？？？？？
def generate_sample(index_words, context_window_size):
    """ Form training pairs according to the skip-gram model. """
    for index, center in enumerate(index_words):
        #index:0 1 2 ...
        #center: 2 3 1 5(word在voca里面的位置(不是one-hot编码))
        context = random.randint(1, context_window_size)
        #为什么要产生随机数？？？？？？？
        # get a random target before the center word
        #取出index_words里，center——word前面的几个词
        for target in index_words[max(0, index - context): index]:
            yield center, target
        # get a random target after the center wrod
        for target in index_words[index + 1: index + context + 1]:
            #[:]切片不包含最后一个数所对应的值
            yield center, target
            #yield生成一个 generator

def get_batch(iterator, batch_size):
    """ Group a numerical stream into batches and yield them as Numpy arrays. """
    while True:
        #为何需要加入此循环？？？？？？？
        center_batch = np.zeros(batch_size, dtype=np.int32)
        target_batch = np.zeros([batch_size, 1])
        for index in range(batch_size):
            center_batch[index], target_batch[index] = next(iterator)
        yield center_batch, target_batch
        #yield在for循环之外

def process_data(vocab_size, batch_size, skip_window):
    file_path = download(FILE_NAME, EXPECTED_BYTES)
    words = read_data(file_path)
    dictionary, _ = build_vocab(words, vocab_size)
    index_words = convert_words_to_index(words, dictionary)
    del words # to save memory
    
    single_gen = generate_sample(index_words, skip_window)
    #生成一个随机采样点
    return get_batch(single_gen, batch_size)

    #此处功能在build_vocab中已经实现
def get_index_vocab(vocab_size):
    file_path = download(FILE_NAME, EXPECTED_BYTES)
    words = read_data(file_path)
    return build_vocab(words, vocab_size)

