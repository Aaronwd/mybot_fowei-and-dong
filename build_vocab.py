# -*- coding: utf-8 -*-
"""
Created on Sun Oct 23 20:28:38 2016

@author: shjtdx
"""

import numpy as np

PAD = '<pad>'
UNK = '<unk>'


# build vocab
file = open('cut_res.txt' , 'r')
text = file.readlines()
text = text[0]
lists = list(text)

word = '/'

vocab = []
old = -1

for i in range(0,len(lists)):
    v = lists[i]
    if word == v:
        words = "".join(lists[old+1:i]) 
        str_len = i-(old+1)
        old = i
        if words not in vocab: #and str_len <= 6:
            vocab.append(words)

vectors = []
words_list = []

#loading vectors
file = open('vectors.txt' , 'r')
lines = file.readlines()

asks = []
answers = []
content = []
Qa_file = open('db.txt', 'r')
content = Qa_file.readlines()    
for i in range(len(content)):
    if i%2 == 0:
        asks.append(content[i])
    elif content[i] not in answers:
        answers.append(content[i])
answers.append('您/问/的/问题/很/有趣/，/不如/您/问/一些/关于/我们/公司/的/问题/吧\n')

for i in range(0,len(lines)):
    line = lines[i]
    line_parts = line.split()
    # The first part is the word.
    word = line_parts[0]
    if word in vocab:
        word_vector = np.array(map(float, line_parts[1:]))
        words_list.append(word)
        vectors.append(word_vector)

#标记单词在词典中的位置
words_list.insert(2, PAD)
vectors.insert(2, np.ones(200, 'float64')*0.005)
encoder_sequence_batch = []
symbol = '/'
input_len = 15

for ask in asks:
    ret = []
    old = -1
    for i in range(0,len(ask)):
        v = ask[i]
        if symbol == v:
            words = "".join(ask[old+1:i]) 
            old = i
            if words in words_list:
                ret.append(words_list.index(words))
                ret.reverse()
    encoder_sequence = [words_list.index(PAD)] * (input_len - len(ret)) + ret
    encoder_sequence_batch.append(encoder_sequence)
    encoder = np.asarray(encoder_sequence_batch).T