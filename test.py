# -*- coding: utf-8 -*-
"""
Created on Sun Oct 23 20:28:38 2016

@author: shjtdx
"""

import tensorflow as tf
import numpy as np
import os
import jieba

PAD = '<pad>'

os.chdir('/home/wfw/wfw/5.my_bot')
answers = np.load('answers.npy')
answers = list(answers)

while True:
    #分词
    sentence = input('说：')
    sentence = sentence.strip()

    if sentence in ('quit', 'exit'):
        break
    if len(sentence) <= 0:
        break

    seg_list = jieba.cut(sentence, cut_all=True)
    seg_sentence = "/".join(seg_list)
    seg_sentence = seg_sentence + '/'

    #encoder_inputs
    words_list = np.load('words_list.npy')
    words_list = list(words_list)
    encoder_sequence_batch = []
    symbol = '/'
    input_len = 15

    ret = []
    old = -1
    for i in range(0,len(seg_sentence)):
        v = seg_sentence[i]
        if symbol == v:
            words = "".join(seg_sentence[old+1:i])
            word = words.encode("utf-8")
            old = i
            if word in words_list:
                ret.append(words_list.index(word))
                ret.reverse()
    encoder_sequence = [words_list.index(PAD)] * (input_len - len(ret)) + ret
    encoder_sequence_batch.append(encoder_sequence)
    encoder = np.asarray(encoder_sequence_batch).T

    #test
    input_len = 15
    encoders = tf.placeholder(tf.int32, [15, None])
    vector = np.load('vectors.npy')

    session = tf.InteractiveSession()
    # set embeddings
    embedding = tf.Variable(tf.random_uniform(vector.shape, minval=-0.1, maxval=0.1), trainable=False)
    session.run(embedding.assign(vector))

    cell = tf.nn.rnn_cell.BasicLSTMCell(200)
    cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=1.0)
    cell = tf.nn.rnn_cell.MultiRNNCell([cell] * 2)

    encoder_cell = tf.nn.rnn_cell.EmbeddingWrapper(
        cell, embedding_classes=63,
        embedding_size=200)

    encoder_inputs = []
    for i in range(input_len):
        encoder_inputs.append(encoders[i,:])

    encoder_outputs, encoder_state = tf.nn.rnn(encoder_cell, encoder_inputs, dtype ='float32')

    W = tf.Variable(tf.zeros([800, 5]))
    b = tf.Variable(tf.zeros([5]))
    predict_labels = tf.nn.softmax(tf.matmul(encoder_state, W) + b)

    ckpt_dir = "./ckpt_dir"
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    saver = tf.train.Saver()

    print('start test')
    init = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init)

        ckpt = tf.train.get_checkpoint_state(ckpt_dir)
        if ckpt and ckpt.model_checkpoint_path:
           # print(ckpt.model_checkpoint_path)
            saver.restore(sess, ckpt.model_checkpoint_path)  # 调取之前的model，从上一个训练开始

        predict_label = sess.run(predict_labels, feed_dict={encoders: encoder})
        label= np.argmax(predict_label)
        print answers[label]