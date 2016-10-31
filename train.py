# -*- coding: utf-8 -*-
"""
Created on Sun Oct 23 20:28:38 2016

@author: shjtdx
"""

import tensorflow as tf
import numpy as np
import os
from data_save import *
# import build_vocab

os.chdir('/home/wfw/wfw/5.my_bot')

#vectors = build_vocab.vectors
# vector = np.zeros([200,63])
# for i in range(63):
#     vector[:,i] = vectors[i]
input_len = 15
vector = np.load('vectors.npy')
encoder = np.load('encoder.npy')
encoders = tf.placeholder(tf.int32, [15, None])

labels = [1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,3,3,4,4,4,4,4,4,4,4,4,4,5,5,5,5,5,5,5,5,5,5]
target_label = np.zeros([len(labels), 5], 'float32')
for i in range(len(labels)):
    target_label[i, labels[i]-1] = 1

session = tf.InteractiveSession()
# set embeddings
embedding = tf.Variable(tf.random_uniform(vector.shape, minval=-0.1, maxval=0.1), trainable=False)
#session.run(tf.initialize_all_variables())
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


# encoder_inputs = [tf.placeholder(tf.int32, [None], name='encoder_inputs_{}'.format(i))
#                   for i in range(input_len)]
# encoder_output = tf.placeholder(tf.float32, [None, 800])
target_labels = tf.placeholder(tf.float32, [None, 5])
W = tf.Variable(tf.zeros([800, 5]))
b = tf.Variable(tf.zeros([5]))
predict_labels = tf.nn.softmax(tf.matmul(encoder_state, W) + b)
loss_func = -tf.reduce_sum(target_labels * tf.log(predict_labels))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss_func)

correct_prediction = tf.equal(tf.argmax(predict_labels,1), tf.argmax(target_labels,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

ckpt_dir = "./ckpt_dir"
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)

saver = tf.train.Saver()

print('start train')
init = tf.initialize_all_variables()
epoch = 1000
with tf.Session() as sess:
    sess.run(init)

    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    if ckpt and ckpt.model_checkpoint_path:
        print(ckpt.model_checkpoint_path)
        saver.restore(sess, ckpt.model_checkpoint_path)  # 调取之前的model，从上一个训练开始

    for i in range(1, epoch):
        sess.run(train_step, feed_dict = {encoders: encoder, target_labels: target_label})
        if(i%100==0):
            #save_model(sess)
            print "accuracy", sess.run(accuracy, feed_dict = {encoders: encoder, target_labels: target_label})
            saver.save(sess, ckpt_dir + "/model.ckpt", global_step= i)