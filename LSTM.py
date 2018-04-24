# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 21:51:13 2018

@author: Brian Nguyen
"""

import tensorflow as tf
import numpy as np
from merge_data import tokenise_merged_data
from tensorflow.contrib import rnn
from sentence_embedder import sentence_embedder, extract_categories, stem
from sklearn.model_selection import train_test_split
import pandas as pd
from gensim.models import Word2Vec


def merge_helper(data):
    '''Given a list of lists, append all items in the list together'''
    final = []
    for i in range(len(data)):
        final += data[i]
    return final

def lstm(x, weights, biases):
    '''returns predictions from the result of a long short-term memory rnn'''
    
    x = tf.reshape(x, [-1, INPUT_NUM])
    
    x = tf.split(x, INPUT_NUM, 1)
    
    #rnn_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(HIDDEN_NUM), rnn.BasicLSTMCell(HIDDEN_NUM)])
    rnn_cell = rnn.BasicLSTMCell(HIDDEN_NUM, forget_bias = 1)
    
    outputs, stages = rnn.static_rnn(rnn_cell, x, dtype = tf.float32)
    
    return tf.matmul(outputs[-1], weights["output"]) + biases["output"]

def bi_lstm(x, weights, biases):
    x = tf.reshape(x, [-1, INPUT_NUM])
    
    x = tf.split(x, INPUT_NUM, 1)
    
    #rnn_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(HIDDEN_NUM), rnn.BasicLSTMCell(HIDDEN_NUM)])
    lstm_fw_cell = rnn.BasicLSTMCell(HIDDEN_NUM, forget_bias = 1)
    lstm_bw_cell = rnn.BasicLSTMCell(HIDDEN_NUM, forget_bias = 1)
    
    outputs, _, _ = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x, dtype = tf.float32)
    
    return tf.matmul(outputs[-1], weights["output"]) + biases["output"]

def next_batch(data, feature_columns, target_column, start_index, batch_size):
    return data[feature_columns].iloc[start_index:start_index + batch_size], data[target_column].iloc[start_index:start_index + batch_size]


#Read in all of the data
total_data = tokenise_merged_data(model = "lstm")

#NLP: Create Word Embeddings
total_data.postText = extract_categories(stem(total_data.postText.head(21997)))

word_embeddings = Word2Vec(total_data.postText, min_count = 1, size = 100)

total_data.postText = sentence_embedder(total_data.postText, word2vec = word_embeddings)


X_train, X_test, y_train, y_test = train_test_split(
        total_data.postText, total_data.truthClass, test_size=.2, random_state=0)

train_data = pd.DataFrame(data = {"postText": X_train, "truthClass" : y_train})

test_features = np.asarray(X_test.tolist())
test_labels = tf.one_hot(np.array(y_test), 2)

max_array = len(total_data.postText[0])

HIDDEN_NUM = 25
OUTPUT_NUM = 2
INPUT_NUM = max_array

EPOCHS = 500
LEARNING_RATE = 0.005
TRAIN_STEPS = 100
BATCH_SIZE = 160

x_train = tf.placeholder(tf.float32, [None, INPUT_NUM])
y_train = tf.placeholder(tf.float32, [None, OUTPUT_NUM])

weights = {"hidden": tf.Variable(tf.random_normal([INPUT_NUM, HIDDEN_NUM])),
           "output": tf.Variable(tf.random_normal([HIDDEN_NUM, OUTPUT_NUM]))}
biases = {"hidden": tf.Variable(tf.random_normal([HIDDEN_NUM])),
          "output": tf.Variable(tf.random_normal([OUTPUT_NUM]))}

#pred = tf.nn.softmax(bi_lstm(x_train, weights, biases))

hidden_layer = tf.nn.relu(tf.add(tf.matmul(x_train, weights["hidden"]), biases["hidden"]))
output_layer = tf.matmul(hidden_layer, weights["output"]) + biases["output"]


loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_train, logits = output_layer))
#loss = tf.reduce_mean(tf.contrib.legacy_seq2seq.sequence_loss_by_example(targets = y_train, logits = pred))
optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE).minimize(loss)

correct_prediction = tf.equal(tf.argmax(output_layer,1), tf.argmax(y_train,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#prediction=tf.argmax(output_layer,1)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    test_labels = sess.run(test_labels)
    loss_total = 0
    acc_total = 0
    for i in range(EPOCHS):
        start_index = 0
        for j in range(TRAIN_STEPS):
            x_batch, y_batch = next_batch(train_data, ["postText"], "truthClass", start_index, BATCH_SIZE)
            x_batch = np.asarray(x_batch.postText.tolist())
            y_batch = tf.one_hot(np.array(y_batch), 2)
            y_batch = sess.run(y_batch)
            start_index += BATCH_SIZE
            sess.run([optimizer, loss], feed_dict = {x_train: x_batch, y_train: y_batch})
            #loss_total += loss
            #acc_total += acc
        print("Epoch %d Accuracy: %f" % (i, sess.run(accuracy, feed_dict = {x_train: test_features, y_train: test_labels})))
        #print(sess.run([prediction],feed_dict = {x_train: test_features, y_train:test_labels}))
    
