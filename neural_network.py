# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 21:17:25 2018

@author: Brian Nguyen
"""

import tensorflow as tf
import numpy as np
from merge_data import read_subset, tokenise_merged_data
from sentence_embedder import sentence_embedder, extract_categories, stem
from sklearn.model_selection import train_test_split
import pandas as pd
from gensim.models import Word2Vec
        
def next_batch(data, feature_columns, target_column, start_index, batch_size):
    return data[feature_columns].iloc[start_index:start_index + batch_size], data[target_column].iloc[start_index:start_index + batch_size]

#Read in all of the data
total_data = tokenise_merged_data(model = "nn")

#NLP: Create Word Embeddings
total_data.postText = extract_categories(stem(total_data.postText.head(21997)))

word_embeddings = Word2Vec(total_data.postText, min_count = 1, size = 100)

total_data.postText = sentence_embedder(total_data.postText, word2vec = word_embeddings)

train_data = total_data.iloc[:14001].append(total_data.iloc[19001:]).reset_index(drop = True)

test_data = total_data.iloc[14000:19000].reset_index(drop = True)

X_train = train_data.postText
y_train = train_data.truthClass
X_test = test_data.postText
y_test = test_data.truthClass

train_data = pd.DataFrame(data = {"postText": X_train, "truthClass" : y_train})

del y_train, X_train

test_features = np.asarray(X_test.tolist())
test_labels = tf.one_hot(np.array(y_test), 2)

max_array = len(total_data.postText[0])

HIDDEN_NUM = 15
OUTPUT_NUM = 2
INPUT_NUM = max_array

EPOCHS = 500
LEARNING_RATE = 0.0005
TRAIN_STEPS = 100
BATCH_SIZE = 169

x_train = tf.placeholder(tf.float32, [None, INPUT_NUM])
y_train = tf.placeholder(tf.float32, [None, OUTPUT_NUM])

weights = {"hidden": tf.Variable(tf.random_normal([INPUT_NUM, HIDDEN_NUM])),
           "output": tf.Variable(tf.random_normal([HIDDEN_NUM, OUTPUT_NUM]))}
biases = {"hidden": tf.Variable(tf.random_normal([HIDDEN_NUM])),
          "output": tf.Variable(tf.random_normal([OUTPUT_NUM]))}

hidden_layer = tf.nn.relu(tf.add(tf.matmul(x_train, weights["hidden"]), biases["hidden"]))
output_layer = tf.matmul(hidden_layer, weights["output"]) + biases["output"]

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_train, logits = output_layer))
#loss = tf.reduce_mean(tf.contrib.legacy_seq2seq.sequence_loss_by_example(targets = y_train, logits = pred))
optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE).minimize(loss)

correct_prediction = tf.equal(tf.argmax(output_layer,1), tf.argmax(y_train,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

prediction=tf.argmax(output_layer,1)

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
            sess.run([optimizer], feed_dict = {x_train: x_batch, y_train: y_batch})
            #loss_total += loss
            #acc_total += acc
        print("Epoch %d Accuracy: %f" % (i, sess.run(accuracy, feed_dict = {x_train: test_features, y_train: test_labels})))
        predictions = pd.DataFrame(sess.run([prediction],feed_dict = {x_train: test_features, y_train:test_labels})[0])
        predictions.columns = ["results"]
        predictions["actual"] = y_test
        predictions.to_csv("predictions_2.csv")