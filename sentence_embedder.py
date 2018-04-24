# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 15:23:59 2018

@author: Brian Nguyen
"""
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
import nltk
from nltk.stem.snowball import SnowballStemmer

def sentence_embedder(data, word2vec):
    '''creates sentence embeddings given a column of tokens'''
    word_embeddings = word2vec
    
    #max_tokens = data.apply(len).max()
    max_tokens = 37 #max amount of tokens in postText

    all_arrays = []
    for i in range(len(data)):
        counter = 0
        sentence_array = []
        for j in data[i]:
            counter += 1
            sentence_array += word_embeddings.wv[j].tolist()
        if counter < max_tokens:
            for k in range(max_tokens - counter):
                sentence_array += np.zeros([100]).tolist()
        all_arrays.append(np.array(sentence_array, dtype = np.float32))
    return pd.Series(all_arrays)

def extract_categories(data):
    
    special_tokens = ["<url>", "<user>", "<smile>", "<lolface>", "<sadface>", "<number>", "<hashtag>", "<allcaps>",
                      "<repeat>", "<elong>"]
    
    data = data.apply(nltk.pos_tag)
    
    
    for i in range(len(data)):
        final_tags = []
        for token, category in data[i]:
            if token in special_tokens:
                final_tags.append(token)
            else:
                final_tags.append(category)
                
        data[i] = final_tags
                
    return data
            

def stem(data):
    
    stemmer = SnowballStemmer("english")
    for i in range(len(data)):
        new_words = []
        for word in data[i]:
            new_words.append(stemmer.stem(word))
        data[i] = new_words
        
    return data