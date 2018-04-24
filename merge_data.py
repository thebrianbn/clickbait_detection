# -*- coding: utf-8 -*-
"""
Created on Tue Feb 06 14:19:51 2018

@author: Brian Nguyen
"""
import json
import pandas as pd
from tweet_tokenizer import tokenise

def tokenise_merged_data(model = "nn"):
    '''Merge train/validation data with truth data and apply tweet tokeniser'''
    
    with open('clickbait17-train-170331/clickbait17-train-170331/instances.jsonl', encoding = "utf8") as f:
        instance_data1 = pd.DataFrame(json.loads(line) for line in f)
        
    with open('clickbait17-train-170630/clickbait17-validation-170630/instances.jsonl', encoding = "utf8") as f:
        instance_data2 = pd.DataFrame(json.loads(line) for line in f)    
        
    with open('clickbait17-train-170331/clickbait17-train-170331/truth.jsonl', encoding = "utf8") as f:
        truth_data1 = pd.DataFrame(json.loads(line) for line in f)
    
    with open('clickbait17-train-170630/clickbait17-validation-170630/truth.jsonl', encoding = "utf8") as f:
        truth_data2 = pd.DataFrame(json.loads(line) for line in f)
        
    instance_data = pd.concat([instance_data2, instance_data1])
    truth_data = pd.concat([truth_data2, truth_data1])
    total_data = pd.merge(instance_data, truth_data).reset_index(drop = True)
    
    #total_data.postText = total_data.postText.apply(lambda x: x[0].split())
    
    #del total_data["postTimestamp"], total_data["targetDescription"], total_data["targetKeywords"]
    #del total_data["targetTitle"], total_data["targetParagraphs"], total_data["targetCaptions"]
    
    if model == "nn":
        total_data.truthClass[total_data.truthClass == "no-clickbait"] = 0
        total_data.truthClass[total_data.truthClass == "clickbait"] = 1
        total_data.postText = total_data.postText.apply(lambda x: tokenise(x[0]))
        total_data.targetDescription = total_data.targetDescription.apply(lambda x: tokenise(x))
        total_data.targetCaptions = total_data.targetCaptions.apply(lambda x: tokenise("".join(x)))
        total_data.targetParagraphs = total_data.targetParagraphs.apply(lambda x: tokenise("".join(x)))
        total_data.targetTitle = total_data.targetTitle.apply(lambda x: tokenise(x))
        total_data.targetKeywords = total_data.targetKeywords.apply(lambda x: tokenise(" ".join(x.split(","))))
        #total_data.postText = total_data.postText.apply(lambda x: " ".join(x))
    else:
        total_data.postText = total_data.postText.apply(lambda x: tokenise(x[0]))
        total_data.postText = total_data.postText.apply(lambda x: "".join(x))
    
    return total_data
    
def read_subset(model = "lstm", test = False):
    '''Read only the training data set'''
    
    if test:
        with open('clickbait17-train-170331/clickbait17-train-170331/instances.jsonl', encoding = "utf8") as f:
            instance_data = pd.DataFrame(json.loads(line) for line in f)
        
        with open('clickbait17-train-170331/clickbait17-train-170331/truth.jsonl', encoding = "utf8") as f:
            truth_data = pd.DataFrame(json.loads(line) for line in f)
    else:
    
        with open('clickbait17-train-170630/clickbait17-validation-170630/instances.jsonl', encoding = "utf8") as f:
            instance_data = pd.DataFrame(json.loads(line) for line in f)
            
        with open('clickbait17-train-170630/clickbait17-validation-170630/truth.jsonl', encoding = "utf8") as f:
            truth_data = pd.DataFrame(json.loads(line) for line in f)
        
    total_data = pd.merge(instance_data, truth_data).reset_index(drop = True)
    
    #total_data.postText = total_data.postText.apply(lambda x: "".join(x))
    
    #del total_data["postTimestamp"], total_data["targetDescription"], total_data["targetKeywords"]
    #del total_data["targetTitle"], total_data["targetParagraphs"], total_data["targetCaptions"]
    
# =============================================================================
#     if model == "lstm":
#         total_data.truthClass[total_data.truthClass == "no-clickbait"] = 0
#         total_data.truthClass[total_data.truthClass == "clickbait"] = 1
#         total_data.postText = total_data.postText.apply(lambda x: tokenise(x[0]))
#         total_data.targetDescription = total_data.targetDescription.apply(lambda x: tokenise(x))
#         total_data.targetCaptions = total_data.targetCaptions.apply(lambda x: tokenise("".join(x)))
#         total_data.targetParagraphs = total_data.targetParagraphs.apply(lambda x: tokenise("".join(x)))
#         total_data.targetTitle = total_data.targetTitle.apply(lambda x: tokenise(x))
#         total_data.targetKeywords = total_data.targetKeywords.apply(lambda x: tokenise(" ".join(x.split(","))))
#         #total_data.postText = total_data.postText.apply(lambda x: " ".join(x))
#     else:
#         total_data.postText = total_data.postText.apply(lambda x: tokenise(x[0]))
#         total_data.postText = total_data.postText.apply(lambda x: "".join(x))
# =============================================================================
    
    return total_data
