# -*- coding: utf-8 -*-
"""
Created on Tue Feb 06 13:46:34 2018

@author: Brian Nguyen
"""
import numpy as np
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from merge_data import merge_data
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

def cross_validate(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=.3, random_state=0)
    
    clf = BernoulliNB().fit(X_train, y_train)
    return(clf.score(X_test, y_test) * 100)

def main():
    
    data = merge_data(naive_bayes = True)
    
    counter = CountVectorizer(encoding=u'utf-8', stop_words = "english")
    cv_fit = counter.fit_transform(data.postText.values.tolist())
    
    tf_transformer = TfidfTransformer(use_idf=False).fit(cv_fit)
    words_train_tf = tf_transformer.transform(cv_fit)
    
    clf = BernoulliNB().fit(words_train_tf, data.truthClass)
    
    predicted = clf.predict(words_train_tf)
    
    text_clf = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('clf', BernoulliNB()),])
    text_clf.fit(data.postText.values.tolist(), data.truthClass)
    
    #text_clf.predict(test.postText)
    
    print(confusion_matrix(predicted, data.truthClass))
    
    print("Cross validation with 30 percent test set: %f" % cross_validate(words_train_tf, data.truthClass))
    
    print("Accuracy from training set: %f" % (np.mean(predicted == data.truthClass) * 100))
    
if __name__ == "__main__":
    main()