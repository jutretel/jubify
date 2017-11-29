# Porter Stemmer

import nltk
import string
import re

porter_stemmer = nltk.stem.porter.PorterStemmer()

def porter_tokenizer(text, stemmer=porter_stemmer):
    """
    A Porter-Stemmer-Tokenizer hybrid to splits sentences into words (tokens) 
    and applies the porter stemming algorithm to each of the obtained token. 
    Tokens that are only consisting of punctuation characters are removed as well.
    Only tokens that consist of more than one letter are being kept.
    
    Parameters
    ----------
        
    text : `str`. 
      A sentence that is to split into words.
        
    Returns
    ----------
    
    no_punct : `str`. 
      A list of tokens after stemming and removing Sentence punctuation patterns.
    
    """
    lower_txt = text.lower()
    tokens = nltk.wordpunct_tokenize(lower_txt)
    stems = [porter_stemmer.stem(t) for t in tokens]
    no_punct = [s for s in stems if re.match('^[a-zA-Z]+$', s) is not None]
    return no_punct

# ================================================================================================

from sklearn.feature_extraction.text import CountVectorizer

with open('./stop_words.txt', 'r') as infile:
    stop_words = infile.read().splitlines()

vec = CountVectorizer(
            encoding='utf-8',
            decode_error='replace',
            strip_accents='unicode',
            analyzer='word',
            stop_words=stop_words,
            binary=False,
            tokenizer=porter_tokenizer,
            ngram_range=(2,2))

# ================================================================================================

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(
            encoding='utf-8',
            decode_error='replace',
            strip_accents='unicode',
            analyzer='word',
            stop_words=stop_words,
            binary=False,
            tokenizer=porter_tokenizer)

# ================================================================================================

import numpy as np
import pandas as pd

from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.preprocessing import LabelEncoder

song_data = pd.read_csv('./74-14-10.csv',sep= ',', header= None)

print "Dataset Lenght: ", len(song_data)

Y = song_data.values[:, 0]
X = song_data.values[:, 3]

print "=================== DECISION TREE :: COUNT VECTOR ==================="

X_count = vec.fit_transform(X.ravel())

print X_count

X_train, X_test, y_train, y_test = train_test_split(X_count, Y, test_size = 0.3, random_state = 100)

clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100)
clf_gini.fit(X_train, y_train)

y_pred = clf_gini.predict(X_test)
print "Accuracy for gini is ", accuracy_score(y_test,y_pred)*100

clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100)
clf_entropy.fit(X_train, y_train)

y_pred = clf_entropy.predict(X_test)
print "Accuracy for entropy is ", accuracy_score(y_test,y_pred)*100

print "=================== DECISION TREE :: TFIDF VECTOR ==================="

X_tfidf = tfidf.fit_transform(X.ravel())
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, Y, test_size = 0.3, random_state = 100)

clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100)
clf_gini.fit(X_train, y_train)

y_pred = clf_gini.predict(X_test)
print "Accuracy for gini is ", accuracy_score(y_test,y_pred)*100

clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100)
clf_entropy.fit(X_train, y_train)

y_pred = clf_entropy.predict(X_test)
print "Accuracy for entropy is ", accuracy_score(y_test,y_pred)*100