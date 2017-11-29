#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 08:54:02 2017

@author: luizcelso
"""
import itertools
import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.utils import shuffle
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


df_docs_original = pd.read_csv('./74-14-10.csv', sep= ',', header= None)

# df_docs = df_docs_original.drop_duplicates(u"Link para a notícia")

# df_docs = df_docs[[u"Link para a notícia", u"Texto da notícia", u"Categoria da notícia"]]

# df_docs = df_docs.set_index(u"Link para a notícia")

# data = df_docs.rename(columns={u"Texto da notícia":"text", u"Categoria da notícia":"class"})

data = shuffle(df_docs_original)

labels = np.unique(data.values[:, 0]).tolist()

#pipeline = Pipeline([
#    ('vectorizer',  TfidfVectorizer()),
#    ('classifier',  MultinomialNB()) ])


pipeline = Pipeline([
    ('vectorizer',  CountVectorizer()),
    ('classifier',  SVC(kernel='linear', C=0.01)) ])


#pipeline.fit(data['text'].values, data['class'].values)


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else '.0f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')



from sklearn.cross_validation import KFold
from sklearn.metrics import confusion_matrix, f1_score


k_fold = KFold(n=len(data), n_folds=4)
scores = []

confusion =  np.zeros((len(labels),len(labels)))

for train_indices, test_indices in k_fold:
    train_text = data.iloc[train_indices].values[:, 3]
    train_y = data.iloc[train_indices].values[:, 0]

    test_text = data.iloc[test_indices].values[:, 3]
    test_y = data.iloc[test_indices].values[:, 0]

    pipeline.fit(train_text, train_y)
    predictions = pipeline.predict(test_text)

    confusion += confusion_matrix(test_y, predictions, labels=labels)
    score = f1_score(test_y, predictions, average="micro", labels=labels)
    scores.append(score)


print('Total classified:', len(data))
print('Score:', sum(scores)/len(scores))
print('Scores:', scores)


np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure(figsize=(100,100))
plot_confusion_matrix(confusion, classes=labels, 
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure(figsize=(100,100))
plot_confusion_matrix(confusion, classes=labels, normalize=True,
                      title='Normalized confusion matrix')

plt.show()

# # Following lines show how to make new predictions with the trained model

examples = ["""Numa escala zero a dez eu te dou
Cem
E cem parece pouco pra você
Mais de um milhão de corações eu quero um
E o seu é o bastante pra eu viver
Deixa eu te levar para ver as flores
Colorir nosso jardim de amor
Enquanto o céu vai misturando as cores
A gente ama até o sol se pôr

É você, só você que sabe me fazer feliz
Que chega em meu ouvido e diz
Que o meu desejo é desejar você (2x)

Numa escala zero a dez eu te dou
Cem
E cem parece pouco pra você
Mais de um milhão de corações eu quero um
E o seu é o bastante pra eu viver
Deixa eu te levar para ver as flores
Colorir nosso jardim de amor
Enquanto o céu vai misturando as cores
A gente ama até o sol se pôr

É você, só você que sabe me fazer feliz
Que chega em meu ouvido e diz
Que o meu desejo é desejar você (2x)

Que o meu desejo é desejar você
É desejar você""","""Light breaks underneath
A heavy door
And i try to keep myself awake
Fall all around us on a hotel floor
And you think that you've made a mistake
And theres a pain in my stomach
From another sleepless binge
And i struggle to get myself up again
I want to hang onto something
That won't break away or fall apart
Like the pieces of my heart

And globes and maps are all around me now
I want to feel you breathe me
Globes and maps i see surround you here
Why won't you believe me?
Globes and maps they chartered your way back home
Do you want to leave or something?

Dreams came around you
In a hazy rain
You open your mouth wide to feel them fall
And I write a letter from a one-way train
But I don't think youll read it at all

And globes and maps are all around me now
I want to feel you breathe me
And globes and maps I see surround you here
Why won't you believe me?
Globes and maps they charter your way back home
So do you want to leave or something?

I can't take this anymore
I know that I can't take this anymore
I can't take this anymore
'Cause I know someday I'll see you walk out that door

Globes and Maps are all around me now
I want to feel you breathe me
Globes and maps I see surround you here
Why won't you believe me
Globes and maps they charter your way back home
So do you want to leave, do you want to leave
Globes and maps they charter your way back home
Do you want to leave or something?"""]

predictions = pipeline.predict(examples)
print predictions # [1, 0]
