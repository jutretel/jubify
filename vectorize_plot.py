import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.manifold import MDS

df = pd.DataFrame.from_csv('./train_lyrics_1000.csv')

X_train = df['lyrics'].values
names = df['title'].values

count_vect = CountVectorizer()
dtm = count_vect.fit_transform(X_train.ravel())

vocab = count_vect.get_feature_names()

dtm = dtm.toarray()
vocab = np.array(vocab)

dist = euclidean_distances(dtm)

dist = np.round(dist, 1)

mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)
pos = mds.fit_transform(dist)  # shape (n_components, n_samples)

xs, ys = pos[:, 0], pos[:, 1]

for x, y, name in zip(xs, ys, names):
    color = 'skyblue'
    plt.scatter(x, y, c=color)
    plt.text(x, y, name)

plt.show()
