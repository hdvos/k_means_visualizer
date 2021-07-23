from kmeans_plotter import k_means_gif

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer

import sys

categories = [
    'alt.atheism',
    'comp.graphics',
    'comp.sys.ibm.pc.hardware',
    'misc.forsale',
    'rec.autos',
    'sci.space',
    'talk.religion.misc',
]

dataset = fetch_20newsgroups(subset='all', categories=categories,
                             shuffle=True, random_state=42)


# hasher = HashingVectorizer(n_features=1000,
#                            stop_words='english', alternate_sign=False,
#                            norm=None)

vectorizer = TfidfVectorizer(min_df=5)

X = vectorizer.fit_transform(dataset.data)

k_means_gif(6, X.toarray(), "figures_examples/20_newsgroups_k6.gif", seed=1, max_difference=0.0001)