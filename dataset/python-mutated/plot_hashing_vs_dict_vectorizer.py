"""
===========================================
FeatureHasher and DictVectorizer Comparison
===========================================

In this example we illustrate text vectorization, which is the process of
representing non-numerical input data (such as dictionaries or text documents)
as vectors of real numbers.

We first compare :func:`~sklearn.feature_extraction.FeatureHasher` and
:func:`~sklearn.feature_extraction.DictVectorizer` by using both methods to
vectorize text documents that are preprocessed (tokenized) with the help of a
custom Python function.

Later we introduce and analyze the text-specific vectorizers
:func:`~sklearn.feature_extraction.text.HashingVectorizer`,
:func:`~sklearn.feature_extraction.text.CountVectorizer` and
:func:`~sklearn.feature_extraction.text.TfidfVectorizer` that handle both the
tokenization and the assembling of the feature matrix within a single class.

The objective of the example is to demonstrate the usage of text vectorization
API and to compare their processing time. See the example scripts
:ref:`sphx_glr_auto_examples_text_plot_document_classification_20newsgroups.py`
and :ref:`sphx_glr_auto_examples_text_plot_document_clustering.py` for actual
learning on text documents.

"""
from sklearn.datasets import fetch_20newsgroups
categories = ['alt.atheism', 'comp.graphics', 'comp.sys.ibm.pc.hardware', 'misc.forsale', 'rec.autos', 'sci.space', 'talk.religion.misc']
print('Loading 20 newsgroups training data')
(raw_data, _) = fetch_20newsgroups(subset='train', categories=categories, return_X_y=True)
data_size_mb = sum((len(s.encode('utf-8')) for s in raw_data)) / 1000000.0
print(f'{len(raw_data)} documents - {data_size_mb:.3f}MB')
import re

def tokenize(doc):
    if False:
        i = 10
        return i + 15
    'Extract tokens from doc.\n\n    This uses a simple regex that matches word characters to break strings\n    into tokens. For a more principled approach, see CountVectorizer or\n    TfidfVectorizer.\n    '
    return (tok.lower() for tok in re.findall('\\w+', doc))
list(tokenize("This is a simple example, isn't it?"))
from collections import defaultdict

def token_freqs(doc):
    if False:
        print('Hello World!')
    'Extract a dict mapping tokens from doc to their occurrences.'
    freq = defaultdict(int)
    for tok in tokenize(doc):
        freq[tok] += 1
    return freq
token_freqs('That is one example, but this is another one')
from time import time
from sklearn.feature_extraction import DictVectorizer
dict_count_vectorizers = defaultdict(list)
t0 = time()
vectorizer = DictVectorizer()
vectorizer.fit_transform((token_freqs(d) for d in raw_data))
duration = time() - t0
dict_count_vectorizers['vectorizer'].append(vectorizer.__class__.__name__ + '\non freq dicts')
dict_count_vectorizers['speed'].append(data_size_mb / duration)
print(f'done in {duration:.3f} s at {data_size_mb / duration:.1f} MB/s')
print(f'Found {len(vectorizer.get_feature_names_out())} unique terms')
type(vectorizer.vocabulary_)
len(vectorizer.vocabulary_)
vectorizer.vocabulary_['example']
import numpy as np

def n_nonzero_columns(X):
    if False:
        while True:
            i = 10
    'Number of columns with at least one non-zero value in a CSR matrix.\n\n    This is useful to count the number of features columns that are effectively\n    active when using the FeatureHasher.\n    '
    return len(np.unique(X.nonzero()[1]))
from sklearn.feature_extraction import FeatureHasher
t0 = time()
hasher = FeatureHasher(n_features=2 ** 18)
X = hasher.transform((token_freqs(d) for d in raw_data))
duration = time() - t0
dict_count_vectorizers['vectorizer'].append(hasher.__class__.__name__ + '\non freq dicts')
dict_count_vectorizers['speed'].append(data_size_mb / duration)
print(f'done in {duration:.3f} s at {data_size_mb / duration:.1f} MB/s')
print(f'Found {n_nonzero_columns(X)} unique tokens')
t0 = time()
hasher = FeatureHasher(n_features=2 ** 22)
X = hasher.transform((token_freqs(d) for d in raw_data))
duration = time() - t0
print(f'done in {duration:.3f} s at {data_size_mb / duration:.1f} MB/s')
print(f'Found {n_nonzero_columns(X)} unique tokens')
t0 = time()
hasher = FeatureHasher(n_features=2 ** 18, input_type='string')
X = hasher.transform((tokenize(d) for d in raw_data))
duration = time() - t0
dict_count_vectorizers['vectorizer'].append(hasher.__class__.__name__ + '\non raw tokens')
dict_count_vectorizers['speed'].append(data_size_mb / duration)
print(f'done in {duration:.3f} s at {data_size_mb / duration:.1f} MB/s')
print(f'Found {n_nonzero_columns(X)} unique tokens')
import matplotlib.pyplot as plt
(fig, ax) = plt.subplots(figsize=(12, 6))
y_pos = np.arange(len(dict_count_vectorizers['vectorizer']))
ax.barh(y_pos, dict_count_vectorizers['speed'], align='center')
ax.set_yticks(y_pos)
ax.set_yticklabels(dict_count_vectorizers['vectorizer'])
ax.invert_yaxis()
_ = ax.set_xlabel('speed (MB/s)')
from sklearn.feature_extraction.text import CountVectorizer
t0 = time()
vectorizer = CountVectorizer()
vectorizer.fit_transform(raw_data)
duration = time() - t0
dict_count_vectorizers['vectorizer'].append(vectorizer.__class__.__name__)
dict_count_vectorizers['speed'].append(data_size_mb / duration)
print(f'done in {duration:.3f} s at {data_size_mb / duration:.1f} MB/s')
print(f'Found {len(vectorizer.get_feature_names_out())} unique terms')
from sklearn.feature_extraction.text import HashingVectorizer
t0 = time()
vectorizer = HashingVectorizer(n_features=2 ** 18)
vectorizer.fit_transform(raw_data)
duration = time() - t0
dict_count_vectorizers['vectorizer'].append(vectorizer.__class__.__name__)
dict_count_vectorizers['speed'].append(data_size_mb / duration)
print(f'done in {duration:.3f} s at {data_size_mb / duration:.1f} MB/s')
from sklearn.feature_extraction.text import TfidfVectorizer
t0 = time()
vectorizer = TfidfVectorizer()
vectorizer.fit_transform(raw_data)
duration = time() - t0
dict_count_vectorizers['vectorizer'].append(vectorizer.__class__.__name__)
dict_count_vectorizers['speed'].append(data_size_mb / duration)
print(f'done in {duration:.3f} s at {data_size_mb / duration:.1f} MB/s')
print(f'Found {len(vectorizer.get_feature_names_out())} unique terms')
(fig, ax) = plt.subplots(figsize=(12, 6))
y_pos = np.arange(len(dict_count_vectorizers['vectorizer']))
ax.barh(y_pos, dict_count_vectorizers['speed'], align='center')
ax.set_yticks(y_pos)
ax.set_yticklabels(dict_count_vectorizers['vectorizer'])
ax.invert_yaxis()
_ = ax.set_xlabel('speed (MB/s)')