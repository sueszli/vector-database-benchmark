"""
Corpora and Vector Spaces
=========================

Demonstrates transforming text into a vector space representation.

Also introduces corpus streaming and persistence to disk in various formats.
"""
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
documents = ['Human machine interface for lab abc computer applications', 'A survey of user opinion of computer system response time', 'The EPS user interface management system', 'System and human system engineering testing of EPS', 'Relation of user perceived response time to error measurement', 'The generation of random binary unordered trees', 'The intersection graph of paths in trees', 'Graph minors IV Widths of trees and well quasi ordering', 'Graph minors A survey']
from pprint import pprint
from collections import defaultdict
stoplist = set('for a of the and to in'.split())
texts = [[word for word in document.lower().split() if word not in stoplist] for document in documents]
frequency = defaultdict(int)
for text in texts:
    for token in text:
        frequency[token] += 1
texts = [[token for token in text if frequency[token] > 1] for text in texts]
pprint(texts)
from gensim import corpora
dictionary = corpora.Dictionary(texts)
dictionary.save('/tmp/deerwester.dict')
print(dictionary)
print(dictionary.token2id)
new_doc = 'Human computer interaction'
new_vec = dictionary.doc2bow(new_doc.lower().split())
print(new_vec)
corpus = [dictionary.doc2bow(text) for text in texts]
corpora.MmCorpus.serialize('/tmp/deerwester.mm', corpus)
print(corpus)
from smart_open import open

class MyCorpus:

    def __iter__(self):
        if False:
            for i in range(10):
                print('nop')
        for line in open('https://radimrehurek.com/mycorpus.txt'):
            yield dictionary.doc2bow(line.lower().split())
corpus_memory_friendly = MyCorpus()
print(corpus_memory_friendly)
for vector in corpus_memory_friendly:
    print(vector)
dictionary = corpora.Dictionary((line.lower().split() for line in open('https://radimrehurek.com/mycorpus.txt')))
stop_ids = [dictionary.token2id[stopword] for stopword in stoplist if stopword in dictionary.token2id]
once_ids = [tokenid for (tokenid, docfreq) in dictionary.dfs.items() if docfreq == 1]
dictionary.filter_tokens(stop_ids + once_ids)
dictionary.compactify()
print(dictionary)
corpus = [[(1, 0.5)], []]
corpora.MmCorpus.serialize('/tmp/corpus.mm', corpus)
corpora.SvmLightCorpus.serialize('/tmp/corpus.svmlight', corpus)
corpora.BleiCorpus.serialize('/tmp/corpus.lda-c', corpus)
corpora.LowCorpus.serialize('/tmp/corpus.low', corpus)
corpus = corpora.MmCorpus('/tmp/corpus.mm')
print(corpus)
print(list(corpus))
for doc in corpus:
    print(doc)
corpora.BleiCorpus.serialize('/tmp/corpus.lda-c', corpus)
import gensim
import numpy as np
numpy_matrix = np.random.randint(10, size=[5, 2])
corpus = gensim.matutils.Dense2Corpus(numpy_matrix)
import scipy.sparse
scipy_sparse_matrix = scipy.sparse.random(5, 2)
corpus = gensim.matutils.Sparse2Corpus(scipy_sparse_matrix)
scipy_csc_matrix = gensim.matutils.corpus2csc(corpus)
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
img = mpimg.imread('run_corpora_and_vector_spaces.png')
imgplot = plt.imshow(img)
_ = plt.axis('off')