"""
Fast Similarity Queries with Annoy and Word2Vec
===============================================

Introduces the Annoy library for similarity queries on top of vectors learned by Word2Vec.
"""
LOGS = False
if LOGS:
    import logging
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
import gensim.downloader as api
text8_path = api.load('text8', return_path=True)
print('Using corpus from', text8_path)
from gensim.models import Word2Vec, KeyedVectors
from gensim.models.word2vec import Text8Corpus
params = {'alpha': 0.05, 'vector_size': 100, 'window': 5, 'epochs': 5, 'min_count': 5, 'sample': 0.0001, 'sg': 1, 'hs': 0, 'negative': 5}
model = Word2Vec(Text8Corpus(text8_path), **params)
wv = model.wv
print('Using trained model', wv)
from gensim.similarities.annoy import AnnoyIndexer
annoy_index = AnnoyIndexer(model, 100)
vector = wv['science']
approximate_neighbors = wv.most_similar([vector], topn=11, indexer=annoy_index)
print('Approximate Neighbors')
for neighbor in approximate_neighbors:
    print(neighbor)
normal_neighbors = wv.most_similar([vector], topn=11)
print('\nExact Neighbors')
for neighbor in normal_neighbors:
    print(neighbor)
annoy_index = AnnoyIndexer(model, 100)
normed_vectors = wv.get_normed_vectors()
vector = normed_vectors[0]
wv.most_similar([vector], topn=5, indexer=annoy_index)
wv.most_similar([vector], topn=5)
import time
import numpy as np

def avg_query_time(annoy_index=None, queries=1000):
    if False:
        print('Hello World!')
    'Average query time of a most_similar method over 1000 random queries.'
    total_time = 0
    for _ in range(queries):
        rand_vec = normed_vectors[np.random.randint(0, len(wv))]
        start_time = time.process_time()
        wv.most_similar([rand_vec], topn=5, indexer=annoy_index)
        total_time += time.process_time() - start_time
    return total_time / queries
queries = 1000
gensim_time = avg_query_time(queries=queries)
annoy_time = avg_query_time(annoy_index, queries=queries)
print('Gensim (s/query):\t{0:.5f}'.format(gensim_time))
print('Annoy (s/query):\t{0:.5f}'.format(annoy_time))
speed_improvement = gensim_time / annoy_time
print('\nAnnoy is {0:.2f} times faster on average on this particular run'.format(speed_improvement))
fname = '/tmp/mymodel.index'
annoy_index.save(fname)
import os.path
if os.path.exists(fname):
    annoy_index2 = AnnoyIndexer()
    annoy_index2.load(fname)
    annoy_index2.model = model
vector = wv['science']
approximate_neighbors2 = wv.most_similar([vector], topn=11, indexer=annoy_index2)
for neighbor in approximate_neighbors2:
    print(neighbor)
assert approximate_neighbors == approximate_neighbors2
if LOGS:
    logging.disable(logging.CRITICAL)
from multiprocessing import Process
import os
import psutil
model.save('/tmp/mymodel.pkl')

def f(process_id):
    if False:
        i = 10
        return i + 15
    print('Process Id: {}'.format(os.getpid()))
    process = psutil.Process(os.getpid())
    new_model = Word2Vec.load('/tmp/mymodel.pkl')
    vector = new_model.wv['science']
    annoy_index = AnnoyIndexer(new_model, 100)
    approximate_neighbors = new_model.wv.most_similar([vector], topn=5, indexer=annoy_index)
    print('\nMemory used by process {}: {}\n---'.format(os.getpid(), process.memory_info()))
p1 = Process(target=f, args=('1',))
p1.start()
p1.join()
p2 = Process(target=f, args=('2',))
p2.start()
p2.join()
model.save('/tmp/mymodel.pkl')

def f(process_id):
    if False:
        i = 10
        return i + 15
    print('Process Id: {}'.format(os.getpid()))
    process = psutil.Process(os.getpid())
    new_model = Word2Vec.load('/tmp/mymodel.pkl')
    vector = new_model.wv['science']
    annoy_index = AnnoyIndexer()
    annoy_index.load('/tmp/mymodel.index')
    annoy_index.model = new_model
    approximate_neighbors = new_model.wv.most_similar([vector], topn=5, indexer=annoy_index)
    print('\nMemory used by process {}: {}\n---'.format(os.getpid(), process.memory_info()))
p1 = Process(target=f, args=('1',))
p1.start()
p1.join()
p2 = Process(target=f, args=('2',))
p2.start()
p2.join()
import matplotlib.pyplot as plt
exact_results = [element[0] for element in wv.most_similar([normed_vectors[0]], topn=100)]
x_values = []
y_values_init = []
y_values_accuracy = []
for x in range(1, 300, 10):
    x_values.append(x)
    start_time = time.time()
    annoy_index = AnnoyIndexer(model, x)
    y_values_init.append(time.time() - start_time)
    approximate_results = wv.most_similar([normed_vectors[0]], topn=100, indexer=annoy_index)
    top_words = [result[0] for result in approximate_results]
    y_values_accuracy.append(len(set(top_words).intersection(exact_results)))
plt.figure(1, figsize=(12, 6))
plt.subplot(121)
plt.plot(x_values, y_values_init)
plt.title('num_trees vs initalization time')
plt.ylabel('Initialization time (s)')
plt.xlabel('num_trees')
plt.subplot(122)
plt.plot(x_values, y_values_accuracy)
plt.title('num_trees vs accuracy')
plt.ylabel('%% accuracy')
plt.xlabel('num_trees')
plt.tight_layout()
plt.show()
wv.save_word2vec_format('/tmp/vectors.txt', binary=False)
from smart_open import open
with open('/tmp/vectors.txt', encoding='utf8') as myfile:
    for i in range(3):
        print(myfile.readline().strip())
wv = KeyedVectors.load_word2vec_format('/tmp/vectors.txt', binary=False)
wv.save_word2vec_format('/tmp/vectors.bin', binary=True)
wv = KeyedVectors.load_word2vec_format('/tmp/vectors.bin', binary=True)
annoy_index = AnnoyIndexer(wv, 100)
annoy_index.save('/tmp/mymodel.index')
wv = KeyedVectors.load_word2vec_format('/tmp/vectors.bin', binary=True)
annoy_index = AnnoyIndexer()
annoy_index.load('/tmp/mymodel.index')
annoy_index.model = wv
vector = wv['cat']
approximate_neighbors = wv.most_similar([vector], topn=11, indexer=annoy_index)
print('Approximate Neighbors')
for neighbor in approximate_neighbors:
    print(neighbor)
normal_neighbors = wv.most_similar([vector], topn=11)
print('\nExact Neighbors')
for neighbor in normal_neighbors:
    print(neighbor)