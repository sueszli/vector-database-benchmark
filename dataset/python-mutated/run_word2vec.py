"""
Word2Vec Model
==============

Introduces Gensim's Word2Vec model and demonstrates its use on the `Lee Evaluation Corpus
<https://hekyll.services.adelaide.edu.au/dspace/bitstream/2440/28910/1/hdl_28910.pdf>`_.
"""
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
import gensim.downloader as api
wv = api.load('word2vec-google-news-300')
for (index, word) in enumerate(wv.index_to_key):
    if index == 10:
        break
    print(f'word #{index}/{len(wv.index_to_key)} is {word}')
vec_king = wv['king']
try:
    vec_cameroon = wv['cameroon']
except KeyError:
    print("The word 'cameroon' does not appear in this model")
pairs = [('car', 'minivan'), ('car', 'bicycle'), ('car', 'airplane'), ('car', 'cereal'), ('car', 'communism')]
for (w1, w2) in pairs:
    print('%r\t%r\t%.2f' % (w1, w2, wv.similarity(w1, w2)))
print(wv.most_similar(positive=['car', 'minivan'], topn=5))
print(wv.doesnt_match(['fire', 'water', 'land', 'sea', 'air', 'car']))
from gensim.test.utils import datapath
from gensim import utils

class MyCorpus:
    """An iterator that yields sentences (lists of str)."""

    def __iter__(self):
        if False:
            i = 10
            return i + 15
        corpus_path = datapath('lee_background.cor')
        for line in open(corpus_path):
            yield utils.simple_preprocess(line)
import gensim.models
sentences = MyCorpus()
model = gensim.models.Word2Vec(sentences=sentences)
vec_king = model.wv['king']
for (index, word) in enumerate(wv.index_to_key):
    if index == 10:
        break
    print(f'word #{index}/{len(wv.index_to_key)} is {word}')
import tempfile
with tempfile.NamedTemporaryFile(prefix='gensim-model-', delete=False) as tmp:
    temporary_filepath = tmp.name
    model.save(temporary_filepath)
    new_model = gensim.models.Word2Vec.load(temporary_filepath)
model = gensim.models.Word2Vec(sentences, min_count=10)
model = gensim.models.Word2Vec(sentences, vector_size=200)
model = gensim.models.Word2Vec(sentences, workers=4)
model.wv.evaluate_word_analogies(datapath('questions-words.txt'))
model.wv.evaluate_word_pairs(datapath('wordsim353.tsv'))
model = gensim.models.Word2Vec.load(temporary_filepath)
more_sentences = [['Advanced', 'users', 'can', 'load', 'a', 'model', 'and', 'continue', 'training', 'it', 'with', 'more', 'sentences']]
model.build_vocab(more_sentences, update=True)
model.train(more_sentences, total_examples=model.corpus_count, epochs=model.epochs)
import os
os.remove(temporary_filepath)
model_with_loss = gensim.models.Word2Vec(sentences, min_count=1, compute_loss=True, hs=0, sg=1, seed=42)
training_loss = model_with_loss.get_latest_training_loss()
print(training_loss)
import io
import os
import gensim.models.word2vec
import gensim.downloader as api
import smart_open

def head(path, size):
    if False:
        print('Hello World!')
    with smart_open.open(path) as fin:
        return io.StringIO(fin.read(size))

def generate_input_data():
    if False:
        return 10
    lee_path = datapath('lee_background.cor')
    ls = gensim.models.word2vec.LineSentence(lee_path)
    ls.name = '25kB'
    yield ls
    text8_path = api.load('text8').fn
    labels = ('1MB', '10MB', '50MB', '100MB')
    sizes = (1024 ** 2, 10 * 1024 ** 2, 50 * 1024 ** 2, 100 * 1024 ** 2)
    for (l, s) in zip(labels, sizes):
        ls = gensim.models.word2vec.LineSentence(head(text8_path, s))
        ls.name = l
        yield ls
input_data = list(generate_input_data())
logging.root.level = logging.ERROR
import time
import numpy as np
import pandas as pd
train_time_values = []
seed_val = 42
sg_values = [0, 1]
hs_values = [0, 1]
fast = True
if fast:
    input_data_subset = input_data[:3]
else:
    input_data_subset = input_data
for data in input_data_subset:
    for sg_val in sg_values:
        for hs_val in hs_values:
            for loss_flag in [True, False]:
                time_taken_list = []
                for i in range(3):
                    start_time = time.time()
                    w2v_model = gensim.models.Word2Vec(data, compute_loss=loss_flag, sg=sg_val, hs=hs_val, seed=seed_val)
                    time_taken_list.append(time.time() - start_time)
                time_taken_list = np.array(time_taken_list)
                time_mean = np.mean(time_taken_list)
                time_std = np.std(time_taken_list)
                model_result = {'train_data': data.name, 'compute_loss': loss_flag, 'sg': sg_val, 'hs': hs_val, 'train_time_mean': time_mean, 'train_time_std': time_std}
                print('Word2vec model #%i: %s' % (len(train_time_values), model_result))
                train_time_values.append(model_result)
train_times_table = pd.DataFrame(train_time_values)
train_times_table = train_times_table.sort_values(by=['train_data', 'sg', 'hs', 'compute_loss'], ascending=[False, False, True, False])
print(train_times_table)
from sklearn.decomposition import IncrementalPCA
from sklearn.manifold import TSNE
import numpy as np

def reduce_dimensions(model):
    if False:
        return 10
    num_dimensions = 2
    vectors = np.asarray(model.wv.vectors)
    labels = np.asarray(model.wv.index_to_key)
    tsne = TSNE(n_components=num_dimensions, random_state=0)
    vectors = tsne.fit_transform(vectors)
    x_vals = [v[0] for v in vectors]
    y_vals = [v[1] for v in vectors]
    return (x_vals, y_vals, labels)
(x_vals, y_vals, labels) = reduce_dimensions(model)

def plot_with_plotly(x_vals, y_vals, labels, plot_in_notebook=True):
    if False:
        print('Hello World!')
    from plotly.offline import init_notebook_mode, iplot, plot
    import plotly.graph_objs as go
    trace = go.Scatter(x=x_vals, y=y_vals, mode='text', text=labels)
    data = [trace]
    if plot_in_notebook:
        init_notebook_mode(connected=True)
        iplot(data, filename='word-embedding-plot')
    else:
        plot(data, filename='word-embedding-plot.html')

def plot_with_matplotlib(x_vals, y_vals, labels):
    if False:
        i = 10
        return i + 15
    import matplotlib.pyplot as plt
    import random
    random.seed(0)
    plt.figure(figsize=(12, 12))
    plt.scatter(x_vals, y_vals)
    indices = list(range(len(labels)))
    selected_indices = random.sample(indices, 25)
    for i in selected_indices:
        plt.annotate(labels[i], (x_vals[i], y_vals[i]))
try:
    get_ipython()
except Exception:
    plot_function = plot_with_matplotlib
else:
    plot_function = plot_with_plotly
plot_function(x_vals, y_vals, labels)