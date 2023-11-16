"""
How to reproduce the doc2vec 'Paragraph Vector' paper
=====================================================

Shows how to reproduce results of the "Distributed Representation of Sentences and Documents" paper by Le and Mikolov using Gensim.

"""
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
import collections
SentimentDocument = collections.namedtuple('SentimentDocument', 'words tags split sentiment')
import io
import re
import tarfile
import os.path
import smart_open
import gensim.utils

def download_dataset(url='http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'):
    if False:
        i = 10
        return i + 15
    fname = url.split('/')[-1]
    if os.path.isfile(fname):
        return fname
    try:
        kwargs = {'compression': smart_open.compression.NO_COMPRESSION}
        fin = smart_open.open(url, 'rb', **kwargs)
    except (AttributeError, TypeError):
        kwargs = {'ignore_ext': True}
        fin = smart_open.open(url, 'rb', **kwargs)
    if fin:
        with smart_open.open(fname, 'wb', **kwargs) as fout:
            while True:
                buf = fin.read(io.DEFAULT_BUFFER_SIZE)
                if not buf:
                    break
                fout.write(buf)
        fin.close()
    return fname

def create_sentiment_document(name, text, index):
    if False:
        while True:
            i = 10
    (_, split, sentiment_str, _) = name.split('/')
    sentiment = {'pos': 1.0, 'neg': 0.0, 'unsup': None}[sentiment_str]
    if sentiment is None:
        split = 'extra'
    tokens = gensim.utils.to_unicode(text).split()
    return SentimentDocument(tokens, [index], split, sentiment)

def extract_documents():
    if False:
        return 10
    fname = download_dataset()
    index = 0
    with tarfile.open(fname, mode='r:gz') as tar:
        for member in tar.getmembers():
            if re.match('aclImdb/(train|test)/(pos|neg|unsup)/\\d+_\\d+.txt$', member.name):
                member_bytes = tar.extractfile(member).read()
                member_text = member_bytes.decode('utf-8', errors='replace')
                assert member_text.count('\n') == 0
                yield create_sentiment_document(member.name, member_text, index)
                index += 1
alldocs = list(extract_documents())
print(alldocs[27])
train_docs = [doc for doc in alldocs if doc.split == 'train']
test_docs = [doc for doc in alldocs if doc.split == 'test']
print(f'{len(alldocs)} docs: {len(train_docs)} train-sentiment, {len(test_docs)} test-sentiment')
import multiprocessing
from collections import OrderedDict
import gensim.models.doc2vec
assert gensim.models.doc2vec.FAST_VERSION > -1, 'This will be painfully slow otherwise'
from gensim.models.doc2vec import Doc2Vec
common_kwargs = dict(vector_size=100, epochs=20, min_count=2, sample=0, workers=multiprocessing.cpu_count(), negative=5, hs=0)
simple_models = [Doc2Vec(dm=0, **common_kwargs), Doc2Vec(dm=1, window=10, alpha=0.05, comment='alpha=0.05', **common_kwargs), Doc2Vec(dm=1, dm_concat=1, window=5, **common_kwargs)]
for model in simple_models:
    model.build_vocab(alldocs)
    print(f'{model} vocabulary scanned & state initialized')
models_by_name = OrderedDict(((str(model), model) for model in simple_models))
from gensim.test.test_doc2vec import ConcatenatedDoc2Vec
models_by_name['dbow+dmm'] = ConcatenatedDoc2Vec([simple_models[0], simple_models[1]])
models_by_name['dbow+dmc'] = ConcatenatedDoc2Vec([simple_models[0], simple_models[2]])
import numpy as np
import statsmodels.api as sm
from random import sample

def logistic_predictor_from_data(train_targets, train_regressors):
    if False:
        for i in range(10):
            print('nop')
    'Fit a statsmodel logistic predictor on supplied data'
    logit = sm.Logit(train_targets, train_regressors)
    predictor = logit.fit(disp=0)
    return predictor

def error_rate_for_model(test_model, train_set, test_set):
    if False:
        for i in range(10):
            print('nop')
    'Report error rate on test_doc sentiments, using supplied model and train_docs'
    train_targets = [doc.sentiment for doc in train_set]
    train_regressors = [test_model.dv[doc.tags[0]] for doc in train_set]
    train_regressors = sm.add_constant(train_regressors)
    predictor = logistic_predictor_from_data(train_targets, train_regressors)
    test_regressors = [test_model.dv[doc.tags[0]] for doc in test_set]
    test_regressors = sm.add_constant(test_regressors)
    test_predictions = predictor.predict(test_regressors)
    corrects = sum(np.rint(test_predictions) == [doc.sentiment for doc in test_set])
    errors = len(test_predictions) - corrects
    error_rate = float(errors) / len(test_predictions)
    return (error_rate, errors, len(test_predictions), predictor)
from collections import defaultdict
error_rates = defaultdict(lambda : 1.0)
from random import shuffle
shuffled_alldocs = alldocs[:]
shuffle(shuffled_alldocs)
for model in simple_models:
    print(f'Training {model}')
    model.train(shuffled_alldocs, total_examples=len(shuffled_alldocs), epochs=model.epochs)
    print(f'\nEvaluating {model}')
    (err_rate, err_count, test_count, predictor) = error_rate_for_model(model, train_docs, test_docs)
    error_rates[str(model)] = err_rate
    print('\n%f %s\n' % (err_rate, model))
for model in [models_by_name['dbow+dmm'], models_by_name['dbow+dmc']]:
    print(f'\nEvaluating {model}')
    (err_rate, err_count, test_count, predictor) = error_rate_for_model(model, train_docs, test_docs)
    error_rates[str(model)] = err_rate
    print(f'\n{err_rate} {model}\n')
print('Err_rate Model')
for (rate, name) in sorted(((rate, name) for (name, rate) in error_rates.items())):
    print(f'{rate} {name}')
doc_id = np.random.randint(len(simple_models[0].dv))
print(f'for doc {doc_id}...')
for model in simple_models:
    inferred_docvec = model.infer_vector(alldocs[doc_id].words)
    print(f'{model}:\n {model.dv.most_similar([inferred_docvec], topn=3)}')
import random
doc_id = np.random.randint(len(simple_models[0].dv))
model = random.choice(simple_models)
sims = model.dv.most_similar(doc_id, topn=len(model.dv))
print(f"TARGET ({doc_id}): «{' '.join(alldocs[doc_id].words)}»\n")
print(f'SIMILAR/DISSIMILAR DOCS PER MODEL {model}%s:\n')
for (label, index) in [('MOST', 0), ('MEDIAN', len(sims) // 2), ('LEAST', len(sims) - 1)]:
    s = sims[index]
    i = sims[index][0]
    words = ' '.join(alldocs[i].words)
    print(f'{label} {s}: «{words}»\n')
import random
word_models = simple_models[:]

def pick_random_word(model, threshold=10):
    if False:
        i = 10
        return i + 15
    while True:
        word = random.choice(model.wv.index_to_key)
        if model.wv.get_vecattr(word, 'count') > threshold:
            return word
target_word = pick_random_word(word_models[0])
for model in word_models:
    print(f'target_word: {repr(target_word)} model: {model} similar words:')
    for (i, (word, sim)) in enumerate(model.wv.most_similar(target_word, topn=10), 1):
        print(f'    {i}. {sim:.2f} {repr(word)}')
    print()
from gensim.test.utils import datapath
questions_filename = datapath('questions-words.txt')
for model in word_models:
    (score, sections) = model.wv.evaluate_word_analogies(questions_filename)
    (correct, incorrect) = (len(sections[-1]['correct']), len(sections[-1]['incorrect']))
    print(f'{model}: {float(correct * 100) / (correct + incorrect):0.2f}%% correct ({correct} of {correct + incorrect}')