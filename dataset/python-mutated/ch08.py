import os
import sys
import tarfile
import time
import pyprind
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import re
from nltk.stem.porter import PorterStemmer
import nltk
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.decomposition import LatentDirichletAllocation
from distutils.version import LooseVersion as Version
from sklearn import __version__ as sklearn_version
source = 'http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'
target = 'aclImdb_v1.tar.gz'

def reporthook(count, block_size, total_size):
    if False:
        i = 10
        return i + 15
    global start_time
    if count == 0:
        start_time = time.time()
        return
    duration = time.time() - start_time
    progress_size = int(count * block_size)
    speed = progress_size / (1024.0 ** 2 * duration)
    percent = count * block_size * 100.0 / total_size
    sys.stdout.write('\r%d%% | %d MB | %.2f MB/s | %d sec elapsed' % (percent, progress_size / 1024.0 ** 2, speed, duration))
    sys.stdout.flush()
if not os.path.isdir('aclImdb') and (not os.path.isfile('aclImdb_v1.tar.gz')):
    if sys.version_info < (3, 0):
        import urllib
        urllib.urlretrieve(source, target, reporthook)
    else:
        import urllib.request
        urllib.request.urlretrieve(source, target, reporthook)
if not os.path.isdir('aclImdb'):
    with tarfile.open(target, 'r:gz') as tar:
        tar.extractall()
basepath = 'aclImdb'
labels = {'pos': 1, 'neg': 0}
pbar = pyprind.ProgBar(50000)
df = pd.DataFrame()
for s in ('test', 'train'):
    for l in ('pos', 'neg'):
        path = os.path.join(basepath, s, l)
        for file in os.listdir(path):
            with open(os.path.join(path, file), 'r', encoding='utf-8') as infile:
                txt = infile.read()
            df = df.append([[txt, labels[l]]], ignore_index=True)
            pbar.update()
df.columns = ['review', 'sentiment']
np.random.seed(0)
df = df.reindex(np.random.permutation(df.index))
df.to_csv('movie_data.csv', index=False, encoding='utf-8')
df = pd.read_csv('movie_data.csv', encoding='utf-8')
df.head(3)
count = CountVectorizer()
docs = np.array(['The sun is shining', 'The weather is sweet', 'The sun is shining, the weather is sweet, and one and one is two'])
bag = count.fit_transform(docs)
print(count.vocabulary_)
print(bag.toarray())
np.set_printoptions(precision=2)
tfidf = TfidfTransformer(use_idf=True, norm='l2', smooth_idf=True)
print(tfidf.fit_transform(count.fit_transform(docs)).toarray())
tf_is = 3
n_docs = 3
idf_is = np.log((n_docs + 1) / (3 + 1))
tfidf_is = tf_is * (idf_is + 1)
print('tf-idf of term "is" = %.2f' % tfidf_is)
tfidf = TfidfTransformer(use_idf=True, norm=None, smooth_idf=True)
raw_tfidf = tfidf.fit_transform(count.fit_transform(docs)).toarray()[-1]
raw_tfidf
l2_tfidf = raw_tfidf / np.sqrt(np.sum(raw_tfidf ** 2))
l2_tfidf
df.loc[0, 'review'][-50:]

def preprocessor(text):
    if False:
        i = 10
        return i + 15
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\\)|\\(|D|P)', text)
    text = re.sub('[\\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '')
    return text
preprocessor(df.loc[0, 'review'][-50:])
preprocessor('</a>This :) is :( a test :-)!')
df['review'] = df['review'].apply(preprocessor)
porter = PorterStemmer()

def tokenizer(text):
    if False:
        print('Hello World!')
    return text.split()

def tokenizer_porter(text):
    if False:
        return 10
    return [porter.stem(word) for word in text.split()]
tokenizer('runners like running and thus they run')
tokenizer_porter('runners like running and thus they run')
nltk.download('stopwords')
stop = stopwords.words('english')
[w for w in tokenizer_porter('a runner likes running and runs a lot')[-10:] if w not in stop]
X_train = df.loc[:25000, 'review'].values
y_train = df.loc[:25000, 'sentiment'].values
X_test = df.loc[25000:, 'review'].values
y_test = df.loc[25000:, 'sentiment'].values
tfidf = TfidfVectorizer(strip_accents=None, lowercase=False, preprocessor=None)
param_grid = [{'vect__ngram_range': [(1, 1)], 'vect__stop_words': [stop, None], 'vect__tokenizer': [tokenizer, tokenizer_porter], 'clf__penalty': ['l1', 'l2'], 'clf__C': [1.0, 10.0, 100.0]}, {'vect__ngram_range': [(1, 1)], 'vect__stop_words': [stop, None], 'vect__tokenizer': [tokenizer, tokenizer_porter], 'vect__use_idf': [False], 'vect__norm': [None], 'clf__penalty': ['l1', 'l2'], 'clf__C': [1.0, 10.0, 100.0]}]
lr_tfidf = Pipeline([('vect', tfidf), ('clf', LogisticRegression(random_state=0))])
gs_lr_tfidf = GridSearchCV(lr_tfidf, param_grid, scoring='accuracy', cv=5, verbose=1, n_jobs=-1)
if 'TRAVIS' in os.environ:
    gs_lr_tfidf.verbose = 2
    X_train = df.loc[:250, 'review'].values
    y_train = df.loc[:250, 'sentiment'].values
    X_test = df.loc[25000:25250, 'review'].values
    y_test = df.loc[25000:25250, 'sentiment'].values
gs_lr_tfidf.fit(X_train, y_train)
print('Best parameter set: %s ' % gs_lr_tfidf.best_params_)
print('CV Accuracy: %.3f' % gs_lr_tfidf.best_score_)
clf = gs_lr_tfidf.best_estimator_
print('Test Accuracy: %.3f' % clf.score(X_test, y_test))
np.random.seed(0)
np.set_printoptions(precision=6)
y = [np.random.randint(3) for i in range(25)]
X = (y + np.random.randn(25)).reshape(-1, 1)
cv5_idx = list(StratifiedKFold(n_splits=5, shuffle=False, random_state=0).split(X, y))
cross_val_score(LogisticRegression(random_state=123), X, y, cv=cv5_idx)
gs = GridSearchCV(LogisticRegression(), {}, cv=cv5_idx, verbose=3).fit(X, y)
gs.best_score_
cross_val_score(LogisticRegression(), X, y, cv=cv5_idx).mean()

def tokenizer(text):
    if False:
        print('Hello World!')
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\\)|\\(|D|P)', text.lower())
    text = re.sub('[\\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '')
    tokenized = [w for w in text.split() if w not in stop]
    return tokenized

def stream_docs(path):
    if False:
        for i in range(10):
            print('nop')
    with open(path, 'r', encoding='utf-8') as csv:
        next(csv)
        for line in csv:
            (text, label) = (line[:-3], int(line[-2]))
            yield (text, label)
next(stream_docs(path='movie_data.csv'))

def get_minibatch(doc_stream, size):
    if False:
        for i in range(10):
            print('nop')
    (docs, y) = ([], [])
    try:
        for _ in range(size):
            (text, label) = next(doc_stream)
            docs.append(text)
            y.append(label)
    except StopIteration:
        return (None, None)
    return (docs, y)
vect = HashingVectorizer(decode_error='ignore', n_features=2 ** 21, preprocessor=None, tokenizer=tokenizer)
if Version(sklearn_version) < '0.18':
    clf = SGDClassifier(loss='log', random_state=1, n_iter=1)
else:
    clf = SGDClassifier(loss='log', random_state=1, max_iter=1)
doc_stream = stream_docs(path='movie_data.csv')
pbar = pyprind.ProgBar(45)
classes = np.array([0, 1])
for _ in range(45):
    (X_train, y_train) = get_minibatch(doc_stream, size=1000)
    if not X_train:
        break
    X_train = vect.transform(X_train)
    clf.partial_fit(X_train, y_train, classes=classes)
    pbar.update()
(X_test, y_test) = get_minibatch(doc_stream, size=5000)
X_test = vect.transform(X_test)
print('Accuracy: %.3f' % clf.score(X_test, y_test))
clf = clf.partial_fit(X_test, y_test)
df = pd.read_csv('movie_data.csv', encoding='utf-8')
df.head(3)
if 'TRAVIS' in os.environ:
    df.loc[:500].to_csv('movie_data.csv')
    df = pd.read_csv('movie_data.csv', nrows=500)
    print('SMALL DATA SUBSET CREATED FOR TESTING')
count = CountVectorizer(stop_words='english', max_df=0.1, max_features=5000)
X = count.fit_transform(df['review'].values)
lda = LatentDirichletAllocation(n_topics=10, random_state=123, learning_method='batch')
X_topics = lda.fit_transform(X)
lda.components_.shape
n_top_words = 5
feature_names = count.get_feature_names()
for (topic_idx, topic) in enumerate(lda.components_):
    print('Topic %d:' % (topic_idx + 1))
    print(' '.join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]))
horror = X_topics[:, 5].argsort()[::-1]
for (iter_idx, movie_idx) in enumerate(horror[:3]):
    print('\nHorror movie #%d:' % (iter_idx + 1))
    print(df['review'][movie_idx][:300], '...')