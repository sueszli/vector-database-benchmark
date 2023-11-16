"""
How to Compare LDA Models
=========================

Demonstrates how you can visualize and compare trained topic models.

"""
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
from string import punctuation
from nltk import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.datasets import fetch_20newsgroups
newsgroups = fetch_20newsgroups()
eng_stopwords = set(stopwords.words('english'))
tokenizer = RegexpTokenizer('\\s+', gaps=True)
stemmer = PorterStemmer()
translate_tab = {ord(p): u' ' for p in punctuation}

def text2tokens(raw_text):
    if False:
        for i in range(10):
            print('nop')
    'Split the raw_text string into a list of stemmed tokens.'
    clean_text = raw_text.lower().translate(translate_tab)
    tokens = [token.strip() for token in tokenizer.tokenize(clean_text)]
    tokens = [token for token in tokens if token not in eng_stopwords]
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    return [token for token in stemmed_tokens if len(token) > 2]
dataset = [text2tokens(txt) for txt in newsgroups['data']]
from gensim.corpora import Dictionary
dictionary = Dictionary(documents=dataset, prune_at=None)
dictionary.filter_extremes(no_below=5, no_above=0.3, keep_n=None)
dictionary.compactify()
d2b_dataset = [dictionary.doc2bow(doc) for doc in dataset]
from gensim.models import LdaMulticore
num_topics = 15
lda_fst = LdaMulticore(corpus=d2b_dataset, num_topics=num_topics, id2word=dictionary, workers=4, eval_every=None, passes=10, batch=True)
lda_snd = LdaMulticore(corpus=d2b_dataset, num_topics=num_topics, id2word=dictionary, workers=4, eval_every=None, passes=20, batch=True)

def plot_difference_plotly(mdiff, title='', annotation=None):
    if False:
        while True:
            i = 10
    'Plot the difference between models.\n\n    Uses plotly as the backend.'
    import plotly.graph_objs as go
    import plotly.offline as py
    annotation_html = None
    if annotation is not None:
        annotation_html = [['+++ {}<br>--- {}'.format(', '.join(int_tokens), ', '.join(diff_tokens)) for (int_tokens, diff_tokens) in row] for row in annotation]
    data = go.Heatmap(z=mdiff, colorscale='RdBu', text=annotation_html)
    layout = go.Layout(width=950, height=950, title=title, xaxis=dict(title='topic'), yaxis=dict(title='topic'))
    py.iplot(dict(data=[data], layout=layout))

def plot_difference_matplotlib(mdiff, title='', annotation=None):
    if False:
        i = 10
        return i + 15
    'Helper function to plot difference between models.\n\n    Uses matplotlib as the backend.'
    import matplotlib.pyplot as plt
    (fig, ax) = plt.subplots(figsize=(18, 14))
    data = ax.imshow(mdiff, cmap='RdBu_r', origin='lower')
    plt.title(title)
    plt.colorbar(data)
try:
    get_ipython()
    import plotly.offline as py
except Exception:
    plot_difference = plot_difference_matplotlib
else:
    py.init_notebook_mode()
    plot_difference = plot_difference_plotly
print(LdaMulticore.diff.__doc__)
import numpy as np
mdiff = np.ones((num_topics, num_topics))
np.fill_diagonal(mdiff, 0.0)
plot_difference(mdiff, title='Topic difference (one model) in ideal world')
(mdiff, annotation) = lda_fst.diff(lda_fst, distance='jaccard', num_words=50)
plot_difference(mdiff, title='Topic difference (one model) [jaccard distance]', annotation=annotation)
(mdiff, annotation) = lda_fst.diff(lda_fst, distance='hellinger', num_words=50)
plot_difference(mdiff, title='Topic difference (one model)[hellinger distance]', annotation=annotation)
(mdiff, annotation) = lda_fst.diff(lda_snd, distance='jaccard', num_words=50)
plot_difference(mdiff, title='Topic difference (two models)[jaccard distance]', annotation=annotation)