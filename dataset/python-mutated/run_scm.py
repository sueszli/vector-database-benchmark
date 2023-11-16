"""
Soft Cosine Measure
===================

Demonstrates using Gensim's implemenation of the SCM.

"""
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
img = mpimg.imread('scm-hello.png')
imgplot = plt.imshow(img)
plt.axis('off')
plt.show()
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
sentence_obama = 'Obama speaks to the media in Illinois'
sentence_president = 'The president greets the press in Chicago'
sentence_orange = 'Oranges are my favorite fruit'
from nltk.corpus import stopwords
from nltk import download
download('stopwords')
stop_words = stopwords.words('english')

def preprocess(sentence):
    if False:
        print('Hello World!')
    return [w for w in sentence.lower().split() if w not in stop_words]
sentence_obama = preprocess(sentence_obama)
sentence_president = preprocess(sentence_president)
sentence_orange = preprocess(sentence_orange)
from gensim.corpora import Dictionary
documents = [sentence_obama, sentence_president, sentence_orange]
dictionary = Dictionary(documents)
sentence_obama = dictionary.doc2bow(sentence_obama)
sentence_president = dictionary.doc2bow(sentence_president)
sentence_orange = dictionary.doc2bow(sentence_orange)
from gensim.models import TfidfModel
documents = [sentence_obama, sentence_president, sentence_orange]
tfidf = TfidfModel(documents)
sentence_obama = tfidf[sentence_obama]
sentence_president = tfidf[sentence_president]
sentence_orange = tfidf[sentence_orange]
import gensim.downloader as api
model = api.load('word2vec-google-news-300')
from gensim.similarities import SparseTermSimilarityMatrix, WordEmbeddingSimilarityIndex
termsim_index = WordEmbeddingSimilarityIndex(model)
termsim_matrix = SparseTermSimilarityMatrix(termsim_index, dictionary, tfidf)
similarity = termsim_matrix.inner_product(sentence_obama, sentence_president, normalized=(True, True))
print('similarity = %.4f' % similarity)
similarity = termsim_matrix.inner_product(sentence_obama, sentence_orange, normalized=(True, True))
print('similarity = %.4f' % similarity)