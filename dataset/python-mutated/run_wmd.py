"""
Word Mover's Distance
=====================

Demonstrates using Gensim's implemenation of the WMD.

"""
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
img = mpimg.imread('wmd-obama.png')
imgplot = plt.imshow(img)
plt.axis('off')
plt.show()
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
sentence_obama = 'Obama speaks to the media in Illinois'
sentence_president = 'The president greets the press in Chicago'
from nltk.corpus import stopwords
from nltk import download
download('stopwords')
stop_words = stopwords.words('english')

def preprocess(sentence):
    if False:
        for i in range(10):
            print('nop')
    return [w for w in sentence.lower().split() if w not in stop_words]
sentence_obama = preprocess(sentence_obama)
sentence_president = preprocess(sentence_president)
import gensim.downloader as api
model = api.load('word2vec-google-news-300')
distance = model.wmdistance(sentence_obama, sentence_president)
print('distance = %.4f' % distance)
sentence_orange = preprocess('Oranges are my favorite fruit')
distance = model.wmdistance(sentence_obama, sentence_orange)
print('distance = %.4f' % distance)