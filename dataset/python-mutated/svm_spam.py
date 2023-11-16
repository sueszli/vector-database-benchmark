from __future__ import print_function, division
from builtins import range
from sklearn.svm import SVC
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from wordcloud import WordCloud
df = pd.read_csv('../large_files/spam.csv', encoding='ISO-8859-1')
df = df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1)
df.columns = ['labels', 'data']
df['b_labels'] = df['labels'].map({'ham': 0, 'spam': 1})
Y = df['b_labels'].values
tfidf = TfidfVectorizer(decode_error='ignore')
X = tfidf.fit_transform(df['data'])
(Xtrain, Xtest, Ytrain, Ytest) = train_test_split(X, Y, test_size=0.33)
model = SVC(kernel='linear', C=2.0)
t0 = datetime.now()
model.fit(Xtrain, Ytrain)
print('train duration:', datetime.now() - t0)
t0 = datetime.now()
print('train score:', model.score(Xtrain, Ytrain), 'duration:', datetime.now() - t0)
t0 = datetime.now()
print('test score:', model.score(Xtest, Ytest), 'duration:', datetime.now() - t0)

def visualize(label):
    if False:
        print('Hello World!')
    words = ''
    for msg in df[df['labels'] == label]['data']:
        msg = msg.lower()
        words += msg + ' '
    wordcloud = WordCloud(width=600, height=400).generate(words)
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.title(label)
    plt.show()
visualize('spam')
visualize('ham')
df['predictions'] = model.predict(X)
print('*** things that should be spam ***')
sneaky_spam = df[(df['predictions'] == 0) & (df['b_labels'] == 1)]['data']
for msg in sneaky_spam:
    print(msg)
print('*** things that should not be spam ***')
not_actually_spam = df[(df['predictions'] == 1) & (df['b_labels'] == 0)]['data']
for msg in not_actually_spam:
    print(msg)