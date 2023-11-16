import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import string
import numpy as np
import pandas as pd

def tokenize_corpus(corpus, num_words=-1):
    if False:
        i = 10
        return i + 15
    if num_words > -1:
        tokenizer = Tokenizer(num_words=num_words)
    else:
        tokenizer = Tokenizer()
    tokenizer.fit_on_texts(corpus)
    return tokenizer

def create_lyrics_corpus(dataset, field):
    if False:
        print('Hello World!')
    dataset[field] = dataset[field].str.replace('[{}]'.format(string.punctuation), '')
    dataset[field] = dataset[field].str.lower()
    lyrics = dataset[field].str.cat()
    corpus = lyrics.split('\n')
    for l in range(len(corpus)):
        corpus[l] = corpus[l].rstrip()
    corpus = [l for l in corpus if l != '']
    return corpus
dataset = pd.read_csv('/tmp/songdata.csv', dtype=str)
corpus = create_lyrics_corpus(dataset, 'text')
tokenizer = tokenize_corpus(corpus)
max_sequence_len = 0
for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]
    max_sequence_len = max(max_sequence_len, len(token_list))
model = tf.keras.models.load_model('path/to/model')
seed_text = 'im feeling chills'
next_words = 100
for _ in range(next_words):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
    predicted = np.argmax(model.predict(token_list), axis=-1)
    output_word = ''
    for (word, index) in tokenizer.word_index.items():
        if index == predicted:
            output_word = word
            break
    seed_text += ' ' + output_word
print(seed_text)