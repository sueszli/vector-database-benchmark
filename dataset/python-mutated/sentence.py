import nltk
import os
import itertools
import sys
from nltk.tokenize import word_tokenize

def read_localfile(fileName):
    if False:
        for i in range(10):
            print('nop')
    lines = []
    with open(fileName) as f:
        for line in f:
            lines.append(line)
    f.close()
    return lines

def sentences_split(line):
    if False:
        i = 10
        return i + 15
    nltk.data.path.append(os.environ.get('PWD'))
    sent_tokenizer = nltk.tokenize.PunktSentenceTokenizer()
    sentenized = sent_tokenizer.tokenize(line)
    return sentenized

def sentences_bipadding(sent):
    if False:
        while True:
            i = 10
    return 'SENTENCESTART ' + sent + ' SENTENCEEND'

def sentence_tokenizer(sentences):
    if False:
        for i in range(10):
            print('nop')
    tokenized_sents = nltk.word_tokenize(sentences)
    return tokenized_sents