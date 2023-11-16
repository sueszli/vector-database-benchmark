import numpy as np
import re
import itertools
from collections import Counter
'\nOriginal taken from https://github.com/dennybritz/cnn-text-classification-tf\n'

def clean_str(string):
    if False:
        while True:
            i = 10
    '\n    Tokenization/string cleaning for all datasets except for SST.\n    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py\n    '
    string = re.sub("[^A-Za-z0-9(),!?\\'\\`]", ' ', string)
    string = re.sub("\\'s", " 's", string)
    string = re.sub("\\'ve", " 've", string)
    string = re.sub("n\\'t", " n't", string)
    string = re.sub("\\'re", " 're", string)
    string = re.sub("\\'d", " 'd", string)
    string = re.sub("\\'ll", " 'll", string)
    string = re.sub(',', ' , ', string)
    string = re.sub('!', ' ! ', string)
    string = re.sub('\\(', ' \\( ', string)
    string = re.sub('\\)', ' \\) ', string)
    string = re.sub('\\?', ' \\? ', string)
    string = re.sub('\\s{2,}', ' ', string)
    return string.strip().lower()

def load_data_and_labels():
    if False:
        while True:
            i = 10
    '\n    Loads MR polarity data from files, splits the data into words and generates labels.\n    Returns split sentences and labels.\n    '
    positive_examples = list(open('../data/word_embeddings/rt-polarity.pos', encoding='ISO-8859-1').readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open('../data/word_embeddings/rt-polarity.neg', encoding='ISO-8859-1').readlines())
    negative_examples = [s.strip() for s in negative_examples]
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    x_text = [s.split(' ') for s in x_text]
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]

def pad_sentences(sentences, padding_word='<PAD/>'):
    if False:
        return 10
    '\n    Pads all sentences to the same length. The length is defined by the longest sentence.\n    Returns padded sentences.\n    '
    sequence_length = max((len(x) for x in sentences))
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)
    return padded_sentences

def build_vocab(sentences):
    if False:
        i = 10
        return i + 15
    '\n    Builds a vocabulary mapping from word to index based on the sentences.\n    Returns vocabulary mapping and inverse vocabulary mapping.\n    '
    word_counts = Counter(itertools.chain(*sentences))
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    vocabulary = {x: i for (i, x) in enumerate(vocabulary_inv)}
    return [vocabulary, vocabulary_inv]

def build_input_data(sentences, labels, vocabulary):
    if False:
        return 10
    '\n    Maps sentencs and labels to vectors based on a vocabulary.\n    '
    x = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])
    y = np.array(labels)
    return [x, y]

def load_data():
    if False:
        return 10
    '\n    Loads and preprocessed data for the MR dataset.\n    Returns input vectors, labels, vocabulary, and inverse vocabulary.\n    '
    (sentences, labels) = load_data_and_labels()
    sentences_padded = pad_sentences(sentences)
    (vocabulary, vocabulary_inv) = build_vocab(sentences_padded)
    (x, y) = build_input_data(sentences_padded, labels, vocabulary)
    return [x, y, vocabulary, vocabulary_inv]

def batch_iter(data, batch_size, num_epochs):
    if False:
        while True:
            i = 10
    '\n    Generates a batch iterator for a dataset.\n    '
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data) / batch_size) + 1
    for epoch in range(num_epochs):
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_data = data[shuffle_indices]
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]