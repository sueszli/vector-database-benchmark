from Util.Import import load_file, get_files
import json
import math
import re

class Vectoriser:
    """
    Class for building vectors from tweets
    """

    def __init__(self):
        if False:
            return 10
        self.dictionary = {}
        self.id_reference = {}
        self.idf = {}
        self.document_count = 0
        self._id = 0

    def generate_id(self):
        if False:
            print('Hello World!')
        '\n        :return: numbers incrementing by 1 each time function is called\n        '
        self._id += 1
        return self._id

    def vectorise(self, string):
        if False:
            while True:
                i = 10
        '\n        convert string into a bag of words vector\n        :param string: text sentence\n        :return: sparse vector in dictionary format\n        '
        sentence = self.tokeniser(string)
        vector = {}
        processed = []
        self.document_count += 1
        for word in sentence:
            try:
                id = self.dictionary[word]
            except KeyError:
                id = self.generate_id()
                self.dictionary[word] = id
                self.id_reference[id] = word
            if word not in processed:
                processed.append(word)
                try:
                    self.idf[id] += 1
                except KeyError:
                    self.idf[id] = 1
            try:
                vector[id] += 1
            except KeyError:
                vector[id] = 1
        return vector

    def get_idf(self):
        if False:
            return 10
        for id in self.idf.keys():
            self.idf[id] = math.log((self.document_count - self.idf[id] + 0.5) / (self.idf[id] + 0.5), 2)
        return self.idf

    def add_vector(self, df):
        if False:
            for i in range(10):
                print('nop')
        '\n        add vector to dataframe\n        :param df: dataframe with no vector\n        :return: dataframe with vector\n        '
        df['vector'] = df['text'].map(lambda text: self.vectorise(text))
        return df

    def tokeniser(self, string):
        if False:
            while True:
                i = 10
        '\n        split full sentence into list of tokens\n        :param string: text sentence\n        :return: list of tokens\n        '
        string = string.lower()
        list = re.split('[, \\-!?()]+', string)
        numsPunc = [str(i) for i in range(10)] + ['@', '...', ':', "'", '"', '…', '.', ',']
        for i in range(len(list)):
            if '#' in list[i]:
                list[i] = list[i].replace('#', '')
            if 'https://' in list[i]:
                list[i] = list[i].replace(list[i], '')
            for num in numsPunc:
                list[i] = list[i].replace(num, '')
        list = [word for word in list if len(word) > 2]
        return list
files = get_files('../data')
vec = Vectoriser()
no_files = len(files)
for (i, file) in enumerate(files):
    if i % 20 == 0:
        print('Adding vector to data: {}/{} files\r'.format(i + 1, no_files), end='\r')
    data = load_file(file)
    data = data[['username', 'date', 'text', 'profileLocation', 'latitude', 'longitude']]
    data = vec.add_vector(data)
    if i == no_files - 1:
        print('Vectorising: Complete                              ')
    data.to_json(files[i], orient='index')
with open('../dictionaries/term_to_id_dictionary.json', 'w') as fp:
    json.dump(vec.dictionary, fp)
with open('../dictionaries/id_to_term_dictionary.json', 'w') as fp:
    json.dump(vec.id_reference, fp)
vec.get_idf()
with open('../dictionaries/idf_reference.json', 'w') as fp:
    json.dump(vec.idf, fp)