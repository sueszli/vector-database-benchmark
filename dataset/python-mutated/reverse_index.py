from collections import defaultdict
import pickle
from tqdm import tqdm

class ReverseIndex:

    def __init__(self, docs, preprocessing):
        if False:
            i = 10
            return i + 15
        self.lookup = defaultdict(set)
        self.preprocess = preprocessing
        if docs is not None:
            for (title, words) in tqdm(docs):
                self.add(title, self.preprocess(words))

    def add(self, title, words):
        if False:
            i = 10
            return i + 15
        for word in words:
            self.lookup[word].add(title)

    def docs(self, phrase):
        if False:
            return 10
        ret = []
        for word in self.preprocess(phrase):
            ret.extend(self.lookup[word])
        return ret

    def save(self, file):
        if False:
            i = 10
            return i + 15
        with open(file, 'wb+') as f:
            pickle.dump(self.lookup, f)

    def load(self, file):
        if False:
            print('Hello World!')
        with open(file, 'rb') as f:
            self.lookup = pickle.load(f)