from rte.riedel.fever_features import TermFrequencyFeatureFunction

class FNCTermFrequencyFeatureFunction(TermFrequencyFeatureFunction):

    def __init__(self, doc_db, lim_unigram=5000):
        if False:
            while True:
                i = 10
        super().__init__(doc_db, lim_unigram)
        self.ename = 'evidence'

    def bodies(self, data):
        if False:
            return 10
        return [self.doc_db.get_doc_text(id) for id in set(self.body_id(data))]

    def texts(self, data):
        if False:
            for i in range(10):
                print('nop')
        return [self.doc_db.get_doc_text(id) for id in self.body_id(data)]

    def body_id(self, data):
        if False:
            for i in range(10):
                print('nop')
        return [datum[self.ename] for datum in data]