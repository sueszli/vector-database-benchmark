"""
Simple classifier for RTE corpus.

It calculates the overlap in words and named entities between text and
hypothesis, and also whether there are words / named entities in the
hypothesis which fail to occur in the text, since this is an indicator that
the hypothesis is more informative than (i.e not entailed by) the text.

TO DO: better Named Entity classification
TO DO: add lemmatization
"""
from nltk.classify.maxent import MaxentClassifier
from nltk.classify.util import accuracy
from nltk.tokenize import RegexpTokenizer

class RTEFeatureExtractor:
    """
    This builds a bag of words for both the text and the hypothesis after
    throwing away some stopwords, then calculates overlap and difference.
    """

    def __init__(self, rtepair, stop=True, use_lemmatize=False):
        if False:
            i = 10
            return i + 15
        '\n        :param rtepair: a ``RTEPair`` from which features should be extracted\n        :param stop: if ``True``, stopwords are thrown away.\n        :type stop: bool\n        '
        self.stop = stop
        self.stopwords = {'a', 'the', 'it', 'they', 'of', 'in', 'to', 'is', 'have', 'are', 'were', 'and', 'very', '.', ','}
        self.negwords = {'no', 'not', 'never', 'failed', 'rejected', 'denied'}
        tokenizer = RegexpTokenizer('[\\w.@:/]+|\\w+|\\$[\\d.]+')
        self.text_tokens = tokenizer.tokenize(rtepair.text)
        self.hyp_tokens = tokenizer.tokenize(rtepair.hyp)
        self.text_words = set(self.text_tokens)
        self.hyp_words = set(self.hyp_tokens)
        if use_lemmatize:
            self.text_words = {self._lemmatize(token) for token in self.text_tokens}
            self.hyp_words = {self._lemmatize(token) for token in self.hyp_tokens}
        if self.stop:
            self.text_words = self.text_words - self.stopwords
            self.hyp_words = self.hyp_words - self.stopwords
        self._overlap = self.hyp_words & self.text_words
        self._hyp_extra = self.hyp_words - self.text_words
        self._txt_extra = self.text_words - self.hyp_words

    def overlap(self, toktype, debug=False):
        if False:
            i = 10
            return i + 15
        "\n        Compute the overlap between text and hypothesis.\n\n        :param toktype: distinguish Named Entities from ordinary words\n        :type toktype: 'ne' or 'word'\n        "
        ne_overlap = {token for token in self._overlap if self._ne(token)}
        if toktype == 'ne':
            if debug:
                print('ne overlap', ne_overlap)
            return ne_overlap
        elif toktype == 'word':
            if debug:
                print('word overlap', self._overlap - ne_overlap)
            return self._overlap - ne_overlap
        else:
            raise ValueError("Type not recognized:'%s'" % toktype)

    def hyp_extra(self, toktype, debug=True):
        if False:
            for i in range(10):
                print('nop')
        "\n        Compute the extraneous material in the hypothesis.\n\n        :param toktype: distinguish Named Entities from ordinary words\n        :type toktype: 'ne' or 'word'\n        "
        ne_extra = {token for token in self._hyp_extra if self._ne(token)}
        if toktype == 'ne':
            return ne_extra
        elif toktype == 'word':
            return self._hyp_extra - ne_extra
        else:
            raise ValueError("Type not recognized: '%s'" % toktype)

    @staticmethod
    def _ne(token):
        if False:
            return 10
        '\n        This just assumes that words in all caps or titles are\n        named entities.\n\n        :type token: str\n        '
        if token.istitle() or token.isupper():
            return True
        return False

    @staticmethod
    def _lemmatize(word):
        if False:
            return 10
        '\n        Use morphy from WordNet to find the base form of verbs.\n        '
        from nltk.corpus import wordnet as wn
        lemma = wn.morphy(word, pos=wn.VERB)
        if lemma is not None:
            return lemma
        return word

def rte_features(rtepair):
    if False:
        return 10
    extractor = RTEFeatureExtractor(rtepair)
    features = {}
    features['alwayson'] = True
    features['word_overlap'] = len(extractor.overlap('word'))
    features['word_hyp_extra'] = len(extractor.hyp_extra('word'))
    features['ne_overlap'] = len(extractor.overlap('ne'))
    features['ne_hyp_extra'] = len(extractor.hyp_extra('ne'))
    features['neg_txt'] = len(extractor.negwords & extractor.text_words)
    features['neg_hyp'] = len(extractor.negwords & extractor.hyp_words)
    return features

def rte_featurize(rte_pairs):
    if False:
        while True:
            i = 10
    return [(rte_features(pair), pair.value) for pair in rte_pairs]

def rte_classifier(algorithm, sample_N=None):
    if False:
        while True:
            i = 10
    from nltk.corpus import rte as rte_corpus
    train_set = rte_corpus.pairs(['rte1_dev.xml', 'rte2_dev.xml', 'rte3_dev.xml'])
    test_set = rte_corpus.pairs(['rte1_test.xml', 'rte2_test.xml', 'rte3_test.xml'])
    if sample_N is not None:
        train_set = train_set[:sample_N]
        test_set = test_set[:sample_N]
    featurized_train_set = rte_featurize(train_set)
    featurized_test_set = rte_featurize(test_set)
    print('Training classifier...')
    if algorithm in ['megam']:
        clf = MaxentClassifier.train(featurized_train_set, algorithm)
    elif algorithm in ['GIS', 'IIS']:
        clf = MaxentClassifier.train(featurized_train_set, algorithm)
    else:
        err_msg = str("RTEClassifier only supports these algorithms:\n 'megam', 'GIS', 'IIS'.\n")
        raise Exception(err_msg)
    print('Testing classifier...')
    acc = accuracy(clf, featurized_test_set)
    print('Accuracy: %6.4f' % acc)
    return clf