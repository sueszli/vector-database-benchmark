"""
A SentimentAnalyzer is a tool to implement and facilitate Sentiment Analysis tasks
using NLTK features and classifiers, especially for teaching and demonstrative
purposes.
"""
import sys
from collections import defaultdict
from nltk.classify.util import accuracy as eval_accuracy
from nltk.classify.util import apply_features
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk.metrics import f_measure as eval_f_measure
from nltk.metrics import precision as eval_precision
from nltk.metrics import recall as eval_recall
from nltk.probability import FreqDist

class SentimentAnalyzer:
    """
    A Sentiment Analysis tool based on machine learning approaches.
    """

    def __init__(self, classifier=None):
        if False:
            return 10
        self.feat_extractors = defaultdict(list)
        self.classifier = classifier

    def all_words(self, documents, labeled=None):
        if False:
            i = 10
            return i + 15
        '\n        Return all words/tokens from the documents (with duplicates).\n\n        :param documents: a list of (words, label) tuples.\n        :param labeled: if `True`, assume that each document is represented by a\n            (words, label) tuple: (list(str), str). If `False`, each document is\n            considered as being a simple list of strings: list(str).\n        :rtype: list(str)\n        :return: A list of all words/tokens in `documents`.\n        '
        all_words = []
        if labeled is None:
            labeled = documents and isinstance(documents[0], tuple)
        if labeled:
            for (words, _sentiment) in documents:
                all_words.extend(words)
        elif not labeled:
            for words in documents:
                all_words.extend(words)
        return all_words

    def apply_features(self, documents, labeled=None):
        if False:
            return 10
        '\n        Apply all feature extractor functions to the documents. This is a wrapper\n        around `nltk.classify.util.apply_features`.\n\n        If `labeled=False`, return featuresets as:\n            [feature_func(doc) for doc in documents]\n        If `labeled=True`, return featuresets as:\n            [(feature_func(tok), label) for (tok, label) in toks]\n\n        :param documents: a list of documents. `If labeled=True`, the method expects\n            a list of (words, label) tuples.\n        :rtype: LazyMap\n        '
        return apply_features(self.extract_features, documents, labeled)

    def unigram_word_feats(self, words, top_n=None, min_freq=0):
        if False:
            print('Hello World!')
        '\n        Return most common top_n word features.\n\n        :param words: a list of words/tokens.\n        :param top_n: number of best words/tokens to use, sorted by frequency.\n        :rtype: list(str)\n        :return: A list of `top_n` words/tokens (with no duplicates) sorted by\n            frequency.\n        '
        unigram_feats_freqs = FreqDist((word for word in words))
        return [w for (w, f) in unigram_feats_freqs.most_common(top_n) if unigram_feats_freqs[w] > min_freq]

    def bigram_collocation_feats(self, documents, top_n=None, min_freq=3, assoc_measure=BigramAssocMeasures.pmi):
        if False:
            while True:
                i = 10
        '\n        Return `top_n` bigram features (using `assoc_measure`).\n        Note that this method is based on bigram collocations measures, and not\n        on simple bigram frequency.\n\n        :param documents: a list (or iterable) of tokens.\n        :param top_n: number of best words/tokens to use, sorted by association\n            measure.\n        :param assoc_measure: bigram association measure to use as score function.\n        :param min_freq: the minimum number of occurrencies of bigrams to take\n            into consideration.\n\n        :return: `top_n` ngrams scored by the given association measure.\n        '
        finder = BigramCollocationFinder.from_documents(documents)
        finder.apply_freq_filter(min_freq)
        return finder.nbest(assoc_measure, top_n)

    def classify(self, instance):
        if False:
            i = 10
            return i + 15
        '\n        Classify a single instance applying the features that have already been\n        stored in the SentimentAnalyzer.\n\n        :param instance: a list (or iterable) of tokens.\n        :return: the classification result given by applying the classifier.\n        '
        instance_feats = self.apply_features([instance], labeled=False)
        return self.classifier.classify(instance_feats[0])

    def add_feat_extractor(self, function, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Add a new function to extract features from a document. This function will\n        be used in extract_features().\n        Important: in this step our kwargs are only representing additional parameters,\n        and NOT the document we have to parse. The document will always be the first\n        parameter in the parameter list, and it will be added in the extract_features()\n        function.\n\n        :param function: the extractor function to add to the list of feature extractors.\n        :param kwargs: additional parameters required by the `function` function.\n        '
        self.feat_extractors[function].append(kwargs)

    def extract_features(self, document):
        if False:
            i = 10
            return i + 15
        '\n        Apply extractor functions (and their parameters) to the present document.\n        We pass `document` as the first parameter of the extractor functions.\n        If we want to use the same extractor function multiple times, we have to\n        add it to the extractors with `add_feat_extractor` using multiple sets of\n        parameters (one for each call of the extractor function).\n\n        :param document: the document that will be passed as argument to the\n            feature extractor functions.\n        :return: A dictionary of populated features extracted from the document.\n        :rtype: dict\n        '
        all_features = {}
        for extractor in self.feat_extractors:
            for param_set in self.feat_extractors[extractor]:
                feats = extractor(document, **param_set)
            all_features.update(feats)
        return all_features

    def train(self, trainer, training_set, save_classifier=None, **kwargs):
        if False:
            print('Hello World!')
        '\n        Train classifier on the training set, optionally saving the output in the\n        file specified by `save_classifier`.\n        Additional arguments depend on the specific trainer used. For example,\n        a MaxentClassifier can use `max_iter` parameter to specify the number\n        of iterations, while a NaiveBayesClassifier cannot.\n\n        :param trainer: `train` method of a classifier.\n            E.g.: NaiveBayesClassifier.train\n        :param training_set: the training set to be passed as argument to the\n            classifier `train` method.\n        :param save_classifier: the filename of the file where the classifier\n            will be stored (optional).\n        :param kwargs: additional parameters that will be passed as arguments to\n            the classifier `train` function.\n        :return: A classifier instance trained on the training set.\n        :rtype:\n        '
        print('Training classifier')
        self.classifier = trainer(training_set, **kwargs)
        if save_classifier:
            self.save_file(self.classifier, save_classifier)
        return self.classifier

    def save_file(self, content, filename):
        if False:
            return 10
        '\n        Store `content` in `filename`. Can be used to store a SentimentAnalyzer.\n        '
        print('Saving', filename, file=sys.stderr)
        with open(filename, 'wb') as storage_file:
            import pickle
            pickle.dump(content, storage_file, protocol=2)

    def evaluate(self, test_set, classifier=None, accuracy=True, f_measure=True, precision=True, recall=True, verbose=False):
        if False:
            i = 10
            return i + 15
        '\n        Evaluate and print classifier performance on the test set.\n\n        :param test_set: A list of (tokens, label) tuples to use as gold set.\n        :param classifier: a classifier instance (previously trained).\n        :param accuracy: if `True`, evaluate classifier accuracy.\n        :param f_measure: if `True`, evaluate classifier f_measure.\n        :param precision: if `True`, evaluate classifier precision.\n        :param recall: if `True`, evaluate classifier recall.\n        :return: evaluation results.\n        :rtype: dict(str): float\n        '
        if classifier is None:
            classifier = self.classifier
        print(f'Evaluating {type(classifier).__name__} results...')
        metrics_results = {}
        if accuracy:
            accuracy_score = eval_accuracy(classifier, test_set)
            metrics_results['Accuracy'] = accuracy_score
        gold_results = defaultdict(set)
        test_results = defaultdict(set)
        labels = set()
        for (i, (feats, label)) in enumerate(test_set):
            labels.add(label)
            gold_results[label].add(i)
            observed = classifier.classify(feats)
            test_results[observed].add(i)
        for label in labels:
            if precision:
                precision_score = eval_precision(gold_results[label], test_results[label])
                metrics_results[f'Precision [{label}]'] = precision_score
            if recall:
                recall_score = eval_recall(gold_results[label], test_results[label])
                metrics_results[f'Recall [{label}]'] = recall_score
            if f_measure:
                f_measure_score = eval_f_measure(gold_results[label], test_results[label])
                metrics_results[f'F-measure [{label}]'] = f_measure_score
        if verbose:
            for result in sorted(metrics_results):
                print(f'{result}: {metrics_results[result]}')
        return metrics_results