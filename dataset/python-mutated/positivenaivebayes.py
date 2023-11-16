"""
A variant of the Naive Bayes Classifier that performs binary classification with
partially-labeled training sets. In other words, assume we want to build a classifier
that assigns each example to one of two complementary classes (e.g., male names and
female names).
If we have a training set with labeled examples for both classes, we can use a
standard Naive Bayes Classifier. However, consider the case when we only have labeled
examples for one of the classes, and other, unlabeled, examples.
Then, assuming a prior distribution on the two labels, we can use the unlabeled set
to estimate the frequencies of the various features.

Let the two possible labels be 1 and 0, and let's say we only have examples labeled 1
and unlabeled examples. We are also given an estimate of P(1).

We compute P(feature|1) exactly as in the standard case.

To compute P(feature|0), we first estimate P(feature) from the unlabeled set (we are
assuming that the unlabeled examples are drawn according to the given prior distribution)
and then express the conditional probability as:

|                  P(feature) - P(feature|1) * P(1)
|  P(feature|0) = ----------------------------------
|                               P(0)

Example:

    >>> from nltk.classify import PositiveNaiveBayesClassifier

Some sentences about sports:

    >>> sports_sentences = [ 'The team dominated the game',
    ...                      'They lost the ball',
    ...                      'The game was intense',
    ...                      'The goalkeeper catched the ball',
    ...                      'The other team controlled the ball' ]

Mixed topics, including sports:

    >>> various_sentences = [ 'The President did not comment',
    ...                       'I lost the keys',
    ...                       'The team won the game',
    ...                       'Sara has two kids',
    ...                       'The ball went off the court',
    ...                       'They had the ball for the whole game',
    ...                       'The show is over' ]

The features of a sentence are simply the words it contains:

    >>> def features(sentence):
    ...     words = sentence.lower().split()
    ...     return dict(('contains(%s)' % w, True) for w in words)

We use the sports sentences as positive examples, the mixed ones ad unlabeled examples:

    >>> positive_featuresets = map(features, sports_sentences)
    >>> unlabeled_featuresets = map(features, various_sentences)
    >>> classifier = PositiveNaiveBayesClassifier.train(positive_featuresets,
    ...                                                 unlabeled_featuresets)

Is the following sentence about sports?

    >>> classifier.classify(features('The cat is on the table'))
    False

What about this one?

    >>> classifier.classify(features('My team lost the game'))
    True
"""
from collections import defaultdict
from nltk.classify.naivebayes import NaiveBayesClassifier
from nltk.probability import DictionaryProbDist, ELEProbDist, FreqDist

class PositiveNaiveBayesClassifier(NaiveBayesClassifier):

    @staticmethod
    def train(positive_featuresets, unlabeled_featuresets, positive_prob_prior=0.5, estimator=ELEProbDist):
        if False:
            return 10
        '\n        :param positive_featuresets: An iterable of featuresets that are known as positive\n            examples (i.e., their label is ``True``).\n\n        :param unlabeled_featuresets: An iterable of featuresets whose label is unknown.\n\n        :param positive_prob_prior: A prior estimate of the probability of the label\n            ``True`` (default 0.5).\n        '
        positive_feature_freqdist = defaultdict(FreqDist)
        unlabeled_feature_freqdist = defaultdict(FreqDist)
        feature_values = defaultdict(set)
        fnames = set()
        num_positive_examples = 0
        for featureset in positive_featuresets:
            for (fname, fval) in featureset.items():
                positive_feature_freqdist[fname][fval] += 1
                feature_values[fname].add(fval)
                fnames.add(fname)
            num_positive_examples += 1
        num_unlabeled_examples = 0
        for featureset in unlabeled_featuresets:
            for (fname, fval) in featureset.items():
                unlabeled_feature_freqdist[fname][fval] += 1
                feature_values[fname].add(fval)
                fnames.add(fname)
            num_unlabeled_examples += 1
        for fname in fnames:
            count = positive_feature_freqdist[fname].N()
            positive_feature_freqdist[fname][None] += num_positive_examples - count
            feature_values[fname].add(None)
        for fname in fnames:
            count = unlabeled_feature_freqdist[fname].N()
            unlabeled_feature_freqdist[fname][None] += num_unlabeled_examples - count
            feature_values[fname].add(None)
        negative_prob_prior = 1.0 - positive_prob_prior
        label_probdist = DictionaryProbDist({True: positive_prob_prior, False: negative_prob_prior})
        feature_probdist = {}
        for (fname, freqdist) in positive_feature_freqdist.items():
            probdist = estimator(freqdist, bins=len(feature_values[fname]))
            feature_probdist[True, fname] = probdist
        for (fname, freqdist) in unlabeled_feature_freqdist.items():
            global_probdist = estimator(freqdist, bins=len(feature_values[fname]))
            negative_feature_probs = {}
            for fval in feature_values[fname]:
                prob = (global_probdist.prob(fval) - positive_prob_prior * feature_probdist[True, fname].prob(fval)) / negative_prob_prior
                negative_feature_probs[fval] = max(prob, 0.0)
            feature_probdist[False, fname] = DictionaryProbDist(negative_feature_probs, normalize=True)
        return PositiveNaiveBayesClassifier(label_probdist, feature_probdist)

def demo():
    if False:
        i = 10
        return i + 15
    from nltk.classify.util import partial_names_demo
    classifier = partial_names_demo(PositiveNaiveBayesClassifier.train)
    classifier.show_most_informative_features()