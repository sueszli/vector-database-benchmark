"""
A classifier based on the Naive Bayes algorithm.  In order to find the
probability for a label, this algorithm first uses the Bayes rule to
express P(label|features) in terms of P(label) and P(features|label):

|                       P(label) * P(features|label)
|  P(label|features) = ------------------------------
|                              P(features)

The algorithm then makes the 'naive' assumption that all features are
independent, given the label:

|                       P(label) * P(f1|label) * ... * P(fn|label)
|  P(label|features) = --------------------------------------------
|                                         P(features)

Rather than computing P(features) explicitly, the algorithm just
calculates the numerator for each label, and normalizes them so they
sum to one:

|                       P(label) * P(f1|label) * ... * P(fn|label)
|  P(label|features) = --------------------------------------------
|                        SUM[l]( P(l) * P(f1|l) * ... * P(fn|l) )
"""
from collections import defaultdict
from nltk.classify.api import ClassifierI
from nltk.probability import DictionaryProbDist, ELEProbDist, FreqDist, sum_logs

class NaiveBayesClassifier(ClassifierI):
    """
    A Naive Bayes classifier.  Naive Bayes classifiers are
    paramaterized by two probability distributions:

      - P(label) gives the probability that an input will receive each
        label, given no information about the input's features.

      - P(fname=fval|label) gives the probability that a given feature
        (fname) will receive a given value (fval), given that the
        label (label).

    If the classifier encounters an input with a feature that has
    never been seen with any label, then rather than assigning a
    probability of 0 to all labels, it will ignore that feature.

    The feature value 'None' is reserved for unseen feature values;
    you generally should not use 'None' as a feature value for one of
    your own features.
    """

    def __init__(self, label_probdist, feature_probdist):
        if False:
            return 10
        '\n        :param label_probdist: P(label), the probability distribution\n            over labels.  It is expressed as a ``ProbDistI`` whose\n            samples are labels.  I.e., P(label) =\n            ``label_probdist.prob(label)``.\n\n        :param feature_probdist: P(fname=fval|label), the probability\n            distribution for feature values, given labels.  It is\n            expressed as a dictionary whose keys are ``(label, fname)``\n            pairs and whose values are ``ProbDistI`` objects over feature\n            values.  I.e., P(fname=fval|label) =\n            ``feature_probdist[label,fname].prob(fval)``.  If a given\n            ``(label,fname)`` is not a key in ``feature_probdist``, then\n            it is assumed that the corresponding P(fname=fval|label)\n            is 0 for all values of ``fval``.\n        '
        self._label_probdist = label_probdist
        self._feature_probdist = feature_probdist
        self._labels = list(label_probdist.samples())

    def labels(self):
        if False:
            while True:
                i = 10
        return self._labels

    def classify(self, featureset):
        if False:
            for i in range(10):
                print('nop')
        return self.prob_classify(featureset).max()

    def prob_classify(self, featureset):
        if False:
            i = 10
            return i + 15
        featureset = featureset.copy()
        for fname in list(featureset.keys()):
            for label in self._labels:
                if (label, fname) in self._feature_probdist:
                    break
            else:
                del featureset[fname]
        logprob = {}
        for label in self._labels:
            logprob[label] = self._label_probdist.logprob(label)
        for label in self._labels:
            for (fname, fval) in featureset.items():
                if (label, fname) in self._feature_probdist:
                    feature_probs = self._feature_probdist[label, fname]
                    logprob[label] += feature_probs.logprob(fval)
                else:
                    logprob[label] += sum_logs([])
        return DictionaryProbDist(logprob, normalize=True, log=True)

    def show_most_informative_features(self, n=10):
        if False:
            while True:
                i = 10
        cpdist = self._feature_probdist
        print('Most Informative Features')
        for (fname, fval) in self.most_informative_features(n):

            def labelprob(l):
                if False:
                    return 10
                return cpdist[l, fname].prob(fval)
            labels = sorted((l for l in self._labels if fval in cpdist[l, fname].samples()), key=lambda element: (-labelprob(element), element), reverse=True)
            if len(labels) == 1:
                continue
            l0 = labels[0]
            l1 = labels[-1]
            if cpdist[l0, fname].prob(fval) == 0:
                ratio = 'INF'
            else:
                ratio = '%8.1f' % (cpdist[l1, fname].prob(fval) / cpdist[l0, fname].prob(fval))
            print('%24s = %-14r %6s : %-6s = %s : 1.0' % (fname, fval, ('%s' % l1)[:6], ('%s' % l0)[:6], ratio))

    def most_informative_features(self, n=100):
        if False:
            for i in range(10):
                print('nop')
        "\n        Return a list of the 'most informative' features used by this\n        classifier.  For the purpose of this function, the\n        informativeness of a feature ``(fname,fval)`` is equal to the\n        highest value of P(fname=fval|label), for any label, divided by\n        the lowest value of P(fname=fval|label), for any label:\n\n        |  max[ P(fname=fval|label1) / P(fname=fval|label2) ]\n        "
        if hasattr(self, '_most_informative_features'):
            return self._most_informative_features[:n]
        else:
            features = set()
            maxprob = defaultdict(lambda : 0.0)
            minprob = defaultdict(lambda : 1.0)
            for ((label, fname), probdist) in self._feature_probdist.items():
                for fval in probdist.samples():
                    feature = (fname, fval)
                    features.add(feature)
                    p = probdist.prob(fval)
                    maxprob[feature] = max(p, maxprob[feature])
                    minprob[feature] = min(p, minprob[feature])
                    if minprob[feature] == 0:
                        features.discard(feature)
            self._most_informative_features = sorted(features, key=lambda feature_: (minprob[feature_] / maxprob[feature_], feature_[0], feature_[1] in [None, False, True], str(feature_[1]).lower()))
        return self._most_informative_features[:n]

    @classmethod
    def train(cls, labeled_featuresets, estimator=ELEProbDist):
        if False:
            for i in range(10):
                print('nop')
        '\n        :param labeled_featuresets: A list of classified featuresets,\n            i.e., a list of tuples ``(featureset, label)``.\n        '
        label_freqdist = FreqDist()
        feature_freqdist = defaultdict(FreqDist)
        feature_values = defaultdict(set)
        fnames = set()
        for (featureset, label) in labeled_featuresets:
            label_freqdist[label] += 1
            for (fname, fval) in featureset.items():
                feature_freqdist[label, fname][fval] += 1
                feature_values[fname].add(fval)
                fnames.add(fname)
        for label in label_freqdist:
            num_samples = label_freqdist[label]
            for fname in fnames:
                count = feature_freqdist[label, fname].N()
                if num_samples - count > 0:
                    feature_freqdist[label, fname][None] += num_samples - count
                    feature_values[fname].add(None)
        label_probdist = estimator(label_freqdist)
        feature_probdist = {}
        for ((label, fname), freqdist) in feature_freqdist.items():
            probdist = estimator(freqdist, bins=len(feature_values[fname]))
            feature_probdist[label, fname] = probdist
        return cls(label_probdist, feature_probdist)

def demo():
    if False:
        for i in range(10):
            print('nop')
    from nltk.classify.util import names_demo
    classifier = names_demo(NaiveBayesClassifier.train)
    classifier.show_most_informative_features()
if __name__ == '__main__':
    demo()