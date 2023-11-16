"""
A classifier model that decides which label to assign to a token on
the basis of a tree structure, where branches correspond to conditions
on feature values, and leaves correspond to label assignments.
"""
from collections import defaultdict
from nltk.classify.api import ClassifierI
from nltk.probability import FreqDist, MLEProbDist, entropy

class DecisionTreeClassifier(ClassifierI):

    def __init__(self, label, feature_name=None, decisions=None, default=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        :param label: The most likely label for tokens that reach\n            this node in the decision tree.  If this decision tree\n            has no children, then this label will be assigned to\n            any token that reaches this decision tree.\n        :param feature_name: The name of the feature that this\n            decision tree selects for.\n        :param decisions: A dictionary mapping from feature values\n            for the feature identified by ``feature_name`` to\n            child decision trees.\n        :param default: The child that will be used if the value of\n            feature ``feature_name`` does not match any of the keys in\n            ``decisions``.  This is used when constructing binary\n            decision trees.\n        '
        self._label = label
        self._fname = feature_name
        self._decisions = decisions
        self._default = default

    def labels(self):
        if False:
            i = 10
            return i + 15
        labels = [self._label]
        if self._decisions is not None:
            for dt in self._decisions.values():
                labels.extend(dt.labels())
        if self._default is not None:
            labels.extend(self._default.labels())
        return list(set(labels))

    def classify(self, featureset):
        if False:
            i = 10
            return i + 15
        if self._fname is None:
            return self._label
        fval = featureset.get(self._fname)
        if fval in self._decisions:
            return self._decisions[fval].classify(featureset)
        elif self._default is not None:
            return self._default.classify(featureset)
        else:
            return self._label

    def error(self, labeled_featuresets):
        if False:
            return 10
        errors = 0
        for (featureset, label) in labeled_featuresets:
            if self.classify(featureset) != label:
                errors += 1
        return errors / len(labeled_featuresets)

    def pretty_format(self, width=70, prefix='', depth=4):
        if False:
            i = 10
            return i + 15
        '\n        Return a string containing a pretty-printed version of this\n        decision tree.  Each line in this string corresponds to a\n        single decision tree node or leaf, and indentation is used to\n        display the structure of the decision tree.\n        '
        if self._fname is None:
            n = width - len(prefix) - 15
            return '{}{} {}\n'.format(prefix, '.' * n, self._label)
        s = ''
        for (i, (fval, result)) in enumerate(sorted(self._decisions.items(), key=lambda item: (item[0] in [None, False, True], str(item[0]).lower()))):
            hdr = f'{prefix}{self._fname}={fval}? '
            n = width - 15 - len(hdr)
            s += '{}{} {}\n'.format(hdr, '.' * n, result._label)
            if result._fname is not None and depth > 1:
                s += result.pretty_format(width, prefix + '  ', depth - 1)
        if self._default is not None:
            n = width - len(prefix) - 21
            s += '{}else: {} {}\n'.format(prefix, '.' * n, self._default._label)
            if self._default._fname is not None and depth > 1:
                s += self._default.pretty_format(width, prefix + '  ', depth - 1)
        return s

    def pseudocode(self, prefix='', depth=4):
        if False:
            i = 10
            return i + 15
        '\n        Return a string representation of this decision tree that\n        expresses the decisions it makes as a nested set of pseudocode\n        if statements.\n        '
        if self._fname is None:
            return f'{prefix}return {self._label!r}\n'
        s = ''
        for (fval, result) in sorted(self._decisions.items(), key=lambda item: (item[0] in [None, False, True], str(item[0]).lower())):
            s += f'{prefix}if {self._fname} == {fval!r}: '
            if result._fname is not None and depth > 1:
                s += '\n' + result.pseudocode(prefix + '  ', depth - 1)
            else:
                s += f'return {result._label!r}\n'
        if self._default is not None:
            if len(self._decisions) == 1:
                s += '{}if {} != {!r}: '.format(prefix, self._fname, list(self._decisions.keys())[0])
            else:
                s += f'{prefix}else: '
            if self._default._fname is not None and depth > 1:
                s += '\n' + self._default.pseudocode(prefix + '  ', depth - 1)
            else:
                s += f'return {self._default._label!r}\n'
        return s

    def __str__(self):
        if False:
            i = 10
            return i + 15
        return self.pretty_format()

    @staticmethod
    def train(labeled_featuresets, entropy_cutoff=0.05, depth_cutoff=100, support_cutoff=10, binary=False, feature_values=None, verbose=False):
        if False:
            i = 10
            return i + 15
        '\n        :param binary: If true, then treat all feature/value pairs as\n            individual binary features, rather than using a single n-way\n            branch for each feature.\n        '
        feature_names = set()
        for (featureset, label) in labeled_featuresets:
            for fname in featureset:
                feature_names.add(fname)
        if feature_values is None and binary:
            feature_values = defaultdict(set)
            for (featureset, label) in labeled_featuresets:
                for (fname, fval) in featureset.items():
                    feature_values[fname].add(fval)
        if not binary:
            tree = DecisionTreeClassifier.best_stump(feature_names, labeled_featuresets, verbose)
        else:
            tree = DecisionTreeClassifier.best_binary_stump(feature_names, labeled_featuresets, feature_values, verbose)
        tree.refine(labeled_featuresets, entropy_cutoff, depth_cutoff - 1, support_cutoff, binary, feature_values, verbose)
        return tree

    @staticmethod
    def leaf(labeled_featuresets):
        if False:
            for i in range(10):
                print('nop')
        label = FreqDist((label for (featureset, label) in labeled_featuresets)).max()
        return DecisionTreeClassifier(label)

    @staticmethod
    def stump(feature_name, labeled_featuresets):
        if False:
            i = 10
            return i + 15
        label = FreqDist((label for (featureset, label) in labeled_featuresets)).max()
        freqs = defaultdict(FreqDist)
        for (featureset, label) in labeled_featuresets:
            feature_value = featureset.get(feature_name)
            freqs[feature_value][label] += 1
        decisions = {val: DecisionTreeClassifier(freqs[val].max()) for val in freqs}
        return DecisionTreeClassifier(label, feature_name, decisions)

    def refine(self, labeled_featuresets, entropy_cutoff, depth_cutoff, support_cutoff, binary=False, feature_values=None, verbose=False):
        if False:
            print('Hello World!')
        if len(labeled_featuresets) <= support_cutoff:
            return
        if self._fname is None:
            return
        if depth_cutoff <= 0:
            return
        for fval in self._decisions:
            fval_featuresets = [(featureset, label) for (featureset, label) in labeled_featuresets if featureset.get(self._fname) == fval]
            label_freqs = FreqDist((label for (featureset, label) in fval_featuresets))
            if entropy(MLEProbDist(label_freqs)) > entropy_cutoff:
                self._decisions[fval] = DecisionTreeClassifier.train(fval_featuresets, entropy_cutoff, depth_cutoff, support_cutoff, binary, feature_values, verbose)
        if self._default is not None:
            default_featuresets = [(featureset, label) for (featureset, label) in labeled_featuresets if featureset.get(self._fname) not in self._decisions]
            label_freqs = FreqDist((label for (featureset, label) in default_featuresets))
            if entropy(MLEProbDist(label_freqs)) > entropy_cutoff:
                self._default = DecisionTreeClassifier.train(default_featuresets, entropy_cutoff, depth_cutoff, support_cutoff, binary, feature_values, verbose)

    @staticmethod
    def best_stump(feature_names, labeled_featuresets, verbose=False):
        if False:
            return 10
        best_stump = DecisionTreeClassifier.leaf(labeled_featuresets)
        best_error = best_stump.error(labeled_featuresets)
        for fname in feature_names:
            stump = DecisionTreeClassifier.stump(fname, labeled_featuresets)
            stump_error = stump.error(labeled_featuresets)
            if stump_error < best_error:
                best_error = stump_error
                best_stump = stump
        if verbose:
            print('best stump for {:6d} toks uses {:20} err={:6.4f}'.format(len(labeled_featuresets), best_stump._fname, best_error))
        return best_stump

    @staticmethod
    def binary_stump(feature_name, feature_value, labeled_featuresets):
        if False:
            return 10
        label = FreqDist((label for (featureset, label) in labeled_featuresets)).max()
        pos_fdist = FreqDist()
        neg_fdist = FreqDist()
        for (featureset, label) in labeled_featuresets:
            if featureset.get(feature_name) == feature_value:
                pos_fdist[label] += 1
            else:
                neg_fdist[label] += 1
        decisions = {}
        default = label
        if pos_fdist.N() > 0:
            decisions = {feature_value: DecisionTreeClassifier(pos_fdist.max())}
        if neg_fdist.N() > 0:
            default = DecisionTreeClassifier(neg_fdist.max())
        return DecisionTreeClassifier(label, feature_name, decisions, default)

    @staticmethod
    def best_binary_stump(feature_names, labeled_featuresets, feature_values, verbose=False):
        if False:
            i = 10
            return i + 15
        best_stump = DecisionTreeClassifier.leaf(labeled_featuresets)
        best_error = best_stump.error(labeled_featuresets)
        for fname in feature_names:
            for fval in feature_values[fname]:
                stump = DecisionTreeClassifier.binary_stump(fname, fval, labeled_featuresets)
                stump_error = stump.error(labeled_featuresets)
                if stump_error < best_error:
                    best_error = stump_error
                    best_stump = stump
        if verbose:
            if best_stump._decisions:
                descr = '{}={}'.format(best_stump._fname, list(best_stump._decisions.keys())[0])
            else:
                descr = '(default)'
            print('best stump for {:6d} toks uses {:20} err={:6.4f}'.format(len(labeled_featuresets), descr, best_error))
        return best_stump

def f(x):
    if False:
        i = 10
        return i + 15
    return DecisionTreeClassifier.train(x, binary=True, verbose=True)

def demo():
    if False:
        while True:
            i = 10
    from nltk.classify.util import binary_names_demo_features, names_demo
    classifier = names_demo(f, binary_names_demo_features)
    print(classifier.pretty_format(depth=7))
    print(classifier.pseudocode(depth=7))
if __name__ == '__main__':
    demo()