from hashlib import sha1
import numpy as np
from Orange.classification import Learner, Model
from Orange.statistics import distribution
__all__ = ['MajorityLearner']

class MajorityLearner(Learner):
    """
    A majority classifier. Always returns most frequent class from the
    training set, regardless of the attribute values from the test data
    instance. Returns class value distribution if class probabilities
    are requested. Can be used as a baseline when comparing classifiers.

    In the special case of uniform class distribution within the training data,
    class value is selected randomly. In order to produce consistent results on
    the same dataset, this value is selected based on hash of the class vector.
    """

    def fit_storage(self, dat):
        if False:
            return 10
        if not dat.domain.has_discrete_class:
            raise ValueError('classification.MajorityLearner expects a domain with a (single) categorical variable')
        dist = distribution.get_distribution(dat, dat.domain.class_var)
        N = dist.sum()
        if N > 0:
            dist /= N
        else:
            dist.fill(1 / len(dist))
        probs = np.array(dist)
        ties = np.flatnonzero(probs == probs.max())
        if len(ties) > 1:
            random_idx = int(sha1(np.ascontiguousarray(dat.Y).data).hexdigest(), 16) % len(ties)
            unif_maj = ties[random_idx]
        else:
            unif_maj = None
        return ConstantModel(dist=dist, unif_maj=unif_maj)

class ConstantModel(Model):
    """
    A classification model that returns a given class value.
    """

    def __init__(self, dist, unif_maj=None):
        if False:
            while True:
                i = 10
        '\n        Constructs `Orange.classification.MajorityModel` that always\n        returns majority value of given distribution.\n\n        If no or empty distribution given, constructs a model that returns equal\n        probabilities for each class value.\n\n        :param dist: domain for the `Table`\n        :param unif_maj: majority class for the special case of uniform\n            class distribution in the training data\n        :type dist: Orange.statistics.distribution.Discrete\n        :return: regression model that returns majority value\n        :rtype: Orange.classification.Model\n        '
        self.dist = np.array(dist)
        self.unif_maj = unif_maj

    def predict(self, X):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns majority class for each given instance in X.\n\n        :param X: data table for which to make predictions\n        :type X: Orange.data.Table\n        :return: predicted value\n        :rtype: vector of majority values\n        '
        probs = np.tile(self.dist, (X.shape[0], 1))
        if self.unif_maj is not None:
            value = np.tile(self.unif_maj, (X.shape[0],))
            return (value, probs)
        return probs

    def __str__(self):
        if False:
            print('Hello World!')
        return 'ConstantModel {}'.format(self.dist)
MajorityLearner.__returns__ = ConstantModel