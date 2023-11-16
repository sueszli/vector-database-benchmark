"""General Naive Bayes learner.

Naive Bayes is a supervised classification algorithm that uses Bayes
rule to compute the fit between a new observation and some previously
observed data.  The observations are discrete feature vectors, with
the Bayes assumption that the features are independent.  Although this
is hardly ever true, the classifier works well enough in practice.

Glossary:
 - observation - A feature vector of discrete data.
 - class       - A possible classification for an observation.

Classes:
 - NaiveBayes - Holds information for a naive Bayes classifier.

Functions:
 - train     - Train a new naive Bayes classifier.
 - calculate - Calculate the probabilities of each class,
   given an observation.
 - classify  - Classify an observation into a class.

"""
try:
    import numpy as np
except ImportError:
    from Bio import MissingPythonDependencyError
    raise MissingPythonDependencyError('Please install NumPy if you want to use Bio.NaiveBayes. See http://www.numpy.org/') from None

def _contents(items):
    if False:
        while True:
            i = 10
    'Return a dictionary where the key is the item and the value is the probablity associated (PRIVATE).'
    term = 1.0 / len(items)
    counts = {}
    for item in items:
        counts[item] = counts.get(item, 0) + term
    return counts

class NaiveBayes:
    """Hold information for a NaiveBayes classifier.

    Attributes:
     - classes        - List of the possible classes of data.
     - p_conditional  - CLASS x DIM array of dicts of value -> ``P(value|class,dim)``
     - p_prior        - List of the prior probabilities for every class.
     - dimensionality - Dimensionality of the data.

    """

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        'Initialize the class.'
        self.classes = []
        self.p_conditional = None
        self.p_prior = []
        self.dimensionality = None

def calculate(nb, observation, scale=False):
    if False:
        i = 10
        return i + 15
    'Calculate the logarithmic conditional probability for each class.\n\n    Arguments:\n     - nb          - A NaiveBayes classifier that has been trained.\n     - observation - A list representing the observed data.\n     - scale       - Boolean to indicate whether the probability should be\n       scaled by ``P(observation)``.  By default, no scaling is done.\n\n    A dictionary is returned where the key is the class and the value is\n    the log probability of the class.\n    '
    if len(observation) != nb.dimensionality:
        raise ValueError(f'observation in {len(observation)} dimension, but classifier in {nb.dimensionality}')
    n = len(nb.classes)
    lp_observation_class = np.zeros(n)
    for i in range(n):
        probs = [None] * len(observation)
        for j in range(len(observation)):
            probs[j] = nb.p_conditional[i][j].get(observation[j], 0)
        lprobs = np.log(np.clip(probs, 1e-300, 1e+300))
        lp_observation_class[i] = sum(lprobs)
    lp_prior = np.log(nb.p_prior)
    lp_observation = 0.0
    if scale:
        obs = np.exp(np.clip(lp_prior + lp_observation_class, -700, +700))
        lp_observation = np.log(sum(obs))
    lp_class_observation = {}
    for i in range(len(nb.classes)):
        lp_class_observation[nb.classes[i]] = lp_observation_class[i] + lp_prior[i] - lp_observation
    return lp_class_observation

def classify(nb, observation):
    if False:
        return 10
    'Classify an observation into a class.'
    probs = calculate(nb, observation, scale=False)
    max_prob = max_class = None
    for klass in nb.classes:
        if max_prob is None or probs[klass] > max_prob:
            (max_prob, max_class) = (probs[klass], klass)
    return max_class

def train(training_set, results, priors=None, typecode=None):
    if False:
        for i in range(10):
            print('nop')
    'Train a NaiveBayes classifier on a training set.\n\n    Arguments:\n     - training_set - List of observations.\n     - results      - List of the class assignments for each observation.\n       Thus, training_set and results must be the same length.\n     - priors       - Optional dictionary specifying the prior probabilities\n       for each type of result. If not specified, the priors will\n       be estimated from the training results.\n\n    '
    if not len(training_set):
        raise ValueError('No data in the training set.')
    if len(training_set) != len(results):
        raise ValueError('training_set and results should be parallel lists.')
    dimensions = [len(x) for x in training_set]
    if min(dimensions) != max(dimensions):
        raise ValueError('observations have different dimensionality')
    nb = NaiveBayes()
    nb.dimensionality = dimensions[0]
    if priors is not None:
        percs = priors
        nb.classes = list(set(results))
    else:
        class_freq = _contents(results)
        nb.classes = list(class_freq.keys())
        percs = class_freq
    nb.classes.sort()
    nb.p_prior = np.zeros(len(nb.classes))
    for i in range(len(nb.classes)):
        nb.p_prior[i] = percs[nb.classes[i]]
    c2i = {}
    for (index, key) in enumerate(nb.classes):
        c2i[key] = index
    observations = [[] for c in nb.classes]
    for i in range(len(results)):
        (klass, obs) = (results[i], training_set[i])
        observations[c2i[klass]].append(obs)
    for i in range(len(observations)):
        observations[i] = np.asarray(observations[i], typecode)
    nb.p_conditional = []
    for i in range(len(nb.classes)):
        class_observations = observations[i]
        nb.p_conditional.append([None] * nb.dimensionality)
        for j in range(nb.dimensionality):
            values = class_observations[:, j]
            nb.p_conditional[i][j] = _contents(values)
    return nb