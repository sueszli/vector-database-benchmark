"""Maximum Entropy code.

Uses Improved Iterative Scaling.
"""
from functools import reduce
try:
    import numpy as np
except ImportError:
    from Bio import MissingPythonDependencyError
    raise MissingPythonDependencyError('Please install NumPy if you want to use Bio.MaxEntropy. See http://www.numpy.org/') from None

class MaxEntropy:
    """Hold information for a Maximum Entropy classifier.

    Members:
    classes      List of the possible classes of data.
    alphas       List of the weights for each feature.
    feature_fns  List of the feature functions.

    Car data from example Naive Bayes Classifier example by Eric Meisner November 22, 2003
    http://www.inf.u-szeged.hu/~ormandi/teaching

    >>> from Bio.MaxEntropy import train, classify
    >>> xcar = [
    ...     ['Red', 'Sports', 'Domestic'],
    ...     ['Red', 'Sports', 'Domestic'],
    ...     ['Red', 'Sports', 'Domestic'],
    ...     ['Yellow', 'Sports', 'Domestic'],
    ...     ['Yellow', 'Sports', 'Imported'],
    ...     ['Yellow', 'SUV', 'Imported'],
    ...     ['Yellow', 'SUV', 'Imported'],
    ...     ['Yellow', 'SUV', 'Domestic'],
    ...     ['Red', 'SUV', 'Imported'],
    ...     ['Red', 'Sports', 'Imported']]
    >>> ycar = ['Yes','No','Yes','No','Yes','No','Yes','No','No','Yes']

    Requires some rules or features

    >>> def udf1(ts, cl):
    ...     return ts[0] != 'Red'
    ...
    >>> def udf2(ts, cl):
    ...     return ts[1] != 'Sports'
    ...
    >>> def udf3(ts, cl):
    ...     return ts[2] != 'Domestic'
    ...
    >>> user_functions = [udf1, udf2, udf3]  # must be an iterable type
    >>> xe = train(xcar, ycar, user_functions)
    >>> for xv, yv in zip(xcar, ycar):
    ...     xc = classify(xe, xv)
    ...     print('Pred: %s gives %s y is %s' % (xv, xc, yv))
    ...
    Pred: ['Red', 'Sports', 'Domestic'] gives No y is Yes
    Pred: ['Red', 'Sports', 'Domestic'] gives No y is No
    Pred: ['Red', 'Sports', 'Domestic'] gives No y is Yes
    Pred: ['Yellow', 'Sports', 'Domestic'] gives No y is No
    Pred: ['Yellow', 'Sports', 'Imported'] gives No y is Yes
    Pred: ['Yellow', 'SUV', 'Imported'] gives No y is No
    Pred: ['Yellow', 'SUV', 'Imported'] gives No y is Yes
    Pred: ['Yellow', 'SUV', 'Domestic'] gives No y is No
    Pred: ['Red', 'SUV', 'Imported'] gives No y is No
    Pred: ['Red', 'Sports', 'Imported'] gives No y is Yes
    """

    def __init__(self):
        if False:
            return 10
        'Initialize the class.'
        self.classes = []
        self.alphas = []
        self.feature_fns = []

def calculate(me, observation):
    if False:
        print('Hello World!')
    'Calculate the log of the probability for each class.\n\n    me is a MaxEntropy object that has been trained.  observation is a vector\n    representing the observed data.  The return value is a list of\n    unnormalized log probabilities for each class.\n    '
    scores = []
    assert len(me.feature_fns) == len(me.alphas)
    for klass in me.classes:
        lprob = 0.0
        for (fn, alpha) in zip(me.feature_fns, me.alphas):
            lprob += fn(observation, klass) * alpha
        scores.append(lprob)
    return scores

def classify(me, observation):
    if False:
        print('Hello World!')
    'Classify an observation into a class.'
    scores = calculate(me, observation)
    (max_score, klass) = (scores[0], me.classes[0])
    for i in range(1, len(scores)):
        if scores[i] > max_score:
            (max_score, klass) = (scores[i], me.classes[i])
    return klass

def _eval_feature_fn(fn, xs, classes):
    if False:
        while True:
            i = 10
    'Evaluate a feature function on every instance of the training set and class (PRIVATE).\n\n    fn is a callback function that takes two parameters: a\n    training instance and a class.  Return a dictionary of (training\n    set index, class index) -> non-zero value.  Values of 0 are not\n    stored in the dictionary.\n    '
    values = {}
    for i in range(len(xs)):
        for j in range(len(classes)):
            f = fn(xs[i], classes[j])
            if f != 0:
                values[i, j] = f
    return values

def _calc_empirical_expects(xs, ys, classes, features):
    if False:
        while True:
            i = 10
    'Calculate the expectation of each function from the data (PRIVATE).\n\n    This is the constraint for the maximum entropy distribution. Return a\n    list of expectations, parallel to the list of features.\n    '
    class2index = {}
    for (index, key) in enumerate(classes):
        class2index[key] = index
    ys_i = [class2index[y] for y in ys]
    expect = []
    N = len(xs)
    for feature in features:
        s = 0
        for i in range(N):
            s += feature.get((i, ys_i[i]), 0)
        expect.append(s / N)
    return expect

def _calc_model_expects(xs, classes, features, alphas):
    if False:
        i = 10
        return i + 15
    'Calculate the expectation of each feature from the model (PRIVATE).\n\n    This is not used in maximum entropy training, but provides a good function\n    for debugging.\n    '
    p_yx = _calc_p_class_given_x(xs, classes, features, alphas)
    expects = []
    for feature in features:
        sum = 0.0
        for ((i, j), f) in feature.items():
            sum += p_yx[i][j] * f
        expects.append(sum / len(xs))
    return expects

def _calc_p_class_given_x(xs, classes, features, alphas):
    if False:
        return 10
    'Calculate conditional probability P(y|x) (PRIVATE).\n\n    y is the class and x is an instance from the training set.\n    Return a XSxCLASSES matrix of probabilities.\n    '
    prob_yx = np.zeros((len(xs), len(classes)))
    assert len(features) == len(alphas)
    for (feature, alpha) in zip(features, alphas):
        for ((x, y), f) in feature.items():
            prob_yx[x][y] += alpha * f
    prob_yx = np.exp(prob_yx)
    for i in range(len(xs)):
        z = sum(prob_yx[i])
        prob_yx[i] = prob_yx[i] / z
    return prob_yx

def _calc_f_sharp(N, nclasses, features):
    if False:
        return 10
    'Calculate a matrix of f sharp values (PRIVATE).'
    f_sharp = np.zeros((N, nclasses))
    for feature in features:
        for ((i, j), f) in feature.items():
            f_sharp[i][j] += f
    return f_sharp

def _iis_solve_delta(N, feature, f_sharp, empirical, prob_yx, max_newton_iterations, newton_converge):
    if False:
        for i in range(10):
            print('nop')
    "Solve delta using Newton's method (PRIVATE)."
    delta = 0.0
    iters = 0
    while iters < max_newton_iterations:
        f_newton = df_newton = 0.0
        for ((i, j), f) in feature.items():
            prod = prob_yx[i][j] * f * np.exp(delta * f_sharp[i][j])
            f_newton += prod
            df_newton += prod * f_sharp[i][j]
        (f_newton, df_newton) = (empirical - f_newton / N, -df_newton / N)
        ratio = f_newton / df_newton
        delta -= ratio
        if np.fabs(ratio) < newton_converge:
            break
        iters = iters + 1
    else:
        raise RuntimeError("Newton's method did not converge")
    return delta

def _train_iis(xs, classes, features, f_sharp, alphas, e_empirical, max_newton_iterations, newton_converge):
    if False:
        for i in range(10):
            print('nop')
    'Do one iteration of hill climbing to find better alphas (PRIVATE).'
    p_yx = _calc_p_class_given_x(xs, classes, features, alphas)
    N = len(xs)
    newalphas = alphas[:]
    for i in range(len(alphas)):
        delta = _iis_solve_delta(N, features[i], f_sharp, e_empirical[i], p_yx, max_newton_iterations, newton_converge)
        newalphas[i] += delta
    return newalphas

def train(training_set, results, feature_fns, update_fn=None, max_iis_iterations=10000, iis_converge=1e-05, max_newton_iterations=100, newton_converge=1e-10):
    if False:
        return 10
    "Train a maximum entropy classifier, returns MaxEntropy object.\n\n    Train a maximum entropy classifier on a training set.\n    training_set is a list of observations.  results is a list of the\n    class assignments for each observation.  feature_fns is a list of\n    the features.  These are callback functions that take an\n    observation and class and return a 1 or 0.  update_fn is a\n    callback function that is called at each training iteration.  It is\n    passed a MaxEntropy object that encapsulates the current state of\n    the training.\n\n    The maximum number of iterations and the convergence criterion for IIS\n    are given by max_iis_iterations and iis_converge, respectively, while\n    max_newton_iterations and newton_converge are the maximum number\n    of iterations and the convergence criterion for Newton's method.\n    "
    if not training_set:
        raise ValueError('No data in the training set.')
    if len(training_set) != len(results):
        raise ValueError('training_set and results should be parallel lists.')
    (xs, ys) = (training_set, results)
    classes = sorted(set(results))
    features = [_eval_feature_fn(fn, training_set, classes) for fn in feature_fns]
    f_sharp = _calc_f_sharp(len(training_set), len(classes), features)
    e_empirical = _calc_empirical_expects(xs, ys, classes, features)
    alphas = [0.0] * len(features)
    iters = 0
    while iters < max_iis_iterations:
        nalphas = _train_iis(xs, classes, features, f_sharp, alphas, e_empirical, max_newton_iterations, newton_converge)
        diff = [np.fabs(x - y) for (x, y) in zip(alphas, nalphas)]
        diff = reduce(np.add, diff, 0)
        alphas = nalphas
        me = MaxEntropy()
        (me.alphas, me.classes, me.feature_fns) = (alphas, classes, feature_fns)
        if update_fn is not None:
            update_fn(me)
        if diff < iis_converge:
            break
    else:
        raise RuntimeError('IIS did not converge')
    return me
if __name__ == '__main__':
    from Bio._utils import run_doctest
    run_doctest(verbose=0)