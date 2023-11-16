"""
Distance functions and utilities.
"""
from __future__ import print_function as _
from __future__ import division as _
from __future__ import absolute_import as _
import turicreate as _tc

def euclidean(x, y):
    if False:
        return 10
    "\n    Compute the Euclidean distance between two dictionaries or two lists\n    of equal length. Suppose `x` and `y` each contain :math:`d`\n    variables:\n\n    .. math:: D(x, y) = \\sqrt{\\sum_i^d (x_i - y_i)^2}\n\n    Parameters\n    ----------\n    x : dict or list\n        First input vector.\n\n    y : dict or list\n        Second input vector.\n\n    Returns\n    -------\n    out : float\n        Euclidean distance between `x` and `y`.\n\n    Notes\n    -----\n    - If the input vectors are in dictionary form, keys missing in one\n      of the two dictionaries are assumed to have value 0.\n\n    References\n    ----------\n    - `Wikipedia - Euclidean distance\n      <http://en.wikipedia.org/wiki/Euclidean_distance>`_\n\n    Examples\n    --------\n    >>> tc.distances.euclidean([1, 2, 3], [4, 5, 6])\n    5.196152422706632\n    ...\n    >>> tc.distances.euclidean({'a': 2, 'c': 4}, {'b': 3, 'c': 12})\n    8.774964387392123\n    "
    return _tc.extensions._distances.euclidean(x, y)

def gaussian_kernel(x, y):
    if False:
        print('Hello World!')
    "\n    Compute a Gaussian-type distance between two dictionaries or two lists\n    of equal length. Suppose `x` and `y` each contain :math:`d`\n    variables:\n\n    .. math:: D(x, y) = 1 - \\exp{-\\sum_i^d (x_i - y_i)^2}\n\n    Parameters\n    ----------\n    x : dict or list\n        First input vector.\n\n    y : dict or list\n        Second input vector.\n\n    Returns\n    -------\n    out : float\n        Gaussian distance between `x` and `y`.\n\n    Notes\n    -----\n    - If the input vectors are in dictionary form, keys missing in one\n      of the two dictionaries are assumed to have value 0.\n\n    References\n    ----------\n    - `Wikipedia - Euclidean distance\n      <http://en.wikipedia.org/wiki/Euclidean_distance>`_\n\n    Examples\n    --------\n    >>> tc.distances.gaussian([.1, .2, .3], [.4, .5, .6])\n    5.196152422706632\n    ...\n    >>> tc.distances.euclidean({'a': 2, 'c': 4}, {'b': 3, 'c': 12})\n    8.774964387392123\n    "
    return _tc.extensions._distances.gaussian_kernel(x, y)

def squared_euclidean(x, y):
    if False:
        return 10
    "\n    Compute the squared Euclidean distance between two dictionaries or\n    two lists of equal length. Suppose `x` and `y` each contain\n    :math:`d` variables:\n\n    .. math:: D(x, y) = \\sum_i^d (x_i - y_i)^2\n\n    Parameters\n    ----------\n    x : dict or list\n        First input vector.\n\n    y : dict or list\n        Second input vector.\n\n    Returns\n    -------\n    out : float\n        Squared Euclidean distance between `x` and `y`.\n\n    Notes\n    -----\n    - If the input vectors are in dictionary form, keys missing in one\n      of the two dictionaries are assumed to have value 0.\n\n    - Squared Euclidean distance does not satisfy the triangle\n      inequality, so it is not a metric. This means the ball tree cannot\n      be used to compute nearest neighbors based on this distance.\n\n    References\n    ----------\n    - `Wikipedia - Euclidean distance\n      <http://en.wikipedia.org/wiki/Euclidean_distance>`_\n\n    Examples\n    --------\n    >>> tc.distances.squared_euclidean([1, 2, 3], [4, 5, 6])\n    27.0\n    ...\n    >>> tc.distances.squared_euclidean({'a': 2, 'c': 4},\n    ...                                {'b': 3, 'c': 12})\n    77.0\n    "
    return _tc.extensions._distances.squared_euclidean(x, y)

def manhattan(x, y):
    if False:
        print('Hello World!')
    '\n    Compute the Manhattan distance between between two dictionaries or\n    two lists of equal length. Suppose `x` and `y` each contain\n    :math:`d` variables:\n\n    .. math:: D(x, y) = \\sum_i^d |x_i - y_i|\n\n    Parameters\n    ----------\n    x : dict or list\n        First input vector.\n\n    y : dict or list\n        Second input vector.\n\n    Returns\n    -------\n    out : float\n        Manhattan distance between `x` and `y`.\n\n    Notes\n    -----\n    - If the input vectors are in dictionary form, keys missing in one\n      of the two dictionaries are assumed to have value 0.\n\n    - Manhattan distance is also known as "city block" or "taxi cab"\n      distance.\n\n    References\n    ----------\n    - `Wikipedia - taxicab geometry\n      <http://en.wikipedia.org/wiki/Taxicab_geometry>`_\n\n    Examples\n    --------\n    >>> tc.distances.manhattan([1, 2, 3], [4, 5, 6])\n    9.0\n    ...\n    >>> tc.distances.manhattan({\'a\': 2, \'c\': 4}, {\'b\': 3, \'c\': 12})\n    13.0\n    '
    return _tc.extensions._distances.manhattan(x, y)

def cosine(x, y):
    if False:
        i = 10
        return i + 15
    "\n    Compute the cosine distance between between two dictionaries or two\n    lists of equal length. Suppose `x` and `y` each contain\n    :math:`d` variables:\n\n    .. math::\n\n        D(x, y) = 1 - \\frac{\\sum_i^d x_i y_i}\n        {\\sqrt{\\sum_i^d x_i^2}\\sqrt{\\sum_i^d y_i^2}}\n\n    Parameters\n    ----------\n    x : dict or list\n        First input vector.\n\n    y : dict or list\n        Second input vector.\n\n    Returns\n    -------\n    out : float\n        Cosine distance between `x` and `y`.\n\n    Notes\n    -----\n    - If the input vectors are in dictionary form, keys missing in one\n      of the two dictionaries are assumed to have value 0.\n\n    - Cosine distance is not a metric. This means the ball tree cannot\n      be used to compute nearest neighbors based on this distance.\n\n    References\n    ----------\n    - `Wikipedia - cosine similarity\n      <http://en.wikipedia.org/wiki/Cosine_similarity>`_\n\n    Examples\n    --------\n    >>> tc.distances.cosine([1, 2, 3], [4, 5, 6])\n    0.025368153802923787\n    ...\n    >>> tc.distances.cosine({'a': 2, 'c': 4}, {'b': 3, 'c': 12})\n    0.13227816872537534\n    "
    return _tc.extensions._distances.cosine(x, y)

def levenshtein(x, y):
    if False:
        for i in range(10):
            print('nop')
    '\n    Compute the Levenshtein distance between between strings. The\n    distance is the number of insertion, deletion, and substitution edits\n    needed to transform string `x` into string `y`. The mathematical\n    definition of Levenshtein is recursive:\n\n    .. math::\n\n        D(x, y) = d(|x|, |y|)\n\n        d(i, j) = \\max(i, j), \\quad \\mathrm{if } \\min(i, j) = 0\n\n        d(i, j) = \\min \\Big \\{d(i-1, j) + 1, \\ d(i, j-1) + 1, \\ d(i-1, j-1) + I(x_i \\neq y_i) \\Big \\}, \\quad \\mathrm{else}\n\n\n    Parameters\n    ----------\n    x : string\n        First input string.\n\n    y : string\n        Second input string.\n\n    Returns\n    -------\n    out : float\n        Levenshtein distance between `x` and `y`.\n\n    References\n    ----------\n    - `Wikipedia - Levenshtein distance\n      <http://en.wikipedia.org/wiki/Levenshtein_distance>`_\n\n    Examples\n    --------\n    >>> tc.distances.levenshtein("fossa", "fossil")\n    2.0\n    '
    return _tc.extensions._distances.levenshtein(x, y)

def dot_product(x, y):
    if False:
        for i in range(10):
            print('nop')
    "\n    Compute the dot_product between two dictionaries or two lists of\n    equal length. Suppose `x` and `y` each contain :math:`d` variables:\n\n    .. math:: D(x, y) = \\frac{1}{\\sum_i^d x_i y_i}\n\n    .. warning::\n\n        The 'dot_product' distance is deprecated and will be removed in future\n        versions of Turi Create. Please use 'transformed_dot_product'\n        distance instead, although note that this is more than a name change; it\n        is a *different* transformation of the dot product of two vectors.\n        Please see the distances module documentation for more details.\n\n    Parameters\n    ----------\n    x : dict or list\n        First input vector.\n\n    y : dict or list\n        Second input vector.\n\n    Returns\n    -------\n    out : float\n\n    Notes\n    -----\n    - If the input vectors are in dictionary form, keys missing in one\n      of the two dictionaries are assumed to have value 0.\n\n    - Dot product distance is not a metric. This means the ball tree\n      cannot be used to compute nearest neighbors based on this distance.\n\n    Examples\n    --------\n    >>> tc.distances.dot_product([1, 2, 3], [4, 5, 6])\n    0.03125\n    ...\n    >>> tc.distances.dot_product({'a': 2, 'c': 4}, {'b': 3, 'c': 12})\n    0.020833333333333332\n    "
    return _tc.extensions._distances.dot_product(x, y)

def transformed_dot_product(x, y):
    if False:
        for i in range(10):
            print('nop')
    '\n    Compute the "transformed_dot_product" distance between two dictionaries or\n    two lists of equal length. This is a way to transform the dot product of the\n    two inputs---a similarity measure---into a distance measure. Suppose `x` and\n    `y` each contain :math:`d` variables:\n\n    .. math:: D(x, y) = \\log\\{1 + \\exp\\{-\\sum_i^d x_i y_i\\}\\}\n\n    .. warning::\n\n        The \'dot_product\' distance is deprecated and will be removed in future\n        versions of Turi Create. Please use \'transformed_dot_product\'\n        distance instead, although note that this is more than a name change; it\n        is a *different* transformation of the dot product of two vectors.\n        Please see the distances module documentation for more details.\n\n    Parameters\n    ----------\n    x : dict or list\n        First input vector.\n\n    y : dict or list\n        Second input vector.\n\n    Returns\n    -------\n    out : float\n\n    Notes\n    -----\n    - If the input vectors are in dictionary form, keys missing in one\n      of the two dictionaries are assumed to have value 0.\n\n    - Transformed dot product distance is not a metric because the distance from\n      a point to itself is not 0. This means the ball tree cannot be used to\n      compute nearest neighbors based on this distance.\n\n    Examples\n    --------\n    >>> tc.distances.transformed_dot_product([1, 2, 3], [4, 5, 6])\n    0.03125\n    ...\n    >>> tc.distances.transformed_dot_product({\'a\': 2, \'c\': 4}, {\'b\': 3, \'c\': 12})\n    0.020833333333333332\n    '
    return _tc.extensions._distances.transformed_dot_product(x, y)

def jaccard(x, y):
    if False:
        for i in range(10):
            print('nop')
    "\n    Compute the Jaccard distance between between two dictionaries.\n    Suppose :math:`K_x` and :math:`K_y` are the sets of keys from the\n    two input dictionaries.\n\n    .. math:: D(x, y) = 1 - \\frac{|K_x \\cap K_y|}{|K_x \\cup K_y|}\n\n    Parameters\n    ----------\n    x : dict\n        First input dictionary.\n\n    y : dict\n        Second input dictionary.\n\n    Returns\n    -------\n    out : float\n        Jaccard distance between `x` and `y`.\n\n    Notes\n    -----\n    - Jaccard distance treats the keys in the input dictionaries as\n      sets, and ignores the values in the input dictionaries.\n\n    References\n    ----------\n    - `Wikipedia - Jaccard distance\n      <http://en.wikipedia.org/wiki/Jaccard_index>`_\n\n    Examples\n    --------\n    >>> tc.distances.jaccard({'a': 2, 'c': 4}, {'b': 3, 'c': 12})\n    0.6666666666666667\n    "
    return _tc.extensions._distances.jaccard(x, y)

def weighted_jaccard(x, y):
    if False:
        while True:
            i = 10
    "\n    Compute the weighted Jaccard distance between between two\n    dictionaries. Suppose :math:`K_x` and :math:`K_y` are the sets of\n    keys from the two input dictionaries, while :math:`x_k` and\n    :math:`y_k` are the values associated with key :math:`k` in the\n    respective dictionaries. Typically these values are counts, i.e. of\n    words or n-grams.\n\n    .. math::\n\n        D(x, y) = 1 - \\frac{\\sum_{k \\in K_x \\cup K_y} \\min\\{x_k, y_k\\}}\n        {\\sum_{k \\in K_x \\cup K_y} \\max\\{x_k, y_k\\}}\n\n    Parameters\n    ----------\n    x : dict\n        First input dictionary.\n\n    y : dict\n        Second input dictionary.\n\n    Returns\n    -------\n    out : float\n        Weighted jaccard distance between `x` and `y`.\n\n    Notes\n    -----\n    - If a key is missing in one of the two dictionaries, it is assumed\n      to have value 0.\n\n    References\n    ----------\n    - Weighted Jaccard distance: Chierichetti, F., et al. (2010)\n      `Finding the Jaccard Median\n      <http://theory.stanford.edu/~sergei/papers/soda10-jaccard.pdf>`_.\n      Proceedings of the Twenty-First Annual ACM-SIAM Symposium on\n      Discrete Algorithms. Society for Industrial and Applied\n      Mathematics.\n\n    Examples\n    --------\n    >>> tc.distances.weighted_jaccard({'a': 2, 'c': 4},\n    ...                               {'b': 3, 'c': 12})\n    0.7647058823529411\n    "
    return _tc.extensions._distances.weighted_jaccard(x, y)