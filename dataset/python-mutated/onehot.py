import numpy as np

def one_hot(y, num_labels='auto', dtype='float'):
    if False:
        i = 10
        return i + 15
    "One-hot encoding of class labels\n\n    Parameters\n    ----------\n    y : array-like, shape = [n_classlabels]\n        Python list or numpy array consisting of class labels.\n    num_labels : int or 'auto'\n        Number of unique labels in the class label array. Infers the number\n        of unique labels from the input array if set to 'auto'.\n    dtype : str\n        NumPy array type (float, float32, float64) of the output array.\n\n    Returns\n    ----------\n    ary : numpy.ndarray, shape = [n_classlabels]\n        One-hot encoded array, where each sample is represented as\n        a row vector in the returned array.\n\n    Examples\n    ----------\n    For usage examples, please see\n    https://rasbt.github.io/mlxtend/user_guide/preprocessing/one_hot/\n\n    "
    if not (num_labels == 'auto' or isinstance(num_labels, int)):
        raise AttributeError('num_labels must be an integer or "auto"')
    if isinstance(y, list):
        yt = np.asarray(y)
    else:
        yt = y
    if not len(yt.shape) == 1:
        raise AttributeError('y array must be 1-dimensional')
    if num_labels == 'auto':
        uniq = np.max(yt + 1)
    else:
        uniq = num_labels
    if uniq == 1:
        ary = np.array([[0.0]], dtype=dtype)
    else:
        ary = np.zeros((len(y), uniq))
        for (i, val) in enumerate(y):
            ary[i, val] = 1
    return ary.astype(dtype)