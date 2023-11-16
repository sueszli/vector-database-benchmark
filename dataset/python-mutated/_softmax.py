import cupy

def softmax(x, axis=None):
    if False:
        for i in range(10):
            print('nop')
    'Softmax function.\n\n    The softmax function transforms each element of a\n    collection by computing the exponential of each element\n    divided by the sum of the exponentials of all the elements.\n\n    Parameters\n    ----------\n    x : array-like\n        The input array\n    axis : int or tuple of ints, optional\n        Axis to compute values along. Default is None\n\n    Returns\n    -------\n    s : cupy.ndarray\n        Returns an array with same shape as input. The result\n        will sum to 1 along the provided axis\n\n    '
    x_max = cupy.amax(x, axis=axis, keepdims=True)
    exp_x_shifted = cupy.exp(x - x_max)
    return exp_x_shifted / cupy.sum(exp_x_shifted, axis=axis, keepdims=True)