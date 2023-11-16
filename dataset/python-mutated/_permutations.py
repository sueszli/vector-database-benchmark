from cupy.random import _generator

def shuffle(a):
    if False:
        i = 10
        return i + 15
    'Shuffles an array.\n\n    Args:\n        a (cupy.ndarray): The array to be shuffled.\n\n    .. seealso:: :meth:`numpy.random.shuffle`\n\n    '
    rs = _generator.get_random_state()
    return rs.shuffle(a)

def permutation(a):
    if False:
        return 10
    'Returns a permuted range or a permutation of an array.\n\n    Args:\n        a (int or cupy.ndarray): The range or the array to be shuffled.\n\n    Returns:\n        cupy.ndarray: If `a` is an integer, it is permutation range between 0\n        and `a` - 1.\n        Otherwise, it is a permutation of `a`.\n\n    .. seealso:: :meth:`numpy.random.permutation`\n    '
    rs = _generator.get_random_state()
    return rs.permutation(a)