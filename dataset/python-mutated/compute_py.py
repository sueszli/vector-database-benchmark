import numpy as np

def clip(a, min_value, max_value):
    if False:
        while True:
            i = 10
    return min(max(a, min_value), max_value)

def compute(array_1, array_2, a, b, c):
    if False:
        return 10
    '\n    This function must implement the formula\n    np.clip(array_1, 2, 10) * a + array_2 * b + c\n\n    array_1 and array_2 are 2D.\n    '
    x_max = array_1.shape[0]
    y_max = array_1.shape[1]
    assert array_1.shape == array_2.shape
    result = np.zeros((x_max, y_max), dtype=array_1.dtype)
    for x in range(x_max):
        for y in range(y_max):
            tmp = clip(array_1[x, y], 2, 10)
            tmp = tmp * a + array_2[x, y] * b
            result[x, y] = tmp + c
    return result