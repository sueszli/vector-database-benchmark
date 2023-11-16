import math

def distance(x, y):
    if False:
        print('Hello World!')
    '[summary]\n    HELPER-FUNCTION\n    calculates the (eulidean) distance between vector x and y.\n\n    Arguments:\n        x {[tuple]} -- [vector]\n        y {[tuple]} -- [vector]\n    '
    assert len(x) == len(y), 'The vector must have same length'
    result = ()
    sum = 0
    for i in range(len(x)):
        result += (x[i] - y[i],)
    for component in result:
        sum += component ** 2
    return math.sqrt(sum)

def nearest_neighbor(x, tSet):
    if False:
        return 10
    '[summary]\n    Implements the nearest neighbor algorithm\n\n    Arguments:\n        x {[tupel]} -- [vector]\n        tSet {[dict]} -- [training set]\n\n    Returns:\n        [type] -- [result of the AND-function]\n    '
    assert isinstance(x, tuple) and isinstance(tSet, dict)
    current_key = ()
    min_d = float('inf')
    for key in tSet:
        d = distance(x, key)
        if d < min_d:
            min_d = d
            current_key = key
    return tSet[current_key]