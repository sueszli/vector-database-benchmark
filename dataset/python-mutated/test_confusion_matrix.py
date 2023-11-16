import numpy as np
from numpy.testing import assert_array_equal
from mlxtend.evaluate import confusion_matrix

def test_multiclass():
    if False:
        for i in range(10):
            print('nop')
    y_targ = [1, 1, 1, 0, 0, 2, 0, 3]
    y_pred = [1, 0, 1, 0, 0, 2, 1, 3]
    x = np.array([[2, 1, 0, 0], [1, 2, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    y = confusion_matrix(y_targ, y_pred, binary=False, positive_label=1)
    assert_array_equal(x, y)

def test_binary():
    if False:
        print('Hello World!')
    y_targ = [1, 1, 1, 0, 0, 2, 0, 3]
    y_pred = [1, 0, 1, 0, 0, 2, 1, 3]
    x = np.array([[4, 1], [1, 2]])
    y = confusion_matrix(y_targ, y_pred, binary=True, positive_label=1)
    assert_array_equal(x, y)