def sum_array(view: cython.int[:]):
    if False:
        while True:
            i = 10
    "\n    >>> from array import array\n    >>> sum_array( array('i', [1,2,3]) )\n    6\n    "
    total: cython.int = 0
    for i in range(view.shape[0]):
        total += view[i]
    return total