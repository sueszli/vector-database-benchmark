def byte_bounds(a):
    if False:
        for i in range(10):
            print('nop')
    'Returns pointers to the end-points of an array.\n\n    Args:\n        a: ndarray\n    Returns:\n        Tuple[int, int]: pointers to the end-points of an array\n\n    .. seealso:: :func:`numpy.byte_bounds`\n    '
    a_low = a_high = a.data.ptr
    a_strides = a.strides
    a_shape = a.shape
    a_item_bytes = a.itemsize
    for (shape, stride) in zip(a_shape, a_strides):
        if stride < 0:
            a_low += (shape - 1) * stride
        else:
            a_high += (shape - 1) * stride
    a_high += a_item_bytes
    return (a_low, a_high)