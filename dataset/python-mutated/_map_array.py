import numpy as np

def map_array(input_arr, input_vals, output_vals, out=None):
    if False:
        while True:
            i = 10
    'Map values from input array from input_vals to output_vals.\n\n    Parameters\n    ----------\n    input_arr : array of int, shape (M[, ...])\n        The input label image.\n    input_vals : array of int, shape (K,)\n        The values to map from.\n    output_vals : array, shape (K,)\n        The values to map to.\n    out: array, same shape as `input_arr`\n        The output array. Will be created if not provided. It should\n        have the same dtype as `output_vals`.\n\n    Returns\n    -------\n    out : array, same shape as `input_arr`\n        The array of mapped values.\n    '
    from ._remap import _map_array
    if not np.issubdtype(input_arr.dtype, np.integer):
        raise TypeError('The dtype of an array to be remapped should be integer.')
    orig_shape = input_arr.shape
    input_arr = input_arr.reshape(-1)
    if out is None:
        out = np.empty(orig_shape, dtype=output_vals.dtype)
    elif out.shape != orig_shape:
        raise ValueError(f'If out array is provided, it should have the same shape as the input array. Input array has shape {orig_shape}, provided output array has shape {out.shape}.')
    try:
        out_view = out.view()
        out_view.shape = (-1,)
    except AttributeError:
        raise ValueError(f'If out array is provided, it should be either contiguous or 1-dimensional. Got array with shape {out.shape} and strides {out.strides}.')
    input_vals = input_vals.astype(input_arr.dtype, copy=False)
    output_vals = output_vals.astype(out.dtype, copy=False)
    _map_array(input_arr, out_view, input_vals, output_vals)
    return out

class ArrayMap:
    """Class designed to mimic mapping by NumPy array indexing.

    This class is designed to replicate the use of NumPy arrays for mapping
    values with indexing:

    >>> values = np.array([0.25, 0.5, 1.0])
    >>> indices = np.array([[0, 0, 1], [2, 2, 1]])
    >>> values[indices]
    array([[0.25, 0.25, 0.5 ],
           [1.  , 1.  , 0.5 ]])

    The issue with this indexing is that you need a very large ``values``
    array if the values in the ``indices`` array are large.

    >>> values = np.array([0.25, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0])
    >>> indices = np.array([[0, 0, 10], [0, 10, 10]])
    >>> values[indices]
    array([[0.25, 0.25, 1.  ],
           [0.25, 1.  , 1.  ]])

    Using this class, the approach is similar, but there is no need to
    create a large values array:

    >>> in_indices = np.array([0, 10])
    >>> out_values = np.array([0.25, 1.0])
    >>> values = ArrayMap(in_indices, out_values)
    >>> values
    ArrayMap(array([ 0, 10]), array([0.25, 1.  ]))
    >>> print(values)
    ArrayMap:
      0 → 0.25
      10 → 1.0
    >>> indices = np.array([[0, 0, 10], [0, 10, 10]])
    >>> values[indices]
    array([[0.25, 0.25, 1.  ],
           [0.25, 1.  , 1.  ]])

    Parameters
    ----------
    in_values : array of int, shape (K,)
        The source values from which to map.
    out_values : array, shape (K,)
        The destination values from which to map.
    """

    def __init__(self, in_values, out_values):
        if False:
            print('Hello World!')
        self.in_values = in_values
        self.out_values = out_values
        self._max_str_lines = 4
        self._array = None

    def __len__(self):
        if False:
            while True:
                i = 10
        'Return one more than the maximum label value being remapped.'
        return np.max(self.in_values) + 1

    def __array__(self, dtype=None):
        if False:
            while True:
                i = 10
        'Return an array that behaves like the arraymap when indexed.\n\n        This array can be very large: it is the size of the largest value\n        in the ``in_vals`` array, plus one.\n        '
        if dtype is None:
            dtype = self.out_values.dtype
        output = np.zeros(np.max(self.in_values) + 1, dtype=dtype)
        output[self.in_values] = self.out_values
        return output

    @property
    def dtype(self):
        if False:
            i = 10
            return i + 15
        return self.out_values.dtype

    def __repr__(self):
        if False:
            while True:
                i = 10
        return f'ArrayMap({repr(self.in_values)}, {repr(self.out_values)})'

    def __str__(self):
        if False:
            return 10
        if len(self.in_values) <= self._max_str_lines + 1:
            rows = range(len(self.in_values))
            string = '\n'.join(['ArrayMap:'] + [f'  {self.in_values[i]} → {self.out_values[i]}' for i in rows])
        else:
            rows0 = list(range(0, self._max_str_lines // 2))
            rows1 = list(range(-self._max_str_lines // 2, 0))
            string = '\n'.join(['ArrayMap:'] + [f'  {self.in_values[i]} → {self.out_values[i]}' for i in rows0] + ['  ...'] + [f'  {self.in_values[i]} → {self.out_values[i]}' for i in rows1])
        return string

    def __call__(self, arr):
        if False:
            i = 10
            return i + 15
        return self.__getitem__(arr)

    def __getitem__(self, index):
        if False:
            while True:
                i = 10
        scalar = np.isscalar(index)
        if scalar:
            index = np.array([index])
        elif isinstance(index, slice):
            start = index.start or 0
            stop = index.stop if index.stop is not None else len(self)
            step = index.step
            index = np.arange(start, stop, step)
        if index.dtype == bool:
            index = np.flatnonzero(index)
        out = map_array(index, self.in_values.astype(index.dtype, copy=False), self.out_values)
        if scalar:
            out = out[0]
        return out

    def __setitem__(self, indices, values):
        if False:
            for i in range(10):
                print('nop')
        if self._array is None:
            self._array = self.__array__()
        self._array[indices] = values
        self.in_values = np.flatnonzero(self._array)
        self.out_values = self._array[self.in_values]