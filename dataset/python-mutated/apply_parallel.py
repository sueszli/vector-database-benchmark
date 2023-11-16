import numpy
__all__ = ['apply_parallel']

def _get_chunks(shape, ncpu):
    if False:
        i = 10
        return i + 15
    'Split the array into equal sized chunks based on the number of\n    available processors. The last chunk in each dimension absorbs the\n    remainder array elements if the number of CPUs does not divide evenly into\n    the number of array elements.\n\n    Examples\n    --------\n    >>> _get_chunks((4, 4), 4)\n    ((2, 2), (2, 2))\n    >>> _get_chunks((4, 4), 2)\n    ((2, 2), (4,))\n    >>> _get_chunks((5, 5), 2)\n    ((2, 3), (5,))\n    >>> _get_chunks((2, 4), 2)\n    ((1, 1), (4,))\n    '
    from math import ceil
    chunks = []
    nchunks_per_dim = int(ceil(ncpu ** (1.0 / len(shape))))
    used_chunks = 1
    for i in shape:
        if used_chunks < ncpu:
            regular_chunk = i // nchunks_per_dim
            remainder_chunk = regular_chunk + i % nchunks_per_dim
            if regular_chunk == 0:
                chunk_lens = (remainder_chunk,)
            else:
                chunk_lens = (regular_chunk,) * (nchunks_per_dim - 1) + (remainder_chunk,)
        else:
            chunk_lens = (i,)
        chunks.append(chunk_lens)
        used_chunks *= nchunks_per_dim
    return tuple(chunks)

def _ensure_dask_array(array, chunks=None):
    if False:
        i = 10
        return i + 15
    import dask.array as da
    if isinstance(array, da.Array):
        return array
    return da.from_array(array, chunks=chunks)

def apply_parallel(function, array, chunks=None, depth=0, mode=None, extra_arguments=(), extra_keywords=None, *, dtype=None, compute=None, channel_axis=None):
    if False:
        while True:
            i = 10
    "Map a function in parallel across an array.\n\n    Split an array into possibly overlapping chunks of a given depth and\n    boundary type, call the given function in parallel on the chunks, combine\n    the chunks and return the resulting array.\n\n    Parameters\n    ----------\n    function : function\n        Function to be mapped which takes an array as an argument.\n    array : numpy array or dask array\n        Array which the function will be applied to.\n    chunks : int, tuple, or tuple of tuples, optional\n        A single integer is interpreted as the length of one side of a square\n        chunk that should be tiled across the array.  One tuple of length\n        ``array.ndim`` represents the shape of a chunk, and it is tiled across\n        the array.  A list of tuples of length ``ndim``, where each sub-tuple\n        is a sequence of chunk sizes along the corresponding dimension. If\n        None, the array is broken up into chunks based on the number of\n        available cpus. More information about chunks is in the documentation\n        `here <https://dask.pydata.org/en/latest/array-design.html>`_. When\n        `channel_axis` is not None, the tuples can be length ``ndim - 1`` and\n        a single chunk will be used along the channel axis.\n    depth : int or sequence of int, optional\n        The depth of the added boundary cells. A tuple can be used to specify a\n        different depth per array axis. Defaults to zero. When `channel_axis`\n        is not None, and a tuple of length ``ndim - 1`` is provided, a depth of\n        0 will be used along the channel axis.\n    mode : {'reflect', 'symmetric', 'periodic', 'wrap', 'nearest', 'edge'}, optional\n        Type of external boundary padding.\n    extra_arguments : tuple, optional\n        Tuple of arguments to be passed to the function.\n    extra_keywords : dictionary, optional\n        Dictionary of keyword arguments to be passed to the function.\n    dtype : data-type or None, optional\n        The data-type of the `function` output. If None, Dask will attempt to\n        infer this by calling the function on data of shape ``(1,) * ndim``.\n        For functions expecting RGB or multichannel data this may be\n        problematic. In such cases, the user should manually specify this dtype\n        argument instead.\n\n        .. versionadded:: 0.18\n           ``dtype`` was added in 0.18.\n    compute : bool, optional\n        If ``True``, compute eagerly returning a NumPy Array.\n        If ``False``, compute lazily returning a Dask Array.\n        If ``None`` (default), compute based on array type provided\n        (eagerly for NumPy Arrays and lazily for Dask Arrays).\n    channel_axis : int or None, optional\n        If None, the image is assumed to be a grayscale (single channel) image.\n        Otherwise, this parameter indicates which axis of the array corresponds\n        to channels.\n\n    Returns\n    -------\n    out : ndarray or dask Array\n        Returns the result of the applying the operation.\n        Type is dependent on the ``compute`` argument.\n\n    Notes\n    -----\n    Numpy edge modes 'symmetric', 'wrap', and 'edge' are converted to the\n    equivalent ``dask`` boundary modes 'reflect', 'periodic' and 'nearest',\n    respectively.\n    Setting ``compute=False`` can be useful for chaining later operations.\n    For example region selection to preview a result or storing large data\n    to disk instead of loading in memory.\n\n    "
    try:
        import dask.array as da
    except ImportError:
        raise RuntimeError("Could not import 'dask'.  Please install using 'pip install dask'")
    if extra_keywords is None:
        extra_keywords = {}
    if compute is None:
        compute = not isinstance(array, da.Array)
    if channel_axis is not None:
        channel_axis = channel_axis % array.ndim
    if chunks is None:
        shape = array.shape
        try:
            from multiprocessing import cpu_count
            ncpu = cpu_count()
        except NotImplementedError:
            ncpu = 4
        if channel_axis is not None:
            spatial_shape = shape[:channel_axis] + shape[channel_axis + 1:]
            chunks = list(_get_chunks(spatial_shape, ncpu))
            chunks.insert(channel_axis, shape[channel_axis])
            chunks = tuple(chunks)
        else:
            chunks = _get_chunks(shape, ncpu)
    elif channel_axis is not None and len(chunks) == array.ndim - 1:
        chunks = list(chunks)
        chunks.insert(channel_axis, array.shape[channel_axis])
        chunks = tuple(chunks)
    if mode == 'wrap':
        mode = 'periodic'
    elif mode == 'symmetric':
        mode = 'reflect'
    elif mode == 'edge':
        mode = 'nearest'
    elif mode is None:
        mode = 'reflect'
    if channel_axis is not None:
        if numpy.isscalar(depth):
            depth = [depth] * (array.ndim - 1)
        depth = list(depth)
        if len(depth) == array.ndim - 1:
            depth.insert(channel_axis, 0)
        depth = tuple(depth)

    def wrapped_func(arr):
        if False:
            print('Hello World!')
        return function(arr, *extra_arguments, **extra_keywords)
    darr = _ensure_dask_array(array, chunks=chunks)
    res = darr.map_overlap(wrapped_func, depth, boundary=mode, dtype=dtype)
    if compute:
        res = res.compute()
    return res