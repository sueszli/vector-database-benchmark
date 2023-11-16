from __future__ import annotations
from dask.array import core

def _tiledb_to_chunks(tiledb_array):
    if False:
        while True:
            i = 10
    schema = tiledb_array.schema
    return list((schema.domain.dim(i).tile for i in range(schema.ndim)))

def from_tiledb(uri, attribute=None, chunks=None, storage_options=None, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    'Load array from the TileDB storage format\n\n    See https://docs.tiledb.io for more information about TileDB.\n\n    Parameters\n    ----------\n    uri: TileDB array or str\n        Location to save the data\n    attribute: str or None\n        Attribute selection (single-attribute view on multi-attribute array)\n\n\n    Returns\n    -------\n\n    A Dask Array\n\n    Examples\n    --------\n\n    >>> import tempfile, tiledb\n    >>> import dask.array as da, numpy as np\n    >>> uri = tempfile.NamedTemporaryFile().name\n    >>> _ = tiledb.from_numpy(uri, np.arange(0,9).reshape(3,3))  # create a tiledb array\n    >>> tdb_ar = da.from_tiledb(uri)  # read back the array\n    >>> tdb_ar.shape\n    (3, 3)\n    >>> tdb_ar.mean().compute()\n    4.0\n    '
    import tiledb
    tiledb_config = storage_options or dict()
    key = tiledb_config.pop('key', None)
    if isinstance(uri, tiledb.Array):
        tdb = uri
    else:
        tdb = tiledb.open(uri, attr=attribute, config=tiledb_config, key=key)
    if tdb.schema.sparse:
        raise ValueError('Sparse TileDB arrays are not supported')
    if not attribute:
        if tdb.schema.nattr > 1:
            raise TypeError("keyword 'attribute' must be providedwhen loading a multi-attribute TileDB array")
        else:
            attribute = tdb.schema.attr(0).name
    if tdb.iswritable:
        raise ValueError('TileDB array must be open for reading')
    chunks = chunks or _tiledb_to_chunks(tdb)
    assert len(chunks) == tdb.schema.ndim
    return core.from_array(tdb, chunks, name='tiledb-%s' % uri)

def to_tiledb(darray, uri, compute=True, return_stored=False, storage_options=None, key=None, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    "Save array to the TileDB storage format\n\n    Save 'array' using the TileDB storage manager, to any TileDB-supported URI,\n    including local disk, S3, or HDFS.\n\n    See https://docs.tiledb.io for more information about TileDB.\n\n    Parameters\n    ----------\n\n    darray: dask.array\n        A dask array to write.\n    uri:\n        Any supported TileDB storage location.\n    storage_options: dict\n        Dict containing any configuration options for the TileDB backend.\n        see https://docs.tiledb.io/en/stable/tutorials/config.html\n    compute, return_stored: see ``store()``\n    key: str or None\n        Encryption key\n\n    Returns\n    -------\n\n    None\n        Unless ``return_stored`` is set to ``True`` (``False`` by default)\n\n    Notes\n    -----\n\n    TileDB only supports regularly-chunked arrays.\n    TileDB `tile extents`_ correspond to form 2 of the dask\n    `chunk specification`_, and the conversion is\n    done automatically for supported arrays.\n\n    Examples\n    --------\n\n    >>> import dask.array as da, tempfile\n    >>> uri = tempfile.NamedTemporaryFile().name\n    >>> data = da.random.random(5,5)\n    >>> da.to_tiledb(data, uri)\n    >>> import tiledb\n    >>> tdb_ar = tiledb.open(uri)\n    >>> all(tdb_ar == data)\n    True\n\n    .. _chunk specification: https://docs.tiledb.io/en/stable/tutorials/tiling-dense.html\n    .. _tile extents: http://docs.dask.org/en/latest/array-chunks.html\n    "
    import tiledb
    tiledb_config = storage_options or dict()
    key = key or tiledb_config.pop('key', None)
    if not core._check_regular_chunks(darray.chunks):
        raise ValueError('Attempt to save array to TileDB with irregular chunking, please call `arr.rechunk(...)` first.')
    if isinstance(uri, str):
        chunks = [c[0] for c in darray.chunks]
        tdb = tiledb.empty_like(uri, darray, tile=chunks, config=tiledb_config, key=key, **kwargs)
    elif isinstance(uri, tiledb.Array):
        tdb = uri
        if not (darray.dtype == tdb.dtype and darray.ndim == tdb.ndim):
            raise ValueError('Target TileDB array layout is not compatible with source array')
    else:
        raise ValueError("'uri' must be string pointing to supported TileDB store location or an open, writable TileDB array.")
    if not (tdb.isopen and tdb.iswritable):
        raise ValueError('Target TileDB array is not open and writable.')
    return darray.store(tdb, lock=False, compute=compute, return_stored=return_stored)