from __future__ import annotations
import io
import uuid
from fsspec.core import OpenFile, get_fs_token_paths, open_files
from fsspec.utils import read_block
from fsspec.utils import tokenize as fs_tokenize
from dask.highlevelgraph import HighLevelGraph
MAGIC = b'Obj\x01'
SYNC_SIZE = 16

def read_long(fo):
    if False:
        print('Hello World!')
    'variable-length, zig-zag encoding.'
    c = fo.read(1)
    b = ord(c)
    n = b & 127
    shift = 7
    while b & 128 != 0:
        b = ord(fo.read(1))
        n |= (b & 127) << shift
        shift += 7
    return n >> 1 ^ -(n & 1)

def read_bytes(fo):
    if False:
        while True:
            i = 10
    'a long followed by that many bytes of data.'
    size = read_long(fo)
    return fo.read(size)

def read_header(fo):
    if False:
        return 10
    "Extract an avro file's header\n\n    fo: file-like\n        This should be in bytes mode, e.g., io.BytesIO\n\n    Returns dict representing the header\n\n    Parameters\n    ----------\n    fo: file-like\n    "
    assert fo.read(len(MAGIC)) == MAGIC, 'Magic avro bytes missing'
    meta = {}
    out = {'meta': meta}
    while True:
        n_keys = read_long(fo)
        if n_keys == 0:
            break
        for _ in range(n_keys):
            read_bytes(fo)
            read_bytes(fo)
    out['sync'] = fo.read(SYNC_SIZE)
    out['header_size'] = fo.tell()
    fo.seek(0)
    out['head_bytes'] = fo.read(out['header_size'])
    return out

def open_head(fs, path, compression):
    if False:
        print('Hello World!')
    'Open a file just to read its head and size'
    with OpenFile(fs, path, compression=compression) as f:
        head = read_header(f)
    size = fs.info(path)['size']
    return (head, size)

def read_avro(urlpath, blocksize=100000000, storage_options=None, compression=None):
    if False:
        for i in range(10):
            print('nop')
    "Read set of avro files\n\n    Use this with arbitrary nested avro schemas. Please refer to the\n    fastavro documentation for its capabilities:\n    https://github.com/fastavro/fastavro\n\n    Parameters\n    ----------\n    urlpath: string or list\n        Absolute or relative filepath, URL (may include protocols like\n        ``s3://``), or globstring pointing to data.\n    blocksize: int or None\n        Size of chunks in bytes. If None, there will be no chunking and each\n        file will become one partition.\n    storage_options: dict or None\n        passed to backend file-system\n    compression: str or None\n        Compression format of the targe(s), like 'gzip'. Should only be used\n        with blocksize=None.\n    "
    from dask import compute, delayed
    from dask.bag import from_delayed
    from dask.utils import import_required
    import_required('fastavro', 'fastavro is a required dependency for using bag.read_avro().')
    storage_options = storage_options or {}
    if blocksize is not None:
        (fs, fs_token, paths) = get_fs_token_paths(urlpath, mode='rb', storage_options=storage_options)
        dhead = delayed(open_head)
        out = compute(*[dhead(fs, path, compression) for path in paths])
        (heads, sizes) = zip(*out)
        dread = delayed(read_chunk)
        offsets = []
        lengths = []
        for size in sizes:
            off = list(range(0, size, blocksize))
            length = [blocksize] * len(off)
            offsets.append(off)
            lengths.append(length)
        out = []
        for (path, offset, length, head) in zip(paths, offsets, lengths, heads):
            delimiter = head['sync']
            f = OpenFile(fs, path, compression=compression)
            token = fs_tokenize(fs_token, delimiter, path, fs.ukey(path), compression, offset)
            keys = [f'read-avro-{o}-{token}' for o in offset]
            values = [dread(f, o, l, head, dask_key_name=key) for (o, key, l) in zip(offset, keys, length)]
            out.extend(values)
        return from_delayed(out)
    else:
        files = open_files(urlpath, compression=compression, **storage_options)
        dread = delayed(read_file)
        chunks = [dread(fo) for fo in files]
        return from_delayed(chunks)

def read_chunk(fobj, off, l, head):
    if False:
        i = 10
        return i + 15
    'Get rows from raw bytes block'
    import fastavro
    if hasattr(fastavro, 'iter_avro'):
        reader = fastavro.iter_avro
    else:
        reader = fastavro.reader
    with fobj as f:
        chunk = read_block(f, off, l, head['sync'])
    head_bytes = head['head_bytes']
    if not chunk.startswith(MAGIC):
        chunk = head_bytes + chunk
    i = io.BytesIO(chunk)
    return list(reader(i))

def read_file(fo):
    if False:
        while True:
            i = 10
    'Get rows from file-like'
    import fastavro
    if hasattr(fastavro, 'iter_avro'):
        reader = fastavro.iter_avro
    else:
        reader = fastavro.reader
    with fo as f:
        return list(reader(f))

def to_avro(b, filename, schema, name_function=None, storage_options=None, codec='null', sync_interval=16000, metadata=None, compute=True, **kwargs):
    if False:
        while True:
            i = 10
    'Write bag to set of avro files\n\n    The schema is a complex dictionary describing the data, see\n    https://avro.apache.org/docs/1.8.2/gettingstartedpython.html#Defining+a+schema\n    and https://fastavro.readthedocs.io/en/latest/writer.html .\n    It\'s structure is as follows::\n\n        {\'name\': \'Test\',\n         \'namespace\': \'Test\',\n         \'doc\': \'Descriptive text\',\n         \'type\': \'record\',\n         \'fields\': [\n            {\'name\': \'a\', \'type\': \'int\'},\n         ]}\n\n    where the "name" field is required, but "namespace" and "doc" are optional\n    descriptors; "type" must always be "record". The list of fields should\n    have an entry for every key of the input records, and the types are\n    like the primitive, complex or logical types of the Avro spec\n    ( https://avro.apache.org/docs/1.8.2/spec.html ).\n\n    Results in one avro file per input partition.\n\n    Parameters\n    ----------\n    b: dask.bag.Bag\n    filename: list of str or str\n        Filenames to write to. If a list, number must match the number of\n        partitions. If a string, must include a glob character "*", which will\n        be expanded using name_function\n    schema: dict\n        Avro schema dictionary, see above\n    name_function: None or callable\n        Expands integers into strings, see\n        ``dask.bytes.utils.build_name_function``\n    storage_options: None or dict\n        Extra key/value options to pass to the backend file-system\n    codec: \'null\', \'deflate\', or \'snappy\'\n        Compression algorithm\n    sync_interval: int\n        Number of records to include in each block within a file\n    metadata: None or dict\n        Included in the file header\n    compute: bool\n        If True, files are written immediately, and function blocks. If False,\n        returns delayed objects, which can be computed by the user where\n        convenient.\n    kwargs: passed to compute(), if compute=True\n\n    Examples\n    --------\n    >>> import dask.bag as db\n    >>> b = db.from_sequence([{\'name\': \'Alice\', \'value\': 100},\n    ...                       {\'name\': \'Bob\', \'value\': 200}])\n    >>> schema = {\'name\': \'People\', \'doc\': "Set of people\'s scores",\n    ...           \'type\': \'record\',\n    ...           \'fields\': [\n    ...               {\'name\': \'name\', \'type\': \'string\'},\n    ...               {\'name\': \'value\', \'type\': \'int\'}]}\n    >>> b.to_avro(\'my-data.*.avro\', schema)  # doctest: +SKIP\n    [\'my-data.0.avro\', \'my-data.1.avro\']\n    '
    from dask.utils import import_required
    import_required('fastavro', 'fastavro is a required dependency for using bag.to_avro().')
    _verify_schema(schema)
    storage_options = storage_options or {}
    files = open_files(filename, 'wb', name_function=name_function, num=b.npartitions, **storage_options)
    name = 'to-avro-' + uuid.uuid4().hex
    dsk = {(name, i): (_write_avro_part, (b.name, i), f, schema, codec, sync_interval, metadata) for (i, f) in enumerate(files)}
    graph = HighLevelGraph.from_collections(name, dsk, dependencies=[b])
    out = type(b)(graph, name, b.npartitions)
    if compute:
        out.compute(**kwargs)
        return [f.path for f in files]
    else:
        return out.to_delayed()

def _verify_schema(s):
    if False:
        while True:
            i = 10
    assert isinstance(s, dict), 'Schema must be dictionary'
    for field in ['name', 'type', 'fields']:
        assert field in s, "Schema missing '%s' field" % field
    assert s['type'] == 'record', "Schema must be of type 'record'"
    assert isinstance(s['fields'], list), 'Fields entry must be a list'
    for f in s['fields']:
        assert 'name' in f and 'type' in f, 'Field spec incomplete: %s' % f

def _write_avro_part(part, f, schema, codec, sync_interval, metadata):
    if False:
        i = 10
        return i + 15
    'Create single avro file from list of dictionaries'
    import fastavro
    with f as f:
        fastavro.writer(f, schema, part, codec, sync_interval, metadata)