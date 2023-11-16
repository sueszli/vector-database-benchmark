"""Load models from the native binary format released by Facebook.

The main entry point is the :func:`~gensim.models._fasttext_bin.load` function.
It returns a :class:`~gensim.models._fasttext_bin.Model` namedtuple containing everything loaded from the binary.

Examples
--------

Load a model from a binary file:

.. sourcecode:: pycon

    >>> from gensim.test.utils import datapath
    >>> from gensim.models.fasttext_bin import load
    >>> with open(datapath('crime-and-punishment.bin'), 'rb') as fin:
    ...     model = load(fin)
    >>> model.nwords
    291
    >>> model.vectors_ngrams.shape
    (391, 5)
    >>> sorted(model.raw_vocab, key=lambda w: len(w), reverse=True)[:5]
    ['останавливаться', 'изворачиваться,', 'раздражительном', 'exceptionally', 'проскользнуть']

See Also
--------

`FB Implementation <https://github.com/facebookresearch/fastText/blob/master/src/matrix.cc>`_.

"""
import collections
import gzip
import io
import logging
import struct
import numpy as np
_END_OF_WORD_MARKER = b'\x00'
_DICT_WORD_ENTRY_TYPE_MARKER = b'\x00'
logger = logging.getLogger(__name__)
_FASTTEXT_VERSION = np.int32(12)
_FASTTEXT_FILEFORMAT_MAGIC = np.int32(793712314)
_NEW_HEADER_FORMAT = [('dim', 'i'), ('ws', 'i'), ('epoch', 'i'), ('min_count', 'i'), ('neg', 'i'), ('word_ngrams', 'i'), ('loss', 'i'), ('model', 'i'), ('bucket', 'i'), ('minn', 'i'), ('maxn', 'i'), ('lr_update_rate', 'i'), ('t', 'd')]
_OLD_HEADER_FORMAT = [('epoch', 'i'), ('min_count', 'i'), ('neg', 'i'), ('word_ngrams', 'i'), ('loss', 'i'), ('model', 'i'), ('bucket', 'i'), ('minn', 'i'), ('maxn', 'i'), ('lr_update_rate', 'i'), ('t', 'd')]
_FLOAT_SIZE = struct.calcsize('@f')
if _FLOAT_SIZE == 4:
    _FLOAT_DTYPE = np.dtype(np.float32)
elif _FLOAT_SIZE == 8:
    _FLOAT_DTYPE = np.dtype(np.float64)
else:
    _FLOAT_DTYPE = None

def _yield_field_names():
    if False:
        while True:
            i = 10
    for (name, _) in _OLD_HEADER_FORMAT + _NEW_HEADER_FORMAT:
        if not name.startswith('_'):
            yield name
    yield 'raw_vocab'
    yield 'vocab_size'
    yield 'nwords'
    yield 'vectors_ngrams'
    yield 'hidden_output'
    yield 'ntokens'
_FIELD_NAMES = sorted(set(_yield_field_names()))
Model = collections.namedtuple('Model', _FIELD_NAMES)
'Holds data loaded from the Facebook binary.\n\nParameters\n----------\ndim : int\n    The dimensionality of the vectors.\nws : int\n    The window size.\nepoch : int\n    The number of training epochs.\nneg : int\n    If non-zero, indicates that the model uses negative sampling.\nloss : int\n    If equal to 1, indicates that the model uses hierarchical sampling.\nmodel : int\n    If equal to 2, indicates that the model uses skip-grams.\nbucket : int\n    The number of buckets.\nmin_count : int\n    The threshold below which the model ignores terms.\nt : float\n    The sample threshold.\nminn : int\n    The minimum ngram length.\nmaxn : int\n    The maximum ngram length.\nraw_vocab : collections.OrderedDict\n    A map from words (str) to their frequency (int).  The order in the dict\n    corresponds to the order of the words in the Facebook binary.\nnwords : int\n    The number of words.\nvocab_size : int\n    The size of the vocabulary.\nvectors_ngrams : numpy.array\n    This is a matrix that contains vectors learned by the model.\n    Each row corresponds to a vector.\n    The number of vectors is equal to the number of words plus the number of buckets.\n    The number of columns is equal to the vector dimensionality.\nhidden_output : numpy.array\n    This is a matrix that contains the shallow neural network output.\n    This array has the same dimensions as vectors_ngrams.\n    May be None - in that case, it is impossible to continue training the model.\n'

def _struct_unpack(fin, fmt):
    if False:
        i = 10
        return i + 15
    num_bytes = struct.calcsize(fmt)
    return struct.unpack(fmt, fin.read(num_bytes))

def _load_vocab(fin, new_format, encoding='utf-8'):
    if False:
        i = 10
        return i + 15
    'Load a vocabulary from a FB binary.\n\n    Before the vocab is ready for use, call the prepare_vocab function and pass\n    in the relevant parameters from the model.\n\n    Parameters\n    ----------\n    fin : file\n        An open file pointer to the binary.\n    new_format: boolean\n        True if the binary is of the newer format.\n    encoding : str\n        The encoding to use when decoding binary data into words.\n\n    Returns\n    -------\n    tuple\n        The loaded vocabulary.  Keys are words, values are counts.\n        The vocabulary size.\n        The number of words.\n        The number of tokens.\n    '
    (vocab_size, nwords, nlabels) = _struct_unpack(fin, '@3i')
    if nlabels > 0:
        raise NotImplementedError('Supervised fastText models are not supported')
    logger.info('loading %s words for fastText model from %s', vocab_size, fin.name)
    ntokens = _struct_unpack(fin, '@q')[0]
    if new_format:
        (pruneidx_size,) = _struct_unpack(fin, '@q')
    raw_vocab = collections.OrderedDict()
    for i in range(vocab_size):
        word_bytes = io.BytesIO()
        char_byte = fin.read(1)
        while char_byte != _END_OF_WORD_MARKER:
            word_bytes.write(char_byte)
            char_byte = fin.read(1)
        word_bytes = word_bytes.getvalue()
        try:
            word = word_bytes.decode(encoding)
        except UnicodeDecodeError:
            word = word_bytes.decode(encoding, errors='backslashreplace')
            logger.error('failed to decode invalid unicode bytes %r; replacing invalid characters, using %r', word_bytes, word)
        (count, _) = _struct_unpack(fin, '@qb')
        raw_vocab[word] = count
    if new_format:
        for j in range(pruneidx_size):
            _struct_unpack(fin, '@2i')
    return (raw_vocab, vocab_size, nwords, ntokens)

def _load_matrix(fin, new_format=True):
    if False:
        while True:
            i = 10
    'Load a matrix from fastText native format.\n\n    Interprets the matrix dimensions and type from the file stream.\n\n    Parameters\n    ----------\n    fin : file\n        A file handle opened for reading.\n    new_format : bool, optional\n        True if the quant_input variable precedes\n        the matrix declaration.  Should be True for newer versions of fastText.\n\n    Returns\n    -------\n    :class:`numpy.array`\n        The vectors as an array.\n        Each vector will be a row in the array.\n        The number of columns of the array will correspond to the vector size.\n\n    '
    if _FLOAT_DTYPE is None:
        raise ValueError('bad _FLOAT_SIZE: %r' % _FLOAT_SIZE)
    if new_format:
        _struct_unpack(fin, '@?')
    (num_vectors, dim) = _struct_unpack(fin, '@2q')
    count = num_vectors * dim
    if isinstance(fin, gzip.GzipFile):
        logger.warning('Loading model from a compressed .gz file.  This can be slow. This is a work-around for a bug in NumPy: https://github.com/numpy/numpy/issues/13470. Consider decompressing your model file for a faster load. ')
        matrix = _fromfile(fin, _FLOAT_DTYPE, count)
    else:
        matrix = np.fromfile(fin, _FLOAT_DTYPE, count)
    assert matrix.shape == (count,), 'expected (%r,),  got %r' % (count, matrix.shape)
    matrix = matrix.reshape((num_vectors, dim))
    return matrix

def _batched_generator(fin, count, batch_size=1000000.0):
    if False:
        print('Hello World!')
    'Read `count` floats from `fin`.\n\n    Batches up read calls to avoid I/O overhead.  Keeps no more than batch_size\n    floats in memory at once.\n\n    Yields floats.\n\n    '
    while count > batch_size:
        batch = _struct_unpack(fin, '@%df' % batch_size)
        for f in batch:
            yield f
        count -= batch_size
    batch = _struct_unpack(fin, '@%df' % count)
    for f in batch:
        yield f

def _fromfile(fin, dtype, count):
    if False:
        return 10
    'Reimplementation of numpy.fromfile.'
    return np.fromiter(_batched_generator(fin, count), dtype=dtype)

def load(fin, encoding='utf-8', full_model=True):
    if False:
        while True:
            i = 10
    'Load a model from a binary stream.\n\n    Parameters\n    ----------\n    fin : file\n        The readable binary stream.\n    encoding : str, optional\n        The encoding to use for decoding text\n    full_model : boolean, optional\n        If False, skips loading the hidden output matrix.  This saves a fair bit\n        of CPU time and RAM, but prevents training continuation.\n\n    Returns\n    -------\n    :class:`~gensim.models._fasttext_bin.Model`\n        The loaded model.\n\n    '
    if isinstance(fin, str):
        fin = open(fin, 'rb')
    (magic, version) = _struct_unpack(fin, '@2i')
    new_format = magic == _FASTTEXT_FILEFORMAT_MAGIC
    header_spec = _NEW_HEADER_FORMAT if new_format else _OLD_HEADER_FORMAT
    model = {name: _struct_unpack(fin, fmt)[0] for (name, fmt) in header_spec}
    if not new_format:
        model.update(dim=magic, ws=version)
    (raw_vocab, vocab_size, nwords, ntokens) = _load_vocab(fin, new_format, encoding=encoding)
    model.update(raw_vocab=raw_vocab, vocab_size=vocab_size, nwords=nwords, ntokens=ntokens)
    vectors_ngrams = _load_matrix(fin, new_format=new_format)
    if not full_model:
        hidden_output = None
    else:
        hidden_output = _load_matrix(fin, new_format=new_format)
        assert fin.read() == b'', 'expected to reach EOF'
    model.update(vectors_ngrams=vectors_ngrams, hidden_output=hidden_output)
    model = {k: v for (k, v) in model.items() if k in _FIELD_NAMES}
    return Model(**model)

def _backslashreplace_backport(ex):
    if False:
        return 10
    'Replace byte sequences that failed to decode with character escapes.\n\n    Does the same thing as errors="backslashreplace" from Python 3.  Python 2\n    lacks this functionality out of the box, so we need to backport it.\n\n    Parameters\n    ----------\n    ex: UnicodeDecodeError\n        contains arguments of the string and start/end indexes of the bad portion.\n\n    Returns\n    -------\n    text: unicode\n        The Unicode string corresponding to the decoding of the bad section.\n    end: int\n        The index from which to continue decoding.\n\n    Note\n    ----\n    Works on Py2 only.  Py3 already has backslashreplace built-in.\n\n    '
    (bstr, start, end) = (ex.object, ex.start, ex.end)
    text = u''.join(('\\x{:02x}'.format(ord(c)) for c in bstr[start:end]))
    return (text, end)

def _sign_model(fout):
    if False:
        i = 10
        return i + 15
    "\n    Write signature of the file in Facebook's native fastText `.bin` format\n    to the binary output stream `fout`. Signature includes magic bytes and version.\n\n    Name mimics original C++ implementation, see\n    [FastText::signModel](https://github.com/facebookresearch/fastText/blob/master/src/fasttext.cc)\n\n    Parameters\n    ----------\n    fout: writeable binary stream\n    "
    fout.write(_FASTTEXT_FILEFORMAT_MAGIC.tobytes())
    fout.write(_FASTTEXT_VERSION.tobytes())

def _conv_field_to_bytes(field_value, field_type):
    if False:
        for i in range(10):
            print('nop')
    '\n    Auxiliary function that converts `field_value` to bytes based on request `field_type`,\n    for saving to the binary file.\n\n    Parameters\n    ----------\n    field_value: numerical\n        contains arguments of the string and start/end indexes of the bad portion.\n\n    field_type: str\n        currently supported `field_types` are `i` for 32-bit integer and `d` for 64-bit float\n    '
    if field_type == 'i':
        return np.int32(field_value).tobytes()
    elif field_type == 'd':
        return np.float64(field_value).tobytes()
    else:
        raise NotImplementedError('Currently conversion to "%s" type is not implemmented.' % field_type)

def _get_field_from_model(model, field):
    if False:
        return 10
    '\n    Extract `field` from `model`.\n\n    Parameters\n    ----------\n    model: gensim.models.fasttext.FastText\n        model from which `field` is extracted\n    field: str\n        requested field name, fields are listed in the `_NEW_HEADER_FORMAT` list\n    '
    if field == 'bucket':
        return model.wv.bucket
    elif field == 'dim':
        return model.vector_size
    elif field == 'epoch':
        return model.epochs
    elif field == 'loss':
        if model.hs == 1:
            return 1
        elif model.hs == 0:
            return 2
        elif model.hs == 0 and model.negative == 0:
            return 1
    elif field == 'maxn':
        return model.wv.max_n
    elif field == 'minn':
        return model.wv.min_n
    elif field == 'min_count':
        return model.min_count
    elif field == 'model':
        return 2 if model.sg == 1 else 1
    elif field == 'neg':
        return model.negative
    elif field == 't':
        return model.sample
    elif field == 'word_ngrams':
        return 1
    elif field == 'ws':
        return model.window
    elif field == 'lr_update_rate':
        return 100
    else:
        msg = 'Extraction of header field "' + field + '" from Gensim FastText object not implemmented.'
        raise NotImplementedError(msg)

def _args_save(fout, model, fb_fasttext_parameters):
    if False:
        while True:
            i = 10
    "\n    Saves header with `model` parameters to the binary stream `fout` containing a model in the Facebook's\n    native fastText `.bin` format.\n\n    Name mimics original C++ implementation, see\n    [Args::save](https://github.com/facebookresearch/fastText/blob/master/src/args.cc)\n\n    Parameters\n    ----------\n    fout: writeable binary stream\n        stream to which model is saved\n    model: gensim.models.fasttext.FastText\n        saved model\n    fb_fasttext_parameters: dictionary\n        dictionary contain parameters containing `lr_update_rate`, `word_ngrams`\n        unused by gensim implementation, so they have to be provided externally\n    "
    for (field, field_type) in _NEW_HEADER_FORMAT:
        if field in fb_fasttext_parameters:
            field_value = fb_fasttext_parameters[field]
        else:
            field_value = _get_field_from_model(model, field)
        fout.write(_conv_field_to_bytes(field_value, field_type))

def _dict_save(fout, model, encoding):
    if False:
        while True:
            i = 10
    "\n    Saves the dictionary from `model` to the to the binary stream `fout` containing a model in the Facebook's\n    native fastText `.bin` format.\n\n    Name mimics the original C++ implementation\n    [Dictionary::save](https://github.com/facebookresearch/fastText/blob/master/src/dictionary.cc)\n\n    Parameters\n    ----------\n    fout: writeable binary stream\n        stream to which the dictionary from the model is saved\n    model: gensim.models.fasttext.FastText\n        the model that contains the dictionary to save\n    encoding: str\n        string encoding used in the output\n    "
    fout.write(np.int32(len(model.wv)).tobytes())
    fout.write(np.int32(len(model.wv)).tobytes())
    fout.write(np.int32(0).tobytes())
    fout.write(np.int64(model.corpus_total_words).tobytes())
    fout.write(np.int64(-1))
    for word in model.wv.index_to_key:
        word_count = model.wv.get_vecattr(word, 'count')
        fout.write(word.encode(encoding))
        fout.write(_END_OF_WORD_MARKER)
        fout.write(np.int64(word_count).tobytes())
        fout.write(_DICT_WORD_ENTRY_TYPE_MARKER)

def _input_save(fout, model):
    if False:
        print('Hello World!')
    "\n    Saves word and ngram vectors from `model` to the binary stream `fout` containing a model in\n    the Facebook's native fastText `.bin` format.\n\n    Corresponding C++ fastText code:\n    [DenseMatrix::save](https://github.com/facebookresearch/fastText/blob/master/src/densematrix.cc)\n\n    Parameters\n    ----------\n    fout: writeable binary stream\n        stream to which the vectors are saved\n    model: gensim.models.fasttext.FastText\n        the model that contains the vectors to save\n    "
    (vocab_n, vocab_dim) = model.wv.vectors_vocab.shape
    (ngrams_n, ngrams_dim) = model.wv.vectors_ngrams.shape
    assert vocab_dim == ngrams_dim
    assert vocab_n == len(model.wv)
    assert ngrams_n == model.wv.bucket
    fout.write(struct.pack('@2q', vocab_n + ngrams_n, vocab_dim))
    fout.write(model.wv.vectors_vocab.tobytes())
    fout.write(model.wv.vectors_ngrams.tobytes())

def _output_save(fout, model):
    if False:
        print('Hello World!')
    "\n    Saves output layer of `model` to the binary stream `fout` containing a model in\n    the Facebook's native fastText `.bin` format.\n\n    Corresponding C++ fastText code:\n    [DenseMatrix::save](https://github.com/facebookresearch/fastText/blob/master/src/densematrix.cc)\n\n    Parameters\n    ----------\n    fout: writeable binary stream\n        the model that contains the output layer to save\n    model: gensim.models.fasttext.FastText\n        saved model\n    "
    if model.hs:
        hidden_output = model.syn1
    if model.negative:
        hidden_output = model.syn1neg
    (hidden_n, hidden_dim) = hidden_output.shape
    fout.write(struct.pack('@2q', hidden_n, hidden_dim))
    fout.write(hidden_output.tobytes())

def _save_to_stream(model, fout, fb_fasttext_parameters, encoding):
    if False:
        while True:
            i = 10
    "\n    Saves word embeddings to binary stream `fout` using the Facebook's native fasttext `.bin` format.\n\n    Parameters\n    ----------\n    fout: file name or writeable binary stream\n        stream to which the word embeddings are saved\n    model: gensim.models.fasttext.FastText\n        the model that contains the word embeddings to save\n    fb_fasttext_parameters: dictionary\n        dictionary contain parameters containing `lr_update_rate`, `word_ngrams`\n        unused by gensim implementation, so they have to be provided externally\n    encoding: str\n        encoding used in the output file\n    "
    _sign_model(fout)
    _args_save(fout, model, fb_fasttext_parameters)
    _dict_save(fout, model, encoding)
    fout.write(struct.pack('@?', False))
    _input_save(fout, model)
    fout.write(struct.pack('@?', False))
    _output_save(fout, model)

def save(model, fout, fb_fasttext_parameters, encoding):
    if False:
        return 10
    "\n    Saves word embeddings to the Facebook's native fasttext `.bin` format.\n\n    Parameters\n    ----------\n    fout: file name or writeable binary stream\n        stream to which model is saved\n    model: gensim.models.fasttext.FastText\n        saved model\n    fb_fasttext_parameters: dictionary\n        dictionary contain parameters containing `lr_update_rate`, `word_ngrams`\n        unused by gensim implementation, so they have to be provided externally\n    encoding: str\n        encoding used in the output file\n\n    Notes\n    -----\n    Unfortunately, there is no documentation of the Facebook's native fasttext `.bin` format\n\n    This is just reimplementation of\n    [FastText::saveModel](https://github.com/facebookresearch/fastText/blob/master/src/fasttext.cc)\n\n    Based on v0.9.1, more precisely commit da2745fcccb848c7a225a7d558218ee4c64d5333\n\n    Code follows the original C++ code naming.\n    "
    if isinstance(fout, str):
        with open(fout, 'wb') as fout_stream:
            _save_to_stream(model, fout_stream, fb_fasttext_parameters, encoding)
    else:
        _save_to_stream(model, fout, fb_fasttext_parameters, encoding)