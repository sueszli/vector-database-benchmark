"""
This module implements a corpus class that stores its data in separate files called
"shards". This is a compromise between speed (keeping the whole dataset
in memory) and memory footprint (keeping the data on disk and reading from it
on demand).

The corpus is intended for situations where you need to use your data
as numpy arrays for some iterative processing (like training something
using SGD, which usually involves heavy matrix multiplication).

"""
from __future__ import print_function
import logging
import os
import math
import time
import numpy
import scipy.sparse as sparse
import gensim
from gensim.corpora import IndexedCorpus
from gensim.interfaces import TransformedCorpus
logger = logging.getLogger(__name__)
_default_dtype = float
try:
    import theano
    _default_dtype = theano.config.floatX
except ImportError:
    logger.info('Could not import Theano, will use standard float for default ShardedCorpus dtype.')

class ShardedCorpus(IndexedCorpus):
    """
    This corpus is designed for situations where you need to train a model
    on matrices, with a large number of iterations. (It should be faster than
    gensim's other IndexedCorpus implementations for this use case; check the
    `benchmark_datasets.py` script. It should also serialize faster.)

    The corpus stores its data in separate files called
    "shards". This is a compromise between speed (keeping the whole dataset
    in memory) and memory footprint (keeping the data on disk and reading from
    it on demand). Persistence is done using the standard gensim load/save methods.

    .. note::

      The dataset is **read-only**, there is - as opposed to gensim's Similarity
      class, which works similarly - no way of adding documents to the dataset
      (for now).

    You can use ShardedCorpus to serialize your data just like any other gensim
    corpus that implements serialization. However, because the data is saved
    as numpy 2-dimensional ndarrays (or scipy sparse matrices), you need to
    supply the dimension of your data to the corpus. (The dimension of word
    frequency vectors will typically be the size of the vocabulary, etc.)

    .. sourcecode:: pycon

        >>> corpus = gensim.utils.mock_data()
        >>> output_prefix = 'mydata.shdat'
        >>> ShardedCorpus.serialize(output_prefix, corpus, dim=1000)

    The `output_prefix` tells the ShardedCorpus where to put the data.
    Shards are saved as `output_prefix.0`, `output_prefix.1`, etc.
    All shards must be of the same size. The shards can be re-sized (which
    is essentially a re-serialization into new-size shards), but note that
    this operation will temporarily take twice as much disk space, because
    the old shards are not deleted until the new shards are safely in place.

    After serializing the data, the corpus will then save itself to the file
    `output_prefix`.

    On further initialization with the same `output_prefix`, the corpus
    will load the already built dataset unless the `overwrite` option is
    given. (A new object is "cloned" from the one saved to `output_prefix`
    previously.)

    To retrieve data, you can load the corpus and use it like a list:

    .. sourcecode:: pycon

        >>> sh_corpus = ShardedCorpus.load(output_prefix)
        >>> batch = sh_corpus[100:150]

    This will retrieve a numpy 2-dimensional array of 50 rows and 1000
    columns (1000 was the dimension of the data we supplied to the corpus).
    To retrieve gensim-style sparse vectors, set the `gensim` property:

    .. sourcecode:: pycon

        >>> sh_corpus.gensim = True
        >>> batch = sh_corpus[100:150]

    The batch now will be a generator of gensim vectors.

    Since the corpus needs the data serialized in order to be able to operate,
    it will serialize data right away on initialization. Instead of calling
    `ShardedCorpus.serialize()`, you can just initialize and use the corpus
    right away:

    .. sourcecode:: pycon

        >>> corpus = ShardedCorpus(output_prefix, corpus, dim=1000)
        >>> batch = corpus[100:150]

    ShardedCorpus also supports working with scipy sparse matrices, both
    during retrieval and during serialization. If you want to serialize your
    data as sparse matrices, set the `sparse_serialization` flag. For
    retrieving your data as sparse matrices, use the `sparse_retrieval`
    flag. (You can also retrieve densely serialized data as sparse matrices,
    for the sake of completeness, and vice versa.) By default, the corpus
    will retrieve numpy ndarrays even if it was serialized into sparse
    matrices.

    .. sourcecode:: pycon

        >>> sparse_prefix = 'mydata.sparse.shdat'
        >>> ShardedCorpus.serialize(sparse_prefix, corpus, dim=1000, sparse_serialization=True)
        >>> sparse_corpus = ShardedCorpus.load(sparse_prefix)
        >>> batch = sparse_corpus[100:150]
        >>> type(batch)
        <type 'numpy.ndarray'>
        >>> sparse_corpus.sparse_retrieval = True
        >>> batch = sparse_corpus[100:150]
        <class 'scipy.sparse.csr.csr_matrix'>

    While you *can* touch the `sparse_retrieval` attribute during the life
    of a ShardedCorpus object, you should definitely not touch `
    `sharded_serialization`! Changing the attribute will not miraculously
    re-serialize the data in the requested format.

    The CSR format is used for sparse data throughout.

    Internally, to retrieve data, the dataset keeps track of which shard is
    currently open and on a `__getitem__` request, either returns an item from
    the current shard, or opens a new one. The shard size is constant, except
    for the last shard.
    """

    def __init__(self, output_prefix, corpus, dim=None, shardsize=4096, overwrite=False, sparse_serialization=False, sparse_retrieval=False, gensim=False):
        if False:
            while True:
                i = 10
        "Initializes the dataset. If `output_prefix` is not found,\n        builds the shards.\n\n        :type output_prefix: str\n        :param output_prefix: The absolute path to the file from which shard\n            filenames should be derived. The individual shards will be saved\n            as `output_prefix.0`, `output_prefix.1`, etc.\n\n            The `output_prefix` path then works as the filename to which\n            the ShardedCorpus object itself will be automatically saved.\n            Normally, gensim corpora do not do this, but ShardedCorpus needs\n            to remember several serialization settings: namely the shard\n            size and whether it was serialized in dense or sparse format. By\n            saving automatically, any new ShardedCorpus with the same\n            `output_prefix` will be able to find the information about the\n            data serialized with the given prefix.\n\n            If you want to *overwrite* your data serialized with some output\n            prefix, set the `overwrite` flag to True.\n\n            Of course, you can save your corpus separately as well using\n            the `save()` method.\n\n        :type corpus: gensim.interfaces.CorpusABC\n        :param corpus: The source corpus from which to build the dataset.\n\n        :type dim: int\n        :param dim: Specify beforehand what the dimension of a dataset item\n            should be. This is useful when initializing from a corpus that\n            doesn't advertise its dimension, or when it does and you want to\n            check that the corpus matches the expected dimension. **If `dim`\n            is left unused and `corpus` does not provide its dimension in\n            an expected manner, initialization will fail.**\n\n        :type shardsize: int\n        :param shardsize: How many data points should be in one shard. More\n            data per shard means less shard reloading but higher memory usage\n            and vice versa.\n\n        :type overwrite: bool\n        :param overwrite: If set, will build dataset from given corpus even\n            if `output_prefix` already exists.\n\n        :type sparse_serialization: bool\n        :param sparse_serialization: If set, will save the data in a sparse\n            form (as csr matrices). This is to speed up retrieval when you\n            know you will be using sparse matrices.\n\n            ..note::\n\n                This property **should not change** during the lifetime of\n                the dataset. (If you find out you need to change from a sparse\n                to a dense representation, the best practice is to create\n                another ShardedCorpus object.)\n\n        :type sparse_retrieval: bool\n        :param sparse_retrieval: If set, will retrieve data as sparse vectors\n            (numpy csr matrices). If unset, will return ndarrays.\n\n            Note that retrieval speed for this option depends on how the dataset\n            was serialized. If `sparse_serialization` was set, then setting\n            `sparse_retrieval` will be faster. However, if the two settings\n            do not correspond, the conversion on the fly will slow the dataset\n            down.\n\n        :type gensim: bool\n        :param gensim: If set, will convert the output to gensim\n            sparse vectors (list of tuples (id, value)) to make it behave like\n            any other gensim corpus. This **will** slow the dataset down.\n\n        "
        self.output_prefix = output_prefix
        self.shardsize = shardsize
        self.n_docs = 0
        self.offsets = []
        self.n_shards = 0
        self.dim = dim
        self.sparse_serialization = sparse_serialization
        self.sparse_retrieval = sparse_retrieval
        self.gensim = gensim
        self.current_shard = None
        self.current_shard_n = None
        self.current_offset = None
        logger.info('Initializing sharded corpus with prefix %s', output_prefix)
        if not os.path.isfile(output_prefix) or overwrite:
            logger.info('Building from corpus...')
            self.init_shards(output_prefix, corpus, shardsize)
            logger.info('Saving ShardedCorpus object to %s', self.output_prefix)
            self.save()
        else:
            logger.info('Cloning existing...')
            self.init_by_clone()

    def init_shards(self, output_prefix, corpus, shardsize=4096, dtype=_default_dtype):
        if False:
            i = 10
            return i + 15
        'Initialize shards from the corpus.'
        (is_corpus, corpus) = gensim.utils.is_corpus(corpus)
        if not is_corpus:
            raise ValueError('Cannot initialize shards without a corpus to read from! Corpus type: %s' % type(corpus))
        proposed_dim = self._guess_n_features(corpus)
        if proposed_dim != self.dim:
            if self.dim is None:
                logger.info('Deriving dataset dimension from corpus: %d', proposed_dim)
            else:
                logger.warning('Dataset dimension derived from input corpus differs from initialization argument, using corpus. (corpus %d, init arg %d)', proposed_dim, self.dim)
        self.dim = proposed_dim
        self.offsets = [0]
        start_time = time.perf_counter()
        logger.info('Running init from corpus.')
        for (n, doc_chunk) in enumerate(gensim.utils.grouper(corpus, chunksize=shardsize)):
            logger.info('Chunk no. %d at %f s', n, time.perf_counter() - start_time)
            current_shard = numpy.zeros((len(doc_chunk), self.dim), dtype=dtype)
            logger.debug('Current chunk dimension: %d x %d', len(doc_chunk), self.dim)
            for (i, doc) in enumerate(doc_chunk):
                doc = dict(doc)
                current_shard[i][list(doc)] = list(doc.values())
            if self.sparse_serialization:
                current_shard = sparse.csr_matrix(current_shard)
            self.save_shard(current_shard)
        end_time = time.perf_counter()
        logger.info('Built %d shards in %f s.', self.n_shards, end_time - start_time)

    def init_by_clone(self):
        if False:
            i = 10
            return i + 15
        '\n        Initialize by copying over attributes of another ShardedCorpus\n        instance saved to the output_prefix given at __init__().\n\n        '
        temp = self.__class__.load(self.output_prefix)
        self.n_shards = temp.n_shards
        self.n_docs = temp.n_docs
        self.offsets = temp.offsets
        if temp.dim != self.dim:
            if self.dim is None:
                logger.info('Loaded dataset dimension: %d', temp.dim)
            else:
                logger.warning('Loaded dataset dimension differs from init arg dimension, using loaded dim. (loaded %d, init %d)', temp.dim, self.dim)
        self.dim = temp.dim

    def save_shard(self, shard, n=None, filename=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Pickle the given shard. If `n` is not given, will consider the shard\n        a new one.\n\n        If `filename` is given, will use that file name instead of generating\n        one.\n\n        '
        new_shard = False
        if n is None:
            n = self.n_shards
            new_shard = True
        if not filename:
            filename = self._shard_name(n)
        gensim.utils.pickle(shard, filename)
        if new_shard:
            self.offsets.append(self.offsets[-1] + shard.shape[0])
            self.n_docs += shard.shape[0]
            self.n_shards += 1

    def load_shard(self, n):
        if False:
            i = 10
            return i + 15
        '\n        Load (unpickle) the n-th shard as the "live" part of the dataset\n        into the Dataset object.'
        if self.current_shard_n == n:
            return
        filename = self._shard_name(n)
        if not os.path.isfile(filename):
            raise ValueError('Attempting to load nonexistent shard no. %s' % n)
        shard = gensim.utils.unpickle(filename)
        self.current_shard = shard
        self.current_shard_n = n
        self.current_offset = self.offsets[n]

    def reset(self):
        if False:
            while True:
                i = 10
        '\n        Reset to no shard at all. Used for saving.\n\n        '
        self.current_shard = None
        self.current_shard_n = None
        self.current_offset = None

    def shard_by_offset(self, offset):
        if False:
            for i in range(10):
                print('nop')
        '\n        Determine which shard the given offset belongs to. If the offset\n        is greater than the number of available documents, raises a\n        `ValueError`.\n\n        Assumes that all shards have the same size.\n\n        '
        k = int(offset / self.shardsize)
        if offset >= self.n_docs:
            raise ValueError('Too high offset specified (%s), available docs: %s' % (offset, self.n_docs))
        if offset < 0:
            raise ValueError('Negative offset %s currently not supported.' % offset)
        return k

    def in_current(self, offset):
        if False:
            return 10
        '\n        Determine whether the given offset falls within the current shard.\n\n        '
        return self.current_offset <= offset and offset < self.offsets[self.current_shard_n + 1]

    def in_next(self, offset):
        if False:
            for i in range(10):
                print('nop')
        '\n        Determine whether the given offset falls within the next shard.\n        This is a very small speedup: typically, we will be iterating through\n        the data forward. Could save considerable time with a very large number\n        of smaller shards.\n\n        '
        if self.current_shard_n == self.n_shards:
            return False
        return self.offsets[self.current_shard_n + 1] <= offset and offset < self.offsets[self.current_shard_n + 2]

    def resize_shards(self, shardsize):
        if False:
            print('Hello World!')
        "\n        Re-process the dataset to new shard size. This may take pretty long.\n        Also, note that you need some space on disk for this one (we're\n        assuming there is enough disk space for double the size of the dataset\n        and that there is enough memory for old + new shardsize).\n\n        :type shardsize: int\n        :param shardsize: The new shard size.\n\n        "
        n_new_shards = int(math.floor(self.n_docs / float(shardsize)))
        if self.n_docs % shardsize != 0:
            n_new_shards += 1
        new_shard_names = []
        new_offsets = [0]
        for new_shard_idx in range(n_new_shards):
            new_start = shardsize * new_shard_idx
            new_stop = new_start + shardsize
            if new_stop > self.n_docs:
                assert new_shard_idx == n_new_shards - 1, 'Shard no. %r that ends at %r over last document (%r) is not the last projected shard (%r)' % (new_shard_idx, new_stop, self.n_docs, n_new_shards)
                new_stop = self.n_docs
            new_shard = self[new_start:new_stop]
            new_shard_name = self._resized_shard_name(new_shard_idx)
            new_shard_names.append(new_shard_name)
            try:
                self.save_shard(new_shard, new_shard_idx, new_shard_name)
            except Exception:
                for new_shard_name in new_shard_names:
                    os.remove(new_shard_name)
                raise
            new_offsets.append(new_stop)
        old_shard_names = [self._shard_name(n) for n in range(self.n_shards)]
        try:
            for (old_shard_n, old_shard_name) in enumerate(old_shard_names):
                os.remove(old_shard_name)
        except Exception as e:
            logger.exception('Error during old shard no. %d removal: %s.\nAttempting to at least move new shards in.', old_shard_n, str(e))
        finally:
            try:
                for (shard_n, new_shard_name) in enumerate(new_shard_names):
                    os.rename(new_shard_name, self._shard_name(shard_n))
            except Exception as e:
                logger.exception(e)
                raise RuntimeError('Resizing completely failed. Sorry, dataset is probably ruined...')
            finally:
                self.n_shards = n_new_shards
                self.offsets = new_offsets
                self.shardsize = shardsize
                self.reset()

    def _shard_name(self, n):
        if False:
            while True:
                i = 10
        'Generate the name for the n-th shard.'
        return self.output_prefix + '.' + str(n)

    def _resized_shard_name(self, n):
        if False:
            while True:
                i = 10
        '\n        Generate the name for the n-th new shard temporary file when\n        resizing dataset. The file will then be re-named to standard shard name.\n        '
        return self.output_prefix + '.resize-temp.' + str(n)

    def _guess_n_features(self, corpus):
        if False:
            print('Hello World!')
        'Attempt to guess number of features in `corpus`.'
        n_features = None
        if hasattr(corpus, 'dim'):
            n_features = corpus.dim
        elif hasattr(corpus, 'dictionary'):
            n_features = len(corpus.dictionary)
        elif hasattr(corpus, 'n_out'):
            n_features = corpus.n_out
        elif hasattr(corpus, 'num_terms'):
            n_features = corpus.num_terms
        elif isinstance(corpus, TransformedCorpus):
            try:
                return self._guess_n_features(corpus.obj)
            except TypeError:
                return self._guess_n_features(corpus.corpus)
        else:
            if not self.dim:
                raise TypeError("Couldn't find number of features, refusing to guess. Dimension: %s, corpus: %s)" % (self.dim, type(corpus)))
            logger.warning("Couldn't find number of features, trusting supplied dimension (%d)", self.dim)
            n_features = self.dim
        if self.dim and n_features != self.dim:
            logger.warning('Discovered inconsistent dataset dim (%d) and feature count from corpus (%d). Coercing to dimension given by argument.', self.dim, n_features)
        return n_features

    def __len__(self):
        if False:
            while True:
                i = 10
        return self.n_docs

    def _ensure_shard(self, offset):
        if False:
            while True:
                i = 10
        if self.current_shard is None:
            shard_n = self.shard_by_offset(offset)
            self.load_shard(shard_n)
        elif not self.in_current(offset):
            if self.in_next(offset):
                self.load_shard(self.current_shard_n + 1)
            else:
                shard_n = self.shard_by_offset(offset)
                self.load_shard(shard_n)

    def get_by_offset(self, offset):
        if False:
            while True:
                i = 10
        'As opposed to getitem, this one only accepts ints as offsets.'
        self._ensure_shard(offset)
        result = self.current_shard[offset - self.current_offset]
        return result

    def __getitem__(self, offset):
        if False:
            print('Hello World!')
        '\n        Retrieve the given row of the dataset. Supports slice notation.\n\n        '
        if isinstance(offset, list):
            if self.sparse_serialization:
                l_result = sparse.vstack([self.get_by_offset(i) for i in offset])
                if self.gensim:
                    l_result = self._getitem_sparse2gensim(l_result)
                elif not self.sparse_retrieval:
                    l_result = numpy.array(l_result.todense())
            else:
                l_result = numpy.array([self.get_by_offset(i) for i in offset])
                if self.gensim:
                    l_result = self._getitem_dense2gensim(l_result)
                elif self.sparse_retrieval:
                    l_result = sparse.csr_matrix(l_result)
            return l_result
        elif isinstance(offset, slice):
            start = offset.start
            stop = offset.stop
            if stop > self.n_docs:
                raise IndexError('Requested slice offset %s out of range (%s docs)' % (stop, self.n_docs))
            first_shard = self.shard_by_offset(start)
            last_shard = self.n_shards - 1
            if not stop == self.n_docs:
                last_shard = self.shard_by_offset(stop)
            self.load_shard(first_shard)
            if first_shard == last_shard:
                s_result = self.current_shard[start - self.current_offset:stop - self.current_offset]
                s_result = self._getitem_format(s_result)
                return s_result
            s_result = numpy.zeros((stop - start, self.dim), dtype=self.current_shard.dtype)
            if self.sparse_serialization:
                s_result = sparse.csr_matrix((0, self.dim), dtype=self.current_shard.dtype)
            result_start = 0
            result_stop = self.offsets[self.current_shard_n + 1] - start
            shard_start = start - self.current_offset
            shard_stop = self.offsets[self.current_shard_n + 1] - self.current_offset
            s_result = self.__add_to_slice(s_result, result_start, result_stop, shard_start, shard_stop)
            for shard_n in range(first_shard + 1, last_shard):
                self.load_shard(shard_n)
                result_start = result_stop
                result_stop += self.shardsize
                shard_start = 0
                shard_stop = self.shardsize
                s_result = self.__add_to_slice(s_result, result_start, result_stop, shard_start, shard_stop)
            self.load_shard(last_shard)
            result_start = result_stop
            result_stop += stop - self.current_offset
            shard_start = 0
            shard_stop = stop - self.current_offset
            s_result = self.__add_to_slice(s_result, result_start, result_stop, shard_start, shard_stop)
            s_result = self._getitem_format(s_result)
            return s_result
        else:
            s_result = self.get_by_offset(offset)
            s_result = self._getitem_format(s_result)
            return s_result

    def __add_to_slice(self, s_result, result_start, result_stop, start, stop):
        if False:
            return 10
        '\n        Add rows of the current shard from `start` to `stop`\n        into rows `result_start` to `result_stop` of `s_result`.\n\n        Operation is based on the ``self.sparse_serialize`` setting. If the shard\n        contents are dense, then s_result is assumed to be an ndarray that\n        already supports row indices `result_start:result_stop`. If the shard\n        contents are sparse, assumes that s_result has `result_start` rows\n        and we should add them up to `result_stop`.\n\n        Return the resulting ``s_result``.\n\n        '
        if result_stop - result_start != stop - start:
            raise ValueError('Result start/stop range different than stop/start range (%s - %s vs. %s - %s)' % (result_start, result_stop, start, stop))
        if not self.sparse_serialization:
            s_result[result_start:result_stop] = self.current_shard[start:stop]
            return s_result
        if s_result.shape != (result_start, self.dim):
            raise ValueError('Assuption about sparse s_result shape invalid: %s expected rows, %s real rows.' % (result_start, s_result.shape[0]))
        tmp_matrix = self.current_shard[start:stop]
        s_result = sparse.vstack([s_result, tmp_matrix])
        return s_result

    def _getitem_format(self, s_result):
        if False:
            i = 10
            return i + 15
        if self.sparse_serialization:
            if self.gensim:
                s_result = self._getitem_sparse2gensim(s_result)
            elif not self.sparse_retrieval:
                s_result = numpy.array(s_result.todense())
        elif self.gensim:
            s_result = self._getitem_dense2gensim(s_result)
        elif self.sparse_retrieval:
            s_result = sparse.csr_matrix(s_result)
        return s_result

    def _getitem_sparse2gensim(self, result):
        if False:
            while True:
                i = 10
        '\n        Change given sparse result matrix to gensim sparse vectors.\n\n        Uses the internals of the sparse matrix to make this fast.\n\n        '

        def row_sparse2gensim(row_idx, csr_matrix):
            if False:
                print('Hello World!')
            indices = csr_matrix.indices[csr_matrix.indptr[row_idx]:csr_matrix.indptr[row_idx + 1]]
            g_row = [(col_idx, csr_matrix[row_idx, col_idx]) for col_idx in indices]
            return g_row
        output = (row_sparse2gensim(i, result) for i in range(result.shape[0]))
        return output

    def _getitem_dense2gensim(self, result):
        if False:
            print('Hello World!')
        'Change given dense result matrix to gensim sparse vectors.'
        if len(result.shape) == 1:
            output = gensim.matutils.full2sparse(result)
        else:
            output = (gensim.matutils.full2sparse(result[i]) for i in range(result.shape[0]))
        return output

    def __iter__(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Yield dataset items one by one (generator).\n\n        '
        for i in range(len(self)):
            yield self[i]

    def save(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Save itself (the wrapper) in clean state (after calling `reset()`)\n        to the output_prefix file. If you wish to save to a different file,\n        use the `fname` argument as the first positional arg.\n\n        '
        if len(args) == 0:
            args = (self.output_prefix,)
        attrs_to_ignore = ['current_shard', 'current_shard_n', 'current_offset']
        if 'ignore' in kwargs:
            attrs_to_ignore.extend(kwargs['ignore'])
        kwargs['ignore'] = frozenset(attrs_to_ignore)
        super(ShardedCorpus, self).save(*args, **kwargs)

    @classmethod
    def load(cls, fname, mmap=None):
        if False:
            while True:
                i = 10
        '\n        Load itself in clean state. `mmap` has no effect here.\n        '
        return super(ShardedCorpus, cls).load(fname, mmap)

    @staticmethod
    def save_corpus(fname, corpus, id2word=None, progress_cnt=1000, metadata=False, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Implement a serialization interface. Do not call directly;\n        use the `serialize` method instead.\n\n        Note that you might need some ShardedCorpus init parameters, most\n        likely the dimension (`dim`). Again, pass these as `kwargs` to the\n        `serialize` method.\n\n        All this thing does is initialize a ShardedCorpus from a corpus\n        with the `output_prefix` argument set to the `fname` parameter\n        of this method. The initialization of a ShardedCorpus takes care of\n        serializing the data (in dense form) to shards.\n\n        Ignore the parameters id2word, progress_cnt and metadata. They\n        currently do nothing and are here only to provide a compatible\n        method signature with superclass.\n\n        '
        ShardedCorpus(fname, corpus, **kwargs)

    @classmethod
    def serialize(serializer, fname, corpus, id2word=None, index_fname=None, progress_cnt=None, labels=None, metadata=False, **kwargs):
        if False:
            return 10
        '\n        Iterate through the document stream `corpus`, saving the documents\n        as a ShardedCorpus to `fname`.\n\n        Use this method instead of calling `save_corpus` directly.\n        You may need to supply some kwargs that are used upon dataset creation\n        (namely: `dim`, unless the dataset can infer the dimension from the\n        given corpus).\n\n        Ignore the parameters id2word, index_fname, progress_cnt, labels\n        and metadata. They currently do nothing and are here only to\n        provide a compatible method signature with superclass.\n\n        '
        serializer.save_corpus(fname, corpus, id2word=id2word, progress_cnt=progress_cnt, metadata=metadata, **kwargs)