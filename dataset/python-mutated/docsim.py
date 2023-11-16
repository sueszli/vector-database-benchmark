"""Compute similarities across a collection of documents in the Vector Space Model.

The main class is :class:`~gensim.similarities.docsim.Similarity`, which builds an index for a given set of documents.

Once the index is built, you can perform efficient queries like "Tell me how similar is this query document to each
document in the index?". The result is a vector of numbers as large as the size of the initial set of documents,
that is, one float for each index document. Alternatively, you can also request only the top-N most
similar index documents to the query.


How It Works
------------
The :class:`~gensim.similarities.docsim.Similarity` class splits the index into several smaller sub-indexes ("shards"),
which are disk-based. If your entire index fits in memory (~one million documents per 1GB of RAM),
you can also use the :class:`~gensim.similarities.docsim.MatrixSimilarity`
or :class:`~gensim.similarities.docsim.SparseMatrixSimilarity` classes directly.
These are more simple but do not scale as well: they keep the entire index in RAM, no sharding. They also do not
support adding new document to the index dynamically.

Once the index has been initialized, you can query for document similarity simply by

.. sourcecode:: pycon

    >>> from gensim.similarities import Similarity
    >>> from gensim.test.utils import common_corpus, common_dictionary, get_tmpfile
    >>>
    >>> index_tmpfile = get_tmpfile("index")
    >>> query = [(1, 2), (6, 1), (7, 2)]
    >>>
    >>> index = Similarity(index_tmpfile, common_corpus, num_features=len(common_dictionary))  # build the index
    >>> similarities = index[query]  # get similarities between the query and all index documents

If you have more query documents, you can submit them all at once, in a batch

.. sourcecode:: pycon

    >>> from gensim.test.utils import common_corpus, common_dictionary, get_tmpfile
    >>>
    >>> index_tmpfile = get_tmpfile("index")
    >>> batch_of_documents = common_corpus[:]  # only as example
    >>> index = Similarity(index_tmpfile, common_corpus, num_features=len(common_dictionary))  # build the index
    >>>
    >>> # the batch is simply an iterable of documents, aka gensim corpus:
    >>> for similarities in index[batch_of_documents]:
    ...     pass

The benefit of this batch (aka "chunked") querying is a much better performance.
To see the speed-up on your machine, run ``python -m gensim.test.simspeed``
(compare to my results `here <https://groups.google.com/g/gensim/c/9rg5zqoWyDQ/m/yk-ehhoXb08J>`_).

There is also a special syntax for when you need similarity of documents in the index
to the index itself (i.e. queries = the indexed documents themselves). This special syntax
uses the faster, batch queries internally and **is ideal for all-vs-all pairwise similarities**:

.. sourcecode:: pycon

    >>> from gensim.test.utils import common_corpus, common_dictionary, get_tmpfile
    >>>
    >>> index_tmpfile = get_tmpfile("index")
    >>> index = Similarity(index_tmpfile, common_corpus, num_features=len(common_dictionary))  # build the index
    >>>
    >>> for similarities in index:  # yield similarities of the 1st indexed document, then 2nd...
    ...     pass

"""
import logging
import itertools
import os
import heapq
import warnings
import numpy
import scipy.sparse
from gensim import interfaces, utils, matutils
logger = logging.getLogger(__name__)
PARALLEL_SHARDS = False
try:
    import multiprocessing
except ImportError:
    pass

class Shard(utils.SaveLoad):
    """A proxy that represents a single shard instance within :class:`~gensim.similarity.docsim.Similarity` index.

    Basically just wraps :class:`~gensim.similarities.docsim.MatrixSimilarity`,
    :class:`~gensim.similarities.docsim.SparseMatrixSimilarity`, etc, so that it mmaps from disk on request (query).

    """

    def __init__(self, fname, index):
        if False:
            for i in range(10):
                print('nop')
        '\n\n        Parameters\n        ----------\n        fname : str\n            Path to top-level directory (file) to traverse for corpus documents.\n        index : :class:`~gensim.interfaces.SimilarityABC`\n            Index object.\n\n        '
        (self.dirname, self.fname) = os.path.split(fname)
        self.length = len(index)
        self.cls = index.__class__
        logger.info('saving index shard to %s', self.fullname())
        index.save(self.fullname())
        self.index = self.get_index()

    def fullname(self):
        if False:
            while True:
                i = 10
        'Get full path to shard file.\n\n        Return\n        ------\n        str\n            Path to shard instance.\n\n        '
        return os.path.join(self.dirname, self.fname)

    def __len__(self):
        if False:
            for i in range(10):
                print('nop')
        'Get length.'
        return self.length

    def __getstate__(self):
        if False:
            return 10
        'Special handler for pickle.\n\n        Returns\n        -------\n        dict\n            Object that contains state of current instance without `index`.\n\n        '
        result = self.__dict__.copy()
        if 'index' in result:
            del result['index']
        return result

    def __str__(self):
        if False:
            i = 10
            return i + 15
        return '%s<%i documents in %s>' % (self.cls.__name__, len(self), self.fullname())

    def get_index(self):
        if False:
            return 10
        'Load & get index.\n\n        Returns\n        -------\n        :class:`~gensim.interfaces.SimilarityABC`\n            Index instance.\n\n        '
        if not hasattr(self, 'index'):
            logger.debug('mmaping index from %s', self.fullname())
            self.index = self.cls.load(self.fullname(), mmap='r')
        return self.index

    def get_document_id(self, pos):
        if False:
            i = 10
            return i + 15
        'Get index vector at position `pos`.\n\n        Parameters\n        ----------\n        pos : int\n            Vector position.\n\n        Return\n        ------\n        {:class:`scipy.sparse.csr_matrix`, :class:`numpy.ndarray`}\n            Index vector. Type depends on underlying index.\n\n        Notes\n        -----\n        The vector is of the same type as the underlying index (ie., dense for\n        :class:`~gensim.similarities.docsim.MatrixSimilarity`\n        and scipy.sparse for :class:`~gensim.similarities.docsim.SparseMatrixSimilarity`.\n\n        '
        assert 0 <= pos < len(self), 'requested position out of range'
        return self.get_index().index[pos]

    def __getitem__(self, query):
        if False:
            return 10
        'Get similarities of document (or corpus) `query` to all documents in the corpus.\n\n        Parameters\n        ----------\n        query : {iterable of list of (int, number) , list of (int, number))}\n            Document or corpus.\n\n        Returns\n        -------\n        :class:`numpy.ndarray`\n            Similarities of document/corpus if index is :class:`~gensim.similarities.docsim.MatrixSimilarity` **or**\n        :class:`scipy.sparse.csr_matrix`\n            for case if index is :class:`~gensim.similarities.docsim.SparseMatrixSimilarity`.\n\n        '
        index = self.get_index()
        try:
            index.num_best = self.num_best
            index.normalize = self.normalize
        except Exception:
            raise ValueError('num_best and normalize have to be set before querying a proxy Shard object')
        return index[query]

def query_shard(args):
    if False:
        for i in range(10):
            print('nop')
    'Helper for request query from shard, same as shard[query].\n\n    Parameters\n    ---------\n    args : (list of (int, number), :class:`~gensim.interfaces.SimilarityABC`)\n        Query and Shard instances\n\n    Returns\n    -------\n    :class:`numpy.ndarray` or :class:`scipy.sparse.csr_matrix`\n        Similarities of the query against documents indexed in this shard.\n\n    '
    (query, shard) = args
    logger.debug('querying shard %s num_best=%s in process %s', shard, shard.num_best, os.getpid())
    result = shard[query]
    logger.debug('finished querying shard %s in process %s', shard, os.getpid())
    return result

def _nlargest(n, iterable):
    if False:
        return 10
    'Helper for extracting n documents with maximum similarity.\n\n    Parameters\n    ----------\n    n : int\n        Number of elements to be extracted\n    iterable : iterable of list of (int, float)\n        Iterable containing documents with computed similarities\n\n    Returns\n    -------\n    :class:`list`\n        List with the n largest elements from the dataset defined by iterable.\n\n    Notes\n    -----\n    Elements are compared by the absolute value of similarity, because negative value of similarity\n    does not mean some form of dissimilarity.\n\n    '
    return heapq.nlargest(n, itertools.chain(*iterable), key=lambda item: abs(item[1]))

class Similarity(interfaces.SimilarityABC):
    """Compute cosine similarity of a dynamic query against a corpus of documents ('the index').

    The index supports adding new documents dynamically.

    Notes
    -----
    Scalability is achieved by sharding the index into smaller pieces, each of which fits into core memory
    The shards themselves are simply stored as files to disk and mmap'ed back as needed.

    Examples
    --------
    .. sourcecode:: pycon

        >>> from gensim.corpora.textcorpus import TextCorpus
        >>> from gensim.test.utils import datapath, get_tmpfile
        >>> from gensim.similarities import Similarity
        >>>
        >>> corpus = TextCorpus(datapath('testcorpus.mm'))
        >>> index_temp = get_tmpfile("index")
        >>> index = Similarity(index_temp, corpus, num_features=400)  # create index
        >>>
        >>> query = next(iter(corpus))
        >>> result = index[query]  # search similar to `query` in index
        >>>
        >>> for sims in index[corpus]:  # if you have more query documents, you can submit them all at once, in a batch
        ...     pass
        >>>
        >>> # There is also a special syntax for when you need similarity of documents in the index
        >>> # to the index itself (i.e. queries=indexed documents themselves). This special syntax
        >>> # uses the faster, batch queries internally and **is ideal for all-vs-all pairwise similarities**:
        >>> for similarities in index:  # yield similarities of the 1st indexed document, then 2nd...
        ...     pass

    See Also
    --------
    :class:`~gensim.similarities.docsim.MatrixSimilarity`
        Index similarity (dense with cosine distance).
    :class:`~gensim.similarities.docsim.SparseMatrixSimilarity`
        Index similarity (sparse with cosine distance).
    :class:`~gensim.similarities.docsim.WmdSimilarity`
        Index similarity (with word-mover distance).

    """

    def __init__(self, output_prefix, corpus, num_features, num_best=None, chunksize=256, shardsize=32768, norm='l2'):
        if False:
            return 10
        "\n\n        Parameters\n        ----------\n        output_prefix : str\n            Prefix for shard filename. If None, a random filename in temp will be used.\n        corpus : iterable of list of (int, number)\n            Corpus in streamed Gensim bag-of-words format.\n        num_features : int\n            Size of the dictionary (number of features).\n        num_best : int, optional\n            If set, return only the `num_best` most similar documents, always leaving out documents with similarity = 0.\n            Otherwise, return a full vector with one float for every document in the index.\n        chunksize : int, optional\n            Size of query chunks. Used internally when the query is an entire corpus.\n        shardsize : int, optional\n            Maximum shard size, in documents. Choose a value so that a `shardsize x chunksize` matrix of floats fits\n            comfortably into your RAM.\n        norm : {'l1', 'l2'}, optional\n            Normalization to use.\n\n        Notes\n        -----\n        Documents are split (internally, transparently) into shards of `shardsize` documents each, and each shard\n        converted to a matrix, for faster BLAS calls. Each shard is stored to disk under `output_prefix.shard_number`.\n\n        If you don't specify an output prefix, a random filename in temp will be used.\n\n        If your entire index fits in memory (~1 million documents per 1GB of RAM), you can also use the\n        :class:`~gensim.similarities.docsim.MatrixSimilarity` or\n        :class:`~gensim.similarities.docsim.SparseMatrixSimilarity` classes directly.\n        These are more simple but do not scale as well (they keep the entire index in RAM, no sharding).\n        They also do not support adding new document dynamically.\n\n        "
        if output_prefix is None:
            self.output_prefix = utils.randfname(prefix='simserver')
        else:
            self.output_prefix = output_prefix
        logger.info('starting similarity index under %s', self.output_prefix)
        self.num_features = num_features
        self.num_best = num_best
        self.norm = norm
        self.chunksize = int(chunksize)
        self.shardsize = shardsize
        self.shards = []
        (self.fresh_docs, self.fresh_nnz) = ([], 0)
        if corpus is not None:
            self.add_documents(corpus)

    def __len__(self):
        if False:
            i = 10
            return i + 15
        'Get length of index.'
        return len(self.fresh_docs) + sum((len(shard) for shard in self.shards))

    def __str__(self):
        if False:
            print('Hello World!')
        return '%s<%i documents in %i shards stored under %s>' % (self.__class__.__name__, len(self), len(self.shards), self.output_prefix)

    def add_documents(self, corpus):
        if False:
            for i in range(10):
                print('nop')
        'Extend the index with new documents.\n\n        Parameters\n        ----------\n        corpus : iterable of list of (int, number)\n            Corpus in BoW format.\n\n        Notes\n        -----\n        Internally, documents are buffered and then spilled to disk when there\'s `self.shardsize` of them\n        (or when a query is issued).\n\n        Examples\n        --------\n        .. sourcecode:: pycon\n\n            >>> from gensim.corpora.textcorpus import TextCorpus\n            >>> from gensim.test.utils import datapath, get_tmpfile\n            >>> from gensim.similarities import Similarity\n            >>>\n            >>> corpus = TextCorpus(datapath(\'testcorpus.mm\'))\n            >>> index_temp = get_tmpfile("index")\n            >>> index = Similarity(index_temp, corpus, num_features=400)  # create index\n            >>>\n            >>> one_more_corpus = TextCorpus(datapath(\'testcorpus.txt\'))\n            >>> index.add_documents(one_more_corpus)  # add more documents in corpus\n\n        '
        min_ratio = 1.0
        if self.shards and len(self.shards[-1]) < min_ratio * self.shardsize:
            self.reopen_shard()
        for doc in corpus:
            if isinstance(doc, numpy.ndarray):
                doclen = len(doc)
            elif scipy.sparse.issparse(doc):
                doclen = doc.nnz
            else:
                doclen = len(doc)
                if doclen < 0.3 * self.num_features:
                    doc = matutils.unitvec(matutils.corpus2csc([doc], self.num_features).T, self.norm)
                else:
                    doc = matutils.unitvec(matutils.sparse2full(doc, self.num_features), self.norm)
            self.fresh_docs.append(doc)
            self.fresh_nnz += doclen
            if len(self.fresh_docs) >= self.shardsize:
                self.close_shard()
            if len(self.fresh_docs) % 10000 == 0:
                logger.info('PROGRESS: fresh_shard size=%i', len(self.fresh_docs))

    def shardid2filename(self, shardid):
        if False:
            for i in range(10):
                print('nop')
        'Get shard file by `shardid`.\n\n        Parameters\n        ----------\n        shardid : int\n            Shard index.\n\n        Return\n        ------\n        str\n            Path to shard file.\n\n        '
        if self.output_prefix.endswith('.'):
            return '%s%s' % (self.output_prefix, shardid)
        else:
            return '%s.%s' % (self.output_prefix, shardid)

    def close_shard(self):
        if False:
            i = 10
            return i + 15
        'Force the latest shard to close (be converted to a matrix and stored to disk).\n         Do nothing if no new documents added since last call.\n\n        Notes\n        -----\n        The shard is closed even if it is not full yet (its size is smaller than `self.shardsize`).\n        If documents are added later via :meth:`~gensim.similarities.docsim.MatrixSimilarity.add_documents`\n        this incomplete shard will be loaded again and completed.\n\n        '
        if not self.fresh_docs:
            return
        shardid = len(self.shards)
        issparse = 0.3 > 1.0 * self.fresh_nnz / (len(self.fresh_docs) * self.num_features)
        if issparse:
            index = SparseMatrixSimilarity(self.fresh_docs, num_terms=self.num_features, num_docs=len(self.fresh_docs), num_nnz=self.fresh_nnz)
        else:
            index = MatrixSimilarity(self.fresh_docs, num_features=self.num_features)
        logger.info('creating %s shard #%s', 'sparse' if issparse else 'dense', shardid)
        shard = Shard(self.shardid2filename(shardid), index)
        shard.num_best = self.num_best
        shard.num_nnz = self.fresh_nnz
        self.shards.append(shard)
        (self.fresh_docs, self.fresh_nnz) = ([], 0)

    def reopen_shard(self):
        if False:
            print('Hello World!')
        'Reopen an incomplete shard.'
        assert self.shards
        if self.fresh_docs:
            raise ValueError('cannot reopen a shard with fresh documents in index')
        last_shard = self.shards[-1]
        last_index = last_shard.get_index()
        logger.info('reopening an incomplete shard of %i documents', len(last_shard))
        self.fresh_docs = list(last_index.index)
        self.fresh_nnz = last_shard.num_nnz
        del self.shards[-1]
        logger.debug('reopen complete')

    def query_shards(self, query):
        if False:
            for i in range(10):
                print('nop')
        'Apply shard[query] to each shard in `self.shards`. Used internally.\n\n        Parameters\n        ----------\n        query : {iterable of list of (int, number) , list of (int, number))}\n            Document in BoW format or corpus of documents.\n\n        Returns\n        -------\n        (None, list of individual shard query results)\n            Query results.\n\n        '
        args = zip([query] * len(self.shards), self.shards)
        if PARALLEL_SHARDS and PARALLEL_SHARDS > 1:
            logger.debug('spawning %i query processes', PARALLEL_SHARDS)
            pool = multiprocessing.Pool(PARALLEL_SHARDS)
            result = pool.imap(query_shard, args, chunksize=1 + len(self.shards) / PARALLEL_SHARDS)
        else:
            pool = None
            result = map(query_shard, args)
        return (pool, result)

    def __getitem__(self, query):
        if False:
            return 10
        "Get similarities of the document (or corpus) `query` to all documents in the corpus.\n\n        Parameters\n        ----------\n        query : {iterable of list of (int, number) , list of (int, number))}\n            A single document in bag-of-words format, or a corpus (iterable) of such documents.\n\n        Return\n        ------\n        :class:`numpy.ndarray` or :class:`scipy.sparse.csr_matrix`\n            Similarities of the query against this index.\n\n        Notes\n        -----\n        If `query` is a corpus (iterable of documents), return a matrix of similarities of\n        all query documents vs. all corpus document. This batch query is more efficient than computing the similarities\n        one document after another.\n\n        Examples\n        --------\n        .. sourcecode:: pycon\n\n            >>> from gensim.corpora.textcorpus import TextCorpus\n            >>> from gensim.test.utils import datapath\n            >>> from gensim.similarities import Similarity\n            >>>\n            >>> corpus = TextCorpus(datapath('testcorpus.txt'))\n            >>> index = Similarity('temp', corpus, num_features=400)\n            >>> result = index[corpus]  # pairwise similarities of each document against each document\n\n        "
        self.close_shard()
        for shard in self.shards:
            shard.num_best = self.num_best
            shard.normalize = self.norm
        (pool, shard_results) = self.query_shards(query)
        if self.num_best is None:
            result = numpy.hstack(list(shard_results))
        else:
            offsets = numpy.cumsum([0] + [len(shard) for shard in self.shards])

            def convert(shard_no, doc):
                if False:
                    for i in range(10):
                        print('nop')
                return [(doc_index + offsets[shard_no], sim) for (doc_index, sim) in doc]
            (is_corpus, query) = utils.is_corpus(query)
            is_corpus = is_corpus or (hasattr(query, 'ndim') and query.ndim > 1 and (query.shape[0] > 1))
            if not is_corpus:
                results = (convert(shard_no, result) for (shard_no, result) in enumerate(shard_results))
                result = _nlargest(self.num_best, results)
            else:
                results = []
                for (shard_no, result) in enumerate(shard_results):
                    shard_result = [convert(shard_no, doc) for doc in result]
                    results.append(shard_result)
                result = []
                for parts in zip(*results):
                    merged = _nlargest(self.num_best, parts)
                    result.append(merged)
        if pool:
            pool.terminate()
        return result

    def vector_by_id(self, docpos):
        if False:
            i = 10
            return i + 15
        "Get the indexed vector corresponding to the document at position `docpos`.\n\n        Parameters\n        ----------\n        docpos : int\n            Document position\n\n        Return\n        ------\n        :class:`scipy.sparse.csr_matrix`\n            Indexed vector.\n\n        Examples\n        --------\n        .. sourcecode:: pycon\n\n            >>> from gensim.corpora.textcorpus import TextCorpus\n            >>> from gensim.test.utils import datapath\n            >>> from gensim.similarities import Similarity\n            >>>\n            >>> # Create index:\n            >>> corpus = TextCorpus(datapath('testcorpus.txt'))\n            >>> index = Similarity('temp', corpus, num_features=400)\n            >>> vector = index.vector_by_id(1)\n\n        "
        self.close_shard()
        pos = 0
        for shard in self.shards:
            pos += len(shard)
            if docpos < pos:
                break
        if not self.shards or docpos < 0 or docpos >= pos:
            raise ValueError('invalid document position: %s (must be 0 <= x < %s)' % (docpos, len(self)))
        result = shard.get_document_id(docpos - pos + len(shard))
        return result

    def similarity_by_id(self, docpos):
        if False:
            while True:
                i = 10
        "Get similarity of a document specified by its index position `docpos`.\n\n        Parameters\n        ----------\n        docpos : int\n            Document position in the index.\n\n        Return\n        ------\n        :class:`numpy.ndarray` or :class:`scipy.sparse.csr_matrix`\n            Similarities of the given document against this index.\n\n        Examples\n        --------\n        .. sourcecode:: pycon\n\n            >>> from gensim.corpora.textcorpus import TextCorpus\n            >>> from gensim.test.utils import datapath\n            >>> from gensim.similarities import Similarity\n            >>>\n            >>> corpus = TextCorpus(datapath('testcorpus.txt'))\n            >>> index = Similarity('temp', corpus, num_features=400)\n            >>> similarities = index.similarity_by_id(1)\n\n        "
        query = self.vector_by_id(docpos)
        (norm, self.norm) = (self.norm, False)
        result = self[query]
        self.norm = norm
        return result

    def __iter__(self):
        if False:
            print('Hello World!')
        'For each index document in index, compute cosine similarity against all other documents in the index.\n        Uses :meth:`~gensim.similarities.docsim.Similarity.iter_chunks` internally.\n\n        Yields\n        ------\n        :class:`numpy.ndarray` or :class:`scipy.sparse.csr_matrix`\n            Similarities of each document in turn against the index.\n\n        '
        (norm, self.norm) = (self.norm, False)
        for chunk in self.iter_chunks():
            if chunk.shape[0] > 1:
                for sim in self[chunk]:
                    yield sim
            else:
                yield self[chunk]
        self.norm = norm

    def iter_chunks(self, chunksize=None):
        if False:
            return 10
        'Iteratively yield the index as chunks of document vectors, each of size <= chunksize.\n\n        Parameters\n        ----------\n        chunksize : int, optional\n            Size of chunk,, if None - `self.chunksize` will be used.\n\n        Yields\n        ------\n        :class:`numpy.ndarray` or :class:`scipy.sparse.csr_matrix`\n            Chunks of the index as 2D arrays. The arrays are either dense or sparse, depending on\n            whether the shard was storing dense or sparse vectors.\n\n        '
        self.close_shard()
        if chunksize is None:
            chunksize = self.chunksize
        for shard in self.shards:
            query = shard.get_index().index
            for chunk_start in range(0, query.shape[0], chunksize):
                chunk_end = min(query.shape[0], chunk_start + chunksize)
                chunk = query[chunk_start:chunk_end]
                yield chunk

    def check_moved(self):
        if False:
            print('Hello World!')
        'Update shard locations, for case where the server prefix location changed on the filesystem.'
        dirname = os.path.dirname(self.output_prefix)
        for shard in self.shards:
            shard.dirname = dirname

    def save(self, fname=None, *args, **kwargs):
        if False:
            return 10
        'Save the index object via pickling under `fname`. See also :meth:`~gensim.docsim.Similarity.load()`.\n\n        Parameters\n        ----------\n        fname : str, optional\n            Path for save index, if not provided - will be saved to `self.output_prefix`.\n        *args : object\n            Arguments, see :meth:`gensim.utils.SaveLoad.save`.\n        **kwargs : object\n            Keyword arguments, see :meth:`gensim.utils.SaveLoad.save`.\n\n        Notes\n        -----\n        Will call :meth:`~gensim.similarities.Similarity.close_shard` internally to spill\n        any unfinished shards to disk first.\n\n        Examples\n        --------\n        .. sourcecode:: pycon\n\n            >>> from gensim.corpora.textcorpus import TextCorpus\n            >>> from gensim.test.utils import datapath, get_tmpfile\n            >>> from gensim.similarities import Similarity\n            >>>\n            >>> temp_fname = get_tmpfile("index")\n            >>> output_fname = get_tmpfile("saved_index")\n            >>>\n            >>> corpus = TextCorpus(datapath(\'testcorpus.txt\'))\n            >>> index = Similarity(output_fname, corpus, num_features=400)\n            >>>\n            >>> index.save(output_fname)\n            >>> loaded_index = index.load(output_fname)\n\n        '
        self.close_shard()
        if fname is None:
            fname = self.output_prefix
        super(Similarity, self).save(fname, *args, **kwargs)

    def destroy(self):
        if False:
            while True:
                i = 10
        'Delete all files under self.output_prefix\xa0Index is not usable anymore after calling this method.'
        import glob
        for fname in glob.glob(self.output_prefix + '*'):
            logger.info('deleting %s', fname)
            os.remove(fname)

class MatrixSimilarity(interfaces.SimilarityABC):
    """Compute cosine similarity against a corpus of documents by storing the index matrix in memory.

    Unless the entire matrix fits into main memory, use :class:`~gensim.similarities.docsim.Similarity` instead.

    Examples
    --------
    .. sourcecode:: pycon

        >>> from gensim.test.utils import common_corpus, common_dictionary
        >>> from gensim.similarities import MatrixSimilarity
        >>>
        >>> query = [(1, 2), (5, 4)]
        >>> index = MatrixSimilarity(common_corpus, num_features=len(common_dictionary))
        >>> sims = index[query]

    """

    def __init__(self, corpus, num_best=None, dtype=numpy.float32, num_features=None, chunksize=256, corpus_len=None):
        if False:
            i = 10
            return i + 15
        '\n\n        Parameters\n        ----------\n        corpus : iterable of list of (int, number)\n            Corpus in streamed Gensim bag-of-words format.\n        num_best : int, optional\n            If set, return only the `num_best` most similar documents, always leaving out documents with similarity = 0.\n            Otherwise, return a full vector with one float for every document in the index.\n        num_features : int\n            Size of the dictionary (number of features).\n        corpus_len : int, optional\n            Number of documents in `corpus`. If not specified, will scan the corpus to determine the matrix size.\n        chunksize : int, optional\n            Size of query chunks. Used internally when the query is an entire corpus.\n        dtype : numpy.dtype, optional\n            Datatype to store the internal matrix in.\n\n        '
        if num_features is None:
            logger.warning('scanning corpus to determine the number of features (consider setting `num_features` explicitly)')
            num_features = 1 + utils.get_max_id(corpus)
        self.num_features = num_features
        self.num_best = num_best
        self.normalize = True
        self.chunksize = chunksize
        if corpus_len is None:
            corpus_len = len(corpus)
        if corpus is not None:
            if self.num_features <= 0:
                raise ValueError('cannot index a corpus with zero features (you must specify either `num_features` or a non-empty corpus in the constructor)')
            logger.info('creating matrix with %i documents and %i features', corpus_len, num_features)
            self.index = numpy.empty(shape=(corpus_len, num_features), dtype=dtype)
            for (docno, vector) in enumerate(corpus):
                if docno % 1000 == 0:
                    logger.debug('PROGRESS: at document #%i/%i', docno, corpus_len)
                if isinstance(vector, numpy.ndarray):
                    pass
                elif scipy.sparse.issparse(vector):
                    vector = vector.toarray().flatten()
                else:
                    vector = matutils.unitvec(matutils.sparse2full(vector, num_features))
                self.index[docno] = vector

    def __len__(self):
        if False:
            return 10
        return self.index.shape[0]

    def get_similarities(self, query):
        if False:
            while True:
                i = 10
        'Get similarity between `query` and this index.\n\n        Warnings\n        --------\n        Do not use this function directly, use the :class:`~gensim.similarities.docsim.MatrixSimilarity.__getitem__`\n        instead.\n\n        Parameters\n        ----------\n        query : {list of (int, number), iterable of list of (int, number), :class:`scipy.sparse.csr_matrix`}\n            Document or collection of documents.\n\n        Return\n        ------\n        :class:`numpy.ndarray`\n            Similarity matrix.\n\n        '
        (is_corpus, query) = utils.is_corpus(query)
        if is_corpus:
            query = numpy.asarray([matutils.sparse2full(vec, self.num_features) for vec in query], dtype=self.index.dtype)
        else:
            if scipy.sparse.issparse(query):
                query = query.toarray()
            elif isinstance(query, numpy.ndarray):
                pass
            else:
                query = matutils.sparse2full(query, self.num_features)
            query = numpy.asarray(query, dtype=self.index.dtype)
        result = numpy.dot(self.index, query.T).T
        return result

    def __str__(self):
        if False:
            while True:
                i = 10
        return '%s<%i docs, %i features>' % (self.__class__.__name__, len(self), self.index.shape[1])

class SoftCosineSimilarity(interfaces.SimilarityABC):
    """Compute soft cosine similarity against a corpus of documents by storing the index matrix in memory.

    Examples
    --------
    .. sourcecode:: pycon

        >>> from gensim.test.utils import common_texts
        >>> from gensim.corpora import Dictionary
        >>> from gensim.models import Word2Vec
        >>> from gensim.similarities import SoftCosineSimilarity, SparseTermSimilarityMatrix
        >>> from gensim.similarities import WordEmbeddingSimilarityIndex
        >>>
        >>> model = Word2Vec(common_texts, vector_size=20, min_count=1)  # train word-vectors
        >>> termsim_index = WordEmbeddingSimilarityIndex(model.wv)
        >>> dictionary = Dictionary(common_texts)
        >>> bow_corpus = [dictionary.doc2bow(document) for document in common_texts]
        >>> similarity_matrix = SparseTermSimilarityMatrix(termsim_index, dictionary)  # construct similarity matrix
        >>> docsim_index = SoftCosineSimilarity(bow_corpus, similarity_matrix, num_best=10)
        >>>
        >>> query = 'graph trees computer'.split()  # make a query
        >>> sims = docsim_index[dictionary.doc2bow(query)]  # calculate similarity of query to each doc from bow_corpus

    Check out `the Gallery <https://radimrehurek.com/gensim/auto_examples/tutorials/run_scm.html>`__
    for more examples.

    """

    def __init__(self, corpus, similarity_matrix, num_best=None, chunksize=256, normalized=None, normalize_queries=True, normalize_documents=True):
        if False:
            for i in range(10):
                print('nop')
        "\n\n        Parameters\n        ----------\n        corpus: iterable of list of (int, float)\n            A list of documents in the BoW format.\n        similarity_matrix : :class:`gensim.similarities.SparseTermSimilarityMatrix`\n            A term similarity matrix.\n        num_best : int, optional\n            The number of results to retrieve for a query, if None - return similarities with all elements from corpus.\n        chunksize: int, optional\n            Size of one corpus chunk.\n        normalized : tuple of {True, False, 'maintain', None}, optional\n            A deprecated alias for `(normalize_queries, normalize_documents)`. If None, use\n            `normalize_queries` and `normalize_documents`. Default is None.\n        normalize_queries : {True, False, 'maintain'}, optional\n            Whether the query vector in the inner product will be L2-normalized (True; corresponds\n            to the soft cosine similarity measure; default), maintain their L2-norm during change of\n            basis ('maintain'; corresponds to queryexpansion with partial membership), or kept as-is\n            (False;  corresponds to query expansion).\n        normalize_documents : {True, False, 'maintain'}, optional\n            Whether the document vector in the inner product will be L2-normalized (True; corresponds\n            to the soft cosine similarity measure; default), maintain their L2-norm during change of\n            basis ('maintain'; corresponds to queryexpansion with partial membership), or kept as-is\n            (False;  corresponds to query expansion).\n\n        See Also\n        --------\n        :class:`~gensim.similarities.termsim.SparseTermSimilarityMatrix`\n            A sparse term similarity matrix built using a term similarity index.\n        :class:`~gensim.similarities.termsim.LevenshteinSimilarityIndex`\n            A term similarity index that computes Levenshtein similarities between terms.\n        :class:`~gensim.similarities.termsim.WordEmbeddingSimilarityIndex`\n            A term similarity index that computes cosine similarities between word embeddings.\n\n        "
        self.similarity_matrix = similarity_matrix
        self.corpus = list(corpus)
        self.num_best = num_best
        self.chunksize = chunksize
        if normalized is not None:
            warnings.warn('Parameter normalized will be removed in 5.0.0, use normalize_queries and normalize_documents instead', category=DeprecationWarning)
            self.normalized = normalized
        else:
            self.normalized = (normalize_queries, normalize_documents)
        self.normalize = False
        self.index = numpy.arange(len(corpus))

    def __len__(self):
        if False:
            return 10
        return len(self.corpus)

    def get_similarities(self, query):
        if False:
            i = 10
            return i + 15
        'Get similarity between `query` and this index.\n\n        Warnings\n        --------\n        Do not use this function directly; use the `self[query]` syntax instead.\n\n        Parameters\n        ----------\n        query : {list of (int, number), iterable of list of (int, number)}\n            Document or collection of documents.\n\n        Return\n        ------\n        :class:`numpy.ndarray`\n            Similarity matrix.\n\n        '
        if not self.corpus:
            return numpy.array()
        (is_corpus, query) = utils.is_corpus(query)
        if not is_corpus and isinstance(query, numpy.ndarray):
            query = [self.corpus[i] for i in query]
        result = self.similarity_matrix.inner_product(query, self.corpus, normalized=self.normalized)
        if scipy.sparse.issparse(result):
            return numpy.asarray(result.todense())
        if numpy.isscalar(result):
            return numpy.array(result)
        return numpy.asarray(result)[0]

    def __str__(self):
        if False:
            i = 10
            return i + 15
        return '%s<%i docs, %i features>' % (self.__class__.__name__, len(self), self.similarity_matrix.shape[0])

class WmdSimilarity(interfaces.SimilarityABC):
    """Compute negative WMD similarity against a corpus of documents.

    Check out `the Gallery <https://radimrehurek.com/gensim/auto_examples/tutorials/run_wmd.html>`__
    for more examples.

    When using this code, please consider citing the following papers:

    * `Rémi Flamary et al. "POT: Python Optimal Transport"
      <https://jmlr.org/papers/v22/20-451.html>`_
    * `Matt Kusner et al. "From Word Embeddings To Document Distances"
      <http://proceedings.mlr.press/v37/kusnerb15.pdf>`_

    Example
    -------
    .. sourcecode:: pycon

        >>> from gensim.test.utils import common_texts
        >>> from gensim.models import Word2Vec
        >>> from gensim.similarities import WmdSimilarity
        >>>
        >>> model = Word2Vec(common_texts, vector_size=20, min_count=1)  # train word-vectors
        >>>
        >>> index = WmdSimilarity(common_texts, model.wv)
        >>> # Make query.
        >>> query = ['trees']
        >>> sims = index[query]

    """

    def __init__(self, corpus, kv_model, num_best=None, chunksize=256):
        if False:
            i = 10
            return i + 15
        '\n\n        Parameters\n        ----------\n        corpus: iterable of list of str\n            A list of documents, each of which is a list of tokens.\n        kv_model: :class:`~gensim.models.keyedvectors.KeyedVectors`\n            A set of KeyedVectors\n        num_best: int, optional\n            Number of results to retrieve.\n        chunksize : int, optional\n            Size of chunk.\n\n        '
        self.corpus = corpus
        self.wv = kv_model
        self.num_best = num_best
        self.chunksize = chunksize
        self.normalize = False
        self.index = numpy.arange(len(corpus))

    def __len__(self):
        if False:
            print('Hello World!')
        'Get size of corpus.'
        return len(self.corpus)

    def get_similarities(self, query):
        if False:
            for i in range(10):
                print('nop')
        'Get similarity between `query` and this index.\n\n        Warnings\n        --------\n        Do not use this function directly; use the `self[query]` syntax instead.\n\n        Parameters\n        ----------\n        query : {list of str, iterable of list of str}\n            Document or collection of documents.\n\n        Return\n        ------\n        :class:`numpy.ndarray`\n            Similarity matrix.\n\n        '
        if isinstance(query, numpy.ndarray):
            query = [self.corpus[i] for i in query]
        if not query or not isinstance(query[0], list):
            query = [query]
        n_queries = len(query)
        result = []
        for qidx in range(n_queries):
            qresult = [self.wv.wmdistance(document, query[qidx]) for document in self.corpus]
            qresult = numpy.array(qresult)
            qresult = 1.0 / (1.0 + qresult)
            result.append(qresult)
        if len(result) == 1:
            result = result[0]
        else:
            result = numpy.array(result)
        return result

    def __str__(self):
        if False:
            print('Hello World!')
        return '%s<%i docs, %i features>' % (self.__class__.__name__, len(self), self.wv.vectors.shape[1])

class SparseMatrixSimilarity(interfaces.SimilarityABC):
    """Compute cosine similarity against a corpus of documents by storing the index matrix in memory.

    Examples
    --------
    Here is how you would index and query a corpus of documents in the bag-of-words format using the
    cosine similarity:

    .. sourcecode:: pycon

        >>> from gensim.corpora import Dictionary
        >>> from gensim.similarities import SparseMatrixSimilarity
        >>> from gensim.test.utils import common_texts as corpus
        >>>
        >>> dictionary = Dictionary(corpus)  # fit dictionary
        >>> bow_corpus = [dictionary.doc2bow(line) for line in corpus]  # convert corpus to BoW format
        >>> index = SparseMatrixSimilarity(bm25_corpus, num_docs=len(corpus), num_terms=len(dictionary))
        >>>
        >>> query = 'graph trees computer'.split()  # make a query
        >>> bow_query = dictionary.doc2bow(query)
        >>> similarities = index[bow_query]  # calculate similarity of query to each doc from bow_corpus

    Here is how you would index and query a corpus of documents using the Okapi BM25 scoring
    function:

    .. sourcecode:: pycon

        >>> from gensim.corpora import Dictionary
        >>> from gensim.models import TfidfModel, OkapiBM25Model
        >>> from gensim.similarities import SparseMatrixSimilarity
        >>> from gensim.test.utils import common_texts as corpus
        >>>
        >>> dictionary = Dictionary(corpus)  # fit dictionary
        >>> query_model = TfidfModel(dictionary=dictionary, smartirs='bnn')  # enforce binary weights
        >>> document_model = OkapiBM25Model(dictionary=dictionary)  # fit bm25 model
        >>>
        >>> bow_corpus = [dictionary.doc2bow(line) for line in corpus]  # convert corpus to BoW format
        >>> bm25_corpus = document_model[bow_corpus]
        >>> index = SparseMatrixSimilarity(bm25_corpus, num_docs=len(corpus), num_terms=len(dictionary),
        ...                                normalize_queries=False, normalize_documents=False)
        >>>
        >>>
        >>> query = 'graph trees computer'.split()  # make a query
        >>> bow_query = dictionary.doc2bow(query)
        >>> bm25_query = query_model[bow_query]
        >>> similarities = index[bm25_query]  # calculate similarity of query to each doc from bow_corpus

    Notes
    -----
    Use this if your input corpus contains sparse vectors (such as TF-IDF documents) and fits into RAM.

    The matrix is internally stored as a :class:`scipy.sparse.csr_matrix` matrix. Unless the entire
    matrix fits into main memory, use :class:`~gensim.similarities.docsim.Similarity` instead.

    Takes an optional `maintain_sparsity` argument, setting this to True
    causes `get_similarities` to return a sparse matrix instead of a
    dense representation if possible.

    See also
    --------
    :class:`~gensim.similarities.docsim.Similarity`
        Index similarity (wrapper for other inheritors of :class:`~gensim.interfaces.SimilarityABC`).
    :class:`~gensim.similarities.docsim.MatrixSimilarity`
        Index similarity (dense with cosine distance).

    """

    def __init__(self, corpus, num_features=None, num_terms=None, num_docs=None, num_nnz=None, num_best=None, chunksize=500, dtype=numpy.float32, maintain_sparsity=False, normalize_queries=True, normalize_documents=True):
        if False:
            i = 10
            return i + 15
        '\n\n        Parameters\n        ----------\n        corpus: iterable of list of (int, float)\n            A list of documents in the BoW format.\n        num_features : int, optional\n            Size of the dictionary. Must be either specified, or present in `corpus.num_terms`.\n        num_terms : int, optional\n            Alias for `num_features`, you can use either.\n        num_docs : int, optional\n            Number of documents in `corpus`. Will be calculated if not provided.\n        num_nnz : int, optional\n            Number of non-zero elements in `corpus`. Will be calculated if not provided.\n        num_best : int, optional\n            If set, return only the `num_best` most similar documents, always leaving out documents with similarity = 0.\n            Otherwise, return a full vector with one float for every document in the index.\n        chunksize : int, optional\n            Size of query chunks. Used internally when the query is an entire corpus.\n        dtype : numpy.dtype, optional\n            Data type of the internal matrix.\n        maintain_sparsity : bool, optional\n            Return sparse arrays from :meth:`~gensim.similarities.docsim.SparseMatrixSimilarity.get_similarities`?\n        normalize_queries : bool, optional\n            If queries are in bag-of-words (int, float) format, as opposed to a sparse or dense\n            2D arrays, they will be L2-normalized. Default is True.\n        normalize_documents : bool, optional\n            If `corpus` is in bag-of-words (int, float) format, as opposed to a sparse or dense\n            2D arrays, it will be L2-normalized. Default is True.\n        '
        self.num_best = num_best
        self.normalize = normalize_queries
        self.chunksize = chunksize
        self.maintain_sparsity = maintain_sparsity
        if corpus is not None:
            logger.info('creating sparse index')
            try:
                (num_terms, num_docs, num_nnz) = (corpus.num_terms, corpus.num_docs, corpus.num_nnz)
                logger.debug('using efficient sparse index creation')
            except AttributeError:
                pass
            if num_features is not None:
                num_terms = num_features
            if num_terms is None:
                raise ValueError('refusing to guess the number of sparse features: specify num_features explicitly')
            corpus = (matutils.scipy2sparse(v) if scipy.sparse.issparse(v) else matutils.full2sparse(v) if isinstance(v, numpy.ndarray) else matutils.unitvec(v) if normalize_documents else v for v in corpus)
            self.index = matutils.corpus2csc(corpus, num_terms=num_terms, num_docs=num_docs, num_nnz=num_nnz, dtype=dtype, printprogress=10000).T
            self.index = self.index.tocsr()
            logger.info('created %r', self.index)

    def __len__(self):
        if False:
            while True:
                i = 10
        'Get size of index.'
        return self.index.shape[0]

    def get_similarities(self, query):
        if False:
            return 10
        'Get similarity between `query` and this index.\n\n        Warnings\n        --------\n        Do not use this function directly; use the `self[query]` syntax instead.\n\n        Parameters\n        ----------\n        query : {list of (int, number), iterable of list of (int, number), :class:`scipy.sparse.csr_matrix`}\n            Document or collection of documents.\n\n        Return\n        ------\n        :class:`numpy.ndarray`\n            Similarity matrix (if maintain_sparsity=False) **OR**\n        :class:`scipy.sparse.csc`\n            otherwise\n\n        '
        (is_corpus, query) = utils.is_corpus(query)
        if is_corpus:
            query = matutils.corpus2csc(query, self.index.shape[1], dtype=self.index.dtype)
        elif scipy.sparse.issparse(query):
            query = query.T
        elif isinstance(query, numpy.ndarray):
            if query.ndim == 1:
                query.shape = (1, len(query))
            query = scipy.sparse.csr_matrix(query, dtype=self.index.dtype).T
        else:
            query = matutils.corpus2csc([query], self.index.shape[1], dtype=self.index.dtype)
        result = self.index * query.tocsc()
        if result.shape[1] == 1 and (not is_corpus):
            result = result.toarray().flatten()
        elif self.maintain_sparsity:
            result = result.T
        else:
            result = result.toarray().T
        return result