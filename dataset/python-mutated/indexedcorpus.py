"""Base Indexed Corpus class."""
import logging
import numpy
from gensim import interfaces, utils
logger = logging.getLogger(__name__)

class IndexedCorpus(interfaces.CorpusABC):
    """Indexed corpus is a mechanism for random-accessing corpora.

    While the standard corpus interface in gensim allows iterating over corpus,
    we'll show it with :class:`~gensim.corpora.mmcorpus.MmCorpus`.

    .. sourcecode:: pycon

        >>> from gensim.corpora import MmCorpus
        >>> from gensim.test.utils import datapath
        >>>
        >>> corpus = MmCorpus(datapath('testcorpus.mm'))
        >>> for doc in corpus:
        ...     pass

    :class:`~gensim.corpora.indexedcorpus.IndexedCorpus` allows accessing the documents with index
    in :math:`{O}(1)` look-up time.

    .. sourcecode:: pycon

        >>> document_index = 3
        >>> doc = corpus[document_index]

    Notes
    -----
    This functionality is achieved by storing an extra file (by default named the same as the `fname.index`)
    that stores the byte offset of the beginning of each document.

    """

    def __init__(self, fname, index_fname=None):
        if False:
            i = 10
            return i + 15
        '\n\n        Parameters\n        ----------\n        fname : str\n            Path to corpus.\n        index_fname : str, optional\n            Path to index, if not provided - used `fname.index`.\n\n        '
        try:
            if index_fname is None:
                index_fname = utils.smart_extension(fname, '.index')
            self.index = utils.unpickle(index_fname)
            self.index = numpy.asarray(self.index)
            logger.info('loaded corpus index from %s', index_fname)
        except Exception:
            self.index = None
        self.length = None

    @classmethod
    def serialize(serializer, fname, corpus, id2word=None, index_fname=None, progress_cnt=None, labels=None, metadata=False):
        if False:
            print('Hello World!')
        'Serialize corpus with offset metadata, allows to use direct indexes after loading.\n\n        Parameters\n        ----------\n        fname : str\n            Path to output file.\n        corpus : iterable of iterable of (int, float)\n            Corpus in BoW format.\n        id2word : dict of (str, str), optional\n            Mapping id -> word.\n        index_fname : str, optional\n             Where to save resulting index, if None - store index to `fname`.index.\n        progress_cnt : int, optional\n            Number of documents after which progress info is printed.\n        labels : bool, optional\n             If True - ignore first column (class labels).\n        metadata : bool, optional\n            If True - ensure that serialize will write out article titles to a pickle file.\n\n        Examples\n        --------\n        .. sourcecode:: pycon\n\n            >>> from gensim.corpora import MmCorpus\n            >>> from gensim.test.utils import get_tmpfile\n            >>>\n            >>> corpus = [[(1, 0.3), (2, 0.1)], [(1, 0.1)], [(2, 0.3)]]\n            >>> output_fname = get_tmpfile("test.mm")\n            >>>\n            >>> MmCorpus.serialize(output_fname, corpus)\n            >>> mm = MmCorpus(output_fname)  # `mm` document stream now has random access\n            >>> print(mm[1])  # retrieve document no. 42, etc.\n            [(1, 0.1)]\n\n        '
        if getattr(corpus, 'fname', None) == fname:
            raise ValueError('identical input vs. output corpus filename, refusing to serialize: %s' % fname)
        if index_fname is None:
            index_fname = utils.smart_extension(fname, '.index')
        kwargs = {'metadata': metadata}
        if progress_cnt is not None:
            kwargs['progress_cnt'] = progress_cnt
        if labels is not None:
            kwargs['labels'] = labels
        offsets = serializer.save_corpus(fname, corpus, id2word, **kwargs)
        if offsets is None:
            raise NotImplementedError("Called serialize on class %s which doesn't support indexing!" % serializer.__name__)
        logger.info('saving %s index to %s', serializer.__name__, index_fname)
        utils.pickle(offsets, index_fname)

    def __len__(self):
        if False:
            return 10
        'Get the index length.\n\n        Notes\n        -----\n        If the corpus is not indexed, also count corpus length and cache this value.\n\n        Returns\n        -------\n        int\n            Length of index.\n\n        '
        if self.index is not None:
            return len(self.index)
        if self.length is None:
            logger.info('caching corpus length')
            self.length = sum((1 for _ in self))
        return self.length

    def __getitem__(self, docno):
        if False:
            for i in range(10):
                print('nop')
        "Get document by `docno` index.\n\n        Parameters\n        ----------\n        docno : {int, iterable of int}\n            Document number or iterable of numbers (like a list of str).\n\n        Returns\n        -------\n        list of (int, float)\n            If `docno` is int - return document in BoW format.\n\n        :class:`~gensim.utils.SlicedCorpus`\n            If `docno` is iterable of int - return several documents in BoW format\n            wrapped to :class:`~gensim.utils.SlicedCorpus`.\n\n        Raises\n        ------\n        RuntimeError\n            If index isn't exist.\n\n        "
        if self.index is None:
            raise RuntimeError('Cannot call corpus[docid] without an index')
        if isinstance(docno, (slice, list, numpy.ndarray)):
            return utils.SlicedCorpus(self, docno)
        elif isinstance(docno, (int, numpy.integer)):
            return self.docbyoffset(self.index[docno])
        else:
            raise ValueError('Unrecognised value for docno, use either a single integer, a slice or a numpy.ndarray')