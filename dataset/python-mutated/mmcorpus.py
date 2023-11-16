"""Corpus in the `Matrix Market format <https://math.nist.gov/MatrixMarket/formats.html>`_."""
import logging
from gensim import matutils
from gensim.corpora import IndexedCorpus
logger = logging.getLogger(__name__)

class MmCorpus(matutils.MmReader, IndexedCorpus):
    """Corpus serialized using the `sparse coordinate Matrix Market format
    <https://math.nist.gov/MatrixMarket/formats.html>`_.

    Wrap a term-document matrix on disk (in matrix-market format), and present it
    as an object which supports iteration over the matrix rows (~documents).

    Notes
    -----
    The file is read into memory one document at a time, not the whole matrix at once,
    unlike e.g. `scipy.io.mmread` and other implementations. This allows you to **process corpora which are larger
    than the available RAM**, in a streamed manner.

    Example
    --------
    .. sourcecode:: pycon

        >>> from gensim.corpora.mmcorpus import MmCorpus
        >>> from gensim.test.utils import datapath
        >>>
        >>> corpus = MmCorpus(datapath('test_mmcorpus_with_index.mm'))
        >>> for document in corpus:
        ...     pass

    """

    def __init__(self, fname):
        if False:
            for i in range(10):
                print('nop')
        '\n\n        Parameters\n        ----------\n        fname : {str, file-like object}\n            Path to file in MM format or a file-like object that supports `seek()`\n            (e.g. a compressed file opened by `smart_open <https://github.com/RaRe-Technologies/smart_open>`_).\n\n        '
        IndexedCorpus.__init__(self, fname)
        matutils.MmReader.__init__(self, fname)

    def __iter__(self):
        if False:
            print('Hello World!')
        'Iterate through all documents.\n\n        Yields\n        ------\n        list of (int, numeric)\n            Document in the `sparse Gensim bag-of-words format <intro.rst#core-concepts>`__.\n\n        Notes\n        ------\n        The total number of vectors returned is always equal to the number of rows specified in the header.\n        Empty documents are inserted and yielded where appropriate, even if they are not explicitly stored in the\n        (sparse) Matrix Market file.\n\n        '
        for (doc_id, doc) in super(MmCorpus, self).__iter__():
            yield doc

    @staticmethod
    def save_corpus(fname, corpus, id2word=None, progress_cnt=1000, metadata=False):
        if False:
            return 10
        'Save a corpus to disk in the sparse coordinate Matrix Market format.\n\n        Parameters\n        ----------\n        fname : str\n            Path to file.\n        corpus : iterable of list of (int, number)\n            Corpus in Bow format.\n        id2word : dict of (int, str), optional\n            Mapping between word_id -> word. Used to retrieve the total vocabulary size if provided.\n            Otherwise, the total vocabulary size is estimated based on the highest feature id encountered in `corpus`.\n        progress_cnt : int, optional\n            How often to report (log) progress.\n        metadata : bool, optional\n            Writes out additional metadata?\n\n        Warnings\n        --------\n        This function is automatically called by :class:`~gensim.corpora.mmcorpus.MmCorpus.serialize`, don\'t\n        call it directly, call :class:`~gensim.corpora.mmcorpus.MmCorpus.serialize` instead.\n\n        Example\n        -------\n        .. sourcecode:: pycon\n\n            >>> from gensim.corpora.mmcorpus import MmCorpus\n            >>> from gensim.test.utils import datapath\n            >>>\n            >>> corpus = MmCorpus(datapath(\'test_mmcorpus_with_index.mm\'))\n            >>>\n            >>> MmCorpus.save_corpus("random", corpus)  # Do not do it, use `serialize` instead.\n            [97, 121, 169, 201, 225, 249, 258, 276, 303]\n\n        '
        logger.info('storing corpus in Matrix Market format to %s', fname)
        num_terms = len(id2word) if id2word is not None else None
        return matutils.MmWriter.write_corpus(fname, corpus, num_terms=num_terms, index=True, progress_cnt=progress_cnt, metadata=metadata)