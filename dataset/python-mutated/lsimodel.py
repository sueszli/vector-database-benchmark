"""Module for `Latent Semantic Analysis (aka Latent Semantic Indexing)
<https://en.wikipedia.org/wiki/Latent_semantic_analysis#Latent_semantic_indexing>`_.

Implements fast truncated SVD (Singular Value Decomposition). The SVD decomposition can be updated with new observations
at any time, for an online, incremental, memory-efficient training.

This module actually contains several algorithms for decomposition of large corpora, a
combination of which effectively and transparently allows building LSI models for:

* corpora much larger than RAM: only constant memory is needed, independent of
  the corpus size
* corpora that are streamed: documents are only accessed sequentially, no
  random access
* corpora that cannot be even temporarily stored: each document can only be
  seen once and must be processed immediately (one-pass algorithm)
* distributed computing for very large corpora, making use of a cluster of
  machines

Wall-clock `performance on the English Wikipedia <https://radimrehurek.com/gensim/wiki.html>`_
(2G corpus positions, 3.2M documents, 100K features, 0.5G non-zero entries in the final TF-IDF matrix),
requesting the top 400 LSI factors:

====================================================== ============ ==================
 algorithm                                             serial       distributed
====================================================== ============ ==================
 one-pass merge algorithm                              5h14m        1h41m
 multi-pass stochastic algo (with 2 power iterations)  5h39m        N/A [1]_
====================================================== ============ ==================


*serial* = Core 2 Duo MacBook Pro 2.53Ghz, 4GB RAM, libVec

*distributed* = cluster of four logical nodes on three physical machines, each
with dual core Xeon 2.0GHz, 4GB RAM, ATLAS


Examples
--------
.. sourcecode:: pycon

    >>> from gensim.test.utils import common_dictionary, common_corpus
    >>> from gensim.models import LsiModel
    >>>
    >>> model = LsiModel(common_corpus, id2word=common_dictionary)
    >>> vectorized_corpus = model[common_corpus]  # vectorize input copus in BoW format


.. [1] The stochastic algo could be distributed too, but most time is already spent
   reading/decompressing the input from disk in its 4 passes. The extra network
   traffic due to data distribution across cluster nodes would likely make it
   *slower*.

"""
import logging
import sys
import time
import numpy as np
import scipy.linalg
import scipy.sparse
from scipy.sparse import sparsetools
from gensim import interfaces, matutils, utils
from gensim.models import basemodel
from gensim.utils import is_empty
logger = logging.getLogger(__name__)
P2_EXTRA_DIMS = 100
P2_EXTRA_ITERS = 2

def clip_spectrum(s, k, discard=0.001):
    if False:
        i = 10
        return i + 15
    "Find how many factors should be kept to avoid storing spurious (tiny, numerically unstable) values.\n\n    Parameters\n    ----------\n    s : list of float\n        Eigenvalues of the original matrix.\n    k : int\n        Maximum desired rank (number of factors)\n    discard: float\n        Percentage of the spectrum's energy to be discarded.\n\n    Returns\n    -------\n    int\n        Rank (number of factors) of the reduced matrix.\n\n\n    "
    rel_spectrum = np.abs(1.0 - np.cumsum(s / np.sum(s)))
    small = 1 + len(np.where(rel_spectrum > min(discard, 1.0 / k))[0])
    k = min(k, small)
    logger.info('keeping %i factors (discarding %.3f%% of energy spectrum)', k, 100 * rel_spectrum[k - 1])
    return k

def asfarray(a, name=''):
    if False:
        while True:
            i = 10
    'Get an array laid out in Fortran order in memory.\n\n    Parameters\n    ----------\n    a : numpy.ndarray\n        Input array.\n    name : str, optional\n        Array name, used only for logging purposes.\n\n    Returns\n    -------\n    np.ndarray\n        The input `a` in Fortran, or column-major order.\n\n    '
    if not a.flags.f_contiguous:
        logger.debug('converting %s array %s to FORTRAN order', a.shape, name)
        a = np.asfortranarray(a)
    return a

def ascarray(a, name=''):
    if False:
        return 10
    'Return a contiguous array in memory (C order).\n\n    Parameters\n    ----------\n    a : numpy.ndarray\n        Input array.\n    name : str, optional\n        Array name, used for logging purposes.\n\n    Returns\n    -------\n    np.ndarray\n        Contiguous array (row-major order) of same shape and content as `a`.\n\n    '
    if not a.flags.contiguous:
        logger.debug('converting %s array %s to C order', a.shape, name)
        a = np.ascontiguousarray(a)
    return a

class Projection(utils.SaveLoad):
    """Low dimensional projection of a term-document matrix.

    This is the class taking care of the 'core math': interfacing with corpora, splitting large corpora into chunks
    and merging them etc. This done through the higher-level :class:`~gensim.models.lsimodel.LsiModel` class.

    Notes
    -----
    The projection can be later updated by merging it with another :class:`~gensim.models.lsimodel.Projection`
    via  :meth:`~gensim.models.lsimodel.Projection.merge`. This is how incremental training actually happens.

    """

    def __init__(self, m, k, docs=None, use_svdlibc=False, power_iters=P2_EXTRA_ITERS, extra_dims=P2_EXTRA_DIMS, dtype=np.float64, random_seed=None):
        if False:
            return 10
        'Construct the (U, S) projection from a corpus.\n\n        Parameters\n        ----------\n        m : int\n            Number of features (terms) in the corpus.\n        k : int\n            Desired rank of the decomposed matrix.\n        docs : {iterable of list of (int, float), scipy.sparse.csc}\n            Corpus in BoW format or as sparse matrix.\n        use_svdlibc : bool, optional\n            If True - will use `sparsesvd library <https://pypi.org/project/sparsesvd/>`_,\n            otherwise - our own version will be used.\n        power_iters: int, optional\n            Number of power iteration steps to be used. Tune to improve accuracy.\n        extra_dims : int, optional\n            Extra samples to be used besides the rank `k`. Tune to improve accuracy.\n        dtype : numpy.dtype, optional\n            Enforces a type for elements of the decomposed matrix.\n        random_seed: {None, int}, optional\n            Random seed used to initialize the pseudo-random number generator,\n            a local instance of numpy.random.RandomState instance.\n\n        '
        (self.m, self.k) = (m, k)
        self.power_iters = power_iters
        self.extra_dims = extra_dims
        self.random_seed = random_seed
        if docs is not None:
            if not use_svdlibc:
                (u, s) = stochastic_svd(docs, k, chunksize=sys.maxsize, num_terms=m, power_iters=self.power_iters, extra_dims=self.extra_dims, dtype=dtype, random_seed=self.random_seed)
            else:
                try:
                    import sparsesvd
                except ImportError:
                    raise ImportError('`sparsesvd` module requested but not found; run `easy_install sparsesvd`')
                logger.info('computing sparse SVD of %s matrix', str(docs.shape))
                if not scipy.sparse.issparse(docs):
                    docs = matutils.corpus2csc(docs)
                (ut, s, vt) = sparsesvd.sparsesvd(docs, k + 30)
                u = ut.T
                del ut, vt
                k = clip_spectrum(s ** 2, self.k)
            self.u = u[:, :k].copy()
            self.s = s[:k].copy()
        else:
            (self.u, self.s) = (None, None)

    def empty_like(self):
        if False:
            print('Hello World!')
        'Get an empty Projection with the same parameters as the current object.\n\n        Returns\n        -------\n        :class:`~gensim.models.lsimodel.Projection`\n            An empty copy (without corpus) of the current projection.\n\n        '
        return Projection(self.m, self.k, power_iters=self.power_iters, extra_dims=self.extra_dims, random_seed=self.random_seed)

    def merge(self, other, decay=1.0):
        if False:
            return 10
        'Merge current :class:`~gensim.models.lsimodel.Projection` instance with another.\n\n        Warnings\n        --------\n        The content of `other` is destroyed in the process, so pass this function a copy of `other`\n        if you need it further. The `other` :class:`~gensim.models.lsimodel.Projection` is expected to contain\n        the same number of features.\n\n        Parameters\n        ----------\n        other : :class:`~gensim.models.lsimodel.Projection`\n            The Projection object to be merged into the current one. It will be destroyed after merging.\n        decay : float, optional\n            Weight of existing observations relatively to new ones.\n            Setting `decay` < 1.0 causes re-orientation towards new data trends in the input document stream,\n            by giving less emphasis to old observations. This allows LSA to gradually "forget" old observations\n            (documents) and give more preference to new ones.\n\n        '
        if other.u is None:
            return
        if self.u is None:
            self.u = other.u.copy()
            self.s = other.s.copy()
            return
        if self.m != other.m:
            raise ValueError('vector space mismatch: update is using %s features, expected %s' % (other.m, self.m))
        logger.info('merging projections: %s + %s', str(self.u.shape), str(other.u.shape))
        (m, n1, n2) = (self.u.shape[0], self.u.shape[1], other.u.shape[1])
        logger.debug('constructing orthogonal component')
        self.u = asfarray(self.u, 'self.u')
        c = np.dot(self.u.T, other.u)
        self.u = ascarray(self.u, 'self.u')
        other.u -= np.dot(self.u, c)
        other.u = [other.u]
        (q, r) = matutils.qr_destroy(other.u)
        assert not other.u
        k = np.bmat([[np.diag(decay * self.s), np.multiply(c, other.s)], [matutils.pad(np.array([]).reshape(0, 0), min(m, n2), n1), np.multiply(r, other.s)]])
        logger.debug('computing SVD of %s dense matrix', k.shape)
        try:
            (u_k, s_k, _) = scipy.linalg.svd(k, full_matrices=False)
        except scipy.linalg.LinAlgError:
            logger.error('SVD(A) failed; trying SVD(A * A^T)')
            (u_k, s_k, _) = scipy.linalg.svd(np.dot(k, k.T), full_matrices=False)
            s_k = np.sqrt(s_k)
        k = clip_spectrum(s_k ** 2, self.k)
        (u1_k, u2_k, s_k) = (np.array(u_k[:n1, :k]), np.array(u_k[n1:, :k]), s_k[:k])
        logger.debug('updating orthonormal basis U')
        self.s = s_k
        self.u = ascarray(self.u, 'self.u')
        self.u = np.dot(self.u, u1_k)
        q = ascarray(q, 'q')
        q = np.dot(q, u2_k)
        self.u += q
        if self.u.shape[0] > 0:
            for i in range(self.u.shape[1]):
                if self.u[0, i] < 0.0:
                    self.u[:, i] *= -1.0

class LsiModel(interfaces.TransformationABC, basemodel.BaseTopicModel):
    """Model for `Latent Semantic Indexing
    <https://en.wikipedia.org/wiki/Latent_semantic_analysis#Latent_semantic_indexing>`_.

    The decomposition algorithm is described in `"Fast and Faster: A Comparison of Two Streamed
    Matrix Decomposition Algorithms" <https://arxiv.org/pdf/1102.5597.pdf>`_.

    Notes
    -----
    * :attr:`gensim.models.lsimodel.LsiModel.projection.u` - left singular vectors,
    * :attr:`gensim.models.lsimodel.LsiModel.projection.s` - singular values,
    * ``model[training_corpus]`` - right singular vectors (can be reconstructed if needed).

    See Also
    --------
    `FAQ about LSI matrices
    <https://github.com/RaRe-Technologies/gensim/wiki/Recipes-&-FAQ#q4-how-do-you-output-the-u-s-vt-matrices-of-lsi>`_.

    Examples
    --------
    .. sourcecode:: pycon

        >>> from gensim.test.utils import common_corpus, common_dictionary, get_tmpfile
        >>> from gensim.models import LsiModel
        >>>
        >>> model = LsiModel(common_corpus[:3], id2word=common_dictionary)  # train model
        >>> vector = model[common_corpus[4]]  # apply model to BoW document
        >>> model.add_documents(common_corpus[4:])  # update model with new documents
        >>> tmp_fname = get_tmpfile("lsi.model")
        >>> model.save(tmp_fname)  # save model
        >>> loaded_model = LsiModel.load(tmp_fname)  # load model

    """

    def __init__(self, corpus=None, num_topics=200, id2word=None, chunksize=20000, decay=1.0, distributed=False, onepass=True, power_iters=P2_EXTRA_ITERS, extra_samples=P2_EXTRA_DIMS, dtype=np.float64, random_seed=None):
        if False:
            for i in range(10):
                print('nop')
        'Build an LSI model.\n\n        Parameters\n        ----------\n        corpus : {iterable of list of (int, float), scipy.sparse.csc}, optional\n            Stream of document vectors or a sparse matrix of shape (`num_documents`, `num_terms`).\n        num_topics : int, optional\n            Number of requested factors (latent dimensions)\n        id2word : dict of {int: str}, optional\n            ID to word mapping, optional.\n        chunksize :  int, optional\n            Number of documents to be used in each training chunk.\n        decay : float, optional\n            Weight of existing observations relatively to new ones.\n        distributed : bool, optional\n            If True - distributed mode (parallel execution on several machines) will be used.\n        onepass : bool, optional\n            Whether the one-pass algorithm should be used for training.\n            Pass `False` to force a multi-pass stochastic algorithm.\n        power_iters: int, optional\n            Number of power iteration steps to be used.\n            Increasing the number of power iterations improves accuracy, but lowers performance\n        extra_samples : int, optional\n            Extra samples to be used besides the rank `k`. Can improve accuracy.\n        dtype : type, optional\n            Enforces a type for elements of the decomposed matrix.\n        random_seed: {None, int}, optional\n            Random seed used to initialize the pseudo-random number generator,\n            a local instance of numpy.random.RandomState instance.\n\n        '
        self.id2word = id2word
        self.num_topics = int(num_topics)
        self.chunksize = int(chunksize)
        self.decay = float(decay)
        if distributed:
            if not onepass:
                logger.warning('forcing the one-pass algorithm for distributed LSA')
                onepass = True
        self.onepass = onepass
        (self.extra_samples, self.power_iters) = (extra_samples, power_iters)
        self.dtype = dtype
        self.random_seed = random_seed
        if corpus is None and self.id2word is None:
            raise ValueError('at least one of corpus/id2word must be specified, to establish input space dimensionality')
        if self.id2word is None:
            logger.warning('no word id mapping provided; initializing from corpus, assuming identity')
            self.id2word = utils.dict_from_corpus(corpus)
            self.num_terms = len(self.id2word)
        else:
            self.num_terms = 1 + (max(self.id2word.keys()) if self.id2word else -1)
        self.docs_processed = 0
        self.projection = Projection(self.num_terms, self.num_topics, power_iters=self.power_iters, extra_dims=self.extra_samples, dtype=dtype, random_seed=self.random_seed)
        self.numworkers = 1
        if not distributed:
            logger.info('using serial LSI version on this node')
            self.dispatcher = None
        else:
            if not onepass:
                raise NotImplementedError('distributed stochastic LSA not implemented yet; run either distributed one-pass, or serial randomized.')
            try:
                import Pyro4
                dispatcher = Pyro4.Proxy('PYRONAME:gensim.lsi_dispatcher')
                logger.debug('looking for dispatcher at %s', str(dispatcher._pyroUri))
                dispatcher.initialize(id2word=self.id2word, num_topics=num_topics, chunksize=chunksize, decay=decay, power_iters=self.power_iters, extra_samples=self.extra_samples, distributed=False, onepass=onepass)
                self.dispatcher = dispatcher
                self.numworkers = len(dispatcher.getworkers())
                logger.info('using distributed version with %i workers', self.numworkers)
            except Exception as err:
                logger.error('failed to initialize distributed LSI (%s)', err)
                raise RuntimeError('failed to initialize distributed LSI (%s)' % err)
        if corpus is not None:
            start = time.time()
            self.add_documents(corpus)
            self.add_lifecycle_event('created', msg=f'trained {self} in {time.time() - start:.2f}s')

    def add_documents(self, corpus, chunksize=None, decay=None):
        if False:
            print('Hello World!')
        'Update model with new `corpus`.\n\n        Parameters\n        ----------\n        corpus : {iterable of list of (int, float), scipy.sparse.csc}\n            Stream of document vectors or sparse matrix of shape (`num_terms`, num_documents).\n        chunksize : int, optional\n            Number of documents to be used in each training chunk, will use `self.chunksize` if not specified.\n        decay : float, optional\n            Weight of existing observations relatively to new ones,  will use `self.decay` if not specified.\n\n        Notes\n        -----\n        Training proceeds in chunks of `chunksize` documents at a time. The size of `chunksize` is a tradeoff\n        between increased speed (bigger `chunksize`) vs. lower memory footprint (smaller `chunksize`).\n        If the distributed mode is on, each chunk is sent to a different worker/computer.\n\n        '
        logger.info('updating model with new documents')
        if chunksize is None:
            chunksize = self.chunksize
        if decay is None:
            decay = self.decay
        if is_empty(corpus):
            logger.warning('LsiModel.add_documents() called but no documents provided, is this intended?')
        if not scipy.sparse.issparse(corpus):
            if not self.onepass:
                update = Projection(self.num_terms, self.num_topics, None, dtype=self.dtype, random_seed=self.random_seed)
                (update.u, update.s) = stochastic_svd(corpus, self.num_topics, num_terms=self.num_terms, chunksize=chunksize, extra_dims=self.extra_samples, power_iters=self.power_iters, dtype=self.dtype, random_seed=self.random_seed)
                self.projection.merge(update, decay=decay)
                self.docs_processed += len(corpus) if hasattr(corpus, '__len__') else 0
            else:
                doc_no = 0
                if self.dispatcher:
                    logger.info('initializing %s workers', self.numworkers)
                    self.dispatcher.reset()
                for (chunk_no, chunk) in enumerate(utils.grouper(corpus, chunksize)):
                    logger.info('preparing a new chunk of documents')
                    nnz = sum((len(doc) for doc in chunk))
                    logger.debug('converting corpus to csc format')
                    job = matutils.corpus2csc(chunk, num_docs=len(chunk), num_terms=self.num_terms, num_nnz=nnz, dtype=self.dtype)
                    del chunk
                    doc_no += job.shape[1]
                    if self.dispatcher:
                        logger.debug('creating job #%i', chunk_no)
                        self.dispatcher.putjob(job)
                        del job
                        logger.info('dispatched documents up to #%s', doc_no)
                    else:
                        update = Projection(self.num_terms, self.num_topics, job, extra_dims=self.extra_samples, power_iters=self.power_iters, dtype=self.dtype, random_seed=self.random_seed)
                        del job
                        self.projection.merge(update, decay=decay)
                        del update
                        logger.info('processed documents up to #%s', doc_no)
                        self.print_topics(5)
                if self.dispatcher:
                    logger.info('reached the end of input; now waiting for all remaining jobs to finish')
                    self.projection = self.dispatcher.getstate()
                self.docs_processed += doc_no
        else:
            assert not self.dispatcher, 'must be in serial mode to receive jobs'
            update = Projection(self.num_terms, self.num_topics, corpus.tocsc(), extra_dims=self.extra_samples, power_iters=self.power_iters, dtype=self.dtype)
            self.projection.merge(update, decay=decay)
            logger.info('processed sparse job of %i documents', corpus.shape[1])
            self.docs_processed += corpus.shape[1]

    def __str__(self):
        if False:
            print('Hello World!')
        'Get a human readable representation of model.\n\n        Returns\n        -------\n        str\n            A human readable string of the current objects parameters.\n\n        '
        return '%s<num_terms=%s, num_topics=%s, decay=%s, chunksize=%s>' % (self.__class__.__name__, self.num_terms, self.num_topics, self.decay, self.chunksize)

    def __getitem__(self, bow, scaled=False, chunksize=512):
        if False:
            return 10
        'Get the latent representation for `bow`.\n\n        Parameters\n        ----------\n        bow : {list of (int, int), iterable of list of (int, int)}\n            Document or corpus in BoW representation.\n        scaled : bool, optional\n            If True - topics will be scaled by the inverse of singular values.\n        chunksize :  int, optional\n            Number of documents to be used in each applying chunk.\n\n        Returns\n        -------\n        list of (int, float)\n            Latent representation of topics in BoW format for document **OR**\n        :class:`gensim.matutils.Dense2Corpus`\n            Latent representation of corpus in BoW format if `bow` is corpus.\n\n        '
        if self.projection.u is None:
            raise ValueError('No training data provided - LSI model not initialized yet')
        (is_corpus, bow) = utils.is_corpus(bow)
        if is_corpus and chunksize:
            return self._apply(bow, chunksize=chunksize)
        if not is_corpus:
            bow = [bow]
        vec = matutils.corpus2csc(bow, num_terms=self.num_terms, dtype=self.projection.u.dtype)
        topic_dist = (vec.T * self.projection.u[:, :self.num_topics]).T
        if not is_corpus:
            topic_dist = topic_dist.reshape(-1)
        if scaled:
            topic_dist = 1.0 / self.projection.s[:self.num_topics] * topic_dist
        if not is_corpus:
            result = matutils.full2sparse(topic_dist)
        else:
            result = matutils.Dense2Corpus(topic_dist)
        return result

    def get_topics(self):
        if False:
            return 10
        'Get the topic vectors.\n\n        Notes\n        -----\n        The number of topics can actually be smaller than `self.num_topics`, if there were not enough factors\n        in the matrix (real rank of input matrix smaller than `self.num_topics`).\n\n        Returns\n        -------\n        np.ndarray\n            The term topic matrix with shape (`num_topics`, `vocabulary_size`)\n\n        '
        projections = self.projection.u.T
        num_topics = len(projections)
        topics = []
        for i in range(num_topics):
            c = np.asarray(projections[i, :]).flatten()
            norm = np.sqrt(np.sum(np.dot(c, c)))
            topics.append(1.0 * c / norm)
        return np.array(topics)

    def show_topic(self, topicno, topn=10):
        if False:
            i = 10
            return i + 15
        'Get the words that define a topic along with their contribution.\n\n        This is actually the left singular vector of the specified topic.\n\n        The most important words in defining the topic (greatest absolute value) are included\n        in the output, along with their contribution to the topic.\n\n        Parameters\n        ----------\n        topicno : int\n            The topics id number.\n        topn : int\n            Number of words to be included to the result.\n\n        Returns\n        -------\n        list of (str, float)\n            Topic representation in BoW format.\n\n        '
        if topicno >= len(self.projection.u.T):
            return ''
        c = np.asarray(self.projection.u.T[topicno, :]).flatten()
        norm = np.sqrt(np.sum(np.dot(c, c)))
        most = matutils.argsort(np.abs(c), topn, reverse=True)
        return [(self.id2word[val], 1.0 * c[val] / norm) for val in most if val in self.id2word]

    def show_topics(self, num_topics=-1, num_words=10, log=False, formatted=True):
        if False:
            i = 10
            return i + 15
        'Get the most significant topics.\n\n        Parameters\n        ----------\n        num_topics : int, optional\n            The number of topics to be selected, if -1 - all topics will be in result (ordered by significance).\n        num_words : int, optional\n            The number of words to be included per topics (ordered by significance).\n        log : bool, optional\n            If True - log topics with logger.\n        formatted : bool, optional\n            If True - each topic represented as string, otherwise - in BoW format.\n\n        Returns\n        -------\n        list of (int, str)\n            If `formatted=True`, return sequence with (topic_id, string representation of topics) **OR**\n        list of (int, list of (str, float))\n            Otherwise, return sequence with (topic_id, [(word, value), ... ]).\n\n        '
        shown = []
        if num_topics < 0:
            num_topics = self.num_topics
        for i in range(min(num_topics, self.num_topics)):
            if i < len(self.projection.s):
                if formatted:
                    topic = self.print_topic(i, topn=num_words)
                else:
                    topic = self.show_topic(i, topn=num_words)
                shown.append((i, topic))
                if log:
                    logger.info('topic #%i(%.3f): %s', i, self.projection.s[i], topic)
        return shown

    def print_debug(self, num_topics=5, num_words=10):
        if False:
            return 10
        'Print (to log) the most salient words of the first `num_topics` topics.\n\n        Unlike :meth:`~gensim.models.lsimodel.LsiModel.print_topics`, this looks for words that are significant for\n        a particular topic *and* not for others. This *should* result in a\n        more human-interpretable description of topics.\n\n        Alias for :func:`~gensim.models.lsimodel.print_debug`.\n\n        Parameters\n        ----------\n        num_topics : int, optional\n            The number of topics to be selected (ordered by significance).\n        num_words : int, optional\n            The number of words to be included per topics (ordered by significance).\n\n        '
        print_debug(self.id2word, self.projection.u, self.projection.s, range(min(num_topics, len(self.projection.u.T))), num_words=num_words)

    def save(self, fname, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Save the model to a file.\n\n        Notes\n        -----\n        Large internal arrays may be stored into separate files, with `fname` as prefix.\n\n        Warnings\n        --------\n        Do not save as a compressed file if you intend to load the file back with `mmap`.\n\n        Parameters\n        ----------\n        fname : str\n            Path to output file.\n        *args\n            Variable length argument list, see :meth:`gensim.utils.SaveLoad.save`.\n        **kwargs\n            Arbitrary keyword arguments, see :meth:`gensim.utils.SaveLoad.save`.\n\n        See Also\n        --------\n        :meth:`~gensim.models.lsimodel.LsiModel.load`\n\n        '
        if self.projection is not None:
            self.projection.save(utils.smart_extension(fname, '.projection'), *args, **kwargs)
        super(LsiModel, self).save(fname, *args, ignore=['projection', 'dispatcher'], **kwargs)

    @classmethod
    def load(cls, fname, *args, **kwargs):
        if False:
            return 10
        "Load a previously saved object using :meth:`~gensim.models.lsimodel.LsiModel.save` from file.\n\n        Notes\n        -----\n        Large arrays can be memmap'ed back as read-only (shared memory) by setting the `mmap='r'` parameter.\n\n        Parameters\n        ----------\n        fname : str\n            Path to file that contains LsiModel.\n        *args\n            Variable length argument list, see :meth:`gensim.utils.SaveLoad.load`.\n        **kwargs\n            Arbitrary keyword arguments, see :meth:`gensim.utils.SaveLoad.load`.\n\n        See Also\n        --------\n        :meth:`~gensim.models.lsimodel.LsiModel.save`\n\n        Returns\n        -------\n        :class:`~gensim.models.lsimodel.LsiModel`\n            Loaded instance.\n\n        Raises\n        ------\n        IOError\n            When methods are called on instance (should be called from class).\n\n        "
        kwargs['mmap'] = kwargs.get('mmap', None)
        result = super(LsiModel, cls).load(fname, *args, **kwargs)
        projection_fname = utils.smart_extension(fname, '.projection')
        try:
            result.projection = super(LsiModel, cls).load(projection_fname, *args, **kwargs)
        except Exception as e:
            logging.warning('failed to load projection from %s: %s', projection_fname, e)
        return result

def print_debug(id2token, u, s, topics, num_words=10, num_neg=None):
    if False:
        i = 10
        return i + 15
    'Log the most salient words per topic.\n\n    Parameters\n    ----------\n    id2token : :class:`~gensim.corpora.dictionary.Dictionary`\n        Mapping from ID to word in the Dictionary.\n    u : np.ndarray\n        The 2D U decomposition matrix.\n    s : np.ndarray\n        The 1D reduced array of eigenvalues used for decomposition.\n    topics : list of int\n        Sequence of topic IDs to be printed\n    num_words : int, optional\n        Number of words to be included for each topic.\n    num_neg : int, optional\n        Number of words with a negative contribution to a topic that should be included.\n\n    '
    if num_neg is None:
        num_neg = num_words / 2
    logger.info('computing word-topic salience for %i topics', len(topics))
    (topics, result) = (set(topics), {})
    for (uvecno, uvec) in enumerate(u):
        uvec = np.abs(np.asarray(uvec).flatten())
        udiff = uvec / np.sqrt(np.sum(np.dot(uvec, uvec)))
        for topic in topics:
            result.setdefault(topic, []).append((udiff[topic], uvecno))
    logger.debug('printing %i+%i salient words', num_words, num_neg)
    for topic in sorted(result.keys()):
        weights = sorted(result[topic], key=lambda x: -abs(x[0]))
        (_, most) = weights[0]
        if u[most, topic] < 0.0:
            normalize = -1.0
        else:
            normalize = 1.0
        (pos, neg) = ([], [])
        for (weight, uvecno) in weights:
            if normalize * u[uvecno, topic] > 0.0001:
                pos.append('%s(%.3f)' % (id2token[uvecno], u[uvecno, topic]))
                if len(pos) >= num_words:
                    break
        for (weight, uvecno) in weights:
            if normalize * u[uvecno, topic] < -0.0001:
                neg.append('%s(%.3f)' % (id2token[uvecno], u[uvecno, topic]))
                if len(neg) >= num_neg:
                    break
        logger.info('topic #%s(%.3f): %s, ..., %s', topic, s[topic], ', '.join(pos), ', '.join(neg))

def stochastic_svd(corpus, rank, num_terms, chunksize=20000, extra_dims=None, power_iters=0, dtype=np.float64, eps=1e-06, random_seed=None):
    if False:
        i = 10
        return i + 15
    'Run truncated Singular Value Decomposition (SVD) on a sparse input.\n\n    Parameters\n    ----------\n    corpus : {iterable of list of (int, float), scipy.sparse}\n        Input corpus as a stream (does not have to fit in RAM)\n        or a sparse matrix of shape (`num_terms`, num_documents).\n    rank : int\n        Desired number of factors to be retained after decomposition.\n    num_terms : int\n        The number of features (terms) in `corpus`.\n    chunksize :  int, optional\n        Number of documents to be used in each training chunk.\n    extra_dims : int, optional\n        Extra samples to be used besides the rank `k`. Can improve accuracy.\n    power_iters: int, optional\n        Number of power iteration steps to be used. Increasing the number of power iterations improves accuracy,\n        but lowers performance.\n    dtype : numpy.dtype, optional\n        Enforces a type for elements of the decomposed matrix.\n    eps: float, optional\n        Percentage of the spectrum\'s energy to be discarded.\n    random_seed: {None, int}, optional\n        Random seed used to initialize the pseudo-random number generator,\n         a local instance of numpy.random.RandomState instance.\n\n\n    Notes\n    -----\n    The corpus may be larger than RAM (iterator of vectors), if `corpus` is a `scipy.sparse.csc` instead,\n    it is assumed the whole corpus fits into core memory and a different (more efficient) code path is chosen.\n    This may return less than the requested number of top `rank` factors, in case the input itself is of lower rank.\n    The `extra_dims` (oversampling) and especially `power_iters` (power iterations) parameters affect accuracy of the\n    decomposition.\n\n    This algorithm uses `2 + power_iters` passes over the input data. In case you can only afford a single pass,\n    set `onepass=True` in :class:`~gensim.models.lsimodel.LsiModel` and avoid using this function directly.\n\n    The decomposition algorithm is based on `"Finding structure with randomness:\n    Probabilistic algorithms for constructing approximate matrix decompositions" <https://arxiv.org/abs/0909.4061>`_.\n\n\n    Returns\n    -------\n    (np.ndarray 2D, np.ndarray 1D)\n        The left singular vectors and the singular values of the `corpus`.\n\n    '
    rank = int(rank)
    if extra_dims is None:
        samples = max(10, 2 * rank)
    else:
        samples = rank + int(extra_dims)
    logger.info('using %i extra samples and %i power iterations', samples - rank, power_iters)
    num_terms = int(num_terms)
    y = np.zeros(dtype=dtype, shape=(num_terms, samples))
    logger.info('1st phase: constructing %s action matrix', str(y.shape))
    random_state = np.random.RandomState(random_seed)
    if scipy.sparse.issparse(corpus):
        (m, n) = corpus.shape
        assert num_terms == m, f'mismatch in number of features: {m} in sparse matrix vs. {num_terms} parameter'
        o = random_state.normal(0.0, 1.0, (n, samples)).astype(y.dtype)
        sparsetools.csc_matvecs(m, n, samples, corpus.indptr, corpus.indices, corpus.data, o.ravel(), y.ravel())
        del o
        if y.dtype != dtype:
            y = y.astype(dtype)
        logger.info('orthonormalizing %s action matrix', str(y.shape))
        y = [y]
        (q, _) = matutils.qr_destroy(y)
        logger.debug('running %i power iterations', power_iters)
        for _ in range(power_iters):
            q = corpus.T * q
            q = [corpus * q]
            (q, _) = matutils.qr_destroy(q)
    else:
        num_docs = 0
        for (chunk_no, chunk) in enumerate(utils.grouper(corpus, chunksize)):
            logger.info('PROGRESS: at document #%i', chunk_no * chunksize)
            s = sum((len(doc) for doc in chunk))
            chunk = matutils.corpus2csc(chunk, num_terms=num_terms, dtype=dtype)
            (m, n) = chunk.shape
            assert m == num_terms
            assert n <= chunksize
            num_docs += n
            logger.debug('multiplying chunk * gauss')
            o = random_state.normal(0.0, 1.0, (n, samples)).astype(dtype)
            sparsetools.csc_matvecs(m, n, samples, chunk.indptr, chunk.indices, chunk.data, o.ravel(), y.ravel())
            del chunk, o
        y = [y]
        (q, _) = matutils.qr_destroy(y)
        for power_iter in range(power_iters):
            logger.info('running power iteration #%i', power_iter + 1)
            yold = q.copy()
            q[:] = 0.0
            for (chunk_no, chunk) in enumerate(utils.grouper(corpus, chunksize)):
                logger.info('PROGRESS: at document #%i/%i', chunk_no * chunksize, num_docs)
                chunk = matutils.corpus2csc(chunk, num_terms=num_terms, dtype=dtype)
                tmp = chunk.T * yold
                tmp = chunk * tmp
                del chunk
                q += tmp
            del yold
            q = [q]
            (q, _) = matutils.qr_destroy(q)
    qt = q[:, :samples].T.copy()
    del q
    if scipy.sparse.issparse(corpus):
        b = qt * corpus
        logger.info('2nd phase: running dense svd on %s matrix', str(b.shape))
        (u, s, vt) = scipy.linalg.svd(b, full_matrices=False)
        del b, vt
    else:
        x = np.zeros(shape=(qt.shape[0], qt.shape[0]), dtype=dtype)
        logger.info('2nd phase: constructing %s covariance matrix', str(x.shape))
        for (chunk_no, chunk) in enumerate(utils.grouper(corpus, chunksize)):
            logger.info('PROGRESS: at document #%i/%i', chunk_no * chunksize, num_docs)
            chunk = matutils.corpus2csc(chunk, num_terms=num_terms, dtype=qt.dtype)
            b = qt * chunk
            del chunk
            x += np.dot(b, b.T)
            del b
        logger.info('running dense decomposition on %s covariance matrix', str(x.shape))
        (u, s, vt) = scipy.linalg.svd(x)
        s = np.sqrt(s)
    q = qt.T.copy()
    del qt
    logger.info('computing the final decomposition')
    keep = clip_spectrum(s ** 2, rank, discard=eps)
    u = u[:, :keep].copy()
    s = s[:keep]
    u = np.dot(q, u)
    return (u.astype(dtype), s.astype(dtype))