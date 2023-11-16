"""Author-topic model.

This module trains the author-topic model on documents and corresponding author-document dictionaries.
The training is online and is constant in memory w.r.t. the number of documents.
The model is *not* constant in memory w.r.t. the number of authors.

The model can be updated with additional documents after training has been completed. It is
also possible to continue training on the existing data.

The model is closely related to :class:`~gensim.models.ldamodel.LdaModel`.
The :class:`~gensim.models.atmodel.AuthorTopicModel` class inherits  :class:`~gensim.models.ldamodel.LdaModel`,
and its usage is thus similar.

The model was introduced by  `Rosen-Zvi and co-authors: "The Author-Topic Model for Authors and Documents"
<https://arxiv.org/abs/1207.4169>`_. The model correlates the authorship information with the topics to give a better
insight on the subject knowledge of an author.

.. _'Online Learning for LDA' by Hoffman et al.: online-lda_
.. _online-lda: https://papers.neurips.cc/paper/2010/file/71f6278d140af599e06ad9bf1ba03cb0-Paper.pdf

Example
-------
.. sourcecode:: pycon

    >>> from gensim.models import AuthorTopicModel
    >>> from gensim.corpora import mmcorpus
    >>> from gensim.test.utils import common_dictionary, datapath, temporary_file

    >>> author2doc = {
    ...     'john': [0, 1, 2, 3, 4, 5, 6],
    ...     'jane': [2, 3, 4, 5, 6, 7, 8],
    ...     'jack': [0, 2, 4, 6, 8]
    ... }
    >>>
    >>> corpus = mmcorpus.MmCorpus(datapath('testcorpus.mm'))
    >>>
    >>> with temporary_file("serialized") as s_path:
    ...     model = AuthorTopicModel(
    ...         corpus, author2doc=author2doc, id2word=common_dictionary, num_topics=4,
    ...         serialized=True, serialization_path=s_path
    ...     )
    ...
    ...     model.update(corpus, author2doc)  # update the author-topic model with additional documents
    >>>
    >>> # construct vectors for authors
    >>> author_vecs = [model.get_author_topics(author) for author in model.id2author.values()]

"""
import logging
from itertools import chain
from copy import deepcopy
from shutil import copyfile
from os.path import isfile
from os import remove
import numpy as np
from scipy.special import gammaln
from gensim import utils
from gensim.models import LdaModel
from gensim.models.ldamodel import LdaState
from gensim.matutils import dirichlet_expectation, mean_absolute_difference
from gensim.corpora import MmCorpus
logger = logging.getLogger(__name__)

class AuthorTopicState(LdaState):
    """Encapsulate information for computation of :class:`~gensim.models.atmodel.AuthorTopicModel`."""

    def __init__(self, eta, lambda_shape, gamma_shape):
        if False:
            i = 10
            return i + 15
        '\n\n        Parameters\n        ----------\n        eta: numpy.ndarray\n            Dirichlet topic parameter for sparsity.\n        lambda_shape: (int, int)\n            Initialize topic parameters.\n        gamma_shape: int\n            Initialize topic parameters.\n\n        '
        self.eta = eta
        self.sstats = np.zeros(lambda_shape)
        self.gamma = np.zeros(gamma_shape)
        self.numdocs = 0
        self.dtype = np.float64

def construct_doc2author(corpus, author2doc):
    if False:
        i = 10
        return i + 15
    'Create a mapping from document IDs to author IDs.\n\n    Parameters\n    ----------\n    corpus: iterable of list of (int, float)\n        Corpus in BoW format.\n    author2doc: dict of (str, list of int)\n        Mapping of authors to documents.\n\n    Returns\n    -------\n    dict of (int, list of str)\n        Document to Author mapping.\n\n    '
    doc2author = {}
    for (d, _) in enumerate(corpus):
        author_ids = []
        for (a, a_doc_ids) in author2doc.items():
            if d in a_doc_ids:
                author_ids.append(a)
        doc2author[d] = author_ids
    return doc2author

def construct_author2doc(doc2author):
    if False:
        i = 10
        return i + 15
    'Make a mapping from author IDs to document IDs.\n\n    Parameters\n    ----------\n    doc2author: dict of (int, list of str)\n        Mapping of document id to authors.\n\n    Returns\n    -------\n    dict of (str, list of int)\n        Mapping of authors to document ids.\n\n    '
    authors_ids = set()
    for (d, a_doc_ids) in doc2author.items():
        for a in a_doc_ids:
            authors_ids.add(a)
    author2doc = {}
    for a in authors_ids:
        author2doc[a] = []
        for (d, a_ids) in doc2author.items():
            if a in a_ids:
                author2doc[a].append(d)
    return author2doc

class AuthorTopicModel(LdaModel):
    """The constructor estimates the author-topic model parameters based on a training corpus."""

    def __init__(self, corpus=None, num_topics=100, id2word=None, author2doc=None, doc2author=None, chunksize=2000, passes=1, iterations=50, decay=0.5, offset=1.0, alpha='symmetric', eta='symmetric', update_every=1, eval_every=10, gamma_threshold=0.001, serialized=False, serialization_path=None, minimum_probability=0.01, random_state=None):
        if False:
            while True:
                i = 10
        "\n\n        Parameters\n        ----------\n        corpus : iterable of list of (int, float), optional\n            Corpus in BoW format\n        num_topics : int, optional\n            Number of topics to be extracted from the training corpus.\n        id2word : :class:`~gensim.corpora.dictionary.Dictionary`, optional\n            A mapping from word ids (integers) to words (strings).\n        author2doc : dict of (str, list of int), optional\n            A dictionary where keys are the names of authors and values are lists of document IDs that the author\n            contributes to.\n        doc2author : dict of (int, list of str), optional\n            A dictionary where the keys are document IDs and the values are lists of author names.\n        chunksize : int, optional\n            Controls the size of the mini-batches.\n        passes : int, optional\n            Number of times the model makes a pass over the entire training data.\n        iterations : int, optional\n            Maximum number of times the model loops over each document.\n        decay : float, optional\n            A number between (0.5, 1] to weight what percentage of the previous lambda value is forgotten\n            when each new document is examined. Corresponds to :math:`\\kappa` from\n            `'Online Learning for LDA' by Hoffman et al.`_\n        offset : float, optional\n            Hyper-parameter that controls how much we will slow down the first steps the first few iterations.\n            Corresponds to :math:`\\tau_0` from `'Online Learning for LDA' by Hoffman et al.`_\n        alpha : {float, numpy.ndarray of float, list of float, str}, optional\n            A-priori belief on document-topic distribution, this can be:\n                * scalar for a symmetric prior over document-topic distribution,\n                * 1D array of length equal to num_topics to denote an asymmetric user defined prior for each topic.\n\n            Alternatively default prior selecting strategies can be employed by supplying a string:\n                * 'symmetric': (default) Uses a fixed symmetric prior of `1.0 / num_topics`,\n                * 'asymmetric': Uses a fixed normalized asymmetric prior of `1.0 / (topic_index + sqrt(num_topics))`,\n                * 'auto': Learns an asymmetric prior from the corpus (not available if `distributed==True`).\n        eta : {float, numpy.ndarray of float, list of float, str}, optional\n            A-priori belief on topic-word distribution, this can be:\n                * scalar for a symmetric prior over topic-word distribution,\n                * 1D array of length equal to num_words to denote an asymmetric user defined prior for each word,\n                * matrix of shape (num_topics, num_words) to assign a probability for each word-topic combination.\n\n            Alternatively default prior selecting strategies can be employed by supplying a string:\n                * 'symmetric': (default) Uses a fixed symmetric prior of `1.0 / num_topics`,\n                * 'auto': Learns an asymmetric prior from the corpus.\n        update_every : int, optional\n            Make updates in topic probability for latest mini-batch.\n        eval_every : int, optional\n            Calculate and estimate log perplexity for latest mini-batch.\n        gamma_threshold : float, optional\n            Threshold value of gamma(topic difference between consecutive two topics)\n            until which the iterations continue.\n        serialized : bool, optional\n            Indicates whether the input corpora to the model are simple lists\n            or saved to the hard-drive.\n        serialization_path : str, optional\n            Must be set to a filepath, if `serialized = True` is used.\n        minimum_probability : float, optional\n            Controls filtering the topics returned for a document (bow).\n        random_state : {int, numpy.random.RandomState}, optional\n            Set the state of the random number generator inside the author-topic model.\n\n        "
        self.dtype = np.float64
        distributed = False
        self.dispatcher = None
        self.numworkers = 1
        self.id2word = id2word
        if corpus is None and self.id2word is None:
            raise ValueError('at least one of corpus/id2word must be specified, to establish input space dimensionality')
        if self.id2word is None:
            logger.warning('no word id mapping provided; initializing from corpus, assuming identity')
            self.id2word = utils.dict_from_corpus(corpus)
            self.num_terms = len(self.id2word)
        elif len(self.id2word) > 0:
            self.num_terms = 1 + max(self.id2word.keys())
        else:
            self.num_terms = 0
        if self.num_terms == 0:
            raise ValueError('cannot compute the author-topic model over an empty collection (no terms)')
        logger.info('Vocabulary consists of %d words.', self.num_terms)
        self.author2doc = {}
        self.doc2author = {}
        self.distributed = distributed
        self.num_topics = num_topics
        self.num_authors = 0
        self.chunksize = chunksize
        self.decay = decay
        self.offset = offset
        self.minimum_probability = minimum_probability
        self.num_updates = 0
        self.total_docs = 0
        self.passes = passes
        self.update_every = update_every
        self.eval_every = eval_every
        self.author2id = {}
        self.id2author = {}
        self.serialized = serialized
        if serialized and (not serialization_path):
            raise ValueError('If serialized corpora are used, a the path to a folder where the corpus should be saved must be provided (serialized_path).')
        if serialized and serialization_path:
            assert not isfile(serialization_path), 'A file already exists at the serialization_path path; choose a different serialization_path, or delete the file.'
        self.serialization_path = serialization_path
        self.init_empty_corpus()
        (self.alpha, self.optimize_alpha) = self.init_dir_prior(alpha, 'alpha')
        assert self.alpha.shape == (self.num_topics,), 'Invalid alpha shape. Got shape %s, but expected (%d, )' % (str(self.alpha.shape), self.num_topics)
        (self.eta, self.optimize_eta) = self.init_dir_prior(eta, 'eta')
        assert self.eta.shape == (self.num_terms,) or self.eta.shape == (self.num_topics, self.num_terms), 'Invalid eta shape. Got shape %s, but expected (%d, 1) or (%d, %d)' % (str(self.eta.shape), self.num_terms, self.num_topics, self.num_terms)
        self.random_state = utils.get_random_state(random_state)
        self.iterations = iterations
        self.gamma_threshold = gamma_threshold
        self.state = AuthorTopicState(self.eta, (self.num_topics, self.num_terms), (self.num_authors, self.num_topics))
        self.state.sstats = self.random_state.gamma(100.0, 1.0 / 100.0, (self.num_topics, self.num_terms))
        self.expElogbeta = np.exp(dirichlet_expectation(self.state.sstats))
        if corpus is not None and (author2doc is not None or doc2author is not None):
            use_numpy = self.dispatcher is not None
            self.update(corpus, author2doc, doc2author, chunks_as_numpy=use_numpy)

    def __str__(self):
        if False:
            while True:
                i = 10
        'Get a string representation of object.\n\n        Returns\n        -------\n        str\n            String representation of current instance.\n\n        '
        return '%s<num_terms=%s, num_topics=%s, num_authors=%s, decay=%s, chunksize=%s>' % (self.__class__.__name__, self.num_terms, self.num_topics, self.num_authors, self.decay, self.chunksize)

    def init_empty_corpus(self):
        if False:
            while True:
                i = 10
        'Initialize an empty corpus.\n        If the corpora are to be treated as lists, simply initialize an empty list.\n        If serialization is used, initialize an empty corpus using :class:`~gensim.corpora.mmcorpus.MmCorpus`.\n\n        '
        if self.serialized:
            MmCorpus.serialize(self.serialization_path, [])
            self.corpus = MmCorpus(self.serialization_path)
        else:
            self.corpus = []

    def extend_corpus(self, corpus):
        if False:
            for i in range(10):
                print('nop')
        "Add new documents from `corpus` to `self.corpus`.\n\n        If serialization is used, then the entire corpus (`self.corpus`) is re-serialized and the new documents\n        are added in the process. If serialization is not used, the corpus, as a list of documents, is simply extended.\n\n        Parameters\n        ----------\n        corpus : iterable of list of (int, float)\n            Corpus in BoW format\n\n        Raises\n        ------\n        AssertionError\n            If serialized == False and corpus isn't list.\n\n        "
        if self.serialized:
            if isinstance(corpus, MmCorpus):
                assert self.corpus.input != corpus.input, 'Input corpus cannot have the same file path as the model corpus (serialization_path).'
            corpus_chain = chain(self.corpus, corpus)
            copyfile(self.serialization_path, self.serialization_path + '.tmp')
            self.corpus.input = self.serialization_path + '.tmp'
            MmCorpus.serialize(self.serialization_path, corpus_chain)
            self.corpus = MmCorpus(self.serialization_path)
            remove(self.serialization_path + '.tmp')
        else:
            assert isinstance(corpus, list), 'If serialized == False, all input corpora must be lists.'
            self.corpus.extend(corpus)

    def compute_phinorm(self, expElogthetad, expElogbetad):
        if False:
            print('Hello World!')
        'Efficiently computes the normalizing factor in phi.\n\n        Parameters\n        ----------\n        expElogthetad: numpy.ndarray\n            Value of variational distribution :math:`q(\\theta|\\gamma)`.\n        expElogbetad: numpy.ndarray\n            Value of variational distribution :math:`q(\\beta|\\lambda)`.\n\n        Returns\n        -------\n        float\n            Value of normalizing factor.\n\n        '
        expElogtheta_sum = expElogthetad.sum(axis=0)
        phinorm = expElogtheta_sum.dot(expElogbetad) + 1e-100
        return phinorm

    def inference(self, chunk, author2doc, doc2author, rhot, collect_sstats=False, chunk_doc_idx=None):
        if False:
            i = 10
            return i + 15
        'Give a `chunk` of sparse document vectors, update gamma for each author corresponding to the `chuck`.\n\n        Warnings\n        --------\n        The whole input chunk of document is assumed to fit in RAM, chunking of a large corpus must be done earlier\n        in the pipeline.\n\n        Avoids computing the `phi` variational parameter directly using the\n        optimization presented in `Lee, Seung: "Algorithms for non-negative matrix factorization", NIPS 2001\n        <https://papers.nips.cc/paper/1861-algorithms-for-non-negative-matrix-factorization.pdf>`_.\n\n        Parameters\n        ----------\n        chunk : iterable of list of (int, float)\n            Corpus in BoW format.\n        author2doc : dict of (str, list of int), optional\n            A dictionary where keys are the names of authors and values are lists of document IDs that the author\n            contributes to.\n        doc2author : dict of (int, list of str), optional\n            A dictionary where the keys are document IDs and the values are lists of author names.\n        rhot : float\n            Value of rho for conducting inference on documents.\n        collect_sstats : boolean, optional\n            If True - collect sufficient statistics needed to update the model\'s topic-word distributions, and return\n            `(gamma_chunk, sstats)`. Otherwise, return `(gamma_chunk, None)`. `gamma_chunk` is of shape\n            `len(chunk_authors) x self.num_topics`,where `chunk_authors` is the number of authors in the documents in\n            the current chunk.\n        chunk_doc_idx : numpy.ndarray, optional\n            Assigns the value for document index.\n\n        Returns\n        -------\n        (numpy.ndarray, numpy.ndarray)\n            gamma_chunk and sstats (if `collect_sstats == True`, otherwise - None)\n\n        '
        try:
            len(chunk)
        except TypeError:
            chunk = list(chunk)
        if len(chunk) > 1:
            logger.debug('performing inference on a chunk of %i documents', len(chunk))
        if collect_sstats:
            sstats = np.zeros_like(self.expElogbeta)
        else:
            sstats = None
        converged = 0
        gamma_chunk = np.zeros((0, self.num_topics))
        for (d, doc) in enumerate(chunk):
            if chunk_doc_idx is not None:
                doc_no = chunk_doc_idx[d]
            else:
                doc_no = d
            if doc and (not isinstance(doc[0][0], (int, np.integer))):
                ids = [int(idx) for (idx, _) in doc]
            else:
                ids = [idx for (idx, _) in doc]
            ids = np.array(ids, dtype=int)
            cts = np.fromiter((cnt for (_, cnt) in doc), dtype=int, count=len(doc))
            authors_d = np.fromiter((self.author2id[a] for a in self.doc2author[doc_no]), dtype=int)
            gammad = self.state.gamma[authors_d, :]
            tilde_gamma = gammad.copy()
            Elogthetad = dirichlet_expectation(tilde_gamma)
            expElogthetad = np.exp(Elogthetad)
            expElogbetad = self.expElogbeta[:, ids]
            phinorm = self.compute_phinorm(expElogthetad, expElogbetad)
            for _ in range(self.iterations):
                lastgamma = tilde_gamma.copy()
                dot = np.dot(cts / phinorm, expElogbetad.T)
                for (ai, a) in enumerate(authors_d):
                    tilde_gamma[ai, :] = self.alpha + len(self.author2doc[self.id2author[a]]) * expElogthetad[ai, :] * dot
                tilde_gamma = (1 - rhot) * gammad + rhot * tilde_gamma
                Elogthetad = dirichlet_expectation(tilde_gamma)
                expElogthetad = np.exp(Elogthetad)
                phinorm = self.compute_phinorm(expElogthetad, expElogbetad)
                meanchange_gamma = mean_absolute_difference(tilde_gamma.ravel(), lastgamma.ravel())
                gamma_condition = meanchange_gamma < self.gamma_threshold
                if gamma_condition:
                    converged += 1
                    break
            self.state.gamma[authors_d, :] = tilde_gamma
            gamma_chunk = np.vstack([gamma_chunk, tilde_gamma])
            if collect_sstats:
                expElogtheta_sum_a = expElogthetad.sum(axis=0)
                sstats[:, ids] += np.outer(expElogtheta_sum_a.T, cts / phinorm)
        if len(chunk) > 1:
            logger.debug('%i/%i documents converged within %i iterations', converged, len(chunk), self.iterations)
        if collect_sstats:
            sstats *= self.expElogbeta
        return (gamma_chunk, sstats)

    def do_estep(self, chunk, author2doc, doc2author, rhot, state=None, chunk_doc_idx=None):
        if False:
            for i in range(10):
                print('nop')
        'Performs inference (E-step) on a chunk of documents, and accumulate the collected sufficient statistics.\n\n        Parameters\n        ----------\n        chunk : iterable of list of (int, float)\n            Corpus in BoW format.\n        author2doc : dict of (str, list of int), optional\n            A dictionary where keys are the names of authors and values are lists of document IDs that the author\n            contributes to.\n        doc2author : dict of (int, list of str), optional\n            A dictionary where the keys are document IDs and the values are lists of author names.\n        rhot : float\n            Value of rho for conducting inference on documents.\n        state : int, optional\n            Initializes the state for a new E iteration.\n        chunk_doc_idx : numpy.ndarray, optional\n            Assigns the value for document index.\n\n        Returns\n        -------\n        float\n            Value of gamma for training of model.\n\n        '
        if state is None:
            state = self.state
        (gamma, sstats) = self.inference(chunk, author2doc, doc2author, rhot, collect_sstats=True, chunk_doc_idx=chunk_doc_idx)
        state.sstats += sstats
        state.numdocs += len(chunk)
        return gamma

    def log_perplexity(self, chunk, chunk_doc_idx=None, total_docs=None):
        if False:
            print('Hello World!')
        'Calculate per-word likelihood bound, using the `chunk` of documents as evaluation corpus.\n\n        Parameters\n        ----------\n        chunk : iterable of list of (int, float)\n            Corpus in BoW format.\n        chunk_doc_idx : numpy.ndarray, optional\n            Assigns the value for document index.\n        total_docs : int, optional\n            Initializes the value for total number of documents.\n\n        Returns\n        -------\n        float\n            Value of per-word likelihood bound.\n\n        '
        if total_docs is None:
            total_docs = len(chunk)
        corpus_words = sum((cnt for document in chunk for (_, cnt) in document))
        subsample_ratio = 1.0 * total_docs / len(chunk)
        perwordbound = self.bound(chunk, chunk_doc_idx, subsample_ratio=subsample_ratio) / (subsample_ratio * corpus_words)
        logger.info('%.3f per-word bound, %.1f perplexity estimate based on a corpus of %i documents with %i words', perwordbound, np.exp2(-perwordbound), len(chunk), corpus_words)
        return perwordbound

    def update(self, corpus=None, author2doc=None, doc2author=None, chunksize=None, decay=None, offset=None, passes=None, update_every=None, eval_every=None, iterations=None, gamma_threshold=None, chunks_as_numpy=False):
        if False:
            for i in range(10):
                print('nop')
        "Train the model with new documents, by EM-iterating over `corpus` until the topics converge (or until the\n        maximum number of allowed iterations is reached).\n\n        Notes\n        -----\n        This update also supports updating an already trained model (`self`) with new documents from `corpus`;\n        the two models are then merged in proportion to the number of old vs. new documents.\n        This feature is still experimental for non-stationary input streams.\n\n        For stationary input (no topic drift in new documents), on the other hand, this equals the\n        online update of `'Online Learning for LDA' by Hoffman et al.`_\n        and is guaranteed to converge for any `decay` in (0.5, 1]. Additionally, for smaller corpus sizes, an\n        increasing `offset` may be beneficial (see Table 1 in the same paper).\n\n        If update is called with authors that already exist in the model, it will resume training on not only new\n        documents for that author, but also the previously seen documents. This is necessary for those authors' topic\n        distributions to converge.\n\n        Every time `update(corpus, author2doc)` is called, the new documents are to appended to all the previously seen\n        documents, and author2doc is combined with the previously seen authors.\n\n        To resume training on all the data seen by the model, simply call\n        :meth:`~gensim.models.atmodel.AuthorTopicModel.update`.\n\n        It is not possible to add new authors to existing documents, as all documents in `corpus` are assumed to be\n        new documents.\n\n        Parameters\n        ----------\n        corpus : iterable of list of (int, float)\n            The corpus in BoW format.\n        author2doc : dict of (str, list of int), optional\n            A dictionary where keys are the names of authors and values are lists of document IDs that the author\n            contributes to.\n        doc2author : dict of (int, list of str), optional\n            A dictionary where the keys are document IDs and the values are lists of author names.\n        chunksize : int, optional\n            Controls the size of the mini-batches.\n        decay : float, optional\n            A number between (0.5, 1] to weight what percentage of the previous lambda value is forgotten\n            when each new document is examined. Corresponds to :math:`\\kappa` from\n            `'Online Learning for LDA' by Hoffman et al.`_\n        offset : float, optional\n            Hyper-parameter that controls how much we will slow down the first steps the first few iterations.\n            Corresponds to :math:`\\tau_0` from `'Online Learning for LDA' by Hoffman et al.`_\n        passes : int, optional\n            Number of times the model makes a pass over the entire training data.\n        update_every : int, optional\n            Make updates in topic probability for latest mini-batch.\n        eval_every : int, optional\n            Calculate and estimate log perplexity for latest mini-batch.\n        iterations : int, optional\n            Maximum number of times the model loops over each document\n        gamma_threshold : float, optional\n            Threshold value of gamma(topic difference between consecutive two topics)\n            until which the iterations continue.\n        chunks_as_numpy : bool, optional\n            Whether each chunk passed to :meth:`~gensim.models.atmodel.AuthorTopicModel.inference` should be a numpy\n            array of not. Numpy can in some settings turn the term IDs into floats, these will be converted back into\n            integers in inference, which incurs a performance hit. For distributed computing (not supported now)\n            it may be desirable to keep the chunks as numpy arrays.\n\n        "
        if decay is None:
            decay = self.decay
        if offset is None:
            offset = self.offset
        if passes is None:
            passes = self.passes
        if update_every is None:
            update_every = self.update_every
        if eval_every is None:
            eval_every = self.eval_every
        if iterations is None:
            iterations = self.iterations
        if gamma_threshold is None:
            gamma_threshold = self.gamma_threshold
        author2doc = deepcopy(author2doc)
        doc2author = deepcopy(doc2author)
        if corpus is None:
            assert self.total_docs > 0, 'update() was called with no documents to train on.'
            train_corpus_idx = [d for d in range(self.total_docs)]
            num_input_authors = len(self.author2doc)
        else:
            if doc2author is None and author2doc is None:
                raise ValueError('at least one of author2doc/doc2author must be specified, to establish input space dimensionality')
            if doc2author is None:
                doc2author = construct_doc2author(corpus, author2doc)
            elif author2doc is None:
                author2doc = construct_author2doc(doc2author)
            num_input_authors = len(author2doc)
            try:
                len_input_corpus = len(corpus)
            except TypeError:
                logger.warning('input corpus stream has no len(); counting documents')
                len_input_corpus = sum((1 for _ in corpus))
            if len_input_corpus == 0:
                logger.warning('AuthorTopicModel.update() called with an empty corpus')
                return
            self.total_docs += len_input_corpus
            self.extend_corpus(corpus)
            new_authors = []
            for a in sorted(author2doc.keys()):
                if not self.author2doc.get(a):
                    new_authors.append(a)
            num_new_authors = len(new_authors)
            for (a_id, a_name) in enumerate(new_authors):
                self.author2id[a_name] = a_id + self.num_authors
                self.id2author[a_id + self.num_authors] = a_name
            self.num_authors += num_new_authors
            gamma_new = self.random_state.gamma(100.0, 1.0 / 100.0, (num_new_authors, self.num_topics))
            self.state.gamma = np.vstack([self.state.gamma, gamma_new])
            for (a, doc_ids) in author2doc.items():
                doc_ids = [d + self.total_docs - len_input_corpus for d in doc_ids]
            for (a, doc_ids) in author2doc.items():
                if self.author2doc.get(a):
                    self.author2doc[a].extend(doc_ids)
                else:
                    self.author2doc[a] = doc_ids
            for (d, a_list) in doc2author.items():
                self.doc2author[d] = a_list
            train_corpus_idx = set()
            for doc_ids in self.author2doc.values():
                train_corpus_idx.update(doc_ids)
            train_corpus_idx = sorted(train_corpus_idx)
        lencorpus = len(train_corpus_idx)
        if chunksize is None:
            chunksize = min(lencorpus, self.chunksize)
        self.state.numdocs += lencorpus
        if update_every:
            updatetype = 'online'
            updateafter = min(lencorpus, update_every * self.numworkers * chunksize)
        else:
            updatetype = 'batch'
            updateafter = lencorpus
        evalafter = min(lencorpus, (eval_every or 0) * self.numworkers * chunksize)
        updates_per_pass = max(1, lencorpus / updateafter)
        logger.info('running %s author-topic training, %s topics, %s authors, %i passes over the supplied corpus of %i documents, updating model once every %i documents, evaluating perplexity every %i documents, iterating %ix with a convergence threshold of %f', updatetype, self.num_topics, num_input_authors, passes, lencorpus, updateafter, evalafter, iterations, gamma_threshold)
        if updates_per_pass * passes < 10:
            logger.warning('too few updates, training might not converge; consider increasing the number of passes or iterations to improve accuracy')

        def rho():
            if False:
                for i in range(10):
                    print('nop')
            return pow(offset + pass_ + self.num_updates / chunksize, -decay)
        for pass_ in range(passes):
            if self.dispatcher:
                logger.info('initializing %s workers', self.numworkers)
                self.dispatcher.reset(self.state)
            else:
                other = AuthorTopicState(self.eta, self.state.sstats.shape, (0, 0))
            dirty = False
            reallen = 0
            for (chunk_no, chunk_doc_idx) in enumerate(utils.grouper(train_corpus_idx, chunksize, as_numpy=chunks_as_numpy)):
                chunk = [self.corpus[d] for d in chunk_doc_idx]
                reallen += len(chunk)
                if eval_every and (reallen == lencorpus or (chunk_no + 1) % (eval_every * self.numworkers) == 0):
                    self.log_perplexity(chunk, chunk_doc_idx, total_docs=lencorpus)
                if self.dispatcher:
                    logger.info('PROGRESS: pass %i, dispatching documents up to #%i/%i', pass_, chunk_no * chunksize + len(chunk), lencorpus)
                    self.dispatcher.putjob(chunk)
                else:
                    logger.info('PROGRESS: pass %i, at document #%i/%i', pass_, chunk_no * chunksize + len(chunk), lencorpus)
                    gammat = self.do_estep(chunk, self.author2doc, self.doc2author, rho(), other, chunk_doc_idx)
                    if self.optimize_alpha:
                        self.update_alpha(gammat, rho())
                dirty = True
                del chunk
                if update_every and (chunk_no + 1) % (update_every * self.numworkers) == 0:
                    if self.dispatcher:
                        logger.info('reached the end of input; now waiting for all remaining jobs to finish')
                        other = self.dispatcher.getstate()
                    self.do_mstep(rho(), other, pass_ > 0)
                    del other
                    if self.dispatcher:
                        logger.info('initializing workers')
                        self.dispatcher.reset(self.state)
                    else:
                        other = AuthorTopicState(self.eta, self.state.sstats.shape, (0, 0))
                    dirty = False
            if reallen != lencorpus:
                raise RuntimeError("input corpus size changed during training (don't use generators as input)")
            if dirty:
                if self.dispatcher:
                    logger.info('reached the end of input; now waiting for all remaining jobs to finish')
                    other = self.dispatcher.getstate()
                self.do_mstep(rho(), other, pass_ > 0)
                del other

    def bound(self, chunk, chunk_doc_idx=None, subsample_ratio=1.0, author2doc=None, doc2author=None):
        if False:
            return 10
        'Estimate the variational bound of documents from `corpus`.\n\n        :math:`\\mathbb{E_{q}}[\\log p(corpus)] - \\mathbb{E_{q}}[\\log q(corpus)]`\n\n        Notes\n        -----\n        There are basically two use cases of this method:\n\n        #. `chunk` is a subset of the training corpus, and `chunk_doc_idx` is provided,\n           indicating the indexes of the documents in the training corpus.\n        #. `chunk` is a test set (held-out data), and `author2doc` and `doc2author` corresponding to this test set\n           are provided. There must not be any new authors passed to this method, `chunk_doc_idx` is not needed\n           in this case.\n\n        Parameters\n        ----------\n        chunk : iterable of list of (int, float)\n            Corpus in BoW format.\n        chunk_doc_idx : numpy.ndarray, optional\n            Assigns the value for document index.\n        subsample_ratio : float, optional\n            Used for calculation of word score for estimation of variational bound.\n        author2doc : dict of (str, list of int), optional\n            A dictionary where keys are the names of authors and values are lists of documents that the author\n            contributes to.\n        doc2author : dict of (int, list of str), optional\n            A dictionary where the keys are document IDs and the values are lists of author names.\n\n        Returns\n        -------\n        float\n            Value of variational bound score.\n\n        '
        _lambda = self.state.get_lambda()
        Elogbeta = dirichlet_expectation(_lambda)
        expElogbeta = np.exp(Elogbeta)
        gamma = self.state.gamma
        if author2doc is None and doc2author is None:
            author2doc = self.author2doc
            doc2author = self.doc2author
            if not chunk_doc_idx:
                raise ValueError('Either author dictionaries or chunk_doc_idx must be provided. Consult documentation of bound method.')
        elif author2doc is not None and doc2author is not None:
            for a in author2doc.keys():
                if not self.author2doc.get(a):
                    raise ValueError('bound cannot be called with authors not seen during training.')
            if chunk_doc_idx:
                raise ValueError('Either author dictionaries or chunk_doc_idx must be provided, not both. Consult documentation of bound method.')
        else:
            raise ValueError('Either both author2doc and doc2author should be provided, or neither. Consult documentation of bound method.')
        Elogtheta = dirichlet_expectation(gamma)
        expElogtheta = np.exp(Elogtheta)
        word_score = 0.0
        theta_score = 0.0
        for (d, doc) in enumerate(chunk):
            if chunk_doc_idx:
                doc_no = chunk_doc_idx[d]
            else:
                doc_no = d
            authors_d = np.fromiter((self.author2id[a] for a in self.doc2author[doc_no]), dtype=int)
            ids = np.fromiter((id for (id, _) in doc), dtype=int, count=len(doc))
            cts = np.fromiter((cnt for (_, cnt) in doc), dtype=int, count=len(doc))
            if d % self.chunksize == 0:
                logger.debug('bound: at document #%i in chunk', d)
            phinorm = self.compute_phinorm(expElogtheta[authors_d, :], expElogbeta[:, ids])
            word_score += np.log(1.0 / len(authors_d)) * sum(cts) + cts.dot(np.log(phinorm))
        word_score *= subsample_ratio
        for a in author2doc.keys():
            a = self.author2id[a]
            theta_score += np.sum((self.alpha - gamma[a, :]) * Elogtheta[a, :])
            theta_score += np.sum(gammaln(gamma[a, :]) - gammaln(self.alpha))
            theta_score += gammaln(np.sum(self.alpha)) - gammaln(np.sum(gamma[a, :]))
        theta_score *= self.num_authors / len(author2doc)
        beta_score = 0.0
        beta_score += np.sum((self.eta - _lambda) * Elogbeta)
        beta_score += np.sum(gammaln(_lambda) - gammaln(self.eta))
        sum_eta = np.sum(self.eta)
        beta_score += np.sum(gammaln(sum_eta) - gammaln(np.sum(_lambda, 1)))
        total_score = word_score + theta_score + beta_score
        return total_score

    def get_document_topics(self, word_id, minimum_probability=None):
        if False:
            i = 10
            return i + 15
        'Override :meth:`~gensim.models.ldamodel.LdaModel.get_document_topics` and simply raises an exception.\n\n        Warnings\n        --------\n        This method invalid for model, use :meth:`~gensim.models.atmodel.AuthorTopicModel.get_author_topics` or\n        :meth:`~gensim.models.atmodel.AuthorTopicModel.get_new_author_topics` instead.\n\n        Raises\n        ------\n        NotImplementedError\n            Always.\n\n        '
        raise NotImplementedError('Method "get_document_topics" is not valid for the author-topic model. Use the "get_author_topics" method.')

    def get_new_author_topics(self, corpus, minimum_probability=None):
        if False:
            while True:
                i = 10
        'Infers topics for new author.\n\n        Infers a topic distribution for a new author over the passed corpus of docs,\n        assuming that all documents are from this single new author.\n\n        Parameters\n        ----------\n        corpus : iterable of list of (int, float)\n            Corpus in BoW format.\n        minimum_probability : float, optional\n            Ignore topics with probability below this value, if None - 1e-8 is used.\n\n        Returns\n        -------\n        list of (int, float)\n            Topic distribution for the given `corpus`.\n\n        '

        def rho():
            if False:
                return 10
            return pow(self.offset + 1 + 1, -self.decay)

        def rollback_new_author_chages():
            if False:
                for i in range(10):
                    print('nop')
            self.state.gamma = self.state.gamma[0:-1]
            del self.author2doc[new_author_name]
            a_id = self.author2id[new_author_name]
            del self.id2author[a_id]
            del self.author2id[new_author_name]
            for new_doc_id in corpus_doc_idx:
                del self.doc2author[new_doc_id]
        try:
            len_input_corpus = len(corpus)
        except TypeError:
            logger.warning('input corpus stream has no len(); counting documents')
            len_input_corpus = sum((1 for _ in corpus))
        if len_input_corpus == 0:
            raise ValueError('AuthorTopicModel.get_new_author_topics() called with an empty corpus')
        new_author_name = 'placeholder_name'
        corpus_doc_idx = list(range(self.total_docs, self.total_docs + len_input_corpus))
        num_new_authors = 1
        author_id = self.num_authors
        if new_author_name in self.author2id:
            raise ValueError("self.author2id already has 'placeholder_name' author")
        self.author2id[new_author_name] = author_id
        self.id2author[author_id] = new_author_name
        self.author2doc[new_author_name] = corpus_doc_idx
        for new_doc_id in corpus_doc_idx:
            self.doc2author[new_doc_id] = [new_author_name]
        gamma_new = self.random_state.gamma(100.0, 1.0 / 100.0, (num_new_authors, self.num_topics))
        self.state.gamma = np.vstack([self.state.gamma, gamma_new])
        try:
            (gammat, _) = self.inference(corpus, self.author2doc, self.doc2author, rho(), collect_sstats=False, chunk_doc_idx=corpus_doc_idx)
            new_author_topics = self.get_author_topics(new_author_name, minimum_probability)
        finally:
            rollback_new_author_chages()
        return new_author_topics

    def get_author_topics(self, author_name, minimum_probability=None):
        if False:
            while True:
                i = 10
        'Get topic distribution the given author.\n\n        Parameters\n        ----------\n        author_name : str\n            Name of the author for which the topic distribution needs to be estimated.\n        minimum_probability : float, optional\n            Sets the minimum probability value for showing the topics of a given author, topics with probability <\n            `minimum_probability` will be ignored.\n\n        Returns\n        -------\n        list of (int, float)\n            Topic distribution of an author.\n\n        Example\n        -------\n        .. sourcecode:: pycon\n\n            >>> from gensim.models import AuthorTopicModel\n            >>> from gensim.corpora import mmcorpus\n            >>> from gensim.test.utils import common_dictionary, datapath, temporary_file\n\n            >>> author2doc = {\n            ...     \'john\': [0, 1, 2, 3, 4, 5, 6],\n            ...     \'jane\': [2, 3, 4, 5, 6, 7, 8],\n            ...     \'jack\': [0, 2, 4, 6, 8]\n            ... }\n            >>>\n            >>> corpus = mmcorpus.MmCorpus(datapath(\'testcorpus.mm\'))\n            >>>\n            >>> with temporary_file("serialized") as s_path:\n            ...     model = AuthorTopicModel(\n            ...         corpus, author2doc=author2doc, id2word=common_dictionary, num_topics=4,\n            ...         serialized=True, serialization_path=s_path\n            ...     )\n            ...\n            ...     model.update(corpus, author2doc)  # update the author-topic model with additional documents\n            >>>\n            >>> # construct vectors for authors\n            >>> author_vecs = [model.get_author_topics(author) for author in model.id2author.values()]\n\n        '
        author_id = self.author2id[author_name]
        if minimum_probability is None:
            minimum_probability = self.minimum_probability
        minimum_probability = max(minimum_probability, 1e-08)
        topic_dist = self.state.gamma[author_id, :] / sum(self.state.gamma[author_id, :])
        author_topics = [(topicid, topicvalue) for (topicid, topicvalue) in enumerate(topic_dist) if topicvalue >= minimum_probability]
        return author_topics

    def __getitem__(self, author_names, eps=None):
        if False:
            return 10
        'Get topic distribution for input `author_names`.\n\n        Parameters\n        ----------\n        author_names : {str, list of str}\n            Name(s) of the author for which the topic distribution needs to be estimated.\n        eps : float, optional\n            The minimum probability value for showing the topics of a given author, topics with probability < `eps`\n            will be ignored.\n\n        Returns\n        -------\n        list of (int, float) **or** list of list of (int, float)\n            Topic distribution for the author(s), type depends on type of `author_names`.\n\n        '
        if isinstance(author_names, list):
            items = []
            for a in author_names:
                items.append(self.get_author_topics(a, minimum_probability=eps))
        else:
            items = self.get_author_topics(author_names, minimum_probability=eps)
        return items