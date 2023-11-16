"""Optimized `Latent Dirichlet Allocation (LDA) <https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation>`_ in Python.

For a faster implementation of LDA (parallelized for multicore machines), see also :mod:`gensim.models.ldamulticore`.

This module allows both LDA model estimation from a training corpus and inference of topic
distribution on new, unseen documents. The model can also be updated with new documents
for online training.

The core estimation code is based on the `onlineldavb.py script
<https://github.com/blei-lab/onlineldavb/blob/master/onlineldavb.py>`_, by
Matthew D. Hoffman, David M. Blei, Francis Bach:
`'Online Learning for Latent Dirichlet Allocation', NIPS 2010`_.

.. _'Online Learning for Latent Dirichlet Allocation', NIPS 2010: online-lda_
.. _'Online Learning for LDA' by Hoffman et al.: online-lda_
.. _online-lda: https://papers.neurips.cc/paper/2010/file/71f6278d140af599e06ad9bf1ba03cb0-Paper.pdf

The algorithm:

#. Is **streamed**: training documents may come in sequentially, no random access required.
#. Runs in **constant memory** w.r.t. the number of documents: size of the training corpus does not affect memory
   footprint, can process corpora larger than RAM.
#. Is **distributed**: makes use of a cluster of machines, if available, to speed up model estimation.


Usage examples
--------------

Train an LDA model using a Gensim corpus

.. sourcecode:: pycon

    >>> from gensim.test.utils import common_texts
    >>> from gensim.corpora.dictionary import Dictionary
    >>>
    >>> # Create a corpus from a list of texts
    >>> common_dictionary = Dictionary(common_texts)
    >>> common_corpus = [common_dictionary.doc2bow(text) for text in common_texts]
    >>>
    >>> # Train the model on the corpus.
    >>> lda = LdaModel(common_corpus, num_topics=10)

Save a model to disk, or reload a pre-trained model

.. sourcecode:: pycon

    >>> from gensim.test.utils import datapath
    >>>
    >>> # Save model to disk.
    >>> temp_file = datapath("model")
    >>> lda.save(temp_file)
    >>>
    >>> # Load a potentially pretrained model from disk.
    >>> lda = LdaModel.load(temp_file)

Query, the model using new, unseen documents

.. sourcecode:: pycon

    >>> # Create a new corpus, made of previously unseen documents.
    >>> other_texts = [
    ...     ['computer', 'time', 'graph'],
    ...     ['survey', 'response', 'eps'],
    ...     ['human', 'system', 'computer']
    ... ]
    >>> other_corpus = [common_dictionary.doc2bow(text) for text in other_texts]
    >>>
    >>> unseen_doc = other_corpus[0]
    >>> vector = lda[unseen_doc]  # get topic probability distribution for a document

Update the model by incrementally training on the new corpus

.. sourcecode:: pycon

    >>> lda.update(other_corpus)
    >>> vector = lda[unseen_doc]

A lot of parameters can be tuned to optimize training for your specific case

.. sourcecode:: pycon

    >>> lda = LdaModel(common_corpus, num_topics=50, alpha='auto', eval_every=5)  # learn asymmetric alpha from data

"""
import logging
import numbers
import os
import time
from collections import defaultdict
import numpy as np
from scipy.special import gammaln, psi
from scipy.special import polygamma
from gensim import interfaces, utils, matutils
from gensim.matutils import kullback_leibler, hellinger, jaccard_distance, jensen_shannon, dirichlet_expectation, logsumexp, mean_absolute_difference
from gensim.models import basemodel, CoherenceModel
from gensim.models.callbacks import Callback
logger = logging.getLogger(__name__)

def update_dir_prior(prior, N, logphat, rho):
    if False:
        i = 10
        return i + 15
    'Update a given prior using Newton\'s method, described in\n    `J. Huang: "Maximum Likelihood Estimation of Dirichlet Distribution Parameters"\n    <http://jonathan-huang.org/research/dirichlet/dirichlet.pdf>`_.\n\n    Parameters\n    ----------\n    prior : list of float\n        The prior for each possible outcome at the previous iteration (to be updated).\n    N : int\n        Number of observations.\n    logphat : list of float\n        Log probabilities for the current estimation, also called "observed sufficient statistics".\n    rho : float\n        Learning rate.\n\n    Returns\n    -------\n    list of float\n        The updated prior.\n\n    '
    gradf = N * (psi(np.sum(prior)) - psi(prior) + logphat)
    c = N * polygamma(1, np.sum(prior))
    q = -N * polygamma(1, prior)
    b = np.sum(gradf / q) / (1 / c + np.sum(1 / q))
    dprior = -(gradf - b) / q
    updated_prior = rho * dprior + prior
    if all(updated_prior > 0):
        prior = updated_prior
    else:
        logger.warning('updated prior is not positive')
    return prior

class LdaState(utils.SaveLoad):
    """Encapsulate information for distributed computation of :class:`~gensim.models.ldamodel.LdaModel` objects.

    Objects of this class are sent over the network, so try to keep them lean to
    reduce traffic.

    """

    def __init__(self, eta, shape, dtype=np.float32):
        if False:
            while True:
                i = 10
        '\n\n        Parameters\n        ----------\n        eta : numpy.ndarray\n            The prior probabilities assigned to each term.\n        shape : tuple of (int, int)\n            Shape of the sufficient statistics: (number of topics to be found, number of terms in the vocabulary).\n        dtype : type\n            Overrides the numpy array default types.\n\n        '
        self.eta = eta.astype(dtype, copy=False)
        self.sstats = np.zeros(shape, dtype=dtype)
        self.numdocs = 0
        self.dtype = dtype

    def reset(self):
        if False:
            return 10
        'Prepare the state for a new EM iteration (reset sufficient stats).'
        self.sstats[:] = 0.0
        self.numdocs = 0

    def merge(self, other):
        if False:
            print('Hello World!')
        'Merge the result of an E step from one node with that of another node (summing up sufficient statistics).\n\n        The merging is trivial and after merging all cluster nodes, we have the\n        exact same result as if the computation was run on a single node (no\n        approximation).\n\n        Parameters\n        ----------\n        other : :class:`~gensim.models.ldamodel.LdaState`\n            The state object with which the current one will be merged.\n\n        '
        assert other is not None
        self.sstats += other.sstats
        self.numdocs += other.numdocs

    def blend(self, rhot, other, targetsize=None):
        if False:
            print('Hello World!')
        "Merge the current state with another one using a weighted average for the sufficient statistics.\n\n        The number of documents is stretched in both state objects, so that they are of comparable magnitude.\n        This procedure corresponds to the stochastic gradient update from\n        `'Online Learning for LDA' by Hoffman et al.`_, see equations (5) and (9).\n\n        Parameters\n        ----------\n        rhot : float\n            Weight of the `other` state in the computed average. A value of 0.0 means that `other`\n            is completely ignored. A value of 1.0 means `self` is completely ignored.\n        other : :class:`~gensim.models.ldamodel.LdaState`\n            The state object with which the current one will be merged.\n        targetsize : int, optional\n            The number of documents to stretch both states to.\n\n        "
        assert other is not None
        if targetsize is None:
            targetsize = self.numdocs
        if self.numdocs == 0 or targetsize == self.numdocs:
            scale = 1.0
        else:
            scale = 1.0 * targetsize / self.numdocs
        self.sstats *= (1.0 - rhot) * scale
        if other.numdocs == 0 or targetsize == other.numdocs:
            scale = 1.0
        else:
            logger.info('merging changes from %i documents into a model of %i documents', other.numdocs, targetsize)
            scale = 1.0 * targetsize / other.numdocs
        self.sstats += rhot * scale * other.sstats
        self.numdocs = targetsize

    def blend2(self, rhot, other, targetsize=None):
        if False:
            for i in range(10):
                print('nop')
        'Merge the current state with another one using a weighted sum for the sufficient statistics.\n\n        In contrast to :meth:`~gensim.models.ldamodel.LdaState.blend`, the sufficient statistics are not scaled\n        prior to aggregation.\n\n        Parameters\n        ----------\n        rhot : float\n            Unused.\n        other : :class:`~gensim.models.ldamodel.LdaState`\n            The state object with which the current one will be merged.\n        targetsize : int, optional\n            The number of documents to stretch both states to.\n\n        '
        assert other is not None
        if targetsize is None:
            targetsize = self.numdocs
        self.sstats += other.sstats
        self.numdocs = targetsize

    def get_lambda(self):
        if False:
            i = 10
            return i + 15
        'Get the parameters of the posterior over the topics, also referred to as "the topics".\n\n        Returns\n        -------\n        numpy.ndarray\n            Parameters of the posterior probability over topics.\n\n        '
        return self.eta + self.sstats

    def get_Elogbeta(self):
        if False:
            print('Hello World!')
        'Get the log (posterior) probabilities for each topic.\n\n        Returns\n        -------\n        numpy.ndarray\n            Posterior probabilities for each topic.\n        '
        return dirichlet_expectation(self.get_lambda())

    @classmethod
    def load(cls, fname, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Load a previously stored state from disk.\n\n        Overrides :class:`~gensim.utils.SaveLoad.load` by enforcing the `dtype` parameter\n        to ensure backwards compatibility.\n\n        Parameters\n        ----------\n        fname : str\n            Path to file that contains the needed object.\n        args : object\n            Positional parameters to be propagated to class:`~gensim.utils.SaveLoad.load`\n        kwargs : object\n            Key-word parameters to be propagated to class:`~gensim.utils.SaveLoad.load`\n\n        Returns\n        -------\n        :class:`~gensim.models.ldamodel.LdaState`\n            The state loaded from the given file.\n\n        '
        result = super(LdaState, cls).load(fname, *args, **kwargs)
        if not hasattr(result, 'dtype'):
            result.dtype = np.float64
            logging.info('dtype was not set in saved %s file %s, assuming np.float64', result.__class__.__name__, fname)
        return result

class LdaModel(interfaces.TransformationABC, basemodel.BaseTopicModel):
    """Train and use Online Latent Dirichlet Allocation model as presented in
    `'Online Learning for LDA' by Hoffman et al.`_

    Examples
    -------
    Initialize a model using a Gensim corpus

    .. sourcecode:: pycon

        >>> from gensim.test.utils import common_corpus
        >>>
        >>> lda = LdaModel(common_corpus, num_topics=10)

    You can then infer topic distributions on new, unseen documents.

    .. sourcecode:: pycon

        >>> doc_bow = [(1, 0.3), (2, 0.1), (0, 0.09)]
        >>> doc_lda = lda[doc_bow]

    The model can be updated (trained) with new documents.

    .. sourcecode:: pycon

        >>> # In practice (corpus =/= initial training corpus), but we use the same here for simplicity.
        >>> other_corpus = common_corpus
        >>>
        >>> lda.update(other_corpus)

    Model persistency is achieved through :meth:`~gensim.models.ldamodel.LdaModel.load` and
    :meth:`~gensim.models.ldamodel.LdaModel.save` methods.

    """

    def __init__(self, corpus=None, num_topics=100, id2word=None, distributed=False, chunksize=2000, passes=1, update_every=1, alpha='symmetric', eta=None, decay=0.5, offset=1.0, eval_every=10, iterations=50, gamma_threshold=0.001, minimum_probability=0.01, random_state=None, ns_conf=None, minimum_phi_value=0.01, per_word_topics=False, callbacks=None, dtype=np.float32):
        if False:
            print('Hello World!')
        "\n\n        Parameters\n        ----------\n        corpus : iterable of list of (int, float), optional\n            Stream of document vectors or sparse matrix of shape (`num_documents`, `num_terms`).\n            If you have a CSC in-memory matrix, you can convert it to a\n            streamed corpus with the help of gensim.matutils.Sparse2Corpus.\n            If not given, the model is left untrained (presumably because you want to call\n            :meth:`~gensim.models.ldamodel.LdaModel.update` manually).\n        num_topics : int, optional\n            The number of requested latent topics to be extracted from the training corpus.\n        id2word : {dict of (int, str), :class:`gensim.corpora.dictionary.Dictionary`}\n            Mapping from word IDs to words. It is used to determine the vocabulary size, as well as for\n            debugging and topic printing.\n        distributed : bool, optional\n            Whether distributed computing should be used to accelerate training.\n        chunksize :  int, optional\n            Number of documents to be used in each training chunk.\n        passes : int, optional\n            Number of passes through the corpus during training.\n        update_every : int, optional\n            Number of documents to be iterated through for each update.\n            Set to 0 for batch learning, > 1 for online iterative learning.\n        alpha : {float, numpy.ndarray of float, list of float, str}, optional\n            A-priori belief on document-topic distribution, this can be:\n                * scalar for a symmetric prior over document-topic distribution,\n                * 1D array of length equal to num_topics to denote an asymmetric user defined prior for each topic.\n\n            Alternatively default prior selecting strategies can be employed by supplying a string:\n                * 'symmetric': (default) Uses a fixed symmetric prior of `1.0 / num_topics`,\n                * 'asymmetric': Uses a fixed normalized asymmetric prior of `1.0 / (topic_index + sqrt(num_topics))`,\n                * 'auto': Learns an asymmetric prior from the corpus (not available if `distributed==True`).\n        eta : {float, numpy.ndarray of float, list of float, str}, optional\n            A-priori belief on topic-word distribution, this can be:\n                * scalar for a symmetric prior over topic-word distribution,\n                * 1D array of length equal to num_words to denote an asymmetric user defined prior for each word,\n                * matrix of shape (num_topics, num_words) to assign a probability for each word-topic combination.\n\n            Alternatively default prior selecting strategies can be employed by supplying a string:\n                * 'symmetric': (default) Uses a fixed symmetric prior of `1.0 / num_topics`,\n                * 'auto': Learns an asymmetric prior from the corpus.\n        decay : float, optional\n            A number between (0.5, 1] to weight what percentage of the previous lambda value is forgotten\n            when each new document is examined.\n            Corresponds to :math:`\\kappa` from `'Online Learning for LDA' by Hoffman et al.`_\n        offset : float, optional\n            Hyper-parameter that controls how much we will slow down the first steps the first few iterations.\n            Corresponds to :math:`\\tau_0` from `'Online Learning for LDA' by Hoffman et al.`_\n        eval_every : int, optional\n            Log perplexity is estimated every that many updates. Setting this to one slows down training by ~2x.\n        iterations : int, optional\n            Maximum number of iterations through the corpus when inferring the topic distribution of a corpus.\n        gamma_threshold : float, optional\n            Minimum change in the value of the gamma parameters to continue iterating.\n        minimum_probability : float, optional\n            Topics with a probability lower than this threshold will be filtered out.\n        random_state : {np.random.RandomState, int}, optional\n            Either a randomState object or a seed to generate one. Useful for reproducibility.\n        ns_conf : dict of (str, object), optional\n            Key word parameters propagated to :func:`gensim.utils.getNS` to get a Pyro4 nameserver.\n            Only used if `distributed` is set to True.\n        minimum_phi_value : float, optional\n            if `per_word_topics` is True, this represents a lower bound on the term probabilities.\n        per_word_topics : bool\n            If True, the model also computes a list of topics, sorted in descending order of most likely topics for\n            each word, along with their phi values multiplied by the feature length (i.e. word count).\n        callbacks : list of :class:`~gensim.models.callbacks.Callback`\n            Metric callbacks to log and visualize evaluation metrics of the model during training.\n        dtype : {numpy.float16, numpy.float32, numpy.float64}, optional\n            Data-type to use during calculations inside model. All inputs are also converted.\n\n        "
        self.dtype = np.finfo(dtype).dtype
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
            raise ValueError('cannot compute LDA over an empty collection (no terms)')
        self.distributed = bool(distributed)
        self.num_topics = int(num_topics)
        self.chunksize = chunksize
        self.decay = decay
        self.offset = offset
        self.minimum_probability = minimum_probability
        self.num_updates = 0
        self.passes = passes
        self.update_every = update_every
        self.eval_every = eval_every
        self.minimum_phi_value = minimum_phi_value
        self.per_word_topics = per_word_topics
        self.callbacks = callbacks
        (self.alpha, self.optimize_alpha) = self.init_dir_prior(alpha, 'alpha')
        assert self.alpha.shape == (self.num_topics,), 'Invalid alpha shape. Got shape %s, but expected (%d, )' % (str(self.alpha.shape), self.num_topics)
        (self.eta, self.optimize_eta) = self.init_dir_prior(eta, 'eta')
        assert self.eta.shape == (self.num_terms,) or self.eta.shape == (self.num_topics, self.num_terms), 'Invalid eta shape. Got shape %s, but expected (%d, 1) or (%d, %d)' % (str(self.eta.shape), self.num_terms, self.num_topics, self.num_terms)
        self.random_state = utils.get_random_state(random_state)
        self.iterations = iterations
        self.gamma_threshold = gamma_threshold
        if not distributed:
            logger.info('using serial LDA version on this node')
            self.dispatcher = None
            self.numworkers = 1
        else:
            if self.optimize_alpha:
                raise NotImplementedError('auto-optimizing alpha not implemented in distributed LDA')
            try:
                import Pyro4
                if ns_conf is None:
                    ns_conf = {}
                with utils.getNS(**ns_conf) as ns:
                    from gensim.models.lda_dispatcher import LDA_DISPATCHER_PREFIX
                    self.dispatcher = Pyro4.Proxy(ns.list(prefix=LDA_DISPATCHER_PREFIX)[LDA_DISPATCHER_PREFIX])
                    logger.debug('looking for dispatcher at %s' % str(self.dispatcher._pyroUri))
                    self.dispatcher.initialize(id2word=self.id2word, num_topics=self.num_topics, chunksize=chunksize, alpha=alpha, eta=eta, distributed=False)
                    self.numworkers = len(self.dispatcher.getworkers())
                    logger.info('using distributed version with %i workers', self.numworkers)
            except Exception as err:
                logger.error('failed to initialize distributed LDA (%s)', err)
                raise RuntimeError('failed to initialize distributed LDA (%s)' % err)
        self.state = LdaState(self.eta, (self.num_topics, self.num_terms), dtype=self.dtype)
        self.state.sstats[...] = self.random_state.gamma(100.0, 1.0 / 100.0, (self.num_topics, self.num_terms))
        self.expElogbeta = np.exp(dirichlet_expectation(self.state.sstats))
        assert self.eta.dtype == self.dtype
        assert self.expElogbeta.dtype == self.dtype
        if corpus is not None:
            use_numpy = self.dispatcher is not None
            start = time.time()
            self.update(corpus, chunks_as_numpy=use_numpy)
            self.add_lifecycle_event('created', msg=f'trained {self} in {time.time() - start:.2f}s')

    def init_dir_prior(self, prior, name):
        if False:
            while True:
                i = 10
        "Initialize priors for the Dirichlet distribution.\n\n        Parameters\n        ----------\n        prior : {float, numpy.ndarray of float, list of float, str}\n            A-priori belief on document-topic distribution. If `name` == 'alpha', then the prior can be:\n                * scalar for a symmetric prior over document-topic distribution,\n                * 1D array of length equal to num_topics to denote an asymmetric user defined prior for each topic.\n\n            Alternatively default prior selecting strategies can be employed by supplying a string:\n                * 'symmetric': (default) Uses a fixed symmetric prior of `1.0 / num_topics`,\n                * 'asymmetric': Uses a fixed normalized asymmetric prior of `1.0 / (topic_index + sqrt(num_topics))`,\n                * 'auto': Learns an asymmetric prior from the corpus (not available if `distributed==True`).\n\n            A-priori belief on topic-word distribution. If `name` == 'eta' then the prior can be:\n                * scalar for a symmetric prior over topic-word distribution,\n                * 1D array of length equal to num_words to denote an asymmetric user defined prior for each word,\n                * matrix of shape (num_topics, num_words) to assign a probability for each word-topic combination.\n\n            Alternatively default prior selecting strategies can be employed by supplying a string:\n                * 'symmetric': (default) Uses a fixed symmetric prior of `1.0 / num_topics`,\n                * 'auto': Learns an asymmetric prior from the corpus.\n        name : {'alpha', 'eta'}\n            Whether the `prior` is parameterized by the alpha vector (1 parameter per topic)\n            or by the eta (1 parameter per unique term in the vocabulary).\n\n        Returns\n        -------\n        init_prior: numpy.ndarray\n            Initialized Dirichlet prior:\n            If 'alpha' was provided as `name` the shape is (self.num_topics, ).\n            If 'eta' was provided as `name` the shape is (len(self.id2word), ).\n        is_auto: bool\n            Flag that shows if hyperparameter optimization should be used or not.\n        "
        if prior is None:
            prior = 'symmetric'
        if name == 'alpha':
            prior_shape = self.num_topics
        elif name == 'eta':
            prior_shape = self.num_terms
        else:
            raise ValueError("'name' must be 'alpha' or 'eta'")
        is_auto = False
        if isinstance(prior, str):
            if prior == 'symmetric':
                logger.info('using symmetric %s at %s', name, 1.0 / self.num_topics)
                init_prior = np.fromiter((1.0 / self.num_topics for i in range(prior_shape)), dtype=self.dtype, count=prior_shape)
            elif prior == 'asymmetric':
                if name == 'eta':
                    raise ValueError("The 'asymmetric' option cannot be used for eta")
                init_prior = np.fromiter((1.0 / (i + np.sqrt(prior_shape)) for i in range(prior_shape)), dtype=self.dtype, count=prior_shape)
                init_prior /= init_prior.sum()
                logger.info('using asymmetric %s %s', name, list(init_prior))
            elif prior == 'auto':
                is_auto = True
                init_prior = np.fromiter((1.0 / self.num_topics for i in range(prior_shape)), dtype=self.dtype, count=prior_shape)
                if name == 'alpha':
                    logger.info('using autotuned %s, starting with %s', name, list(init_prior))
            else:
                raise ValueError("Unable to determine proper %s value given '%s'" % (name, prior))
        elif isinstance(prior, list):
            init_prior = np.asarray(prior, dtype=self.dtype)
        elif isinstance(prior, np.ndarray):
            init_prior = prior.astype(self.dtype, copy=False)
        elif isinstance(prior, (np.number, numbers.Real)):
            init_prior = np.fromiter((prior for i in range(prior_shape)), dtype=self.dtype)
        else:
            raise ValueError('%s must be either a np array of scalars, list of scalars, or scalar' % name)
        return (init_prior, is_auto)

    def __str__(self):
        if False:
            return 10
        'Get a string representation of the current object.\n\n        Returns\n        -------\n        str\n            Human readable representation of the most important model parameters.\n\n        '
        return '%s<num_terms=%s, num_topics=%s, decay=%s, chunksize=%s>' % (self.__class__.__name__, self.num_terms, self.num_topics, self.decay, self.chunksize)

    def sync_state(self, current_Elogbeta=None):
        if False:
            for i in range(10):
                print('nop')
        "Propagate the states topic probabilities to the inner object's attribute.\n\n        Parameters\n        ----------\n        current_Elogbeta: numpy.ndarray\n            Posterior probabilities for each topic, optional.\n            If omitted, it will get Elogbeta from state.\n\n        "
        if current_Elogbeta is None:
            current_Elogbeta = self.state.get_Elogbeta()
        self.expElogbeta = np.exp(current_Elogbeta)
        assert self.expElogbeta.dtype == self.dtype

    def clear(self):
        if False:
            while True:
                i = 10
        "Clear the model's state to free some memory. Used in the distributed implementation."
        self.state = None
        self.Elogbeta = None

    def inference(self, chunk, collect_sstats=False):
        if False:
            return 10
        'Given a chunk of sparse document vectors, estimate gamma (parameters controlling the topic weights)\n        for each document in the chunk.\n\n        This function does not modify the model. The whole input chunk of document is assumed to fit in RAM;\n        chunking of a large corpus must be done earlier in the pipeline. Avoids computing the `phi` variational\n        parameter directly using the optimization presented in\n        `Lee, Seung: Algorithms for non-negative matrix factorization"\n        <https://papers.nips.cc/paper/1861-algorithms-for-non-negative-matrix-factorization.pdf>`_.\n\n        Parameters\n        ----------\n        chunk : list of list of (int, float)\n            The corpus chunk on which the inference step will be performed.\n        collect_sstats : bool, optional\n            If set to True, also collect (and return) sufficient statistics needed to update the model\'s topic-word\n            distributions.\n\n        Returns\n        -------\n        (numpy.ndarray, {numpy.ndarray, None})\n            The first element is always returned and it corresponds to the states gamma matrix. The second element is\n            only returned if `collect_sstats` == True and corresponds to the sufficient statistics for the M step.\n\n        '
        try:
            len(chunk)
        except TypeError:
            chunk = list(chunk)
        if len(chunk) > 1:
            logger.debug('performing inference on a chunk of %i documents', len(chunk))
        gamma = self.random_state.gamma(100.0, 1.0 / 100.0, (len(chunk), self.num_topics)).astype(self.dtype, copy=False)
        Elogtheta = dirichlet_expectation(gamma)
        expElogtheta = np.exp(Elogtheta)
        assert Elogtheta.dtype == self.dtype
        assert expElogtheta.dtype == self.dtype
        if collect_sstats:
            sstats = np.zeros_like(self.expElogbeta, dtype=self.dtype)
        else:
            sstats = None
        converged = 0
        integer_types = (int, np.integer)
        epsilon = np.finfo(self.dtype).eps
        for (d, doc) in enumerate(chunk):
            if len(doc) > 0 and (not isinstance(doc[0][0], integer_types)):
                ids = [int(idx) for (idx, _) in doc]
            else:
                ids = [idx for (idx, _) in doc]
            cts = np.fromiter((cnt for (_, cnt) in doc), dtype=self.dtype, count=len(doc))
            gammad = gamma[d, :]
            Elogthetad = Elogtheta[d, :]
            expElogthetad = expElogtheta[d, :]
            expElogbetad = self.expElogbeta[:, ids]
            phinorm = np.dot(expElogthetad, expElogbetad) + epsilon
            for _ in range(self.iterations):
                lastgamma = gammad
                gammad = self.alpha + expElogthetad * np.dot(cts / phinorm, expElogbetad.T)
                Elogthetad = dirichlet_expectation(gammad)
                expElogthetad = np.exp(Elogthetad)
                phinorm = np.dot(expElogthetad, expElogbetad) + epsilon
                meanchange = mean_absolute_difference(gammad, lastgamma)
                if meanchange < self.gamma_threshold:
                    converged += 1
                    break
            gamma[d, :] = gammad
            assert gammad.dtype == self.dtype
            if collect_sstats:
                sstats[:, ids] += np.outer(expElogthetad.T, cts / phinorm)
        if len(chunk) > 1:
            logger.debug('%i/%i documents converged within %i iterations', converged, len(chunk), self.iterations)
        if collect_sstats:
            sstats *= self.expElogbeta
            assert sstats.dtype == self.dtype
        assert gamma.dtype == self.dtype
        return (gamma, sstats)

    def do_estep(self, chunk, state=None):
        if False:
            for i in range(10):
                print('nop')
        'Perform inference on a chunk of documents, and accumulate the collected sufficient statistics.\n\n        Parameters\n        ----------\n        chunk : list of list of (int, float)\n            The corpus chunk on which the inference step will be performed.\n        state : :class:`~gensim.models.ldamodel.LdaState`, optional\n            The state to be updated with the newly accumulated sufficient statistics. If none, the models\n            `self.state` is updated.\n\n        Returns\n        -------\n        numpy.ndarray\n            Gamma parameters controlling the topic weights, shape (`len(chunk)`, `self.num_topics`).\n\n        '
        if state is None:
            state = self.state
        (gamma, sstats) = self.inference(chunk, collect_sstats=True)
        state.sstats += sstats
        state.numdocs += gamma.shape[0]
        assert gamma.dtype == self.dtype
        return gamma

    def update_alpha(self, gammat, rho):
        if False:
            print('Hello World!')
        'Update parameters for the Dirichlet prior on the per-document topic weights.\n\n        Parameters\n        ----------\n        gammat : numpy.ndarray\n            Previous topic weight parameters.\n        rho : float\n            Learning rate.\n\n        Returns\n        -------\n        numpy.ndarray\n            Sequence of alpha parameters.\n\n        '
        N = float(len(gammat))
        logphat = sum((dirichlet_expectation(gamma) for gamma in gammat)) / N
        assert logphat.dtype == self.dtype
        self.alpha = update_dir_prior(self.alpha, N, logphat, rho)
        logger.info('optimized alpha %s', list(self.alpha))
        assert self.alpha.dtype == self.dtype
        return self.alpha

    def update_eta(self, lambdat, rho):
        if False:
            print('Hello World!')
        'Update parameters for the Dirichlet prior on the per-topic word weights.\n\n        Parameters\n        ----------\n        lambdat : numpy.ndarray\n            Previous lambda parameters.\n        rho : float\n            Learning rate.\n\n        Returns\n        -------\n        numpy.ndarray\n            The updated eta parameters.\n\n        '
        N = float(lambdat.shape[0])
        logphat = (sum((dirichlet_expectation(lambda_) for lambda_ in lambdat)) / N).reshape((self.num_terms,))
        assert logphat.dtype == self.dtype
        self.eta = update_dir_prior(self.eta, N, logphat, rho)
        assert self.eta.dtype == self.dtype
        return self.eta

    def log_perplexity(self, chunk, total_docs=None):
        if False:
            i = 10
            return i + 15
        'Calculate and return per-word likelihood bound, using a chunk of documents as evaluation corpus.\n\n        Also output the calculated statistics, including the perplexity=2^(-bound), to log at INFO level.\n\n        Parameters\n        ----------\n        chunk : list of list of (int, float)\n            The corpus chunk on which the inference step will be performed.\n        total_docs : int, optional\n            Number of docs used for evaluation of the perplexity.\n\n        Returns\n        -------\n        numpy.ndarray\n            The variational bound score calculated for each word.\n\n        '
        if total_docs is None:
            total_docs = len(chunk)
        corpus_words = sum((cnt for document in chunk for (_, cnt) in document))
        subsample_ratio = 1.0 * total_docs / len(chunk)
        perwordbound = self.bound(chunk, subsample_ratio=subsample_ratio) / (subsample_ratio * corpus_words)
        logger.info('%.3f per-word bound, %.1f perplexity estimate based on a held-out corpus of %i documents with %i words', perwordbound, np.exp2(-perwordbound), len(chunk), corpus_words)
        return perwordbound

    def update(self, corpus, chunksize=None, decay=None, offset=None, passes=None, update_every=None, eval_every=None, iterations=None, gamma_threshold=None, chunks_as_numpy=False):
        if False:
            i = 10
            return i + 15
        "Train the model with new documents, by EM-iterating over the corpus until the topics converge, or until\n        the maximum number of allowed iterations is reached. `corpus` must be an iterable.\n\n        In distributed mode, the E step is distributed over a cluster of machines.\n\n        Notes\n        -----\n        This update also supports updating an already trained model (`self`) with new documents from `corpus`;\n        the two models are then merged in proportion to the number of old vs. new documents.\n        This feature is still experimental for non-stationary input streams.\n\n        For stationary input (no topic drift in new documents), on the other hand,\n        this equals the online update of `'Online Learning for LDA' by Hoffman et al.`_\n        and is guaranteed to converge for any `decay` in (0.5, 1].\n        Additionally, for smaller corpus sizes,\n        an increasing `offset` may be beneficial (see Table 1 in the same paper).\n\n        Parameters\n        ----------\n        corpus : iterable of list of (int, float), optional\n            Stream of document vectors or sparse matrix of shape (`num_documents`, `num_terms`) used to update the\n            model.\n        chunksize :  int, optional\n            Number of documents to be used in each training chunk.\n        decay : float, optional\n            A number between (0.5, 1] to weight what percentage of the previous lambda value is forgotten\n            when each new document is examined. Corresponds to :math:`\\kappa` from\n            `'Online Learning for LDA' by Hoffman et al.`_\n        offset : float, optional\n            Hyper-parameter that controls how much we will slow down the first steps the first few iterations.\n            Corresponds to :math:`\\tau_0` from `'Online Learning for LDA' by Hoffman et al.`_\n        passes : int, optional\n            Number of passes through the corpus during training.\n        update_every : int, optional\n            Number of documents to be iterated through for each update.\n            Set to 0 for batch learning, > 1 for online iterative learning.\n        eval_every : int, optional\n            Log perplexity is estimated every that many updates. Setting this to one slows down training by ~2x.\n        iterations : int, optional\n            Maximum number of iterations through the corpus when inferring the topic distribution of a corpus.\n        gamma_threshold : float, optional\n            Minimum change in the value of the gamma parameters to continue iterating.\n        chunks_as_numpy : bool, optional\n            Whether each chunk passed to the inference step should be a numpy.ndarray or not. Numpy can in some settings\n            turn the term IDs into floats, these will be converted back into integers in inference, which incurs a\n            performance hit. For distributed computing it may be desirable to keep the chunks as `numpy.ndarray`.\n\n        "
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
        try:
            lencorpus = len(corpus)
        except Exception:
            logger.warning('input corpus stream has no len(); counting documents')
            lencorpus = sum((1 for _ in corpus))
        if lencorpus == 0:
            logger.warning('LdaModel.update() called with an empty corpus')
            return
        if chunksize is None:
            chunksize = min(lencorpus, self.chunksize)
        self.state.numdocs += lencorpus
        if update_every:
            updatetype = 'online'
            if passes == 1:
                updatetype += ' (single-pass)'
            else:
                updatetype += ' (multi-pass)'
            updateafter = min(lencorpus, update_every * self.numworkers * chunksize)
        else:
            updatetype = 'batch'
            updateafter = lencorpus
        evalafter = min(lencorpus, (eval_every or 0) * self.numworkers * chunksize)
        updates_per_pass = max(1, lencorpus / updateafter)
        logger.info('running %s LDA training, %s topics, %i passes over the supplied corpus of %i documents, updating model once every %i documents, evaluating perplexity every %i documents, iterating %ix with a convergence threshold of %f', updatetype, self.num_topics, passes, lencorpus, updateafter, evalafter, iterations, gamma_threshold)
        if updates_per_pass * passes < 10:
            logger.warning('too few updates, training might not converge; consider increasing the number of passes or iterations to improve accuracy')

        def rho():
            if False:
                for i in range(10):
                    print('nop')
            return pow(offset + pass_ + self.num_updates / chunksize, -decay)
        if self.callbacks:
            callback = Callback(self.callbacks)
            callback.set_model(self)
            self.metrics = defaultdict(list)
        for pass_ in range(passes):
            if self.dispatcher:
                logger.info('initializing %s workers', self.numworkers)
                self.dispatcher.reset(self.state)
            else:
                other = LdaState(self.eta, self.state.sstats.shape, self.dtype)
            dirty = False
            reallen = 0
            chunks = utils.grouper(corpus, chunksize, as_numpy=chunks_as_numpy, dtype=self.dtype)
            for (chunk_no, chunk) in enumerate(chunks):
                reallen += len(chunk)
                if eval_every and (reallen == lencorpus or (chunk_no + 1) % (eval_every * self.numworkers) == 0):
                    self.log_perplexity(chunk, total_docs=lencorpus)
                if self.dispatcher:
                    logger.info('PROGRESS: pass %i, dispatching documents up to #%i/%i', pass_, chunk_no * chunksize + len(chunk), lencorpus)
                    self.dispatcher.putjob(chunk)
                else:
                    logger.info('PROGRESS: pass %i, at document #%i/%i', pass_, chunk_no * chunksize + len(chunk), lencorpus)
                    gammat = self.do_estep(chunk, other)
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
                        other = LdaState(self.eta, self.state.sstats.shape, self.dtype)
                    dirty = False
            if reallen != lencorpus:
                raise RuntimeError("input corpus size changed during training (don't use generators as input)")
            if self.callbacks:
                current_metrics = callback.on_epoch_end(pass_)
                for (metric, value) in current_metrics.items():
                    self.metrics[metric].append(value)
            if dirty:
                if self.dispatcher:
                    logger.info('reached the end of input; now waiting for all remaining jobs to finish')
                    other = self.dispatcher.getstate()
                self.do_mstep(rho(), other, pass_ > 0)
                del other
                dirty = False

    def do_mstep(self, rho, other, extra_pass=False):
        if False:
            i = 10
            return i + 15
        'Maximization step: use linear interpolation between the existing topics and\n        collected sufficient statistics in `other` to update the topics.\n\n        Parameters\n        ----------\n        rho : float\n            Learning rate.\n        other : :class:`~gensim.models.ldamodel.LdaModel`\n            The model whose sufficient statistics will be used to update the topics.\n        extra_pass : bool, optional\n            Whether this step required an additional pass over the corpus.\n\n        '
        logger.debug('updating topics')
        previous_Elogbeta = self.state.get_Elogbeta()
        self.state.blend(rho, other)
        current_Elogbeta = self.state.get_Elogbeta()
        self.sync_state(current_Elogbeta)
        self.print_topics(5)
        diff = mean_absolute_difference(previous_Elogbeta.ravel(), current_Elogbeta.ravel())
        logger.info('topic diff=%f, rho=%f', diff, rho)
        if self.optimize_eta:
            self.update_eta(self.state.get_lambda(), rho)
        if not extra_pass:
            self.num_updates += other.numdocs

    def bound(self, corpus, gamma=None, subsample_ratio=1.0):
        if False:
            for i in range(10):
                print('nop')
        'Estimate the variational bound of documents from the corpus as E_q[log p(corpus)] - E_q[log q(corpus)].\n\n        Parameters\n        ----------\n        corpus : iterable of list of (int, float), optional\n            Stream of document vectors or sparse matrix of shape (`num_documents`, `num_terms`) used to estimate the\n            variational bounds.\n        gamma : numpy.ndarray, optional\n            Topic weight variational parameters for each document. If not supplied, it will be inferred from the model.\n        subsample_ratio : float, optional\n            Percentage of the whole corpus represented by the passed `corpus` argument (in case this was a sample).\n            Set to 1.0 if the whole corpus was passed.This is used as a multiplicative factor to scale the likelihood\n            appropriately.\n\n        Returns\n        -------\n        numpy.ndarray\n            The variational bound score calculated for each document.\n\n        '
        score = 0.0
        _lambda = self.state.get_lambda()
        Elogbeta = dirichlet_expectation(_lambda)
        for (d, doc) in enumerate(corpus):
            if d % self.chunksize == 0:
                logger.debug('bound: at document #%i', d)
            if gamma is None:
                (gammad, _) = self.inference([doc])
            else:
                gammad = gamma[d]
            Elogthetad = dirichlet_expectation(gammad)
            assert gammad.dtype == self.dtype
            assert Elogthetad.dtype == self.dtype
            score += sum((cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for (id, cnt) in doc))
            score += np.sum((self.alpha - gammad) * Elogthetad)
            score += np.sum(gammaln(gammad) - gammaln(self.alpha))
            score += gammaln(np.sum(self.alpha)) - gammaln(np.sum(gammad))
        score *= subsample_ratio
        score += np.sum((self.eta - _lambda) * Elogbeta)
        score += np.sum(gammaln(_lambda) - gammaln(self.eta))
        if np.ndim(self.eta) == 0:
            sum_eta = self.eta * self.num_terms
        else:
            sum_eta = np.sum(self.eta)
        score += np.sum(gammaln(sum_eta) - gammaln(np.sum(_lambda, 1)))
        return score

    def show_topics(self, num_topics=10, num_words=10, log=False, formatted=True):
        if False:
            i = 10
            return i + 15
        'Get a representation for selected topics.\n\n        Parameters\n        ----------\n        num_topics : int, optional\n            Number of topics to be returned. Unlike LSA, there is no natural ordering between the topics in LDA.\n            The returned topics subset of all topics is therefore arbitrary and may change between two LDA\n            training runs.\n        num_words : int, optional\n            Number of words to be presented for each topic. These will be the most relevant words (assigned the highest\n            probability for each topic).\n        log : bool, optional\n            Whether the output is also logged, besides being returned.\n        formatted : bool, optional\n            Whether the topic representations should be formatted as strings. If False, they are returned as\n            2 tuples of (word, probability).\n\n        Returns\n        -------\n        list of {str, tuple of (str, float)}\n            a list of topics, each represented either as a string (when `formatted` == True) or word-probability\n            pairs.\n\n        '
        if num_topics < 0 or num_topics >= self.num_topics:
            num_topics = self.num_topics
            chosen_topics = range(num_topics)
        else:
            num_topics = min(num_topics, self.num_topics)
            sort_alpha = self.alpha + 0.0001 * self.random_state.rand(len(self.alpha))
            sorted_topics = list(matutils.argsort(sort_alpha))
            chosen_topics = sorted_topics[:num_topics // 2] + sorted_topics[-num_topics // 2:]
        shown = []
        topic = self.state.get_lambda()
        for i in chosen_topics:
            topic_ = topic[i]
            topic_ = topic_ / topic_.sum()
            bestn = matutils.argsort(topic_, num_words, reverse=True)
            topic_ = [(self.id2word[id], topic_[id]) for id in bestn]
            if formatted:
                topic_ = ' + '.join(('%.3f*"%s"' % (v, k) for (k, v) in topic_))
            shown.append((i, topic_))
            if log:
                logger.info('topic #%i (%.3f): %s', i, self.alpha[i], topic_)
        return shown

    def show_topic(self, topicid, topn=10):
        if False:
            return 10
        'Get the representation for a single topic. Words here are the actual strings, in constrast to\n        :meth:`~gensim.models.ldamodel.LdaModel.get_topic_terms` that represents words by their vocabulary ID.\n\n        Parameters\n        ----------\n        topicid : int\n            The ID of the topic to be returned\n        topn : int, optional\n            Number of the most significant words that are associated with the topic.\n\n        Returns\n        -------\n        list of (str, float)\n            Word - probability pairs for the most relevant words generated by the topic.\n\n        '
        return [(self.id2word[id], value) for (id, value) in self.get_topic_terms(topicid, topn)]

    def get_topics(self):
        if False:
            return 10
        'Get the term-topic matrix learned during inference.\n\n        Returns\n        -------\n        numpy.ndarray\n            The probability for each word in each topic, shape (`num_topics`, `vocabulary_size`).\n\n        '
        topics = self.state.get_lambda()
        return topics / topics.sum(axis=1)[:, None]

    def get_topic_terms(self, topicid, topn=10):
        if False:
            while True:
                i = 10
        'Get the representation for a single topic. Words the integer IDs, in constrast to\n        :meth:`~gensim.models.ldamodel.LdaModel.show_topic` that represents words by the actual strings.\n\n        Parameters\n        ----------\n        topicid : int\n            The ID of the topic to be returned\n        topn : int, optional\n            Number of the most significant words that are associated with the topic.\n\n        Returns\n        -------\n        list of (int, float)\n            Word ID - probability pairs for the most relevant words generated by the topic.\n\n        '
        topic = self.get_topics()[topicid]
        topic = topic / topic.sum()
        bestn = matutils.argsort(topic, topn, reverse=True)
        return [(idx, topic[idx]) for idx in bestn]

    def top_topics(self, corpus=None, texts=None, dictionary=None, window_size=None, coherence='u_mass', topn=20, processes=-1):
        if False:
            print('Hello World!')
        "Get the topics with the highest coherence score the coherence for each topic.\n\n        Parameters\n        ----------\n        corpus : iterable of list of (int, float), optional\n            Corpus in BoW format.\n        texts : list of list of str, optional\n            Tokenized texts, needed for coherence models that use sliding window based (i.e. coherence=`c_something`)\n            probability estimator .\n        dictionary : :class:`~gensim.corpora.dictionary.Dictionary`, optional\n            Gensim dictionary mapping of id word to create corpus.\n            If `model.id2word` is present, this is not needed. If both are provided, passed `dictionary` will be used.\n        window_size : int, optional\n            Is the size of the window to be used for coherence measures using boolean sliding window as their\n            probability estimator. For 'u_mass' this doesn't matter.\n            If None - the default window sizes are used which are: 'c_v' - 110, 'c_uci' - 10, 'c_npmi' - 10.\n        coherence : {'u_mass', 'c_v', 'c_uci', 'c_npmi'}, optional\n            Coherence measure to be used.\n            Fastest method - 'u_mass', 'c_uci' also known as `c_pmi`.\n            For 'u_mass' corpus should be provided, if texts is provided, it will be converted to corpus\n            using the dictionary. For 'c_v', 'c_uci' and 'c_npmi' `texts` should be provided (`corpus` isn't needed)\n        topn : int, optional\n            Integer corresponding to the number of top words to be extracted from each topic.\n        processes : int, optional\n            Number of processes to use for probability estimation phase, any value less than 1 will be interpreted as\n            num_cpus - 1.\n\n        Returns\n        -------\n        list of (list of (int, str), float)\n            Each element in the list is a pair of a topic representation and its coherence score. Topic representations\n            are distributions of words, represented as a list of pairs of word IDs and their probabilities.\n\n        "
        cm = CoherenceModel(model=self, corpus=corpus, texts=texts, dictionary=dictionary, window_size=window_size, coherence=coherence, topn=topn, processes=processes)
        coherence_scores = cm.get_coherence_per_topic()
        str_topics = []
        for topic in self.get_topics():
            bestn = matutils.argsort(topic, topn=topn, reverse=True)
            beststr = [(topic[_id], self.id2word[_id]) for _id in bestn]
            str_topics.append(beststr)
        scored_topics = zip(str_topics, coherence_scores)
        return sorted(scored_topics, key=lambda tup: tup[1], reverse=True)

    def get_document_topics(self, bow, minimum_probability=None, minimum_phi_value=None, per_word_topics=False):
        if False:
            i = 10
            return i + 15
        'Get the topic distribution for the given document.\n\n        Parameters\n        ----------\n        bow : corpus : list of (int, float)\n            The document in BOW format.\n        minimum_probability : float\n            Topics with an assigned probability lower than this threshold will be discarded.\n        minimum_phi_value : float\n            If `per_word_topics` is True, this represents a lower bound on the term probabilities that are included.\n             If set to None, a value of 1e-8 is used to prevent 0s.\n        per_word_topics : bool\n            If True, this function will also return two extra lists as explained in the "Returns" section.\n\n        Returns\n        -------\n        list of (int, float)\n            Topic distribution for the whole document. Each element in the list is a pair of a topic\'s id, and\n            the probability that was assigned to it.\n        list of (int, list of (int, float), optional\n            Most probable topics per word. Each element in the list is a pair of a word\'s id, and a list of\n            topics sorted by their relevance to this word. Only returned if `per_word_topics` was set to True.\n        list of (int, list of float), optional\n            Phi relevance values, multiplied by the feature length, for each word-topic combination.\n            Each element in the list is a pair of a word\'s id and a list of the phi values between this word and\n            each topic. Only returned if `per_word_topics` was set to True.\n\n        '
        if minimum_probability is None:
            minimum_probability = self.minimum_probability
        minimum_probability = max(minimum_probability, 1e-08)
        if minimum_phi_value is None:
            minimum_phi_value = self.minimum_probability
        minimum_phi_value = max(minimum_phi_value, 1e-08)
        (is_corpus, corpus) = utils.is_corpus(bow)
        if is_corpus:
            kwargs = dict(per_word_topics=per_word_topics, minimum_probability=minimum_probability, minimum_phi_value=minimum_phi_value)
            return self._apply(corpus, **kwargs)
        (gamma, phis) = self.inference([bow], collect_sstats=per_word_topics)
        topic_dist = gamma[0] / sum(gamma[0])
        document_topics = [(topicid, topicvalue) for (topicid, topicvalue) in enumerate(topic_dist) if topicvalue >= minimum_probability]
        if not per_word_topics:
            return document_topics
        word_topic = []
        word_phi = []
        for (word_type, weight) in bow:
            phi_values = []
            phi_topic = []
            for topic_id in range(0, self.num_topics):
                if phis[topic_id][word_type] >= minimum_phi_value:
                    phi_values.append((phis[topic_id][word_type], topic_id))
                    phi_topic.append((topic_id, phis[topic_id][word_type]))
            word_phi.append((word_type, phi_topic))
            sorted_phi_values = sorted(phi_values, reverse=True)
            topics_sorted = [x[1] for x in sorted_phi_values]
            word_topic.append((word_type, topics_sorted))
        return (document_topics, word_topic, word_phi)

    def get_term_topics(self, word_id, minimum_probability=None):
        if False:
            return 10
        'Get the most relevant topics to the given word.\n\n        Parameters\n        ----------\n        word_id : int\n            The word for which the topic distribution will be computed.\n        minimum_probability : float, optional\n            Topics with an assigned probability below this threshold will be discarded.\n\n        Returns\n        -------\n        list of (int, float)\n            The relevant topics represented as pairs of their ID and their assigned probability, sorted\n            by relevance to the given word.\n\n        '
        if minimum_probability is None:
            minimum_probability = self.minimum_probability
        minimum_probability = max(minimum_probability, 1e-08)
        if isinstance(word_id, str):
            word_id = self.id2word.doc2bow([word_id])[0][0]
        values = []
        for topic_id in range(0, self.num_topics):
            if self.expElogbeta[topic_id][word_id] >= minimum_probability:
                values.append((topic_id, self.expElogbeta[topic_id][word_id]))
        return values

    def diff(self, other, distance='kullback_leibler', num_words=100, n_ann_terms=10, diagonal=False, annotation=True, normed=True):
        if False:
            for i in range(10):
                print('nop')
        'Calculate the difference in topic distributions between two models: `self` and `other`.\n\n        Parameters\n        ----------\n        other : :class:`~gensim.models.ldamodel.LdaModel`\n            The model which will be compared against the current object.\n        distance : {\'kullback_leibler\', \'hellinger\', \'jaccard\', \'jensen_shannon\'}\n            The distance metric to calculate the difference with.\n        num_words : int, optional\n            The number of most relevant words used if `distance == \'jaccard\'`. Also used for annotating topics.\n        n_ann_terms : int, optional\n            Max number of words in intersection/symmetric difference between topics. Used for annotation.\n        diagonal : bool, optional\n            Whether we need the difference between identical topics (the diagonal of the difference matrix).\n        annotation : bool, optional\n            Whether the intersection or difference of words between two topics should be returned.\n        normed : bool, optional\n            Whether the matrix should be normalized or not.\n\n        Returns\n        -------\n        numpy.ndarray\n            A difference matrix. Each element corresponds to the difference between the two topics,\n            shape (`self.num_topics`, `other.num_topics`)\n        numpy.ndarray, optional\n            Annotation matrix where for each pair we include the word from the intersection of the two topics,\n            and the word from the symmetric difference of the two topics. Only included if `annotation == True`.\n            Shape (`self.num_topics`, `other_model.num_topics`, 2).\n\n        Examples\n        --------\n        Get the differences between each pair of topics inferred by two models\n\n        .. sourcecode:: pycon\n\n            >>> from gensim.models.ldamulticore import LdaMulticore\n            >>> from gensim.test.utils import datapath\n            >>>\n            >>> m1 = LdaMulticore.load(datapath("lda_3_0_1_model"))\n            >>> m2 = LdaMulticore.load(datapath("ldamodel_python_3_5"))\n            >>> mdiff, annotation = m1.diff(m2)\n            >>> topic_diff = mdiff  # get matrix with difference for each topic pair from `m1` and `m2`\n\n        '
        distances = {'kullback_leibler': kullback_leibler, 'hellinger': hellinger, 'jaccard': jaccard_distance, 'jensen_shannon': jensen_shannon}
        if distance not in distances:
            valid_keys = ', '.join(('`{}`'.format(x) for x in distances.keys()))
            raise ValueError('Incorrect distance, valid only {}'.format(valid_keys))
        if not isinstance(other, self.__class__):
            raise ValueError('The parameter `other` must be of type `{}`'.format(self.__name__))
        distance_func = distances[distance]
        (d1, d2) = (self.get_topics(), other.get_topics())
        (t1_size, t2_size) = (d1.shape[0], d2.shape[0])
        annotation_terms = None
        fst_topics = [{w for (w, _) in self.show_topic(topic, topn=num_words)} for topic in range(t1_size)]
        snd_topics = [{w for (w, _) in other.show_topic(topic, topn=num_words)} for topic in range(t2_size)]
        if distance == 'jaccard':
            (d1, d2) = (fst_topics, snd_topics)
        if diagonal:
            assert t1_size == t2_size, 'Both input models should have same no. of topics, as the diagonal will only be valid in a square matrix'
            z = np.zeros(t1_size)
            if annotation:
                annotation_terms = np.zeros(t1_size, dtype=list)
        else:
            z = np.zeros((t1_size, t2_size))
            if annotation:
                annotation_terms = np.zeros((t1_size, t2_size), dtype=list)
        for topic in np.ndindex(z.shape):
            topic1 = topic[0]
            if diagonal:
                topic2 = topic1
            else:
                topic2 = topic[1]
            z[topic] = distance_func(d1[topic1], d2[topic2])
            if annotation:
                pos_tokens = fst_topics[topic1] & snd_topics[topic2]
                neg_tokens = fst_topics[topic1].symmetric_difference(snd_topics[topic2])
                pos_tokens = list(pos_tokens)[:min(len(pos_tokens), n_ann_terms)]
                neg_tokens = list(neg_tokens)[:min(len(neg_tokens), n_ann_terms)]
                annotation_terms[topic] = [pos_tokens, neg_tokens]
        if normed:
            if np.abs(np.max(z)) > 1e-08:
                z /= np.max(z)
        return (z, annotation_terms)

    def __getitem__(self, bow, eps=None):
        if False:
            for i in range(10):
                print('nop')
        "Get the topic distribution for the given document.\n\n        Wraps :meth:`~gensim.models.ldamodel.LdaModel.get_document_topics` to support an operator style call.\n        Uses the model's current state (set using constructor arguments) to fill in the additional arguments of the\n        wrapper method.\n\n        Parameters\n        ---------\n        bow : list of (int, float)\n            The document in BOW format.\n        eps : float, optional\n            Topics with an assigned probability lower than this threshold will be discarded.\n\n        Returns\n        -------\n        list of (int, float)\n            Topic distribution for the given document. Each topic is represented as a pair of its ID and the probability\n            assigned to it.\n\n        "
        return self.get_document_topics(bow, eps, self.minimum_phi_value, self.per_word_topics)

    def save(self, fname, ignore=('state', 'dispatcher'), separately=None, *args, **kwargs):
        if False:
            while True:
                i = 10
        "Save the model to a file.\n\n        Large internal arrays may be stored into separate files, with `fname` as prefix.\n\n        Notes\n        -----\n        If you intend to use models across Python 2/3 versions there are a few things to\n        keep in mind:\n\n          1. The pickled Python dictionaries will not work across Python versions\n          2. The `save` method does not automatically save all numpy arrays separately, only\n             those ones that exceed `sep_limit` set in :meth:`~gensim.utils.SaveLoad.save`. The main\n             concern here is the `alpha` array if for instance using `alpha='auto'`.\n\n        Please refer to the `wiki recipes section\n        <https://github.com/RaRe-Technologies/gensim/wiki/\n        Recipes-&-FAQ#q9-how-do-i-load-a-model-in-python-3-that-was-trained-and-saved-using-python-2>`_\n        for an example on how to work around these issues.\n\n        See Also\n        --------\n        :meth:`~gensim.models.ldamodel.LdaModel.load`\n            Load model.\n\n        Parameters\n        ----------\n        fname : str\n            Path to the system file where the model will be persisted.\n        ignore : tuple of str, optional\n            The named attributes in the tuple will be left out of the pickled model. The reason why\n            the internal `state` is ignored by default is that it uses its own serialisation rather than the one\n            provided by this method.\n        separately : {list of str, None}, optional\n            If None -  automatically detect large numpy/scipy.sparse arrays in the object being stored, and store\n            them into separate files. This avoids pickle memory errors and allows `mmap`'ing large arrays\n            back on load efficiently. If list of str - this attributes will be stored in separate files,\n            the automatic check is not performed in this case.\n        *args\n            Positional arguments propagated to :meth:`~gensim.utils.SaveLoad.save`.\n        **kwargs\n            Key word arguments propagated to :meth:`~gensim.utils.SaveLoad.save`.\n\n        "
        if self.state is not None:
            self.state.save(utils.smart_extension(fname, '.state'), *args, **kwargs)
        if 'id2word' not in ignore:
            utils.pickle(self.id2word, utils.smart_extension(fname, '.id2word'))
        if ignore is not None and ignore:
            if isinstance(ignore, str):
                ignore = [ignore]
            ignore = [e for e in ignore if e]
            ignore = list({'state', 'dispatcher', 'id2word'} | set(ignore))
        else:
            ignore = ['state', 'dispatcher', 'id2word']
        separately_explicit = ['expElogbeta', 'sstats']
        if isinstance(self.alpha, str) and self.alpha == 'auto' or (isinstance(self.alpha, np.ndarray) and len(self.alpha.shape) != 1):
            separately_explicit.append('alpha')
        if isinstance(self.eta, str) and self.eta == 'auto' or (isinstance(self.eta, np.ndarray) and len(self.eta.shape) != 1):
            separately_explicit.append('eta')
        if separately:
            if isinstance(separately, str):
                separately = [separately]
            separately = [e for e in separately if e]
            separately = list(set(separately_explicit) | set(separately))
        else:
            separately = separately_explicit
        super(LdaModel, self).save(fname, *args, ignore=ignore, separately=separately, **kwargs)

    @classmethod
    def load(cls, fname, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        'Load a previously saved :class:`gensim.models.ldamodel.LdaModel` from file.\n\n        See Also\n        --------\n        :meth:`~gensim.models.ldamodel.LdaModel.save`\n            Save model.\n\n        Parameters\n        ----------\n        fname : str\n            Path to the file where the model is stored.\n        *args\n            Positional arguments propagated to :meth:`~gensim.utils.SaveLoad.load`.\n        **kwargs\n            Key word arguments propagated to :meth:`~gensim.utils.SaveLoad.load`.\n\n        Examples\n        --------\n        Large arrays can be memmap\'ed back as read-only (shared memory) by setting `mmap=\'r\'`:\n\n        .. sourcecode:: pycon\n\n            >>> from gensim.test.utils import datapath\n            >>>\n            >>> fname = datapath("lda_3_0_1_model")\n            >>> lda = LdaModel.load(fname, mmap=\'r\')\n\n        '
        kwargs['mmap'] = kwargs.get('mmap', None)
        result = super(LdaModel, cls).load(fname, *args, **kwargs)
        if not hasattr(result, 'random_state'):
            result.random_state = utils.get_random_state(None)
            logging.warning('random_state not set so using default value')
        if not hasattr(result, 'dtype'):
            result.dtype = np.float64
            logging.info('dtype was not set in saved %s file %s, assuming np.float64', result.__class__.__name__, fname)
        state_fname = utils.smart_extension(fname, '.state')
        try:
            result.state = LdaState.load(state_fname, *args, **kwargs)
        except Exception as e:
            logging.warning('failed to load state from %s: %s', state_fname, e)
        id2word_fname = utils.smart_extension(fname, '.id2word')
        if os.path.isfile(id2word_fname):
            try:
                result.id2word = utils.unpickle(id2word_fname)
            except Exception as e:
                logging.warning('failed to load id2word dictionary from %s: %s', id2word_fname, e)
        return result