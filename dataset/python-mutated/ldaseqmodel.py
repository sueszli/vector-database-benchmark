"""Lda Sequence model, inspired by
`David M. Blei, John D. Lafferty: "Dynamic Topic Models"
<https://mimno.infosci.cornell.edu/info6150/readings/dynamic_topic_models.pdf>`_.
The original C/C++ implementation can be found on `blei-lab/dtm <https://github.com/blei-lab/dtm>`_.


TODO: The next steps to take this forward would be:

#. Include DIM mode. Most of the infrastructure for this is in place.
#. See if LdaPost can be replaced by LdaModel completely without breaking anything.
#. Heavy lifting going on in the Sslm class - efforts can be made to cythonise mathematical methods, in particular,
   update_obs and the optimization takes a lot time.
#. Try and make it distributed, especially around the E and M step.
#. Remove all C/C++ coding style/syntax.

Examples
--------

Set up a model using 9 documents, with 2 in the first time-slice, 4 in the second, and 3 in the third

.. sourcecode:: pycon

    >>> from gensim.test.utils import common_corpus
    >>> from gensim.models import LdaSeqModel
    >>>
    >>> ldaseq = LdaSeqModel(corpus=common_corpus, time_slice=[2, 4, 3], num_topics=2, chunksize=1)

Persist a model to disk and reload it later

.. sourcecode:: pycon

    >>> from gensim.test.utils import datapath
    >>>
    >>> temp_file = datapath("model")
    >>> ldaseq.save(temp_file)
    >>>
    >>> # Load a potentially pre-trained model from disk.
    >>> ldaseq = LdaSeqModel.load(temp_file)

Access the document embeddings generated from the DTM

.. sourcecode:: pycon

    >>> doc = common_corpus[1]
    >>>
    >>> embedding = ldaseq[doc]

"""
import logging
import numpy as np
from scipy.special import digamma, gammaln
from scipy import optimize
from gensim import utils, matutils
from gensim.models import ldamodel
logger = logging.getLogger(__name__)

class LdaSeqModel(utils.SaveLoad):
    """Estimate Dynamic Topic Model parameters based on a training corpus."""

    def __init__(self, corpus=None, time_slice=None, id2word=None, alphas=0.01, num_topics=10, initialize='gensim', sstats=None, lda_model=None, obs_variance=0.5, chain_variance=0.005, passes=10, random_state=None, lda_inference_max_iter=25, em_min_iter=6, em_max_iter=20, chunksize=100):
        if False:
            return 10
        '\n\n        Parameters\n        ----------\n        corpus : {iterable of list of (int, float), scipy.sparse.csc}, optional\n            Stream of document vectors or sparse matrix of shape (`num_documents`, `num_terms`).\n            If not given, the model is left untrained (presumably because you want to call\n            :meth:`~gensim.models.ldamodel.LdaSeqModel.update` manually).\n        time_slice : list of int, optional\n            Number of documents in each time-slice. Each time slice could for example represent a year\'s published\n            papers, in case the corpus comes from a journal publishing over multiple years.\n            It is assumed that `sum(time_slice) == num_documents`.\n        id2word : dict of (int, str), optional\n            Mapping from word IDs to words. It is used to determine the vocabulary size, as well as for\n            debugging and topic printing.\n        alphas : float, optional\n            The prior probability for the model.\n        num_topics : int, optional\n            The number of requested latent topics to be extracted from the training corpus.\n        initialize : {\'gensim\', \'own\', \'ldamodel\'}, optional\n            Controls the initialization of the DTM model. Supports three different modes:\n                * \'gensim\': Uses gensim\'s LDA initialization.\n                * \'own\': Uses your own initialization matrix of an LDA model that has been previously trained.\n                * \'lda_model\': Use a previously used LDA model, passing it through the `lda_model` argument.\n        sstats : numpy.ndarray , optional\n            Sufficient statistics used for initializing the model if `initialize == \'own\'`. Corresponds to matrix\n            beta in the linked paper for time slice 0, expected shape (`self.vocab_len`, `num_topics`).\n        lda_model : :class:`~gensim.models.ldamodel.LdaModel`\n            Model whose sufficient statistics will be used to initialize the current object if `initialize == \'gensim\'`.\n        obs_variance : float, optional\n            Observed variance used to approximate the true and forward variance as shown in\n            `David M. Blei, John D. Lafferty: "Dynamic Topic Models"\n            <https://mimno.infosci.cornell.edu/info6150/readings/dynamic_topic_models.pdf>`_.\n        chain_variance : float, optional\n            Gaussian parameter defined in the beta distribution to dictate how the beta values evolve over time.\n        passes : int, optional\n            Number of passes over the corpus for the initial :class:`~gensim.models.ldamodel.LdaModel`\n        random_state : {numpy.random.RandomState, int}, optional\n            Can be a np.random.RandomState object, or the seed to generate one. Used for reproducibility of results.\n        lda_inference_max_iter : int, optional\n            Maximum number of iterations in the inference step of the LDA training.\n        em_min_iter : int, optional\n            Minimum number of iterations until converge of the Expectation-Maximization algorithm\n        em_max_iter : int, optional\n            Maximum number of iterations until converge of the Expectation-Maximization algorithm.\n        chunksize : int, optional\n            Number of documents in the corpus do be processed in in a chunk.\n\n        '
        self.id2word = id2word
        if corpus is None and self.id2word is None:
            raise ValueError('at least one of corpus/id2word must be specified, to establish input space dimensionality')
        if self.id2word is None:
            logger.warning('no word id mapping provided; initializing from corpus, assuming identity')
            self.id2word = utils.dict_from_corpus(corpus)
            self.vocab_len = len(self.id2word)
        elif self.id2word:
            self.vocab_len = len(self.id2word)
        else:
            self.vocab_len = 0
        if corpus is not None:
            try:
                self.corpus_len = len(corpus)
            except TypeError:
                logger.warning('input corpus stream has no len(); counting documents')
                self.corpus_len = sum((1 for _ in corpus))
        self.time_slice = time_slice
        if self.time_slice is not None:
            self.num_time_slices = len(time_slice)
        self.num_topics = num_topics
        self.num_time_slices = len(time_slice)
        self.alphas = np.full(num_topics, alphas)
        self.topic_chains = []
        for topic in range(num_topics):
            sslm_ = sslm(num_time_slices=self.num_time_slices, vocab_len=self.vocab_len, num_topics=self.num_topics, chain_variance=chain_variance, obs_variance=obs_variance)
            self.topic_chains.append(sslm_)
        self.top_doc_phis = None
        self.influence = None
        self.renormalized_influence = None
        self.influence_sum_lgl = None
        if corpus is not None and time_slice is not None:
            self.max_doc_len = max((len(line) for line in corpus))
            if initialize == 'gensim':
                lda_model = ldamodel.LdaModel(corpus, id2word=self.id2word, num_topics=self.num_topics, passes=passes, alpha=self.alphas, random_state=random_state, dtype=np.float64)
                self.sstats = np.transpose(lda_model.state.sstats)
            if initialize == 'ldamodel':
                self.sstats = np.transpose(lda_model.state.sstats)
            if initialize == 'own':
                self.sstats = sstats
            self.init_ldaseq_ss(chain_variance, obs_variance, self.alphas, self.sstats)
            self.fit_lda_seq(corpus, lda_inference_max_iter, em_min_iter, em_max_iter, chunksize)

    def init_ldaseq_ss(self, topic_chain_variance, topic_obs_variance, alpha, init_suffstats):
        if False:
            while True:
                i = 10
        'Initialize State Space Language Model, topic-wise.\n\n        Parameters\n        ----------\n        topic_chain_variance : float\n            Gaussian parameter defined in the beta distribution to dictate how the beta values evolve.\n        topic_obs_variance : float\n            Observed variance used to approximate the true and forward variance as shown in\n            `David M. Blei, John D. Lafferty: "Dynamic Topic Models"\n            <https://mimno.infosci.cornell.edu/info6150/readings/dynamic_topic_models.pdf>`_.\n        alpha : float\n            The prior probability for the model.\n        init_suffstats : numpy.ndarray\n            Sufficient statistics used for initializing the model, expected shape (`self.vocab_len`, `num_topics`).\n\n        '
        self.alphas = alpha
        for (k, chain) in enumerate(self.topic_chains):
            sstats = init_suffstats[:, k]
            sslm.sslm_counts_init(chain, topic_obs_variance, topic_chain_variance, sstats)

    def fit_lda_seq(self, corpus, lda_inference_max_iter, em_min_iter, em_max_iter, chunksize):
        if False:
            return 10
        'Fit a LDA Sequence model (DTM).\n\n        This method will iteratively setup LDA models and perform EM steps until the sufficient statistics convergence,\n        or until the maximum number of iterations is reached. Because the true posterior is intractable, an\n        appropriately tight lower bound must be used instead. This function will optimize this bound, by minimizing\n        its true Kullback-Liebler Divergence with the true posterior.\n\n        Parameters\n        ----------\n        corpus : {iterable of list of (int, float), scipy.sparse.csc}\n            Stream of document vectors or sparse matrix of shape (`num_documents`, `num_terms`).\n        lda_inference_max_iter : int\n            Maximum number of iterations for the inference step of LDA.\n        em_min_iter : int\n            Minimum number of time slices to be inspected.\n        em_max_iter : int\n            Maximum number of time slices to be inspected.\n        chunksize : int\n            Number of documents to be processed in each chunk.\n\n        Returns\n        -------\n        float\n            The highest lower bound for the true posterior produced after all iterations.\n\n       '
        LDASQE_EM_THRESHOLD = 0.0001
        LOWER_ITER = 10
        ITER_MULT_LOW = 2
        MAX_ITER = 500
        num_topics = self.num_topics
        vocab_len = self.vocab_len
        data_len = self.num_time_slices
        corpus_len = self.corpus_len
        bound = 0
        convergence = LDASQE_EM_THRESHOLD + 1
        iter_ = 0
        while iter_ < em_min_iter or (convergence > LDASQE_EM_THRESHOLD and iter_ <= em_max_iter):
            logger.info(' EM iter %i', iter_)
            logger.info('E Step')
            old_bound = bound
            topic_suffstats = []
            for topic in range(num_topics):
                topic_suffstats.append(np.zeros((vocab_len, data_len)))
            gammas = np.zeros((corpus_len, num_topics))
            lhoods = np.zeros((corpus_len, num_topics + 1))
            (bound, gammas) = self.lda_seq_infer(corpus, topic_suffstats, gammas, lhoods, iter_, lda_inference_max_iter, chunksize)
            self.gammas = gammas
            logger.info('M Step')
            topic_bound = self.fit_lda_seq_topics(topic_suffstats)
            bound += topic_bound
            if bound - old_bound < 0:
                if lda_inference_max_iter < LOWER_ITER:
                    lda_inference_max_iter *= ITER_MULT_LOW
                logger.info('Bound went down, increasing iterations to %i', lda_inference_max_iter)
            convergence = np.fabs((bound - old_bound) / old_bound)
            if convergence < LDASQE_EM_THRESHOLD:
                lda_inference_max_iter = MAX_ITER
                logger.info('Starting final iterations, max iter is %i', lda_inference_max_iter)
                convergence = 1.0
            logger.info('iteration %i iteration lda seq bound is %f convergence is %f', iter_, bound, convergence)
            iter_ += 1
        return bound

    def lda_seq_infer(self, corpus, topic_suffstats, gammas, lhoods, iter_, lda_inference_max_iter, chunksize):
        if False:
            for i in range(10):
                print('nop')
        "Inference (or E-step) for the lower bound EM optimization.\n\n        This is used to set up the gensim :class:`~gensim.models.ldamodel.LdaModel` to be used for each time-slice.\n        It also allows for Document Influence Model code to be written in.\n\n        Parameters\n        ----------\n        corpus : {iterable of list of (int, float), scipy.sparse.csc}\n            Stream of document vectors or sparse matrix of shape (`num_documents`, `num_terms`).\n        topic_suffstats : numpy.ndarray\n            Sufficient statistics for time slice 0, used for initializing the model if `initialize == 'own'`,\n            expected shape (`self.vocab_len`, `num_topics`).\n        gammas : numpy.ndarray\n            Topic weight variational parameters for each document. If not supplied, it will be inferred from the model.\n        lhoods : list of float\n            The total log probability lower bound for each topic. Corresponds to the phi variational parameters in the\n            linked paper.\n        iter_ : int\n            Current iteration.\n        lda_inference_max_iter : int\n            Maximum number of iterations for the inference step of LDA.\n        chunksize : int\n            Number of documents to be processed in each chunk.\n\n        Returns\n        -------\n        (float, list of float)\n            The first value is the highest lower bound for the true posterior.\n            The second value is the list of optimized dirichlet variational parameters for the approximation of\n            the posterior.\n\n        "
        num_topics = self.num_topics
        vocab_len = self.vocab_len
        bound = 0.0
        lda = ldamodel.LdaModel(num_topics=num_topics, alpha=self.alphas, id2word=self.id2word, dtype=np.float64)
        lda.topics = np.zeros((vocab_len, num_topics))
        ldapost = LdaPost(max_doc_len=self.max_doc_len, num_topics=num_topics, lda=lda)
        model = 'DTM'
        if model == 'DTM':
            (bound, gammas) = self.inferDTMseq(corpus, topic_suffstats, gammas, lhoods, lda, ldapost, iter_, bound, lda_inference_max_iter, chunksize)
        elif model == 'DIM':
            self.InfluenceTotalFixed(corpus)
            (bound, gammas) = self.inferDIMseq(corpus, topic_suffstats, gammas, lhoods, lda, ldapost, iter_, bound, lda_inference_max_iter, chunksize)
        return (bound, gammas)

    def inferDTMseq(self, corpus, topic_suffstats, gammas, lhoods, lda, ldapost, iter_, bound, lda_inference_max_iter, chunksize):
        if False:
            return 10
        'Compute the likelihood of a sequential corpus under an LDA seq model, and reports the likelihood bound.\n\n        Parameters\n        ----------\n        corpus : {iterable of list of (int, float), scipy.sparse.csc}\n            Stream of document vectors or sparse matrix of shape (`num_documents`, `num_terms`).\n        topic_suffstats : numpy.ndarray\n            Sufficient statistics of the current model, expected shape (`self.vocab_len`, `num_topics`).\n        gammas : numpy.ndarray\n            Topic weight variational parameters for each document. If not supplied, it will be inferred from the model.\n        lhoods : list of float of length `self.num_topics`\n            The total log probability bound for each topic. Corresponds to phi from the linked paper.\n        lda : :class:`~gensim.models.ldamodel.LdaModel`\n            The trained LDA model of the previous iteration.\n        ldapost : :class:`~gensim.models.ldaseqmodel.LdaPost`\n            Posterior probability variables for the given LDA model. This will be used as the true (but intractable)\n            posterior.\n        iter_ : int\n            The current iteration.\n        bound : float\n            The LDA bound produced after all iterations.\n        lda_inference_max_iter : int\n            Maximum number of iterations for the inference step of LDA.\n        chunksize : int\n            Number of documents to be processed in each chunk.\n\n        Returns\n        -------\n        (float, list of float)\n            The first value is the highest lower bound for the true posterior.\n            The second value is the list of optimized dirichlet variational parameters for the approximation of\n            the posterior.\n\n        '
        doc_index = 0
        time = 0
        doc_num = 0
        lda = self.make_lda_seq_slice(lda, time)
        time_slice = np.cumsum(np.array(self.time_slice))
        for (chunk_no, chunk) in enumerate(utils.grouper(corpus, chunksize)):
            for doc in chunk:
                if doc_index > time_slice[time]:
                    time += 1
                    lda = self.make_lda_seq_slice(lda, time)
                    doc_num = 0
                gam = gammas[doc_index]
                lhood = lhoods[doc_index]
                ldapost.gamma = gam
                ldapost.lhood = lhood
                ldapost.doc = doc
                if iter_ == 0:
                    doc_lhood = LdaPost.fit_lda_post(ldapost, doc_num, time, None, lda_inference_max_iter=lda_inference_max_iter)
                else:
                    doc_lhood = LdaPost.fit_lda_post(ldapost, doc_num, time, self, lda_inference_max_iter=lda_inference_max_iter)
                if topic_suffstats is not None:
                    topic_suffstats = LdaPost.update_lda_seq_ss(ldapost, time, doc, topic_suffstats)
                gammas[doc_index] = ldapost.gamma
                bound += doc_lhood
                doc_index += 1
                doc_num += 1
        return (bound, gammas)

    def make_lda_seq_slice(self, lda, time):
        if False:
            print('Hello World!')
        'Update the LDA model topic-word values using time slices.\n\n        Parameters\n        ----------\n\n        lda : :class:`~gensim.models.ldamodel.LdaModel`\n            The stationary model to be updated\n        time : int\n            The time slice assigned to the stationary model.\n\n        Returns\n        -------\n        lda : :class:`~gensim.models.ldamodel.LdaModel`\n            The stationary model updated to reflect the passed time slice.\n\n        '
        for k in range(self.num_topics):
            lda.topics[:, k] = self.topic_chains[k].e_log_prob[:, time]
        lda.alpha = np.copy(self.alphas)
        return lda

    def fit_lda_seq_topics(self, topic_suffstats):
        if False:
            while True:
                i = 10
        'Fit the sequential model topic-wise.\n\n        Parameters\n        ----------\n        topic_suffstats : numpy.ndarray\n            Sufficient statistics of the current model, expected shape (`self.vocab_len`, `num_topics`).\n\n        Returns\n        -------\n        float\n            The sum of the optimized lower bounds for all topics.\n\n        '
        lhood = 0
        for (k, chain) in enumerate(self.topic_chains):
            logger.info('Fitting topic number %i', k)
            lhood_term = sslm.fit_sslm(chain, topic_suffstats[k])
            lhood += lhood_term
        return lhood

    def print_topic_times(self, topic, top_terms=20):
        if False:
            return 10
        'Get the most relevant words for a topic, for each timeslice. This can be used to inspect the evolution of a\n        topic through time.\n\n        Parameters\n        ----------\n        topic : int\n            The index of the topic.\n        top_terms : int, optional\n            Number of most relevant words associated with the topic to be returned.\n\n        Returns\n        -------\n        list of list of str\n            Top `top_terms` relevant terms for the topic for each time slice.\n\n        '
        topics = []
        for time in range(self.num_time_slices):
            topics.append(self.print_topic(topic, time, top_terms))
        return topics

    def print_topics(self, time=0, top_terms=20):
        if False:
            while True:
                i = 10
        'Get the most relevant words for every topic.\n\n        Parameters\n        ----------\n        time : int, optional\n            The time slice in which we are interested in (since topics evolve over time, it is expected that the most\n            relevant words will also gradually change).\n        top_terms : int, optional\n            Number of most relevant words to be returned for each topic.\n\n        Returns\n        -------\n        list of list of (str, float)\n            Representation of all topics. Each of them is represented by a list of pairs of words and their assigned\n            probability.\n\n        '
        return [self.print_topic(topic, time, top_terms) for topic in range(self.num_topics)]

    def print_topic(self, topic, time=0, top_terms=20):
        if False:
            return 10
        'Get the list of words most relevant to the given topic.\n\n        Parameters\n        ----------\n        topic : int\n            The index of the topic to be inspected.\n        time : int, optional\n            The time slice in which we are interested in (since topics evolve over time, it is expected that the most\n            relevant words will also gradually change).\n        top_terms : int, optional\n            Number of words associated with the topic to be returned.\n\n        Returns\n        -------\n        list of (str, float)\n            The representation of this topic. Each element in the list includes the word itself, along with the\n            probability assigned to it by the topic.\n\n        '
        topic = self.topic_chains[topic].e_log_prob
        topic = np.transpose(topic)
        topic = np.exp(topic[time])
        topic = topic / topic.sum()
        bestn = matutils.argsort(topic, top_terms, reverse=True)
        beststr = [(self.id2word[id_], topic[id_]) for id_ in bestn]
        return beststr

    def doc_topics(self, doc_number):
        if False:
            print('Hello World!')
        'Get the topic mixture for a document.\n\n        Uses the priors for the dirichlet distribution that approximates the true posterior with the optimal\n        lower bound, and therefore requires the model to be already trained.\n\n\n        Parameters\n        ----------\n        doc_number : int\n            Index of the document for which the mixture is returned.\n\n        Returns\n        -------\n        list of length `self.num_topics`\n            Probability for each topic in the mixture (essentially a point in the `self.num_topics - 1` simplex.\n\n        '
        doc_topic = self.gammas / self.gammas.sum(axis=1)[:, np.newaxis]
        return doc_topic[doc_number]

    def dtm_vis(self, time, corpus):
        if False:
            while True:
                i = 10
        "Get the information needed to visualize the corpus model at a given time slice, using the pyLDAvis format.\n\n        Parameters\n        ----------\n        time : int\n            The time slice we are interested in.\n        corpus : {iterable of list of (int, float), scipy.sparse.csc}, optional\n            The corpus we want to visualize at the given time slice.\n\n        Returns\n        -------\n        doc_topics : list of length `self.num_topics`\n            Probability for each topic in the mixture (essentially a point in the `self.num_topics - 1` simplex.\n        topic_term : numpy.ndarray\n            The representation of each topic as a multinomial over words in the vocabulary,\n            expected shape (`num_topics`, vocabulary length).\n        doc_lengths : list of int\n            The number of words in each document. These could be fixed, or drawn from a Poisson distribution.\n        term_frequency : numpy.ndarray\n            The term frequency matrix (denoted as beta in the original Blei paper). This could also be the TF-IDF\n            representation of the corpus, expected shape (number of documents, length of vocabulary).\n        vocab : list of str\n            The set of unique terms existing in the cropuse's vocabulary.\n\n        "
        doc_topic = self.gammas / self.gammas.sum(axis=1)[:, np.newaxis]

        def normalize(x):
            if False:
                for i in range(10):
                    print('nop')
            return x / x.sum()
        topic_term = [normalize(np.exp(chain.e_log_prob.T[time])) for (k, chain) in enumerate(self.topic_chains)]
        doc_lengths = []
        term_frequency = np.zeros(self.vocab_len)
        for (doc_no, doc) in enumerate(corpus):
            doc_lengths.append(len(doc))
            for (term, freq) in doc:
                term_frequency[term] += freq
        vocab = [self.id2word[i] for i in range(len(self.id2word))]
        return (doc_topic, np.array(topic_term), doc_lengths, term_frequency, vocab)

    def dtm_coherence(self, time):
        if False:
            for i in range(10):
                print('nop')
        'Get the coherence for each topic.\n\n        Can be used to measure the quality of the model, or to inspect the convergence through training via a callback.\n\n        Parameters\n        ----------\n        time : int\n            The time slice.\n\n        Returns\n        -------\n        list of list of str\n            The word representation for each topic, for each time slice. This can be used to check the time coherence\n            of topics as time evolves: If the most relevant words remain the same then the topic has somehow\n            converged or is relatively static, if they change rapidly the topic is evolving.\n\n        '
        coherence_topics = []
        for topics in self.print_topics(time):
            coherence_topic = []
            for (word, dist) in topics:
                coherence_topic.append(word)
            coherence_topics.append(coherence_topic)
        return coherence_topics

    def __getitem__(self, doc):
        if False:
            while True:
                i = 10
        'Get the topic mixture for the given document, using the inferred approximation of the true posterior.\n\n        Parameters\n        ----------\n        doc : list of (int, float)\n            The doc in BOW format. Can be an unseen document.\n\n        Returns\n        -------\n        list of float\n            Probabilities for each topic in the mixture. This is essentially a point in the `num_topics - 1` simplex.\n\n        '
        lda_model = ldamodel.LdaModel(num_topics=self.num_topics, alpha=self.alphas, id2word=self.id2word, dtype=np.float64)
        lda_model.topics = np.zeros((self.vocab_len, self.num_topics))
        ldapost = LdaPost(num_topics=self.num_topics, max_doc_len=len(doc), lda=lda_model, doc=doc)
        time_lhoods = []
        for time in range(self.num_time_slices):
            lda_model = self.make_lda_seq_slice(lda_model, time)
            lhood = LdaPost.fit_lda_post(ldapost, 0, time, self)
            time_lhoods.append(lhood)
        doc_topic = ldapost.gamma / ldapost.gamma.sum()
        return doc_topic

class sslm(utils.SaveLoad):
    """Encapsulate the inner State Space Language Model for DTM.

    Some important attributes of this class:

        * `obs` is a matrix containing the document to topic ratios.
        * `e_log_prob` is a matrix containing the topic to word ratios.
        * `mean` contains the mean values to be used for inference for each word for a time slice.
        * `variance` contains the variance values to be used for inference of word in a time slice.
        * `fwd_mean` and`fwd_variance` are the forward posterior values for the mean and the variance.
        * `zeta` is an extra variational parameter with a value for each time slice.

    """

    def __init__(self, vocab_len=None, num_time_slices=None, num_topics=None, obs_variance=0.5, chain_variance=0.005):
        if False:
            for i in range(10):
                print('nop')
        self.vocab_len = vocab_len
        self.num_time_slices = num_time_slices
        self.obs_variance = obs_variance
        self.chain_variance = chain_variance
        self.num_topics = num_topics
        self.obs = np.zeros((vocab_len, num_time_slices))
        self.e_log_prob = np.zeros((vocab_len, num_time_slices))
        self.mean = np.zeros((vocab_len, num_time_slices + 1))
        self.fwd_mean = np.zeros((vocab_len, num_time_slices + 1))
        self.fwd_variance = np.zeros((vocab_len, num_time_slices + 1))
        self.variance = np.zeros((vocab_len, num_time_slices + 1))
        self.zeta = np.zeros(num_time_slices)
        self.m_update_coeff = None
        self.mean_t = None
        self.variance_t = None
        self.influence_sum_lgl = None
        self.w_phi_l = None
        self.w_phi_sum = None
        self.w_phi_l_sq = None
        self.m_update_coeff_g = None

    def update_zeta(self):
        if False:
            while True:
                i = 10
        'Update the Zeta variational parameter.\n\n        Zeta is described in the appendix and is equal to sum (exp(mean[word] + Variance[word] / 2)),\n        over every time-slice. It is the value of variational parameter zeta which maximizes the lower bound.\n\n        Returns\n        -------\n        list of float\n            The updated zeta values for each time slice.\n\n        '
        for (j, val) in enumerate(self.zeta):
            self.zeta[j] = np.sum(np.exp(self.mean[:, j + 1] + self.variance[:, j + 1] / 2))
        return self.zeta

    def compute_post_variance(self, word, chain_variance):
        if False:
            for i in range(10):
                print('nop')
        "Get the variance, based on the\n        `Variational Kalman Filtering approach for Approximate Inference (section 3.1)\n        <https://mimno.infosci.cornell.edu/info6150/readings/dynamic_topic_models.pdf>`_.\n\n        This function accepts the word to compute variance for, along with the associated sslm class object,\n        and returns the `variance` and the posterior approximation `fwd_variance`.\n\n        Notes\n        -----\n        This function essentially computes Var[\\beta_{t,w}] for t = 1:T\n\n        .. :math::\n\n            fwd\\_variance[t] \\equiv E((beta_{t,w}-mean_{t,w})^2 |beta_{t}\\ for\\ 1:t) =\n            (obs\\_variance / fwd\\_variance[t - 1] + chain\\_variance + obs\\_variance ) *\n            (fwd\\_variance[t - 1] + obs\\_variance)\n\n        .. :math::\n\n            variance[t] \\equiv E((beta_{t,w}-mean\\_cap_{t,w})^2 |beta\\_cap_{t}\\ for\\ 1:t) =\n            fwd\\_variance[t - 1] + (fwd\\_variance[t - 1] / fwd\\_variance[t - 1] + obs\\_variance)^2 *\n            (variance[t - 1] - (fwd\\_variance[t-1] + obs\\_variance))\n\n        Parameters\n        ----------\n        word: int\n            The word's ID.\n        chain_variance : float\n            Gaussian parameter defined in the beta distribution to dictate how the beta values evolve over time.\n\n        Returns\n        -------\n        (numpy.ndarray, numpy.ndarray)\n            The first returned value is the variance of each word in each time slice, the second value is the\n            inferred posterior variance for the same pairs.\n\n        "
        INIT_VARIANCE_CONST = 1000
        T = self.num_time_slices
        variance = self.variance[word]
        fwd_variance = self.fwd_variance[word]
        fwd_variance[0] = chain_variance * INIT_VARIANCE_CONST
        for t in range(1, T + 1):
            if self.obs_variance:
                c = self.obs_variance / (fwd_variance[t - 1] + chain_variance + self.obs_variance)
            else:
                c = 0
            fwd_variance[t] = c * (fwd_variance[t - 1] + chain_variance)
        variance[T] = fwd_variance[T]
        for t in range(T - 1, -1, -1):
            if fwd_variance[t] > 0.0:
                c = np.power(fwd_variance[t] / (fwd_variance[t] + chain_variance), 2)
            else:
                c = 0
            variance[t] = c * (variance[t + 1] - chain_variance) + (1 - c) * fwd_variance[t]
        return (variance, fwd_variance)

    def compute_post_mean(self, word, chain_variance):
        if False:
            while True:
                i = 10
        "Get the mean, based on the `Variational Kalman Filtering approach for Approximate Inference (section 3.1)\n        <https://mimno.infosci.cornell.edu/info6150/readings/dynamic_topic_models.pdf>`_.\n\n        Notes\n        -----\n        This function essentially computes E[\x08eta_{t,w}] for t = 1:T.\n\n        .. :math::\n\n            Fwd_Mean(t) ≡  E(beta_{t,w} | beta_ˆ 1:t )\n            = (obs_variance / fwd_variance[t - 1] + chain_variance + obs_variance ) * fwd_mean[t - 1] +\n            (1 - (obs_variance / fwd_variance[t - 1] + chain_variance + obs_variance)) * beta\n\n        .. :math::\n\n            Mean(t) ≡ E(beta_{t,w} | beta_ˆ 1:T )\n            = fwd_mean[t - 1] + (obs_variance / fwd_variance[t - 1] + obs_variance) +\n            (1 - obs_variance / fwd_variance[t - 1] + obs_variance)) * mean[t]\n\n        Parameters\n        ----------\n        word: int\n            The word's ID.\n        chain_variance : float\n            Gaussian parameter defined in the beta distribution to dictate how the beta values evolve over time.\n\n        Returns\n        -------\n        (numpy.ndarray, numpy.ndarray)\n            The first returned value is the mean of each word in each time slice, the second value is the\n            inferred posterior mean for the same pairs.\n\n        "
        T = self.num_time_slices
        obs = self.obs[word]
        fwd_variance = self.fwd_variance[word]
        mean = self.mean[word]
        fwd_mean = self.fwd_mean[word]
        fwd_mean[0] = 0
        for t in range(1, T + 1):
            c = self.obs_variance / (fwd_variance[t - 1] + chain_variance + self.obs_variance)
            fwd_mean[t] = c * fwd_mean[t - 1] + (1 - c) * obs[t - 1]
        mean[T] = fwd_mean[T]
        for t in range(T - 1, -1, -1):
            if chain_variance == 0.0:
                c = 0.0
            else:
                c = chain_variance / (fwd_variance[t] + chain_variance)
            mean[t] = c * fwd_mean[t] + (1 - c) * mean[t + 1]
        return (mean, fwd_mean)

    def compute_expected_log_prob(self):
        if False:
            i = 10
            return i + 15
        'Compute the expected log probability given values of m.\n\n        The appendix describes the Expectation of log-probabilities in equation 5 of the DTM paper;\n        The below implementation is the result of solving the equation and is implemented as in the original\n        Blei DTM code.\n\n        Returns\n        -------\n        numpy.ndarray of float\n            The expected value for the log probabilities for each word and time slice.\n\n        '
        for ((w, t), val) in np.ndenumerate(self.e_log_prob):
            self.e_log_prob[w][t] = self.mean[w][t + 1] - np.log(self.zeta[t])
        return self.e_log_prob

    def sslm_counts_init(self, obs_variance, chain_variance, sstats):
        if False:
            while True:
                i = 10
        'Initialize the State Space Language Model with LDA sufficient statistics.\n\n        Called for each topic-chain and initializes initial mean, variance and Topic-Word probabilities\n        for the first time-slice.\n\n        Parameters\n        ----------\n        obs_variance : float, optional\n            Observed variance used to approximate the true and forward variance.\n        chain_variance : float\n            Gaussian parameter defined in the beta distribution to dictate how the beta values evolve over time.\n        sstats : numpy.ndarray\n            Sufficient statistics of the LDA model. Corresponds to matrix beta in the linked paper for time slice 0,\n            expected shape (`self.vocab_len`, `num_topics`).\n\n        '
        W = self.vocab_len
        T = self.num_time_slices
        log_norm_counts = np.copy(sstats)
        log_norm_counts /= sum(log_norm_counts)
        log_norm_counts += 1.0 / W
        log_norm_counts /= sum(log_norm_counts)
        log_norm_counts = np.log(log_norm_counts)
        self.obs = np.repeat(log_norm_counts, T, axis=0).reshape(W, T)
        self.obs_variance = obs_variance
        self.chain_variance = chain_variance
        for w in range(W):
            (self.variance[w], self.fwd_variance[w]) = self.compute_post_variance(w, self.chain_variance)
            (self.mean[w], self.fwd_mean[w]) = self.compute_post_mean(w, self.chain_variance)
        self.zeta = self.update_zeta()
        self.e_log_prob = self.compute_expected_log_prob()

    def fit_sslm(self, sstats):
        if False:
            print('Hello World!')
        'Fits variational distribution.\n\n        This is essentially the m-step.\n        Maximizes the approximation of the true posterior for a particular topic using the provided sufficient\n        statistics. Updates the values using :meth:`~gensim.models.ldaseqmodel.sslm.update_obs` and\n        :meth:`~gensim.models.ldaseqmodel.sslm.compute_expected_log_prob`.\n\n        Parameters\n        ----------\n        sstats : numpy.ndarray\n            Sufficient statistics for a particular topic. Corresponds to matrix beta in the linked paper for the\n            current time slice, expected shape (`self.vocab_len`, `num_topics`).\n\n        Returns\n        -------\n        float\n            The lower bound for the true posterior achieved using the fitted approximate distribution.\n\n        '
        W = self.vocab_len
        bound = 0
        old_bound = 0
        sslm_fit_threshold = 1e-06
        sslm_max_iter = 2
        converged = sslm_fit_threshold + 1
        (self.variance, self.fwd_variance) = (np.array(x) for x in zip(*(self.compute_post_variance(w, self.chain_variance) for w in range(W))))
        totals = sstats.sum(axis=0)
        iter_ = 0
        model = 'DTM'
        if model == 'DTM':
            bound = self.compute_bound(sstats, totals)
        if model == 'DIM':
            bound = self.compute_bound_fixed(sstats, totals)
        logger.info('initial sslm bound is %f', bound)
        while converged > sslm_fit_threshold and iter_ < sslm_max_iter:
            iter_ += 1
            old_bound = bound
            (self.obs, self.zeta) = self.update_obs(sstats, totals)
            if model == 'DTM':
                bound = self.compute_bound(sstats, totals)
            if model == 'DIM':
                bound = self.compute_bound_fixed(sstats, totals)
            converged = np.fabs((bound - old_bound) / old_bound)
            logger.info('iteration %i iteration lda seq bound is %f convergence is %f', iter_, bound, converged)
        self.e_log_prob = self.compute_expected_log_prob()
        return bound

    def compute_bound(self, sstats, totals):
        if False:
            i = 10
            return i + 15
        'Compute the maximized lower bound achieved for the log probability of the true posterior.\n\n        Uses the formula presented in the appendix of the DTM paper (formula no. 5).\n\n        Parameters\n        ----------\n        sstats : numpy.ndarray\n            Sufficient statistics for a particular topic. Corresponds to matrix beta in the linked paper for the first\n            time slice, expected shape (`self.vocab_len`, `num_topics`).\n        totals : list of int of length `len(self.time_slice)`\n            The totals for each time slice.\n\n        Returns\n        -------\n        float\n            The maximized lower bound.\n\n        '
        w = self.vocab_len
        t = self.num_time_slices
        term_1 = 0
        term_2 = 0
        term_3 = 0
        val = 0
        ent = 0
        chain_variance = self.chain_variance
        (self.mean, self.fwd_mean) = (np.array(x) for x in zip(*(self.compute_post_mean(w, self.chain_variance) for w in range(w))))
        self.zeta = self.update_zeta()
        val = sum((self.variance[w][0] - self.variance[w][t] for w in range(w))) / 2 * chain_variance
        logger.info('Computing bound, all times')
        for t in range(1, t + 1):
            term_1 = 0.0
            term_2 = 0.0
            ent = 0.0
            for w in range(w):
                m = self.mean[w][t]
                prev_m = self.mean[w][t - 1]
                v = self.variance[w][t]
                term_1 += np.power(m - prev_m, 2) / (2 * chain_variance) - v / chain_variance - np.log(chain_variance)
                term_2 += sstats[w][t - 1] * m
                ent += np.log(v) / 2
            term_3 = -totals[t - 1] * np.log(self.zeta[t - 1])
            val += term_2 + term_3 + ent - term_1
        return val

    def update_obs(self, sstats, totals):
        if False:
            while True:
                i = 10
        'Optimize the bound with respect to the observed variables.\n\n        TODO:\n        This is by far the slowest function in the whole algorithm.\n        Replacing or improving the performance of this would greatly speed things up.\n\n        Parameters\n        ----------\n        sstats : numpy.ndarray\n            Sufficient statistics for a particular topic. Corresponds to matrix beta in the linked paper for the first\n            time slice, expected shape (`self.vocab_len`, `num_topics`).\n        totals : list of int of length `len(self.time_slice)`\n            The totals for each time slice.\n\n        Returns\n        -------\n        (numpy.ndarray of float, numpy.ndarray of float)\n            The updated optimized values for obs and the zeta variational parameter.\n\n        '
        OBS_NORM_CUTOFF = 2
        STEP_SIZE = 0.01
        TOL = 0.001
        W = self.vocab_len
        T = self.num_time_slices
        runs = 0
        mean_deriv_mtx = np.zeros((T, T + 1))
        norm_cutoff_obs = None
        for w in range(W):
            w_counts = sstats[w]
            counts_norm = 0
            for i in range(len(w_counts)):
                counts_norm += w_counts[i] * w_counts[i]
            counts_norm = np.sqrt(counts_norm)
            if counts_norm < OBS_NORM_CUTOFF and norm_cutoff_obs is not None:
                obs = self.obs[w]
                norm_cutoff_obs = np.copy(obs)
            else:
                if counts_norm < OBS_NORM_CUTOFF:
                    w_counts = np.zeros(len(w_counts))
                for t in range(T):
                    mean_deriv_mtx[t] = self.compute_mean_deriv(w, t, mean_deriv_mtx[t])
                deriv = np.zeros(T)
                args = (self, w_counts, totals, mean_deriv_mtx, w, deriv)
                obs = self.obs[w]
                model = 'DTM'
                if model == 'DTM':
                    obs = optimize.fmin_cg(f=f_obs, fprime=df_obs, x0=obs, gtol=TOL, args=args, epsilon=STEP_SIZE, disp=0)
                if model == 'DIM':
                    pass
                runs += 1
                if counts_norm < OBS_NORM_CUTOFF:
                    norm_cutoff_obs = obs
                self.obs[w] = obs
        self.zeta = self.update_zeta()
        return (self.obs, self.zeta)

    def compute_mean_deriv(self, word, time, deriv):
        if False:
            for i in range(10):
                print('nop')
        "Helper functions for optimizing a function.\n\n        Compute the derivative of:\n\n        .. :math::\n\n            E[\x08eta_{t,w}]/d obs_{s,w} for t = 1:T.\n\n        Parameters\n        ----------\n        word : int\n            The word's ID.\n        time : int\n            The time slice.\n        deriv : list of float\n            Derivative for each time slice.\n\n        Returns\n        -------\n        list of float\n            Mean derivative for each time slice.\n\n        "
        T = self.num_time_slices
        fwd_variance = self.variance[word]
        deriv[0] = 0
        for t in range(1, T + 1):
            if self.obs_variance > 0.0:
                w = self.obs_variance / (fwd_variance[t - 1] + self.chain_variance + self.obs_variance)
            else:
                w = 0.0
            val = w * deriv[t - 1]
            if time == t - 1:
                val += 1 - w
            deriv[t] = val
        for t in range(T - 1, -1, -1):
            if self.chain_variance == 0.0:
                w = 0.0
            else:
                w = self.chain_variance / (fwd_variance[t] + self.chain_variance)
            deriv[t] = w * deriv[t] + (1 - w) * deriv[t + 1]
        return deriv

    def compute_obs_deriv(self, word, word_counts, totals, mean_deriv_mtx, deriv):
        if False:
            for i in range(10):
                print('nop')
        "Derivation of obs which is used in derivative function `df_obs` while optimizing.\n\n        Parameters\n        ----------\n        word : int\n            The word's ID.\n        word_counts : list of int\n            Total word counts for each time slice.\n        totals : list of int of length `len(self.time_slice)`\n            The totals for each time slice.\n        mean_deriv_mtx : list of float\n            Mean derivative for each time slice.\n        deriv : list of float\n            Mean derivative for each time slice.\n\n        Returns\n        -------\n        list of float\n            Mean derivative for each time slice.\n\n        "
        init_mult = 1000
        T = self.num_time_slices
        mean = self.mean[word]
        variance = self.variance[word]
        self.temp_vect = np.zeros(T)
        for u in range(T):
            self.temp_vect[u] = np.exp(mean[u + 1] + variance[u + 1] / 2)
        for t in range(T):
            mean_deriv = mean_deriv_mtx[t]
            term1 = 0
            term2 = 0
            term3 = 0
            term4 = 0
            for u in range(1, T + 1):
                mean_u = mean[u]
                mean_u_prev = mean[u - 1]
                dmean_u = mean_deriv[u]
                dmean_u_prev = mean_deriv[u - 1]
                term1 += (mean_u - mean_u_prev) * (dmean_u - dmean_u_prev)
                term2 += (word_counts[u - 1] - totals[u - 1] * self.temp_vect[u - 1] / self.zeta[u - 1]) * dmean_u
                model = 'DTM'
                if model == 'DIM':
                    pass
            if self.chain_variance:
                term1 = -(term1 / self.chain_variance)
                term1 = term1 - mean[0] * mean_deriv[0] / (init_mult * self.chain_variance)
            else:
                term1 = 0.0
            deriv[t] = term1 + term2 + term3 + term4
        return deriv

class LdaPost(utils.SaveLoad):
    """Posterior values associated with each set of documents.

    TODO: use **Hoffman, Blei, Bach: Online Learning for Latent Dirichlet Allocation, NIPS 2010.**
    to update phi, gamma. End game would be to somehow replace LdaPost entirely with LdaModel.

    """

    def __init__(self, doc=None, lda=None, max_doc_len=None, num_topics=None, gamma=None, lhood=None):
        if False:
            for i in range(10):
                print('nop')
        "Initialize the posterior value structure for the given LDA model.\n\n        Parameters\n        ----------\n        doc : list of (int, int)\n            A BOW representation of the document. Each element in the list is a pair of a word's ID and its number\n            of occurences in the document.\n        lda : :class:`~gensim.models.ldamodel.LdaModel`, optional\n            The underlying LDA model.\n        max_doc_len : int, optional\n            The maximum number of words in a document.\n        num_topics : int, optional\n            Number of topics discovered by the LDA model.\n        gamma : numpy.ndarray, optional\n            Topic weight variational parameters for each document. If not supplied, it will be inferred from the model.\n        lhood : float, optional\n            The log likelihood lower bound.\n\n        "
        self.doc = doc
        self.lda = lda
        self.gamma = gamma
        self.lhood = lhood
        if self.gamma is None:
            self.gamma = np.zeros(num_topics)
        if self.lhood is None:
            self.lhood = np.zeros(num_topics + 1)
        if max_doc_len is not None and num_topics is not None:
            self.phi = np.zeros((max_doc_len, num_topics))
            self.log_phi = np.zeros((max_doc_len, num_topics))
        self.doc_weight = None
        self.renormalized_doc_weight = None

    def update_phi(self, doc_number, time):
        if False:
            for i in range(10):
                print('nop')
        'Update variational multinomial parameters, based on a document and a time-slice.\n\n        This is done based on the original Blei-LDA paper, where:\n        log_phi := beta * exp(Ψ(gamma)), over every topic for every word.\n\n        TODO: incorporate lee-sueng trick used in\n        **Lee, Seung: Algorithms for non-negative matrix factorization, NIPS 2001**.\n\n        Parameters\n        ----------\n        doc_number : int\n            Document number. Unused.\n        time : int\n            Time slice. Unused.\n\n        Returns\n        -------\n        (list of float, list of float)\n            Multinomial parameters, and their logarithm, for each word in the document.\n\n        '
        num_topics = self.lda.num_topics
        dig = np.zeros(num_topics)
        for k in range(num_topics):
            dig[k] = digamma(self.gamma[k])
        n = 0
        for (word_id, count) in self.doc:
            for k in range(num_topics):
                self.log_phi[n][k] = dig[k] + self.lda.topics[word_id][k]
            log_phi_row = self.log_phi[n]
            phi_row = self.phi[n]
            v = log_phi_row[0]
            for i in range(1, len(log_phi_row)):
                v = np.logaddexp(v, log_phi_row[i])
            log_phi_row = log_phi_row - v
            phi_row = np.exp(log_phi_row)
            self.log_phi[n] = log_phi_row
            self.phi[n] = phi_row
            n += 1
        return (self.phi, self.log_phi)

    def update_gamma(self):
        if False:
            while True:
                i = 10
        'Update variational dirichlet parameters.\n\n        This operations is described in the original Blei LDA paper:\n        gamma = alpha + sum(phi), over every topic for every word.\n\n        Returns\n        -------\n        list of float\n            The updated gamma parameters for each word in the document.\n\n        '
        self.gamma = np.copy(self.lda.alpha)
        n = 0
        for (word_id, count) in self.doc:
            phi_row = self.phi[n]
            for k in range(self.lda.num_topics):
                self.gamma[k] += phi_row[k] * count
            n += 1
        return self.gamma

    def init_lda_post(self):
        if False:
            i = 10
            return i + 15
        'Initialize variational posterior. '
        total = sum((count for (word_id, count) in self.doc))
        self.gamma.fill(self.lda.alpha[0] + float(total) / self.lda.num_topics)
        self.phi[:len(self.doc), :] = 1.0 / self.lda.num_topics

    def compute_lda_lhood(self):
        if False:
            i = 10
            return i + 15
        'Compute the log likelihood bound.\n\n        Returns\n        -------\n        float\n            The optimal lower bound for the true posterior using the approximate distribution.\n\n        '
        num_topics = self.lda.num_topics
        gamma_sum = np.sum(self.gamma)
        lhood = gammaln(np.sum(self.lda.alpha)) - gammaln(gamma_sum)
        self.lhood[num_topics] = lhood
        digsum = digamma(gamma_sum)
        model = 'DTM'
        for k in range(num_topics):
            e_log_theta_k = digamma(self.gamma[k]) - digsum
            lhood_term = (self.lda.alpha[k] - self.gamma[k]) * e_log_theta_k + gammaln(self.gamma[k]) - gammaln(self.lda.alpha[k])
            n = 0
            for (word_id, count) in self.doc:
                if self.phi[n][k] > 0:
                    lhood_term += count * self.phi[n][k] * (e_log_theta_k + self.lda.topics[word_id][k] - self.log_phi[n][k])
                n += 1
            self.lhood[k] = lhood_term
            lhood += lhood_term
        return lhood

    def fit_lda_post(self, doc_number, time, ldaseq, LDA_INFERENCE_CONVERGED=1e-08, lda_inference_max_iter=25, g=None, g3_matrix=None, g4_matrix=None, g5_matrix=None):
        if False:
            for i in range(10):
                print('nop')
        'Posterior inference for lda.\n\n        Parameters\n        ----------\n        doc_number : int\n            The documents number.\n        time : int\n            Time slice.\n        ldaseq : object\n            Unused.\n        LDA_INFERENCE_CONVERGED : float\n            Epsilon value used to check whether the inference step has sufficiently converged.\n        lda_inference_max_iter : int\n            Maximum number of iterations in the inference step.\n        g : object\n            Unused. Will be useful when the DIM model is implemented.\n        g3_matrix: object\n            Unused. Will be useful when the DIM model is implemented.\n        g4_matrix: object\n            Unused. Will be useful when the DIM model is implemented.\n        g5_matrix: object\n            Unused. Will be useful when the DIM model is implemented.\n\n        Returns\n        -------\n        float\n            The optimal lower bound for the true posterior using the approximate distribution.\n        '
        self.init_lda_post()
        total = sum((count for (word_id, count) in self.doc))
        model = 'DTM'
        if model == 'DIM':
            pass
        lhood = self.compute_lda_lhood()
        lhood_old = 0
        converged = 0
        iter_ = 0
        iter_ += 1
        lhood_old = lhood
        self.gamma = self.update_gamma()
        model = 'DTM'
        if model == 'DTM' or sslm is None:
            (self.phi, self.log_phi) = self.update_phi(doc_number, time)
        elif model == 'DIM' and sslm is not None:
            (self.phi, self.log_phi) = self.update_phi_fixed(doc_number, time, sslm, g3_matrix, g4_matrix, g5_matrix)
        lhood = self.compute_lda_lhood()
        converged = np.fabs((lhood_old - lhood) / (lhood_old * total))
        while converged > LDA_INFERENCE_CONVERGED and iter_ <= lda_inference_max_iter:
            iter_ += 1
            lhood_old = lhood
            self.gamma = self.update_gamma()
            model = 'DTM'
            if model == 'DTM' or sslm is None:
                (self.phi, self.log_phi) = self.update_phi(doc_number, time)
            elif model == 'DIM' and sslm is not None:
                (self.phi, self.log_phi) = self.update_phi_fixed(doc_number, time, sslm, g3_matrix, g4_matrix, g5_matrix)
            lhood = self.compute_lda_lhood()
            converged = np.fabs((lhood_old - lhood) / (lhood_old * total))
        return lhood

    def update_lda_seq_ss(self, time, doc, topic_suffstats):
        if False:
            print('Hello World!')
        'Update lda sequence sufficient statistics from an lda posterior.\n\n        This is very similar to the :meth:`~gensim.models.ldaseqmodel.LdaPost.update_gamma` method and uses\n        the same formula.\n\n        Parameters\n        ----------\n        time : int\n            The time slice.\n        doc : list of (int, float)\n            Unused but kept here for backwards compatibility. The document set in the constructor (`self.doc`) is used\n            instead.\n        topic_suffstats : list of float\n            Sufficient statistics for each topic.\n\n        Returns\n        -------\n        list of float\n            The updated sufficient statistics for each topic.\n\n        '
        num_topics = self.lda.num_topics
        for k in range(num_topics):
            topic_ss = topic_suffstats[k]
            n = 0
            for (word_id, count) in self.doc:
                topic_ss[word_id][time] += count * self.phi[n][k]
                n += 1
            topic_suffstats[k] = topic_ss
        return topic_suffstats

def f_obs(x, *args):
    if False:
        i = 10
        return i + 15
    "Function which we are optimising for minimizing obs.\n\n    Parameters\n    ----------\n    x : list of float\n        The obs values for this word.\n    sslm : :class:`~gensim.models.ldaseqmodel.sslm`\n        The State Space Language Model for DTM.\n    word_counts : list of int\n        Total word counts for each time slice.\n    totals : list of int of length `len(self.time_slice)`\n        The totals for each time slice.\n    mean_deriv_mtx : list of float\n        Mean derivative for each time slice.\n    word : int\n        The word's ID.\n    deriv : list of float\n        Mean derivative for each time slice.\n\n    Returns\n    -------\n    list of float\n        The value of the objective function evaluated at point `x`.\n\n    "
    (sslm, word_counts, totals, mean_deriv_mtx, word, deriv) = args
    init_mult = 1000
    T = len(x)
    val = 0
    term1 = 0
    term2 = 0
    term3 = 0
    term4 = 0
    sslm.obs[word] = x
    (sslm.mean[word], sslm.fwd_mean[word]) = sslm.compute_post_mean(word, sslm.chain_variance)
    mean = sslm.mean[word]
    variance = sslm.variance[word]
    for t in range(1, T + 1):
        mean_t = mean[t]
        mean_t_prev = mean[t - 1]
        val = mean_t - mean_t_prev
        term1 += val * val
        term2 += word_counts[t - 1] * mean_t - totals[t - 1] * np.exp(mean_t + variance[t] / 2) / sslm.zeta[t - 1]
        model = 'DTM'
        if model == 'DIM':
            pass
    if sslm.chain_variance > 0.0:
        term1 = -(term1 / (2 * sslm.chain_variance))
        term1 = term1 - mean[0] * mean[0] / (2 * init_mult * sslm.chain_variance)
    else:
        term1 = 0.0
    final = -(term1 + term2 + term3 + term4)
    return final

def df_obs(x, *args):
    if False:
        return 10
    "Derivative of the objective function which optimises obs.\n\n    Parameters\n    ----------\n    x : list of float\n        The obs values for this word.\n    sslm : :class:`~gensim.models.ldaseqmodel.sslm`\n        The State Space Language Model for DTM.\n    word_counts : list of int\n        Total word counts for each time slice.\n    totals : list of int of length `len(self.time_slice)`\n        The totals for each time slice.\n    mean_deriv_mtx : list of float\n        Mean derivative for each time slice.\n    word : int\n        The word's ID.\n    deriv : list of float\n        Mean derivative for each time slice.\n\n    Returns\n    -------\n    list of float\n        The derivative of the objective function evaluated at point `x`.\n\n    "
    (sslm, word_counts, totals, mean_deriv_mtx, word, deriv) = args
    sslm.obs[word] = x
    (sslm.mean[word], sslm.fwd_mean[word]) = sslm.compute_post_mean(word, sslm.chain_variance)
    model = 'DTM'
    if model == 'DTM':
        deriv = sslm.compute_obs_deriv(word, word_counts, totals, mean_deriv_mtx, deriv)
    elif model == 'DIM':
        deriv = sslm.compute_obs_deriv_fixed(p.word, p.word_counts, p.totals, p.sslm, p.mean_deriv_mtx, deriv)
    return np.negative(deriv)