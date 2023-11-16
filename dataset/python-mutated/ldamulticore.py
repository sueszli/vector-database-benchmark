"""Online Latent Dirichlet Allocation (LDA) in Python, using all CPU cores to parallelize and speed up model training.

The parallelization uses multiprocessing; in case this doesn't work for you for some reason,
try the :class:`gensim.models.ldamodel.LdaModel` class which is an equivalent, but more straightforward and single-core
implementation.

The training algorithm:

* is **streamed**: training documents may come in sequentially, no random access required,
* runs in **constant memory** w.r.t. the number of documents: size of the
  training corpus does not affect memory footprint, can process corpora larger than RAM

Wall-clock `performance on the English Wikipedia <https://radimrehurek.com/gensim/wiki.html>`_ (2G corpus positions,
3.5M documents, 100K features, 0.54G non-zero entries in the final bag-of-words matrix), requesting 100 topics:


====================================================== ==============
 algorithm                                             training time
====================================================== ==============
 LdaMulticore(workers=1)                               2h30m
 LdaMulticore(workers=2)                               1h24m
 LdaMulticore(workers=3)                               1h6m
 old LdaModel()                                        3h44m
 simply iterating over input corpus = I/O overhead     20m
====================================================== ==============

(Measured on `this i7 server <http://www.hetzner.de/en/hosting/produkte_rootserver/ex40ssd>`_
with 4 physical cores, so that optimal `workers=3`, one less than the number of cores.)

This module allows both LDA model estimation from a training corpus and inference of topic distribution on new,
unseen documents. The model can also be updated with new documents for online training.

The core estimation code is based on the `onlineldavb.py script
<https://github.com/blei-lab/onlineldavb/blob/master/onlineldavb.py>`_, by
Matthew D. Hoffman, David M. Blei, Francis Bach:
`'Online Learning for Latent Dirichlet Allocation', NIPS 2010`_.

.. _'Online Learning for Latent Dirichlet Allocation', NIPS 2010: online-lda_
.. _'Online Learning for LDA' by Hoffman et al.: online-lda_
.. _online-lda: https://papers.neurips.cc/paper/2010/file/71f6278d140af599e06ad9bf1ba03cb0-Paper.pdf

Usage examples
--------------
The constructor estimates Latent Dirichlet Allocation model parameters based on a training corpus

.. sourcecode:: pycon

    >>> from gensim.test.utils import common_corpus, common_dictionary
    >>>
    >>> lda = LdaMulticore(common_corpus, id2word=common_dictionary, num_topics=10)

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

Query, or update the model using new, unseen documents

.. sourcecode:: pycon

    >>> other_texts = [
    ...     ['computer', 'time', 'graph'],
    ...     ['survey', 'response', 'eps'],
    ...     ['human', 'system', 'computer']
    ... ]
    >>> other_corpus = [common_dictionary.doc2bow(text) for text in other_texts]
    >>>
    >>> unseen_doc = other_corpus[0]
    >>> vector = lda[unseen_doc]  # get topic probability distribution for a document
    >>>
    >>> # Update the model by incrementally training on the new corpus.
    >>> lda.update(other_corpus)  # update the LDA model with additional documents

"""
import logging
import queue
from multiprocessing import Pool, Queue, cpu_count
import numpy as np
from gensim import utils
from gensim.models.ldamodel import LdaModel, LdaState
logger = logging.getLogger(__name__)

class LdaMulticore(LdaModel):
    """An optimized implementation of the LDA algorithm, able to harness the power of multicore CPUs.
    Follows the similar API as the parent class :class:`~gensim.models.ldamodel.LdaModel`.

    """

    def __init__(self, corpus=None, num_topics=100, id2word=None, workers=None, chunksize=2000, passes=1, batch=False, alpha='symmetric', eta=None, decay=0.5, offset=1.0, eval_every=10, iterations=50, gamma_threshold=0.001, random_state=None, minimum_probability=0.01, minimum_phi_value=0.01, per_word_topics=False, dtype=np.float32):
        if False:
            i = 10
            return i + 15
        "\n\n        Parameters\n        ----------\n        corpus : {iterable of list of (int, float), scipy.sparse.csc}, optional\n            Stream of document vectors or sparse matrix of shape (`num_documents`, `num_terms`).\n            If not given, the model is left untrained (presumably because you want to call\n            :meth:`~gensim.models.ldamodel.LdaModel.update` manually).\n        num_topics : int, optional\n            The number of requested latent topics to be extracted from the training corpus.\n        id2word : {dict of (int, str),  :class:`gensim.corpora.dictionary.Dictionary`}\n            Mapping from word IDs to words. It is used to determine the vocabulary size, as well as for\n            debugging and topic printing.\n        workers : int, optional\n            Number of workers processes to be used for parallelization. If None all available cores\n            (as estimated by `workers=cpu_count()-1` will be used. **Note** however that for\n            hyper-threaded CPUs, this estimation returns a too high number -- set `workers`\n            directly to the number of your **real** cores (not hyperthreads) minus one, for optimal performance.\n        chunksize :  int, optional\n            Number of documents to be used in each training chunk.\n        passes : int, optional\n            Number of passes through the corpus during training.\n        alpha : {float, numpy.ndarray of float, list of float, str}, optional\n            A-priori belief on document-topic distribution, this can be:\n                * scalar for a symmetric prior over document-topic distribution,\n                * 1D array of length equal to num_topics to denote an asymmetric user defined prior for each topic.\n\n            Alternatively default prior selecting strategies can be employed by supplying a string:\n                * 'symmetric': (default) Uses a fixed symmetric prior of `1.0 / num_topics`,\n                * 'asymmetric': Uses a fixed normalized asymmetric prior of `1.0 / (topic_index + sqrt(num_topics))`.\n        eta : {float, numpy.ndarray of float, list of float, str}, optional\n            A-priori belief on topic-word distribution, this can be:\n                * scalar for a symmetric prior over topic-word distribution,\n                * 1D array of length equal to num_words to denote an asymmetric user defined prior for each word,\n                * matrix of shape (num_topics, num_words) to assign a probability for each word-topic combination.\n\n            Alternatively default prior selecting strategies can be employed by supplying a string:\n                * 'symmetric': (default) Uses a fixed symmetric prior of `1.0 / num_topics`,\n                * 'auto': Learns an asymmetric prior from the corpus.\n        decay : float, optional\n            A number between (0.5, 1] to weight what percentage of the previous lambda value is forgotten\n            when each new document is examined. Corresponds to :math:`\\kappa` from\n            `'Online Learning for LDA' by Hoffman et al.`_\n        offset : float, optional\n            Hyper-parameter that controls how much we will slow down the first steps the first few iterations.\n            Corresponds to :math:`\\tau_0` from `'Online Learning for LDA' by Hoffman et al.`_\n        eval_every : int, optional\n            Log perplexity is estimated every that many updates. Setting this to one slows down training by ~2x.\n        iterations : int, optional\n            Maximum number of iterations through the corpus when inferring the topic distribution of a corpus.\n        gamma_threshold : float, optional\n            Minimum change in the value of the gamma parameters to continue iterating.\n        minimum_probability : float, optional\n            Topics with a probability lower than this threshold will be filtered out.\n        random_state : {np.random.RandomState, int}, optional\n            Either a randomState object or a seed to generate one. Useful for reproducibility.\n            Note that results can still vary due to non-determinism in OS scheduling of the worker processes.\n        minimum_phi_value : float, optional\n            if `per_word_topics` is True, this represents a lower bound on the term probabilities.\n        per_word_topics : bool\n            If True, the model also computes a list of topics, sorted in descending order of most likely topics for\n            each word, along with their phi values multiplied by the feature length (i.e. word count).\n        dtype : {numpy.float16, numpy.float32, numpy.float64}, optional\n            Data-type to use during calculations inside model. All inputs are also converted.\n\n        "
        self.workers = max(1, cpu_count() - 1) if workers is None else workers
        self.batch = batch
        if isinstance(alpha, str) and alpha == 'auto':
            raise NotImplementedError('auto-tuning alpha not implemented in LdaMulticore; use plain LdaModel.')
        super(LdaMulticore, self).__init__(corpus=corpus, num_topics=num_topics, id2word=id2word, chunksize=chunksize, passes=passes, alpha=alpha, eta=eta, decay=decay, offset=offset, eval_every=eval_every, iterations=iterations, gamma_threshold=gamma_threshold, random_state=random_state, minimum_probability=minimum_probability, minimum_phi_value=minimum_phi_value, per_word_topics=per_word_topics, dtype=dtype)

    def update(self, corpus, chunks_as_numpy=False):
        if False:
            i = 10
            return i + 15
        "Train the model with new documents, by EM-iterating over `corpus` until the topics converge\n        (or until the maximum number of allowed iterations is reached).\n\n        Train the model with new documents, by EM-iterating over the corpus until the topics converge, or until\n        the maximum number of allowed iterations is reached. `corpus` must be an iterable. The E step is distributed\n        into the several processes.\n\n        Notes\n        -----\n        This update also supports updating an already trained model (`self`) with new documents from `corpus`;\n        the two models are then merged in proportion to the number of old vs. new documents.\n        This feature is still experimental for non-stationary input streams.\n\n        For stationary input (no topic drift in new documents), on the other hand,\n        this equals the online update of `'Online Learning for LDA' by Hoffman et al.`_\n        and is guaranteed to converge for any `decay` in (0.5, 1].\n\n        Parameters\n        ----------\n        corpus : {iterable of list of (int, float), scipy.sparse.csc}, optional\n            Stream of document vectors or sparse matrix of shape (`num_documents`, `num_terms`) used to update the\n            model.\n        chunks_as_numpy : bool\n            Whether each chunk passed to the inference step should be a np.ndarray or not. Numpy can in some settings\n            turn the term IDs into floats, these will be converted back into integers in inference, which incurs a\n            performance hit. For distributed computing it may be desirable to keep the chunks as `numpy.ndarray`.\n\n        "
        try:
            lencorpus = len(corpus)
        except TypeError:
            logger.warning('input corpus stream has no len(); counting documents')
            lencorpus = sum((1 for _ in corpus))
        if lencorpus == 0:
            logger.warning('LdaMulticore.update() called with an empty corpus')
            return
        self.state.numdocs += lencorpus
        if self.batch:
            updatetype = 'batch'
            updateafter = lencorpus
        else:
            updatetype = 'online'
            updateafter = self.chunksize * self.workers
        eval_every = self.eval_every or 0
        evalafter = min(lencorpus, eval_every * updateafter)
        updates_per_pass = max(1, lencorpus / updateafter)
        logger.info('running %s LDA training, %s topics, %i passes over the supplied corpus of %i documents, updating every %i documents, evaluating every ~%i documents, iterating %ix with a convergence threshold of %f', updatetype, self.num_topics, self.passes, lencorpus, updateafter, evalafter, self.iterations, self.gamma_threshold)
        if updates_per_pass * self.passes < 10:
            logger.warning('too few updates, training might not converge; consider increasing the number of passes or iterations to improve accuracy')
        job_queue = Queue(maxsize=2 * self.workers)
        result_queue = Queue()

        def rho():
            if False:
                print('Hello World!')
            return pow(self.offset + pass_ + self.num_updates / self.chunksize, -self.decay)

        def process_result_queue(force=False):
            if False:
                return 10
            '\n            Clear the result queue, merging all intermediate results, and update the\n            LDA model if necessary.\n\n            '
            merged_new = False
            while not result_queue.empty():
                other.merge(result_queue.get())
                queue_size[0] -= 1
                merged_new = True
            if force and merged_new and (queue_size[0] == 0) or other.numdocs >= updateafter:
                self.do_mstep(rho(), other, pass_ > 0)
                other.reset()
                if eval_every > 0 and (force or self.num_updates / updateafter % eval_every == 0):
                    self.log_perplexity(chunk, total_docs=lencorpus)
        logger.info('training LDA model using %i processes', self.workers)
        pool = Pool(self.workers, worker_e_step, (job_queue, result_queue, self))
        for pass_ in range(self.passes):
            (queue_size, reallen) = ([0], 0)
            other = LdaState(self.eta, self.state.sstats.shape)
            chunk_stream = utils.grouper(corpus, self.chunksize, as_numpy=chunks_as_numpy)
            for (chunk_no, chunk) in enumerate(chunk_stream):
                reallen += len(chunk)
                while True:
                    try:
                        job_queue.put((chunk_no, chunk, self.state), block=False)
                        queue_size[0] += 1
                        logger.info('PROGRESS: pass %i, dispatched chunk #%i = documents up to #%i/%i, outstanding queue size %i', pass_, chunk_no, chunk_no * self.chunksize + len(chunk), lencorpus, queue_size[0])
                        break
                    except queue.Full:
                        process_result_queue()
                process_result_queue()
            while queue_size[0] > 0:
                process_result_queue(force=True)
            if reallen != lencorpus:
                raise RuntimeError("input corpus size changed during training (don't use generators as input)")
        pool.terminate()

def worker_e_step(input_queue, result_queue, worker_lda):
    if False:
        print('Hello World!')
    'Perform E-step for each job.\n\n    Parameters\n    ----------\n    input_queue : queue of (int, list of (int, float), :class:`~gensim.models.lda_worker.Worker`)\n        Each element is a job characterized by its ID, the corpus chunk to be processed in BOW format and the worker\n        responsible for processing it.\n    result_queue : queue of :class:`~gensim.models.ldamodel.LdaState`\n        After the worker finished the job, the state of the resulting (trained) worker model is appended to this queue.\n    worker_lda : :class:`~gensim.models.ldamulticore.LdaMulticore`\n        LDA instance which performed e step\n    '
    logger.debug('worker process entering E-step loop')
    while True:
        logger.debug('getting a new job')
        (chunk_no, chunk, w_state) = input_queue.get()
        logger.debug('processing chunk #%i of %i documents', chunk_no, len(chunk))
        worker_lda.state = w_state
        worker_lda.sync_state()
        worker_lda.state.reset()
        worker_lda.do_estep(chunk)
        del chunk
        logger.debug('processed chunk, queuing the result')
        result_queue.put(worker_lda.state)
        worker_lda.state = None
        logger.debug('result put')