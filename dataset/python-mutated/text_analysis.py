"""This module contains classes for analyzing the texts of a corpus to accumulate
statistical information about word occurrences."""
import itertools
import logging
import multiprocessing as mp
import sys
from collections import Counter
import numpy as np
import scipy.sparse as sps
from gensim import utils
from gensim.models.word2vec import Word2Vec
logger = logging.getLogger(__name__)

def _ids_to_words(ids, dictionary):
    if False:
        return 10
    "Convert an iterable of ids to their corresponding words using a dictionary.\n    Abstract away the differences between the HashDictionary and the standard one.\n\n    Parameters\n    ----------\n    ids: dict\n        Dictionary of ids and their words.\n    dictionary: :class:`~gensim.corpora.dictionary.Dictionary`\n        Input gensim dictionary\n\n    Returns\n    -------\n    set\n        Corresponding words.\n\n    Examples\n    --------\n    .. sourcecode:: pycon\n\n        >>> from gensim.corpora.dictionary import Dictionary\n        >>> from gensim.topic_coherence import text_analysis\n        >>>\n        >>> dictionary = Dictionary()\n        >>> ids = {1: 'fake', 4: 'cats'}\n        >>> dictionary.id2token = {1: 'fake', 2: 'tokens', 3: 'rabbids', 4: 'cats'}\n        >>>\n        >>> text_analysis._ids_to_words(ids, dictionary)\n        set(['cats', 'fake'])\n\n    "
    if not dictionary.id2token:
        setattr(dictionary, 'id2token', {v: k for (k, v) in dictionary.token2id.items()})
    top_words = set()
    for word_id in ids:
        word = dictionary.id2token[word_id]
        if isinstance(word, set):
            top_words = top_words.union(word)
        else:
            top_words.add(word)
    return top_words

class BaseAnalyzer:
    """Base class for corpus and text analyzers.

    Attributes
    ----------
    relevant_ids : dict
        Mapping
    _vocab_size : int
        Size of vocabulary.
    id2contiguous : dict
        Mapping word_id -> number.
    log_every : int
        Interval for logging.
    _num_docs : int
        Number of documents.

    """

    def __init__(self, relevant_ids):
        if False:
            while True:
                i = 10
        "\n\n        Parameters\n        ----------\n        relevant_ids : dict\n            Mapping\n\n        Examples\n        --------\n        .. sourcecode:: pycon\n\n            >>> from gensim.topic_coherence import text_analysis\n            >>> ids = {1: 'fake', 4: 'cats'}\n            >>> base = text_analysis.BaseAnalyzer(ids)\n            >>> # should return {1: 'fake', 4: 'cats'} 2 {1: 0, 4: 1} 1000 0\n            >>> print(base.relevant_ids, base._vocab_size, base.id2contiguous, base.log_every, base._num_docs)\n            {1: 'fake', 4: 'cats'} 2 {1: 0, 4: 1} 1000 0\n\n        "
        self.relevant_ids = relevant_ids
        self._vocab_size = len(self.relevant_ids)
        self.id2contiguous = {word_id: n for (n, word_id) in enumerate(self.relevant_ids)}
        self.log_every = 1000
        self._num_docs = 0

    @property
    def num_docs(self):
        if False:
            print('Hello World!')
        return self._num_docs

    @num_docs.setter
    def num_docs(self, num):
        if False:
            for i in range(10):
                print('nop')
        self._num_docs = num
        if self._num_docs % self.log_every == 0:
            logger.info('%s accumulated stats from %d documents', self.__class__.__name__, self._num_docs)

    def analyze_text(self, text, doc_num=None):
        if False:
            return 10
        raise NotImplementedError('Base classes should implement analyze_text.')

    def __getitem__(self, word_or_words):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(word_or_words, str) or not hasattr(word_or_words, '__iter__'):
            return self.get_occurrences(word_or_words)
        else:
            return self.get_co_occurrences(*word_or_words)

    def get_occurrences(self, word_id):
        if False:
            i = 10
            return i + 15
        'Return number of docs the word occurs in, once `accumulate` has been called.'
        return self._get_occurrences(self.id2contiguous[word_id])

    def _get_occurrences(self, word_id):
        if False:
            return 10
        raise NotImplementedError('Base classes should implement occurrences')

    def get_co_occurrences(self, word_id1, word_id2):
        if False:
            while True:
                i = 10
        'Return number of docs the words co-occur in, once `accumulate` has been called.'
        return self._get_co_occurrences(self.id2contiguous[word_id1], self.id2contiguous[word_id2])

    def _get_co_occurrences(self, word_id1, word_id2):
        if False:
            return 10
        raise NotImplementedError('Base classes should implement co_occurrences')

class UsesDictionary(BaseAnalyzer):
    """A BaseAnalyzer that uses a Dictionary, hence can translate tokens to counts.
    The standard BaseAnalyzer can only deal with token ids since it doesn't have the token2id
    mapping.

    Attributes
    ----------
    relevant_words : set
        Set of words that occurrences should be accumulated for.
    dictionary : :class:`~gensim.corpora.dictionary.Dictionary`
        Dictionary based on text
    token2id : dict
        Mapping from :class:`~gensim.corpora.dictionary.Dictionary`

    """

    def __init__(self, relevant_ids, dictionary):
        if False:
            for i in range(10):
                print('nop')
        "\n\n        Parameters\n        ----------\n        relevant_ids : dict\n            Mapping\n        dictionary : :class:`~gensim.corpora.dictionary.Dictionary`\n            Dictionary based on text\n\n        Examples\n        --------\n        .. sourcecode:: pycon\n\n            >>> from gensim.topic_coherence import text_analysis\n            >>> from gensim.corpora.dictionary import Dictionary\n            >>>\n            >>> ids = {1: 'foo', 2: 'bar'}\n            >>> dictionary = Dictionary([['foo', 'bar', 'baz'], ['foo', 'bar', 'bar', 'baz']])\n            >>> udict = text_analysis.UsesDictionary(ids, dictionary)\n            >>>\n            >>> print(udict.relevant_words)\n            set([u'foo', u'baz'])\n\n        "
        super(UsesDictionary, self).__init__(relevant_ids)
        self.relevant_words = _ids_to_words(self.relevant_ids, dictionary)
        self.dictionary = dictionary
        self.token2id = dictionary.token2id

    def get_occurrences(self, word):
        if False:
            i = 10
            return i + 15
        'Return number of docs the word occurs in, once `accumulate` has been called.'
        try:
            word_id = self.token2id[word]
        except KeyError:
            word_id = word
        return self._get_occurrences(self.id2contiguous[word_id])

    def _word2_contiguous_id(self, word):
        if False:
            print('Hello World!')
        try:
            word_id = self.token2id[word]
        except KeyError:
            word_id = word
        return self.id2contiguous[word_id]

    def get_co_occurrences(self, word1, word2):
        if False:
            i = 10
            return i + 15
        'Return number of docs the words co-occur in, once `accumulate` has been called.'
        word_id1 = self._word2_contiguous_id(word1)
        word_id2 = self._word2_contiguous_id(word2)
        return self._get_co_occurrences(word_id1, word_id2)

class InvertedIndexBased(BaseAnalyzer):
    """Analyzer that builds up an inverted index to accumulate stats."""

    def __init__(self, *args):
        if False:
            while True:
                i = 10
        "\n\n        Parameters\n        ----------\n        args : dict\n            Look at :class:`~gensim.topic_coherence.text_analysis.BaseAnalyzer`\n\n        Examples\n        --------\n        .. sourcecode:: pycon\n\n            >>> from gensim.topic_coherence import text_analysis\n            >>>\n            >>> ids = {1: 'fake', 4: 'cats'}\n            >>> ininb = text_analysis.InvertedIndexBased(ids)\n            >>>\n            >>> print(ininb._inverted_index)\n            [set([]) set([])]\n\n        "
        super(InvertedIndexBased, self).__init__(*args)
        self._inverted_index = np.array([set() for _ in range(self._vocab_size)])

    def _get_occurrences(self, word_id):
        if False:
            print('Hello World!')
        return len(self._inverted_index[word_id])

    def _get_co_occurrences(self, word_id1, word_id2):
        if False:
            for i in range(10):
                print('nop')
        s1 = self._inverted_index[word_id1]
        s2 = self._inverted_index[word_id2]
        return len(s1.intersection(s2))

    def index_to_dict(self):
        if False:
            while True:
                i = 10
        contiguous2id = {n: word_id for (word_id, n) in self.id2contiguous.items()}
        return {contiguous2id[n]: doc_id_set for (n, doc_id_set) in enumerate(self._inverted_index)}

class CorpusAccumulator(InvertedIndexBased):
    """Gather word occurrence stats from a corpus by iterating over its BoW representation."""

    def analyze_text(self, text, doc_num=None):
        if False:
            for i in range(10):
                print('nop')
        'Build an inverted index from a sequence of corpus texts.'
        doc_words = frozenset((x[0] for x in text))
        top_ids_in_doc = self.relevant_ids.intersection(doc_words)
        for word_id in top_ids_in_doc:
            self._inverted_index[self.id2contiguous[word_id]].add(self._num_docs)

    def accumulate(self, corpus):
        if False:
            print('Hello World!')
        for document in corpus:
            self.analyze_text(document)
            self.num_docs += 1
        return self

class WindowedTextsAnalyzer(UsesDictionary):
    """Gather some stats about relevant terms of a corpus by iterating over windows of texts."""

    def __init__(self, relevant_ids, dictionary):
        if False:
            while True:
                i = 10
        '\n\n        Parameters\n        ----------\n        relevant_ids : set of int\n            Relevant id\n        dictionary : :class:`~gensim.corpora.dictionary.Dictionary`\n            Dictionary instance with mappings for the relevant_ids.\n\n        '
        super(WindowedTextsAnalyzer, self).__init__(relevant_ids, dictionary)
        self._none_token = self._vocab_size

    def accumulate(self, texts, window_size):
        if False:
            while True:
                i = 10
        relevant_texts = self._iter_texts(texts)
        windows = utils.iter_windows(relevant_texts, window_size, ignore_below_size=False, include_doc_num=True)
        for (doc_num, virtual_document) in windows:
            if len(virtual_document) > 0:
                self.analyze_text(virtual_document, doc_num)
            self.num_docs += 1
        return self

    def _iter_texts(self, texts):
        if False:
            for i in range(10):
                print('nop')
        dtype = np.uint16 if np.iinfo(np.uint16).max >= self._vocab_size else np.uint32
        for text in texts:
            ids = (self.id2contiguous[self.token2id[w]] if w in self.relevant_words else self._none_token for w in text)
            yield np.fromiter(ids, dtype=dtype, count=len(text))

class InvertedIndexAccumulator(WindowedTextsAnalyzer, InvertedIndexBased):
    """Build an inverted index from a sequence of corpus texts."""

    def analyze_text(self, window, doc_num=None):
        if False:
            for i in range(10):
                print('nop')
        for word_id in window:
            if word_id is not self._none_token:
                self._inverted_index[word_id].add(self._num_docs)

class WordOccurrenceAccumulator(WindowedTextsAnalyzer):
    """Accumulate word occurrences and co-occurrences from a sequence of corpus texts."""

    def __init__(self, *args):
        if False:
            print('Hello World!')
        super(WordOccurrenceAccumulator, self).__init__(*args)
        self._occurrences = np.zeros(self._vocab_size, dtype='uint32')
        self._co_occurrences = sps.lil_matrix((self._vocab_size, self._vocab_size), dtype='uint32')
        self._uniq_words = np.zeros((self._vocab_size + 1,), dtype=bool)
        self._counter = Counter()

    def __str__(self):
        if False:
            while True:
                i = 10
        return self.__class__.__name__

    def accumulate(self, texts, window_size):
        if False:
            return 10
        self._co_occurrences = self._co_occurrences.tolil()
        self.partial_accumulate(texts, window_size)
        self._symmetrize()
        return self

    def partial_accumulate(self, texts, window_size):
        if False:
            return 10
        'Meant to be called several times to accumulate partial results.\n\n        Notes\n        -----\n        The final accumulation should be performed with the `accumulate` method as opposed to this one.\n        This method does not ensure the co-occurrence matrix is in lil format and does not\n        symmetrize it after accumulation.\n\n        '
        self._current_doc_num = -1
        self._token_at_edge = None
        self._counter.clear()
        super(WordOccurrenceAccumulator, self).accumulate(texts, window_size)
        for (combo, count) in self._counter.items():
            self._co_occurrences[combo] += count
        return self

    def analyze_text(self, window, doc_num=None):
        if False:
            return 10
        self._slide_window(window, doc_num)
        mask = self._uniq_words[:-1]
        if mask.any():
            self._occurrences[mask] += 1
            self._counter.update(itertools.combinations(np.nonzero(mask)[0], 2))

    def _slide_window(self, window, doc_num):
        if False:
            while True:
                i = 10
        if doc_num != self._current_doc_num:
            self._uniq_words[:] = False
            self._uniq_words[np.unique(window)] = True
            self._current_doc_num = doc_num
        else:
            self._uniq_words[self._token_at_edge] = False
            self._uniq_words[window[-1]] = True
        self._token_at_edge = window[0]

    def _symmetrize(self):
        if False:
            print('Hello World!')
        'Word pairs may have been encountered in (i, j) and (j, i) order.\n\n        Notes\n        -----\n        Rather than enforcing a particular ordering during the update process,\n        we choose to symmetrize the co-occurrence matrix after accumulation has completed.\n\n        '
        co_occ = self._co_occurrences
        co_occ.setdiag(self._occurrences)
        self._co_occurrences = co_occ + co_occ.T - sps.diags(co_occ.diagonal(), offsets=0, dtype='uint32')

    def _get_occurrences(self, word_id):
        if False:
            return 10
        return self._occurrences[word_id]

    def _get_co_occurrences(self, word_id1, word_id2):
        if False:
            return 10
        return self._co_occurrences[word_id1, word_id2]

    def merge(self, other):
        if False:
            return 10
        self._occurrences += other._occurrences
        self._co_occurrences += other._co_occurrences
        self._num_docs += other._num_docs

class PatchedWordOccurrenceAccumulator(WordOccurrenceAccumulator):
    """Monkey patched for multiprocessing worker usage, to move some of the logic to the master process."""

    def _iter_texts(self, texts):
        if False:
            return 10
        return texts

class ParallelWordOccurrenceAccumulator(WindowedTextsAnalyzer):
    """Accumulate word occurrences in parallel.

    Attributes
    ----------
    processes : int
        Number of processes to use; must be at least two.
    args :
        Should include `relevant_ids` and `dictionary` (see :class:`~UsesDictionary.__init__`).
    kwargs :
        Can include `batch_size`, which is the number of docs to send to a worker at a time.
        If not included, it defaults to 64.
    """

    def __init__(self, processes, *args, **kwargs):
        if False:
            return 10
        super(ParallelWordOccurrenceAccumulator, self).__init__(*args)
        if processes < 2:
            raise ValueError('Must have at least 2 processes to run in parallel; got %d' % processes)
        self.processes = processes
        self.batch_size = kwargs.get('batch_size', 64)

    def __str__(self):
        if False:
            return 10
        return '%s<processes=%s, batch_size=%s>' % (self.__class__.__name__, self.processes, self.batch_size)

    def accumulate(self, texts, window_size):
        if False:
            return 10
        (workers, input_q, output_q) = self.start_workers(window_size)
        try:
            self.queue_all_texts(input_q, texts, window_size)
            interrupted = False
        except KeyboardInterrupt:
            logger.warn('stats accumulation interrupted; <= %d documents processed', self._num_docs)
            interrupted = True
        accumulators = self.terminate_workers(input_q, output_q, workers, interrupted)
        return self.merge_accumulators(accumulators)

    def start_workers(self, window_size):
        if False:
            while True:
                i = 10
        'Set up an input and output queue and start processes for each worker.\n\n        Notes\n        -----\n        The input queue is used to transmit batches of documents to the workers.\n        The output queue is used by workers to transmit the WordOccurrenceAccumulator instances.\n\n        Parameters\n        ----------\n        window_size : int\n\n        Returns\n        -------\n        (list of lists)\n            Tuple of (list of workers, input queue, output queue).\n        '
        input_q = mp.Queue(maxsize=self.processes)
        output_q = mp.Queue()
        workers = []
        for _ in range(self.processes):
            accumulator = PatchedWordOccurrenceAccumulator(self.relevant_ids, self.dictionary)
            worker = AccumulatingWorker(input_q, output_q, accumulator, window_size)
            worker.start()
            workers.append(worker)
        return (workers, input_q, output_q)

    def yield_batches(self, texts):
        if False:
            while True:
                i = 10
        'Return a generator over the given texts that yields batches of `batch_size` texts at a time.'
        batch = []
        for text in self._iter_texts(texts):
            batch.append(text)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    def queue_all_texts(self, q, texts, window_size):
        if False:
            i = 10
            return i + 15
        'Sequentially place batches of texts on the given queue until `texts` is consumed.\n        The texts are filtered so that only those with at least one relevant token are queued.\n        '
        for (batch_num, batch) in enumerate(self.yield_batches(texts)):
            q.put(batch, block=True)
            before = self._num_docs / self.log_every
            self._num_docs += sum((len(doc) - window_size + 1 for doc in batch))
            if before < self._num_docs / self.log_every:
                logger.info('%d batches submitted to accumulate stats from %d documents (%d virtual)', batch_num + 1, (batch_num + 1) * self.batch_size, self._num_docs)

    def terminate_workers(self, input_q, output_q, workers, interrupted=False):
        if False:
            for i in range(10):
                print('nop')
        'Wait until all workers have transmitted their WordOccurrenceAccumulator instances, then terminate each.\n\n        Warnings\n        --------\n        We do not use join here because it has been shown to have some issues\n        in Python 2.7 (and even in later versions). This method also closes both the input and output queue.\n        If `interrupted` is False (normal execution), a None value is placed on the input queue for\n        each worker. The workers are looking for this sentinel value and interpret it as a signal to\n        terminate themselves. If `interrupted` is True, a KeyboardInterrupt occurred. The workers are\n        programmed to recover from this and continue on to transmit their results before terminating.\n        So in this instance, the sentinel values are not queued, but the rest of the execution\n        continues as usual.\n\n        '
        if not interrupted:
            for _ in workers:
                input_q.put(None, block=True)
        accumulators = []
        while len(accumulators) != len(workers):
            accumulators.append(output_q.get())
        logger.info('%d accumulators retrieved from output queue', len(accumulators))
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
        input_q.close()
        output_q.close()
        return accumulators

    def merge_accumulators(self, accumulators):
        if False:
            i = 10
            return i + 15
        'Merge the list of accumulators into a single `WordOccurrenceAccumulator` with all\n        occurrence and co-occurrence counts, and a `num_docs` that reflects the total observed\n        by all the individual accumulators.\n\n        '
        accumulator = WordOccurrenceAccumulator(self.relevant_ids, self.dictionary)
        for other_accumulator in accumulators:
            accumulator.merge(other_accumulator)
        accumulator._symmetrize()
        logger.info('accumulated word occurrence stats for %d virtual documents', accumulator.num_docs)
        return accumulator

class AccumulatingWorker(mp.Process):
    """Accumulate stats from texts fed in from queue."""

    def __init__(self, input_q, output_q, accumulator, window_size):
        if False:
            for i in range(10):
                print('nop')
        super(AccumulatingWorker, self).__init__()
        self.input_q = input_q
        self.output_q = output_q
        self.accumulator = accumulator
        self.accumulator.log_every = sys.maxsize
        self.window_size = window_size

    def run(self):
        if False:
            print('Hello World!')
        try:
            self._run()
        except KeyboardInterrupt:
            logger.info('%s interrupted after processing %d documents', self.__class__.__name__, self.accumulator.num_docs)
        except Exception:
            logger.exception('worker encountered unexpected exception')
        finally:
            self.reply_to_master()

    def _run(self):
        if False:
            return 10
        batch_num = -1
        n_docs = 0
        while True:
            batch_num += 1
            docs = self.input_q.get(block=True)
            if docs is None:
                logger.debug('observed sentinel value; terminating')
                break
            self.accumulator.partial_accumulate(docs, self.window_size)
            n_docs += len(docs)
            logger.debug('completed batch %d; %d documents processed (%d virtual)', batch_num, n_docs, self.accumulator.num_docs)
        logger.debug('finished all batches; %d documents processed (%d virtual)', n_docs, self.accumulator.num_docs)

    def reply_to_master(self):
        if False:
            print('Hello World!')
        logger.info('serializing accumulator to return to master...')
        self.output_q.put(self.accumulator, block=False)
        logger.info('accumulator serialized')

class WordVectorsAccumulator(UsesDictionary):
    """Accumulate context vectors for words using word vector embeddings.

    Attributes
    ----------
    model: Word2Vec (:class:`~gensim.models.keyedvectors.KeyedVectors`)
        If None, a new Word2Vec model is trained on the given text corpus. Otherwise,
        it should be a pre-trained Word2Vec context vectors.
    model_kwargs:
        if model is None, these keyword arguments will be passed through to the Word2Vec constructor.
    """

    def __init__(self, relevant_ids, dictionary, model=None, **model_kwargs):
        if False:
            print('Hello World!')
        super(WordVectorsAccumulator, self).__init__(relevant_ids, dictionary)
        self.model = model
        self.model_kwargs = model_kwargs

    def not_in_vocab(self, words):
        if False:
            return 10
        uniq_words = set(utils.flatten(words))
        return set((word for word in uniq_words if word not in self.model))

    def get_occurrences(self, word):
        if False:
            while True:
                i = 10
        'Return number of docs the word occurs in, once `accumulate` has been called.'
        try:
            self.token2id[word]
        except KeyError:
            word = self.dictionary.id2token[word]
        return self.model.get_vecattr(word, 'count')

    def get_co_occurrences(self, word1, word2):
        if False:
            return 10
        'Return number of docs the words co-occur in, once `accumulate` has been called.'
        raise NotImplementedError('Word2Vec model does not support co-occurrence counting')

    def accumulate(self, texts, window_size):
        if False:
            while True:
                i = 10
        if self.model is not None:
            logger.debug('model is already trained; no accumulation necessary')
            return self
        kwargs = self.model_kwargs.copy()
        if window_size is not None:
            kwargs['window'] = window_size
        kwargs['min_count'] = kwargs.get('min_count', 1)
        kwargs['sg'] = kwargs.get('sg', 1)
        kwargs['hs'] = kwargs.get('hw', 0)
        self.model = Word2Vec(**kwargs)
        self.model.build_vocab(texts)
        self.model.train(texts, total_examples=self.model.corpus_count, epochs=self.model.epochs)
        self.model = self.model.wv
        return self

    def ids_similarity(self, ids1, ids2):
        if False:
            for i in range(10):
                print('nop')
        words1 = self._words_with_embeddings(ids1)
        words2 = self._words_with_embeddings(ids2)
        return self.model.n_similarity(words1, words2)

    def _words_with_embeddings(self, ids):
        if False:
            for i in range(10):
                print('nop')
        if not hasattr(ids, '__iter__'):
            ids = [ids]
        words = [self.dictionary.id2token[word_id] for word_id in ids]
        return [word for word in words if word in self.model]