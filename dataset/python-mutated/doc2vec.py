"""
Introduction
============

Learn paragraph and document embeddings via the distributed memory and distributed bag of words models from
`Quoc Le and Tomas Mikolov: "Distributed Representations of Sentences and Documents"
<http://arxiv.org/pdf/1405.4053v2.pdf>`_.

The algorithms use either hierarchical softmax or negative sampling; see
`Tomas Mikolov, Kai Chen, Greg Corrado, and Jeffrey Dean: "Efficient Estimation of Word Representations in
Vector Space, in Proceedings of Workshop at ICLR, 2013" <https://arxiv.org/pdf/1301.3781.pdf>`_ and
`Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg Corrado, and Jeffrey Dean: "Distributed Representations of Words
and Phrases and their Compositionality. In Proceedings of NIPS, 2013"
<https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf>`_.

For a usage example, see the `Doc2vec tutorial
<https://radimrehurek.com/gensim/auto_examples/tutorials/run_doc2vec_lee.html#sphx-glr-auto-examples-tutorials-run-doc2vec-lee-py>`_.

**Make sure you have a C compiler before installing Gensim, to use the optimized doc2vec routines** (70x speedup
compared to plain NumPy implementation, https://rare-technologies.com/parallelizing-word2vec-in-python/).


Usage examples
==============

Initialize & train a model:

.. sourcecode:: pycon

    >>> from gensim.test.utils import common_texts
    >>> from gensim.models.doc2vec import Doc2Vec, TaggedDocument
    >>>
    >>> documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(common_texts)]
    >>> model = Doc2Vec(documents, vector_size=5, window=2, min_count=1, workers=4)

Persist a model to disk:

.. sourcecode:: pycon

    >>> from gensim.test.utils import get_tmpfile
    >>>
    >>> fname = get_tmpfile("my_doc2vec_model")
    >>>
    >>> model.save(fname)
    >>> model = Doc2Vec.load(fname)  # you can continue training with the loaded model!

Infer vector for a new document:

.. sourcecode:: pycon

    >>> vector = model.infer_vector(["system", "response"])

"""
import logging
import os
from collections import namedtuple, defaultdict
from collections.abc import Iterable
from timeit import default_timer
from dataclasses import dataclass
from numpy import zeros, float32 as REAL, vstack, integer, dtype
import numpy as np
from gensim import utils, matutils
from gensim.utils import deprecated
from gensim.models import Word2Vec, FAST_VERSION
from gensim.models.keyedvectors import KeyedVectors, pseudorandom_weak_vector
logger = logging.getLogger(__name__)
try:
    from gensim.models.doc2vec_inner import train_document_dbow, train_document_dm, train_document_dm_concat
except ImportError:
    raise utils.NO_CYTHON
try:
    from gensim.models.doc2vec_corpusfile import d2v_train_epoch_dbow, d2v_train_epoch_dm_concat, d2v_train_epoch_dm, CORPUSFILE_VERSION
except ImportError:
    CORPUSFILE_VERSION = -1

    def d2v_train_epoch_dbow(model, corpus_file, offset, start_doctag, _cython_vocab, _cur_epoch, _expected_examples, _expected_words, work, _neu1, docvecs_count, word_vectors=None, word_locks=None, train_words=False, learn_doctags=True, learn_words=True, learn_hidden=True, doctag_vectors=None, doctag_locks=None):
        if False:
            return 10
        raise NotImplementedError('Training with corpus_file argument is not supported.')

    def d2v_train_epoch_dm_concat(model, corpus_file, offset, start_doctag, _cython_vocab, _cur_epoch, _expected_examples, _expected_words, work, _neu1, docvecs_count, word_vectors=None, word_locks=None, learn_doctags=True, learn_words=True, learn_hidden=True, doctag_vectors=None, doctag_locks=None):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError('Training with corpus_file argument is not supported.')

    def d2v_train_epoch_dm(model, corpus_file, offset, start_doctag, _cython_vocab, _cur_epoch, _expected_examples, _expected_words, work, _neu1, docvecs_count, word_vectors=None, word_locks=None, learn_doctags=True, learn_words=True, learn_hidden=True, doctag_vectors=None, doctag_locks=None):
        if False:
            print('Hello World!')
        raise NotImplementedError('Training with corpus_file argument is not supported.')

class TaggedDocument(namedtuple('TaggedDocument', 'words tags')):
    """Represents a document along with a tag, input document format for :class:`~gensim.models.doc2vec.Doc2Vec`.

    A single document, made up of `words` (a list of unicode string tokens) and `tags` (a list of tokens).
    Tags may be one or more unicode string tokens, but typical practice (which will also be the most memory-efficient)
    is for the tags list to include a unique integer id as the only tag.

    Replaces "sentence as a list of words" from :class:`gensim.models.word2vec.Word2Vec`.

    """

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        "Human readable representation of the object's state, used for debugging.\n\n        Returns\n        -------\n        str\n           Human readable representation of the object's state (words and tags).\n\n        "
        return '%s<%s, %s>' % (self.__class__.__name__, self.words, self.tags)

@dataclass
class Doctag:
    """A dataclass shape-compatible with keyedvectors.SimpleVocab, extended to record
    details of string document tags discovered during the initial vocabulary scan.

    Will not be used if all presented document tags are ints. No longer used in a
    completed model: just used during initial scan, and for backward compatibility.
    """
    __slots__ = ('doc_count', 'index', 'word_count')
    doc_count: int
    index: int
    word_count: int

    @property
    def count(self):
        if False:
            while True:
                i = 10
        return self.doc_count

    @count.setter
    def count(self, new_val):
        if False:
            while True:
                i = 10
        self.doc_count = new_val

class Doc2Vec(Word2Vec):

    def __init__(self, documents=None, corpus_file=None, vector_size=100, dm_mean=None, dm=1, dbow_words=0, dm_concat=0, dm_tag_count=1, dv=None, dv_mapfile=None, comment=None, trim_rule=None, callbacks=(), window=5, epochs=10, shrink_windows=True, **kwargs):
        if False:
            print('Hello World!')
        'Class for training, using and evaluating neural networks described in\n        `Distributed Representations of Sentences and Documents <http://arxiv.org/abs/1405.4053v2>`_.\n\n        Parameters\n        ----------\n        documents : iterable of list of :class:`~gensim.models.doc2vec.TaggedDocument`, optional\n            Input corpus, can be simply a list of elements, but for larger corpora,consider an iterable that streams\n            the documents directly from disk/network. If you don\'t supply `documents` (or `corpus_file`), the model is\n            left uninitialized -- use if you plan to initialize it in some other way.\n        corpus_file : str, optional\n            Path to a corpus file in :class:`~gensim.models.word2vec.LineSentence` format.\n            You may use this argument instead of `documents` to get performance boost. Only one of `documents` or\n            `corpus_file` arguments need to be passed (or none of them, in that case, the model is left uninitialized).\n            Documents\' tags are assigned automatically and are equal to line number, as in\n            :class:`~gensim.models.doc2vec.TaggedLineDocument`.\n        dm : {1,0}, optional\n            Defines the training algorithm. If `dm=1`, \'distributed memory\' (PV-DM) is used.\n            Otherwise, `distributed bag of words` (PV-DBOW) is employed.\n        vector_size : int, optional\n            Dimensionality of the feature vectors.\n        window : int, optional\n            The maximum distance between the current and predicted word within a sentence.\n        alpha : float, optional\n            The initial learning rate.\n        min_alpha : float, optional\n            Learning rate will linearly drop to `min_alpha` as training progresses.\n        seed : int, optional\n            Seed for the random number generator. Initial vectors for each word are seeded with a hash of\n            the concatenation of word + `str(seed)`. Note that for a fully deterministically-reproducible run,\n            you must also limit the model to a single worker thread (`workers=1`), to eliminate ordering jitter\n            from OS thread scheduling.\n            In Python 3, reproducibility between interpreter launches also requires use of the `PYTHONHASHSEED`\n            environment variable to control hash randomization.\n        min_count : int, optional\n            Ignores all words with total frequency lower than this.\n        max_vocab_size : int, optional\n            Limits the RAM during vocabulary building; if there are more unique\n            words than this, then prune the infrequent ones. Every 10 million word types need about 1GB of RAM.\n            Set to `None` for no limit.\n        sample : float, optional\n            The threshold for configuring which higher-frequency words are randomly downsampled,\n            useful range is (0, 1e-5).\n        workers : int, optional\n            Use these many worker threads to train the model (=faster training with multicore machines).\n        epochs : int, optional\n            Number of iterations (epochs) over the corpus. Defaults to 10 for Doc2Vec.\n        hs : {1,0}, optional\n            If 1, hierarchical softmax will be used for model training.\n            If set to 0, and `negative` is non-zero, negative sampling will be used.\n        negative : int, optional\n            If > 0, negative sampling will be used, the int for negative specifies how many "noise words"\n            should be drawn (usually between 5-20).\n            If set to 0, no negative sampling is used.\n        ns_exponent : float, optional\n            The exponent used to shape the negative sampling distribution. A value of 1.0 samples exactly in proportion\n            to the frequencies, 0.0 samples all words equally, while a negative value samples low-frequency words more\n            than high-frequency words. The popular default value of 0.75 was chosen by the original Word2Vec paper.\n            More recently, in https://arxiv.org/abs/1804.04212, Caselles-Dupré, Lesaint, & Royo-Letelier suggest that\n            other values may perform better for recommendation applications.\n        dm_mean : {1,0}, optional\n            If 0, use the sum of the context word vectors. If 1, use the mean.\n            Only applies when `dm` is used in non-concatenative mode.\n        dm_concat : {1,0}, optional\n            If 1, use concatenation of context vectors rather than sum/average;\n            Note concatenation results in a much-larger model, as the input\n            is no longer the size of one (sampled or arithmetically combined) word vector, but the\n            size of the tag(s) and all words in the context strung together.\n        dm_tag_count : int, optional\n            Expected constant number of document tags per document, when using\n            dm_concat mode.\n        dbow_words : {1,0}, optional\n            If set to 1 trains word-vectors (in skip-gram fashion) simultaneous with DBOW\n            doc-vector training; If 0, only trains doc-vectors (faster).\n        trim_rule : function, optional\n            Vocabulary trimming rule, specifies whether certain words should remain in the vocabulary,\n            be trimmed away, or handled using the default (discard if word count < min_count).\n            Can be None (min_count will be used, look to :func:`~gensim.utils.keep_vocab_item`),\n            or a callable that accepts parameters (word, count, min_count) and returns either\n            :attr:`gensim.utils.RULE_DISCARD`, :attr:`gensim.utils.RULE_KEEP` or :attr:`gensim.utils.RULE_DEFAULT`.\n            The rule, if given, is only used to prune vocabulary during current method call and is not stored as part\n            of the model.\n\n            The input parameters are of the following types:\n                * `word` (str) - the word we are examining\n                * `count` (int) - the word\'s frequency count in the corpus\n                * `min_count` (int) - the minimum count threshold.\n\n        callbacks : :obj: `list` of :obj: `~gensim.models.callbacks.CallbackAny2Vec`, optional\n            List of callbacks that need to be executed/run at specific stages during training.\n        shrink_windows : bool, optional\n            New in 4.1. Experimental.\n            If True, the effective window size is uniformly sampled from  [1, `window`]\n            for each target word during training, to match the original word2vec algorithm\'s\n            approximate weighting of context words by distance. Otherwise, the effective\n            window size is always fixed to `window` words to either side.\n\n        Some important internal attributes are the following:\n\n        Attributes\n        ----------\n        wv : :class:`~gensim.models.keyedvectors.KeyedVectors`\n            This object essentially contains the mapping between words and embeddings. After training, it can be used\n            directly to query those embeddings in various ways. See the module level docstring for examples.\n\n        dv : :class:`~gensim.models.keyedvectors.KeyedVectors`\n            This object contains the paragraph vectors learned from the training data. There will be one such vector\n            for each unique document tag supplied during training. They may be individually accessed using the tag\n            as an indexed-access key. For example, if one of the training documents used a tag of \'doc003\':\n\n            .. sourcecode:: pycon\n\n                >>> model.dv[\'doc003\']\n\n        '
        corpus_iterable = documents
        if dm_mean is not None:
            self.cbow_mean = dm_mean
        self.dbow_words = int(dbow_words)
        self.dm_concat = int(dm_concat)
        self.dm_tag_count = int(dm_tag_count)
        if dm and dm_concat:
            self.layer1_size = (dm_tag_count + 2 * window) * vector_size
            logger.info('using concatenative %d-dimensional layer1', self.layer1_size)
        self.vector_size = vector_size
        self.dv = dv or KeyedVectors(self.vector_size, mapfile_path=dv_mapfile)
        self.dv.vectors_lockf = np.ones(1, dtype=REAL)
        super(Doc2Vec, self).__init__(sentences=corpus_iterable, corpus_file=corpus_file, vector_size=self.vector_size, sg=(1 + dm) % 2, null_word=self.dm_concat, callbacks=callbacks, window=window, epochs=epochs, shrink_windows=shrink_windows, **kwargs)

    @property
    def dm(self):
        if False:
            for i in range(10):
                print('nop')
        "Indicates whether 'distributed memory' (PV-DM) will be used, else 'distributed bag of words'\n        (PV-DBOW) is used.\n\n        "
        return not self.sg

    @property
    def dbow(self):
        if False:
            return 10
        "Indicates whether 'distributed bag of words' (PV-DBOW) will be used, else 'distributed memory'\n        (PV-DM) is used.\n\n        "
        return self.sg

    @property
    @deprecated('The `docvecs` property has been renamed `dv`.')
    def docvecs(self):
        if False:
            i = 10
            return i + 15
        return self.dv

    @docvecs.setter
    @deprecated('The `docvecs` property has been renamed `dv`.')
    def docvecs(self, value):
        if False:
            print('Hello World!')
        self.dv = value

    def _clear_post_train(self):
        if False:
            i = 10
            return i + 15
        'Resets the current word vectors. '
        self.wv.norms = None
        self.dv.norms = None

    def init_weights(self):
        if False:
            return 10
        super(Doc2Vec, self).init_weights()
        self.dv.resize_vectors(seed=self.seed + 7919)

    def reset_from(self, other_model):
        if False:
            print('Hello World!')
        "Copy shareable data structures from another (possibly pre-trained) model.\n\n        This specifically causes some structures to be shared, so is limited to\n        structures (like those rleated to the known word/tag vocabularies) that\n        won't change during training or thereafter. Beware vocabulary edits/updates\n        to either model afterwards: the partial sharing and out-of-band modification\n        may leave the other model in a broken state.\n\n        Parameters\n        ----------\n        other_model : :class:`~gensim.models.doc2vec.Doc2Vec`\n            Other model whose internal data structures will be copied over to the current object.\n\n        "
        self.wv.key_to_index = other_model.wv.key_to_index
        self.wv.index_to_key = other_model.wv.index_to_key
        self.wv.expandos = other_model.wv.expandos
        self.cum_table = other_model.cum_table
        self.corpus_count = other_model.corpus_count
        self.dv.key_to_index = other_model.dv.key_to_index
        self.dv.index_to_key = other_model.dv.index_to_key
        self.dv.expandos = other_model.dv.expandos
        self.init_weights()

    def _do_train_epoch(self, corpus_file, thread_id, offset, cython_vocab, thread_private_mem, cur_epoch, total_examples=None, total_words=None, offsets=None, start_doctags=None, **kwargs):
        if False:
            print('Hello World!')
        (work, neu1) = thread_private_mem
        doctag_vectors = self.dv.vectors
        doctags_lockf = self.dv.vectors_lockf
        offset = offsets[thread_id]
        start_doctag = start_doctags[thread_id]
        if self.sg:
            (examples, tally, raw_tally) = d2v_train_epoch_dbow(self, corpus_file, offset, start_doctag, cython_vocab, cur_epoch, total_examples, total_words, work, neu1, len(self.dv), doctag_vectors=doctag_vectors, doctags_lockf=doctags_lockf, train_words=self.dbow_words)
        elif self.dm_concat:
            (examples, tally, raw_tally) = d2v_train_epoch_dm_concat(self, corpus_file, offset, start_doctag, cython_vocab, cur_epoch, total_examples, total_words, work, neu1, len(self.dv), doctag_vectors=doctag_vectors, doctags_lockf=doctags_lockf)
        else:
            (examples, tally, raw_tally) = d2v_train_epoch_dm(self, corpus_file, offset, start_doctag, cython_vocab, cur_epoch, total_examples, total_words, work, neu1, len(self.dv), doctag_vectors=doctag_vectors, doctags_lockf=doctags_lockf)
        return (examples, tally, raw_tally)

    def _do_train_job(self, job, alpha, inits):
        if False:
            return 10
        'Train model using `job` data.\n\n        Parameters\n        ----------\n        job : iterable of list of :class:`~gensim.models.doc2vec.TaggedDocument`\n            The corpus chunk to be used for training this batch.\n        alpha : float\n            Learning rate to be used for training this batch.\n        inits : (np.ndarray, np.ndarray)\n            Each worker threads private work memory.\n\n        Returns\n        -------\n        (int, int)\n             2-tuple (effective word count after ignoring unknown words and sentence length trimming, total word count).\n\n        '
        (work, neu1) = inits
        tally = 0
        for doc in job:
            doctag_indexes = [self.dv.get_index(tag) for tag in doc.tags if tag in self.dv]
            doctag_vectors = self.dv.vectors
            doctags_lockf = self.dv.vectors_lockf
            if self.sg:
                tally += train_document_dbow(self, doc.words, doctag_indexes, alpha, work, train_words=self.dbow_words, doctag_vectors=doctag_vectors, doctags_lockf=doctags_lockf)
            elif self.dm_concat:
                tally += train_document_dm_concat(self, doc.words, doctag_indexes, alpha, work, neu1, doctag_vectors=doctag_vectors, doctags_lockf=doctags_lockf)
            else:
                tally += train_document_dm(self, doc.words, doctag_indexes, alpha, work, neu1, doctag_vectors=doctag_vectors, doctags_lockf=doctags_lockf)
        return (tally, self._raw_word_count(job))

    def train(self, corpus_iterable=None, corpus_file=None, total_examples=None, total_words=None, epochs=None, start_alpha=None, end_alpha=None, word_count=0, queue_factor=2, report_delay=1.0, callbacks=(), **kwargs):
        if False:
            return 10
        "Update the model's neural weights.\n\n        To support linear learning-rate decay from (initial) `alpha` to `min_alpha`, and accurate\n        progress-percentage logging, either `total_examples` (count of documents) or `total_words` (count of\n        raw words in documents) **MUST** be provided. If `documents` is the same corpus\n        that was provided to :meth:`~gensim.models.word2vec.Word2Vec.build_vocab` earlier,\n        you can simply use `total_examples=self.corpus_count`.\n\n        To avoid common mistakes around the model's ability to do multiple training passes itself, an\n        explicit `epochs` argument **MUST** be provided. In the common and recommended case\n        where :meth:`~gensim.models.word2vec.Word2Vec.train` is only called once,\n        you can set `epochs=self.iter`.\n\n        Parameters\n        ----------\n        corpus_iterable : iterable of list of :class:`~gensim.models.doc2vec.TaggedDocument`, optional\n            Can be simply a list of elements, but for larger corpora,consider an iterable that streams\n            the documents directly from disk/network. If you don't supply `documents` (or `corpus_file`), the model is\n            left uninitialized -- use if you plan to initialize it in some other way.\n        corpus_file : str, optional\n            Path to a corpus file in :class:`~gensim.models.word2vec.LineSentence` format.\n            You may use this argument instead of `documents` to get performance boost. Only one of `documents` or\n            `corpus_file` arguments need to be passed (not both of them). Documents' tags are assigned automatically\n            and are equal to line number, as in :class:`~gensim.models.doc2vec.TaggedLineDocument`.\n        total_examples : int, optional\n            Count of documents.\n        total_words : int, optional\n            Count of raw words in documents.\n        epochs : int, optional\n            Number of iterations (epochs) over the corpus.\n        start_alpha : float, optional\n            Initial learning rate. If supplied, replaces the starting `alpha` from the constructor,\n            for this one call to `train`.\n            Use only if making multiple calls to `train`, when you want to manage the alpha learning-rate yourself\n            (not recommended).\n        end_alpha : float, optional\n            Final learning rate. Drops linearly from `start_alpha`.\n            If supplied, this replaces the final `min_alpha` from the constructor, for this one call to\n            :meth:`~gensim.models.doc2vec.Doc2Vec.train`.\n            Use only if making multiple calls to :meth:`~gensim.models.doc2vec.Doc2Vec.train`, when you want to manage\n            the alpha learning-rate yourself (not recommended).\n        word_count : int, optional\n            Count of words already trained. Set this to 0 for the usual\n            case of training on all words in documents.\n        queue_factor : int, optional\n            Multiplier for size of queue (number of workers * queue_factor).\n        report_delay : float, optional\n            Seconds to wait before reporting progress.\n        callbacks : :obj: `list` of :obj: `~gensim.models.callbacks.CallbackAny2Vec`, optional\n            List of callbacks that need to be executed/run at specific stages during training.\n\n        "
        if corpus_file is None and corpus_iterable is None:
            raise TypeError('Either one of corpus_file or corpus_iterable value must be provided')
        if corpus_file is not None and corpus_iterable is not None:
            raise TypeError('Both corpus_file and corpus_iterable must not be provided at the same time')
        if corpus_iterable is None and (not os.path.isfile(corpus_file)):
            raise TypeError('Parameter corpus_file must be a valid path to a file, got %r instead' % corpus_file)
        if corpus_iterable is not None and (not isinstance(corpus_iterable, Iterable)):
            raise TypeError('corpus_iterable must be an iterable of TaggedDocument, got %r instead' % corpus_iterable)
        if corpus_file is not None:
            (offsets, start_doctags) = self._get_offsets_and_start_doctags_for_corpusfile(corpus_file, self.workers)
            kwargs['offsets'] = offsets
            kwargs['start_doctags'] = start_doctags
        super(Doc2Vec, self).train(corpus_iterable=corpus_iterable, corpus_file=corpus_file, total_examples=total_examples, total_words=total_words, epochs=epochs, start_alpha=start_alpha, end_alpha=end_alpha, word_count=word_count, queue_factor=queue_factor, report_delay=report_delay, callbacks=callbacks, **kwargs)

    @classmethod
    def _get_offsets_and_start_doctags_for_corpusfile(cls, corpus_file, workers):
        if False:
            i = 10
            return i + 15
        'Get offset and initial document tag in a corpus_file for each worker.\n\n        Firstly, approximate offsets are calculated based on number of workers and corpus_file size.\n        Secondly, for each approximate offset we find the maximum offset which points to the beginning of line and\n        less than approximate offset.\n\n        Parameters\n        ----------\n        corpus_file : str\n            Path to a corpus file in :class:`~gensim.models.word2vec.LineSentence` format.\n        workers : int\n            Number of workers.\n\n        Returns\n        -------\n        list of int, list of int\n            Lists with offsets and document tags with length = number of workers.\n        '
        corpus_file_size = os.path.getsize(corpus_file)
        approx_offsets = [int(corpus_file_size // workers * i) for i in range(workers)]
        offsets = []
        start_doctags = []
        with utils.open(corpus_file, mode='rb') as fin:
            curr_offset_idx = 0
            prev_filepos = 0
            for (line_no, line) in enumerate(fin):
                if curr_offset_idx == len(approx_offsets):
                    break
                curr_filepos = prev_filepos + len(line)
                while curr_offset_idx != len(approx_offsets) and approx_offsets[curr_offset_idx] < curr_filepos:
                    offsets.append(prev_filepos)
                    start_doctags.append(line_no)
                    curr_offset_idx += 1
                prev_filepos = curr_filepos
        return (offsets, start_doctags)

    def _raw_word_count(self, job):
        if False:
            i = 10
            return i + 15
        'Get the number of words in a given job.\n\n        Parameters\n        ----------\n        job : iterable of list of :class:`~gensim.models.doc2vec.TaggedDocument`\n            Corpus chunk.\n\n        Returns\n        -------\n        int\n            Number of raw words in the corpus chunk.\n\n        '
        return sum((len(sentence.words) for sentence in job))

    def estimated_lookup_memory(self):
        if False:
            i = 10
            return i + 15
        'Get estimated memory for tag lookup, 0 if using pure int tags.\n\n        Returns\n        -------\n        int\n            The estimated RAM required to look up a tag in bytes.\n\n        '
        return 60 * len(self.dv) + 140 * len(self.dv)

    def infer_vector(self, doc_words, alpha=None, min_alpha=None, epochs=None):
        if False:
            while True:
                i = 10
        'Infer a vector for given post-bulk training document.\n\n        Notes\n        -----\n        Subsequent calls to this function may infer different representations for the same document.\n        For a more stable representation, increase the number of epochs to assert a stricter convergence.\n\n        Parameters\n        ----------\n        doc_words : list of str\n            A document for which the vector representation will be inferred.\n        alpha : float, optional\n            The initial learning rate. If unspecified, value from model initialization will be reused.\n        min_alpha : float, optional\n            Learning rate will linearly drop to `min_alpha` over all inference epochs. If unspecified,\n            value from model initialization will be reused.\n        epochs : int, optional\n            Number of times to train the new document. Larger values take more time, but may improve\n            quality and run-to-run stability of inferred vectors. If unspecified, the `epochs` value\n            from model initialization will be reused.\n\n        Returns\n        -------\n        np.ndarray\n            The inferred paragraph vector for the new document.\n\n        '
        if isinstance(doc_words, str):
            raise TypeError('Parameter doc_words of infer_vector() must be a list of strings (not a single string).')
        alpha = alpha or self.alpha
        min_alpha = min_alpha or self.min_alpha
        epochs = epochs or self.epochs
        doctag_vectors = pseudorandom_weak_vector(self.dv.vector_size, seed_string=' '.join(doc_words))
        doctag_vectors = doctag_vectors.reshape(1, self.dv.vector_size)
        doctags_lockf = np.ones(1, dtype=REAL)
        doctag_indexes = [0]
        work = zeros(self.layer1_size, dtype=REAL)
        if not self.sg:
            neu1 = matutils.zeros_aligned(self.layer1_size, dtype=REAL)
        alpha_delta = (alpha - min_alpha) / max(epochs - 1, 1)
        for i in range(epochs):
            if self.sg:
                train_document_dbow(self, doc_words, doctag_indexes, alpha, work, learn_words=False, learn_hidden=False, doctag_vectors=doctag_vectors, doctags_lockf=doctags_lockf)
            elif self.dm_concat:
                train_document_dm_concat(self, doc_words, doctag_indexes, alpha, work, neu1, learn_words=False, learn_hidden=False, doctag_vectors=doctag_vectors, doctags_lockf=doctags_lockf)
            else:
                train_document_dm(self, doc_words, doctag_indexes, alpha, work, neu1, learn_words=False, learn_hidden=False, doctag_vectors=doctag_vectors, doctags_lockf=doctags_lockf)
            alpha -= alpha_delta
        return doctag_vectors[0]

    def __getitem__(self, tag):
        if False:
            while True:
                i = 10
        'Get the vector representation of (possibly multi-term) tag.\n\n        Parameters\n        ----------\n        tag : {str, int, list of str, list of int}\n            The tag (or tags) to be looked up in the model.\n\n        Returns\n        -------\n        np.ndarray\n            The vector representations of each tag as a matrix (will be 1D if `tag` was a single tag)\n\n        '
        if isinstance(tag, (str, int, integer)):
            if tag not in self.wv:
                return self.dv[tag]
            return self.wv[tag]
        return vstack([self[i] for i in tag])

    def __str__(self):
        if False:
            return 10
        'Abbreviated name reflecting major configuration parameters.\n\n        Returns\n        -------\n        str\n            Human readable representation of the models internal state.\n\n        '
        segments = []
        if self.comment:
            segments.append('"%s"' % self.comment)
        if self.sg:
            if self.dbow_words:
                segments.append('dbow+w')
            else:
                segments.append('dbow')
        elif self.dm_concat:
            segments.append('dm/c')
        elif self.cbow_mean:
            segments.append('dm/m')
        else:
            segments.append('dm/s')
        segments.append('d%d' % self.dv.vector_size)
        if self.negative:
            segments.append('n%d' % self.negative)
        if self.hs:
            segments.append('hs')
        if not self.sg or (self.sg and self.dbow_words):
            segments.append('w%d' % self.window)
        if self.min_count > 1:
            segments.append('mc%d' % self.min_count)
        if self.sample > 0:
            segments.append('s%g' % self.sample)
        if self.workers > 1:
            segments.append('t%d' % self.workers)
        return '%s<%s>' % (self.__class__.__name__, ','.join(segments))

    def save_word2vec_format(self, fname, doctag_vec=False, word_vec=True, prefix='*dt_', fvocab=None, binary=False):
        if False:
            print('Hello World!')
        'Store the input-hidden weight matrix in the same format used by the original C word2vec-tool.\n\n        Parameters\n        ----------\n        fname : str\n            The file path used to save the vectors in.\n        doctag_vec : bool, optional\n            Indicates whether to store document vectors.\n        word_vec : bool, optional\n            Indicates whether to store word vectors.\n        prefix : str, optional\n            Uniquely identifies doctags from word vocab, and avoids collision in case of repeated string in doctag\n            and word vocab.\n        fvocab : str, optional\n            Optional file path used to save the vocabulary.\n        binary : bool, optional\n            If True, the data will be saved in binary word2vec format, otherwise - will be saved in plain text.\n\n        '
        total_vec = None
        if word_vec:
            if doctag_vec:
                total_vec = len(self.wv) + len(self.dv)
            self.wv.save_word2vec_format(fname, fvocab, binary, total_vec)
        if doctag_vec:
            write_header = True
            append = False
            if word_vec:
                write_header = False
                append = True
            self.dv.save_word2vec_format(fname, prefix=prefix, fvocab=fvocab, binary=binary, write_header=write_header, append=append, sort_attr='doc_count')

    @deprecated('Gensim 4.0.0 implemented internal optimizations that make calls to init_sims() unnecessary. init_sims() is now obsoleted and will be completely removed in future versions. See https://github.com/RaRe-Technologies/gensim/wiki/Migrating-from-Gensim-3.x-to-4')
    def init_sims(self, replace=False):
        if False:
            print('Hello World!')
        '\n        Precompute L2-normalized vectors. Obsoleted.\n\n        If you need a single unit-normalized vector for some key, call\n        :meth:`~gensim.models.keyedvectors.KeyedVectors.get_vector` instead:\n        ``doc2vec_model.dv.get_vector(key, norm=True)``.\n\n        To refresh norms after you performed some atypical out-of-band vector tampering,\n        call `:meth:`~gensim.models.keyedvectors.KeyedVectors.fill_norms()` instead.\n\n        Parameters\n        ----------\n        replace : bool\n            If True, forget the original trained vectors and only keep the normalized ones.\n            You lose information if you do this.\n\n        '
        self.dv.init_sims(replace=replace)

    @classmethod
    def load(cls, *args, **kwargs):
        if False:
            while True:
                i = 10
        'Load a previously saved :class:`~gensim.models.doc2vec.Doc2Vec` model.\n\n        Parameters\n        ----------\n        fname : str\n            Path to the saved file.\n        *args : object\n            Additional arguments, see `~gensim.models.word2vec.Word2Vec.load`.\n        **kwargs : object\n            Additional arguments, see `~gensim.models.word2vec.Word2Vec.load`.\n\n        See Also\n        --------\n        :meth:`~gensim.models.doc2vec.Doc2Vec.save`\n            Save :class:`~gensim.models.doc2vec.Doc2Vec` model.\n\n        Returns\n        -------\n        :class:`~gensim.models.doc2vec.Doc2Vec`\n            Loaded model.\n\n        '
        try:
            return super(Doc2Vec, cls).load(*args, rethrow=True, **kwargs)
        except AttributeError as ae:
            logger.error('Model load error. Was model saved using code from an older Gensim version? Try loading older model using gensim-3.8.3, then re-saving, to restore compatibility with current code.')
            raise ae

    def estimate_memory(self, vocab_size=None, report=None):
        if False:
            for i in range(10):
                print('nop')
        "Estimate required memory for a model using current settings.\n\n        Parameters\n        ----------\n        vocab_size : int, optional\n            Number of raw words in the vocabulary.\n        report : dict of (str, int), optional\n            A dictionary from string representations of the **specific** model's memory consuming members\n            to their size in bytes.\n\n        Returns\n        -------\n        dict of (str, int), optional\n            A dictionary from string representations of the model's memory consuming members to their size in bytes.\n            Includes members from the base classes as well as weights and tag lookup memory estimation specific to the\n            class.\n\n        "
        report = report or {}
        report['doctag_lookup'] = self.estimated_lookup_memory()
        report['doctag_syn0'] = len(self.dv) * self.vector_size * dtype(REAL).itemsize
        return super(Doc2Vec, self).estimate_memory(vocab_size, report=report)

    def build_vocab(self, corpus_iterable=None, corpus_file=None, update=False, progress_per=10000, keep_raw_vocab=False, trim_rule=None, **kwargs):
        if False:
            print('Hello World!')
        "Build vocabulary from a sequence of documents (can be a once-only generator stream).\n\n        Parameters\n        ----------\n        documents : iterable of list of :class:`~gensim.models.doc2vec.TaggedDocument`, optional\n            Can be simply a list of :class:`~gensim.models.doc2vec.TaggedDocument` elements, but for larger corpora,\n            consider an iterable that streams the documents directly from disk/network.\n            See :class:`~gensim.models.doc2vec.TaggedBrownCorpus` or :class:`~gensim.models.doc2vec.TaggedLineDocument`\n        corpus_file : str, optional\n            Path to a corpus file in :class:`~gensim.models.word2vec.LineSentence` format.\n            You may use this argument instead of `documents` to get performance boost. Only one of `documents` or\n            `corpus_file` arguments need to be passed (not both of them). Documents' tags are assigned automatically\n            and are equal to a line number, as in :class:`~gensim.models.doc2vec.TaggedLineDocument`.\n        update : bool\n            If true, the new words in `documents` will be added to model's vocab.\n        progress_per : int\n            Indicates how many words to process before showing/updating the progress.\n        keep_raw_vocab : bool\n            If not true, delete the raw vocabulary after the scaling is done and free up RAM.\n        trim_rule : function, optional\n            Vocabulary trimming rule, specifies whether certain words should remain in the vocabulary,\n            be trimmed away, or handled using the default (discard if word count < min_count).\n            Can be None (min_count will be used, look to :func:`~gensim.utils.keep_vocab_item`),\n            or a callable that accepts parameters (word, count, min_count) and returns either\n            :attr:`gensim.utils.RULE_DISCARD`, :attr:`gensim.utils.RULE_KEEP` or :attr:`gensim.utils.RULE_DEFAULT`.\n            The rule, if given, is only used to prune vocabulary during current method call and is not stored as part\n            of the model.\n\n            The input parameters are of the following types:\n                * `word` (str) - the word we are examining\n                * `count` (int) - the word's frequency count in the corpus\n                * `min_count` (int) - the minimum count threshold.\n\n        **kwargs\n            Additional key word arguments passed to the internal vocabulary construction.\n\n        "
        (total_words, corpus_count) = self.scan_vocab(corpus_iterable=corpus_iterable, corpus_file=corpus_file, progress_per=progress_per, trim_rule=trim_rule)
        self.corpus_count = corpus_count
        self.corpus_total_words = total_words
        report_values = self.prepare_vocab(update=update, keep_raw_vocab=keep_raw_vocab, trim_rule=trim_rule, **kwargs)
        report_values['memory'] = self.estimate_memory(vocab_size=report_values['num_retained_words'])
        self.prepare_weights(update=update)

    def build_vocab_from_freq(self, word_freq, keep_raw_vocab=False, corpus_count=None, trim_rule=None, update=False):
        if False:
            i = 10
            return i + 15
        "Build vocabulary from a dictionary of word frequencies.\n\n        Build model vocabulary from a passed dictionary that contains a (word -> word count) mapping.\n        Words must be of type unicode strings.\n\n        Parameters\n        ----------\n        word_freq : dict of (str, int)\n            Word <-> count mapping.\n        keep_raw_vocab : bool, optional\n            If not true, delete the raw vocabulary after the scaling is done and free up RAM.\n        corpus_count : int, optional\n            Even if no corpus is provided, this argument can set corpus_count explicitly.\n        trim_rule : function, optional\n            Vocabulary trimming rule, specifies whether certain words should remain in the vocabulary,\n            be trimmed away, or handled using the default (discard if word count < min_count).\n            Can be None (min_count will be used, look to :func:`~gensim.utils.keep_vocab_item`),\n            or a callable that accepts parameters (word, count, min_count) and returns either\n            :attr:`gensim.utils.RULE_DISCARD`, :attr:`gensim.utils.RULE_KEEP` or :attr:`gensim.utils.RULE_DEFAULT`.\n            The rule, if given, is only used to prune vocabulary during\n            :meth:`~gensim.models.doc2vec.Doc2Vec.build_vocab` and is not stored as part of the model.\n\n            The input parameters are of the following types:\n                * `word` (str) - the word we are examining\n                * `count` (int) - the word's frequency count in the corpus\n                * `min_count` (int) - the minimum count threshold.\n\n        update : bool, optional\n            If true, the new provided words in `word_freq` dict will be added to model's vocab.\n\n        "
        logger.info('processing provided word frequencies')
        raw_vocab = word_freq
        logger.info('collected %i different raw words, with total frequency of %i', len(raw_vocab), sum(raw_vocab.values()))
        self.corpus_count = corpus_count or 0
        self.raw_vocab = raw_vocab
        report_values = self.prepare_vocab(keep_raw_vocab=keep_raw_vocab, trim_rule=trim_rule, update=update)
        report_values['memory'] = self.estimate_memory(vocab_size=report_values['num_retained_words'])
        self.prepare_weights(update=update)

    def _scan_vocab(self, corpus_iterable, progress_per, trim_rule):
        if False:
            i = 10
            return i + 15
        document_no = -1
        total_words = 0
        min_reduce = 1
        interval_start = default_timer() - 1e-05
        interval_count = 0
        checked_string_types = 0
        vocab = defaultdict(int)
        max_rawint = -1
        doctags_lookup = {}
        doctags_list = []
        for (document_no, document) in enumerate(corpus_iterable):
            if not checked_string_types:
                if isinstance(document.words, str):
                    logger.warning("Each 'words' should be a list of words (usually unicode strings). First 'words' here is instead plain %s.", type(document.words))
                checked_string_types += 1
            if document_no % progress_per == 0:
                interval_rate = (total_words - interval_count) / (default_timer() - interval_start)
                logger.info('PROGRESS: at example #%i, processed %i words (%i words/s), %i word types, %i tags', document_no, total_words, interval_rate, len(vocab), len(doctags_list))
                interval_start = default_timer()
                interval_count = total_words
            document_length = len(document.words)
            for tag in document.tags:
                if isinstance(tag, (int, integer)):
                    max_rawint = max(max_rawint, tag)
                elif tag in doctags_lookup:
                    doctags_lookup[tag].doc_count += 1
                    doctags_lookup[tag].word_count += document_length
                else:
                    doctags_lookup[tag] = Doctag(index=len(doctags_list), word_count=document_length, doc_count=1)
                    doctags_list.append(tag)
            for word in document.words:
                vocab[word] += 1
            total_words += len(document.words)
            if self.max_vocab_size and len(vocab) > self.max_vocab_size:
                utils.prune_vocab(vocab, min_reduce, trim_rule=trim_rule)
                min_reduce += 1
        corpus_count = document_no + 1
        if len(doctags_list) > corpus_count:
            logger.warning('More unique tags (%i) than documents (%i).', len(doctags_list), corpus_count)
        if max_rawint > corpus_count:
            logger.warning('Highest int doctag (%i) larger than count of documents (%i). This means at least %i excess, unused slots (%i bytes) will be allocated for vectors.', max_rawint, corpus_count, max_rawint - corpus_count, (max_rawint - corpus_count) * self.vector_size * dtype(REAL).itemsize)
        if max_rawint > -1:
            for key in doctags_list:
                doctags_lookup[key].index = doctags_lookup[key].index + max_rawint + 1
            doctags_list = list(range(0, max_rawint + 1)) + doctags_list
        self.dv.index_to_key = doctags_list
        for (t, dt) in doctags_lookup.items():
            self.dv.key_to_index[t] = dt.index
            self.dv.set_vecattr(t, 'word_count', dt.word_count)
            self.dv.set_vecattr(t, 'doc_count', dt.doc_count)
        self.raw_vocab = vocab
        return (total_words, corpus_count)

    def scan_vocab(self, corpus_iterable=None, corpus_file=None, progress_per=100000, trim_rule=None):
        if False:
            return 10
        "Create the model's vocabulary: a mapping from unique words in the corpus to their frequency count.\n\n        Parameters\n        ----------\n        documents : iterable of :class:`~gensim.models.doc2vec.TaggedDocument`, optional\n            The tagged documents used to create the vocabulary. Their tags can be either str tokens or ints (faster).\n        corpus_file : str, optional\n            Path to a corpus file in :class:`~gensim.models.word2vec.LineSentence` format.\n            You may use this argument instead of `documents` to get performance boost. Only one of `documents` or\n            `corpus_file` arguments need to be passed (not both of them).\n        progress_per : int\n            Progress will be logged every `progress_per` documents.\n        trim_rule : function, optional\n            Vocabulary trimming rule, specifies whether certain words should remain in the vocabulary,\n            be trimmed away, or handled using the default (discard if word count < min_count).\n            Can be None (min_count will be used, look to :func:`~gensim.utils.keep_vocab_item`),\n            or a callable that accepts parameters (word, count, min_count) and returns either\n            :attr:`gensim.utils.RULE_DISCARD`, :attr:`gensim.utils.RULE_KEEP` or :attr:`gensim.utils.RULE_DEFAULT`.\n            The rule, if given, is only used to prune vocabulary during\n            :meth:`~gensim.models.doc2vec.Doc2Vec.build_vocab` and is not stored as part of the model.\n\n            The input parameters are of the following types:\n                * `word` (str) - the word we are examining\n                * `count` (int) - the word's frequency count in the corpus\n                * `min_count` (int) - the minimum count threshold.\n\n        Returns\n        -------\n        (int, int)\n            Tuple of `(total words in the corpus, number of documents)`.\n\n        "
        logger.info('collecting all words and their counts')
        if corpus_file is not None:
            corpus_iterable = TaggedLineDocument(corpus_file)
        (total_words, corpus_count) = self._scan_vocab(corpus_iterable, progress_per, trim_rule)
        logger.info('collected %i word types and %i unique tags from a corpus of %i examples and %i words', len(self.raw_vocab), len(self.dv), corpus_count, total_words)
        return (total_words, corpus_count)

    def similarity_unseen_docs(self, doc_words1, doc_words2, alpha=None, min_alpha=None, epochs=None):
        if False:
            i = 10
            return i + 15
        'Compute cosine similarity between two post-bulk out of training documents.\n\n        Parameters\n        ----------\n        model : :class:`~gensim.models.doc2vec.Doc2Vec`\n            An instance of a trained `Doc2Vec` model.\n        doc_words1 : list of str\n            Input document.\n        doc_words2 : list of str\n            Input document.\n        alpha : float, optional\n            The initial learning rate.\n        min_alpha : float, optional\n            Learning rate will linearly drop to `min_alpha` as training progresses.\n        epochs : int, optional\n            Number of epoch to train the new document.\n\n        Returns\n        -------\n        float\n            The cosine similarity between `doc_words1` and `doc_words2`.\n\n        '
        d1 = self.infer_vector(doc_words=doc_words1, alpha=alpha, min_alpha=min_alpha, epochs=epochs)
        d2 = self.infer_vector(doc_words=doc_words2, alpha=alpha, min_alpha=min_alpha, epochs=epochs)
        return np.dot(matutils.unitvec(d1), matutils.unitvec(d2))

class Doc2VecVocab(utils.SaveLoad):
    """Obsolete class retained for now as load-compatibility state capture"""

class Doc2VecTrainables(utils.SaveLoad):
    """Obsolete class retained for now as load-compatibility state capture"""

class TaggedBrownCorpus:

    def __init__(self, dirname):
        if False:
            while True:
                i = 10
        'Reader for the `Brown corpus (part of NLTK data) <http://www.nltk.org/book/ch02.html#tab-brown-sources>`_.\n\n        Parameters\n        ----------\n        dirname : str\n            Path to folder with Brown corpus.\n\n        '
        self.dirname = dirname

    def __iter__(self):
        if False:
            while True:
                i = 10
        'Iterate through the corpus.\n\n        Yields\n        ------\n        :class:`~gensim.models.doc2vec.TaggedDocument`\n            Document from `source`.\n\n        '
        for fname in os.listdir(self.dirname):
            fname = os.path.join(self.dirname, fname)
            if not os.path.isfile(fname):
                continue
            with utils.open(fname, 'rb') as fin:
                for (item_no, line) in enumerate(fin):
                    line = utils.to_unicode(line)
                    token_tags = [t.split('/') for t in line.split() if len(t.split('/')) == 2]
                    words = ['%s/%s' % (token.lower(), tag[:2]) for (token, tag) in token_tags if tag[:2].isalpha()]
                    if not words:
                        continue
                    yield TaggedDocument(words, ['%s_SENT_%s' % (fname, item_no)])

class TaggedLineDocument:

    def __init__(self, source):
        if False:
            for i in range(10):
                print('nop')
        'Iterate over a file that contains documents:\n        one line = :class:`~gensim.models.doc2vec.TaggedDocument` object.\n\n        Words are expected to be already preprocessed and separated by whitespace. Document tags are constructed\n        automatically from the document line number (each document gets a unique integer tag).\n\n        Parameters\n        ----------\n        source : string or a file-like object\n            Path to the file on disk, or an already-open file object (must support `seek(0)`).\n\n        Examples\n        --------\n        .. sourcecode:: pycon\n\n            >>> from gensim.test.utils import datapath\n            >>> from gensim.models.doc2vec import TaggedLineDocument\n            >>>\n            >>> for document in TaggedLineDocument(datapath("head500.noblanks.cor")):\n            ...     pass\n\n        '
        self.source = source

    def __iter__(self):
        if False:
            return 10
        'Iterate through the lines in the source.\n\n        Yields\n        ------\n        :class:`~gensim.models.doc2vec.TaggedDocument`\n            Document from `source` specified in the constructor.\n\n        '
        try:
            self.source.seek(0)
            for (item_no, line) in enumerate(self.source):
                yield TaggedDocument(utils.to_unicode(line).split(), [item_no])
        except AttributeError:
            with utils.open(self.source, 'rb') as fin:
                for (item_no, line) in enumerate(fin):
                    yield TaggedDocument(utils.to_unicode(line).split(), [item_no])