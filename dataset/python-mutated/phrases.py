"""
Automatically detect common phrases -- aka multi-word expressions, word n-gram collocations -- from
a stream of sentences.

Inspired by:

* `Mikolov, et. al: "Distributed Representations of Words and Phrases and their Compositionality"
  <https://arxiv.org/abs/1310.4546>`_
* `"Normalized (Pointwise) Mutual Information in Collocation Extraction" by Gerlof Bouma
  <https://svn.spraakdata.gu.se/repos/gerlof/pub/www/Docs/npmi-pfd.pdf>`_


Examples
--------
.. sourcecode:: pycon

    >>> from gensim.test.utils import datapath
    >>> from gensim.models.word2vec import Text8Corpus
    >>> from gensim.models.phrases import Phrases, ENGLISH_CONNECTOR_WORDS
    >>>
    >>> # Create training corpus. Must be a sequence of sentences (e.g. an iterable or a generator).
    >>> sentences = Text8Corpus(datapath('testcorpus.txt'))
    >>> # Each sentence must be a list of string tokens:
    >>> first_sentence = next(iter(sentences))
    >>> print(first_sentence[:10])
    ['computer', 'human', 'interface', 'computer', 'response', 'survey', 'system', 'time', 'user', 'interface']
    >>>
    >>> # Train a toy phrase model on our training corpus.
    >>> phrase_model = Phrases(sentences, min_count=1, threshold=1, connector_words=ENGLISH_CONNECTOR_WORDS)
    >>>
    >>> # Apply the trained phrases model to a new, unseen sentence.
    >>> new_sentence = ['trees', 'graph', 'minors']
    >>> phrase_model[new_sentence]
    ['trees_graph', 'minors']
    >>> # The toy model considered "trees graph" a single phrase => joined the two
    >>> # tokens into a single "phrase" token, using our selected `_` delimiter.
    >>>
    >>> # Apply the trained model to each sentence of a corpus, using the same [] syntax:
    >>> for sent in phrase_model[sentences]:
    ...     pass
    >>>
    >>> # Update the model with two new sentences on the fly.
    >>> phrase_model.add_vocab([["hello", "world"], ["meow"]])
    >>>
    >>> # Export the trained model = use less RAM, faster processing. Model updates no longer possible.
    >>> frozen_model = phrase_model.freeze()
    >>> # Apply the frozen model; same results as before:
    >>> frozen_model[new_sentence]
    ['trees_graph', 'minors']
    >>>
    >>> # Save / load models.
    >>> frozen_model.save("/tmp/my_phrase_model.pkl")
    >>> model_reloaded = Phrases.load("/tmp/my_phrase_model.pkl")
    >>> model_reloaded[['trees', 'graph', 'minors']]  # apply the reloaded model to a sentence
    ['trees_graph', 'minors']

"""
import logging
import itertools
from math import log
import pickle
from inspect import getfullargspec as getargspec
import time
from gensim import utils, interfaces
logger = logging.getLogger(__name__)
NEGATIVE_INFINITY = float('-inf')
ENGLISH_CONNECTOR_WORDS = frozenset(' a an the  for of with without at from to in on by  and or '.split())

def original_scorer(worda_count, wordb_count, bigram_count, len_vocab, min_count, corpus_word_count):
    if False:
        while True:
            i = 10
    'Bigram scoring function, based on the original `Mikolov, et. al: "Distributed Representations\n    of Words and Phrases and their Compositionality" <https://arxiv.org/abs/1310.4546>`_.\n\n    Parameters\n    ----------\n    worda_count : int\n        Number of occurrences for first word.\n    wordb_count : int\n        Number of occurrences for second word.\n    bigram_count : int\n        Number of co-occurrences for phrase "worda_wordb".\n    len_vocab : int\n        Size of vocabulary.\n    min_count: int\n        Minimum collocation count threshold.\n    corpus_word_count : int\n        Not used in this particular scoring technique.\n\n    Returns\n    -------\n    float\n        Score for given phrase. Can be negative.\n\n    Notes\n    -----\n    Formula: :math:`\\frac{(bigram\\_count - min\\_count) * len\\_vocab }{ (worda\\_count * wordb\\_count)}`.\n\n    '
    denom = worda_count * wordb_count
    if denom == 0:
        return NEGATIVE_INFINITY
    return (bigram_count - min_count) / float(denom) * len_vocab

def npmi_scorer(worda_count, wordb_count, bigram_count, len_vocab, min_count, corpus_word_count):
    if False:
        print('Hello World!')
    'Calculation NPMI score based on `"Normalized (Pointwise) Mutual Information in Colocation Extraction"\n    by Gerlof Bouma <https://svn.spraakdata.gu.se/repos/gerlof/pub/www/Docs/npmi-pfd.pdf>`_.\n\n    Parameters\n    ----------\n    worda_count : int\n        Number of occurrences for first word.\n    wordb_count : int\n        Number of occurrences for second word.\n    bigram_count : int\n        Number of co-occurrences for phrase "worda_wordb".\n    len_vocab : int\n        Not used.\n    min_count: int\n        Ignore all bigrams with total collected count lower than this value.\n    corpus_word_count : int\n        Total number of words in the corpus.\n\n    Returns\n    -------\n    float\n        If bigram_count >= min_count, return the collocation score, in the range -1 to 1.\n        Otherwise return -inf.\n\n    Notes\n    -----\n    Formula: :math:`\\frac{ln(prop(word_a, word_b) / (prop(word_a)*prop(word_b)))}{ -ln(prop(word_a, word_b)}`,\n    where :math:`prob(word) = \\frac{word\\_count}{corpus\\_word\\_count}`\n\n    '
    if bigram_count >= min_count:
        corpus_word_count = float(corpus_word_count)
        pa = worda_count / corpus_word_count
        pb = wordb_count / corpus_word_count
        pab = bigram_count / corpus_word_count
        try:
            return log(pab / (pa * pb)) / -log(pab)
        except ValueError:
            return NEGATIVE_INFINITY
    else:
        return NEGATIVE_INFINITY

def _is_single(obj):
    if False:
        i = 10
        return i + 15
    'Check whether `obj` is a single document or an entire corpus.\n\n    Parameters\n    ----------\n    obj : object\n\n    Return\n    ------\n    (bool, object)\n        2-tuple ``(is_single_document, new_obj)`` tuple, where `new_obj`\n        yields the same sequence as the original `obj`.\n\n    Notes\n    -----\n    `obj` is a single document if it is an iterable of strings. It is a corpus if it is an iterable of documents.\n\n    '
    obj_iter = iter(obj)
    temp_iter = obj_iter
    try:
        peek = next(obj_iter)
        obj_iter = itertools.chain([peek], obj_iter)
    except StopIteration:
        return (True, obj)
    if isinstance(peek, str):
        return (True, obj_iter)
    if temp_iter is obj:
        return (False, obj_iter)
    return (False, obj)

class _PhrasesTransformation(interfaces.TransformationABC):
    """
    Abstract base class for :class:`~gensim.models.phrases.Phrases` and
    :class:`~gensim.models.phrases.FrozenPhrases`.

    """

    def __init__(self, connector_words):
        if False:
            print('Hello World!')
        self.connector_words = frozenset(connector_words)

    def score_candidate(self, word_a, word_b, in_between):
        if False:
            i = 10
            return i + 15
        'Score a single phrase candidate.\n\n        Returns\n        -------\n        (str, float)\n            2-tuple of ``(delimiter-joined phrase, phrase score)`` for a phrase,\n            or ``(None, None)`` if not a phrase.\n        '
        raise NotImplementedError('ABC: override this method in child classes')

    def analyze_sentence(self, sentence):
        if False:
            return 10
        'Analyze a sentence, concatenating any detected phrases into a single token.\n\n        Parameters\n        ----------\n        sentence : iterable of str\n            Token sequence representing the sentence to be analyzed.\n\n        Yields\n        ------\n        (str, {float, None})\n            Iterate through the input sentence tokens and yield 2-tuples of:\n            - ``(concatenated_phrase_tokens, score)`` for token sequences that form a phrase.\n            - ``(word, None)`` if the token is not a part of a phrase.\n\n        '
        (start_token, in_between) = (None, [])
        for word in sentence:
            if word not in self.connector_words:
                if start_token:
                    (phrase, score) = self.score_candidate(start_token, word, in_between)
                    if score is not None:
                        yield (phrase, score)
                        (start_token, in_between) = (None, [])
                    else:
                        yield (start_token, None)
                        for w in in_between:
                            yield (w, None)
                        (start_token, in_between) = (word, [])
                else:
                    (start_token, in_between) = (word, [])
            elif start_token:
                in_between.append(word)
            else:
                yield (word, None)
        if start_token:
            yield (start_token, None)
            for w in in_between:
                yield (w, None)

    def __getitem__(self, sentence):
        if False:
            while True:
                i = 10
        "Convert the input sequence of tokens ``sentence`` into a sequence of tokens where adjacent\n        tokens are replaced by a single token if they form a bigram collocation.\n\n        If `sentence` is an entire corpus (iterable of sentences rather than a single\n        sentence), return an iterable that converts each of the corpus' sentences\n        into phrases on the fly, one after another.\n\n        Parameters\n        ----------\n        sentence : {list of str, iterable of list of str}\n            Input sentence or a stream of sentences.\n\n        Return\n        ------\n        {list of str, iterable of list of str}\n            Sentence with phrase tokens joined by ``self.delimiter``, if input was a single sentence.\n            A generator of such sentences if input was a corpus.\n\ns        "
        (is_single, sentence) = _is_single(sentence)
        if not is_single:
            return self._apply(sentence)
        return [token for (token, _) in self.analyze_sentence(sentence)]

    def find_phrases(self, sentences):
        if False:
            print('Hello World!')
        "Get all unique phrases (multi-word expressions) that appear in ``sentences``, and their scores.\n\n        Parameters\n        ----------\n        sentences : iterable of list of str\n            Text corpus.\n\n        Returns\n        -------\n        dict(str, float)\n           Unique phrases found in ``sentences``, mapped to their scores.\n\n        Example\n        -------\n        .. sourcecode:: pycon\n\n            >>> from gensim.test.utils import datapath\n            >>> from gensim.models.word2vec import Text8Corpus\n            >>> from gensim.models.phrases import Phrases, ENGLISH_CONNECTOR_WORDS\n            >>>\n            >>> sentences = Text8Corpus(datapath('testcorpus.txt'))\n            >>> phrases = Phrases(sentences, min_count=1, threshold=0.1, connector_words=ENGLISH_CONNECTOR_WORDS)\n            >>>\n            >>> for phrase, score in phrases.find_phrases(sentences).items():\n            ...     print(phrase, score)\n        "
        result = {}
        for sentence in sentences:
            for (phrase, score) in self.analyze_sentence(sentence):
                if score is not None:
                    result[phrase] = score
        return result

    @classmethod
    def load(cls, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Load a previously saved :class:`~gensim.models.phrases.Phrases` /\n        :class:`~gensim.models.phrases.FrozenPhrases` model.\n\n        Handles backwards compatibility from older versions which did not support pluggable scoring functions.\n\n        Parameters\n        ----------\n        args : object\n            See :class:`~gensim.utils.SaveLoad.load`.\n        kwargs : object\n            See :class:`~gensim.utils.SaveLoad.load`.\n\n        '
        model = super(_PhrasesTransformation, cls).load(*args, **kwargs)
        try:
            phrasegrams = getattr(model, 'phrasegrams', {})
            (component, score) = next(iter(phrasegrams.items()))
            if isinstance(score, tuple):
                model.phrasegrams = {str(model.delimiter.join(key), encoding='utf8'): val[1] for (key, val) in phrasegrams.items()}
            elif isinstance(component, tuple):
                model.phrasegrams = {str(model.delimiter.join(key), encoding='utf8'): val for (key, val) in phrasegrams.items()}
        except StopIteration:
            pass
        if not hasattr(model, 'scoring'):
            logger.warning('older version of %s loaded without scoring function', cls.__name__)
            logger.warning('setting pluggable scoring method to original_scorer for compatibility')
            model.scoring = original_scorer
        if hasattr(model, 'scoring'):
            if isinstance(model.scoring, str):
                if model.scoring == 'default':
                    logger.warning('older version of %s loaded with "default" scoring parameter', cls.__name__)
                    logger.warning('setting scoring method to original_scorer for compatibility')
                    model.scoring = original_scorer
                elif model.scoring == 'npmi':
                    logger.warning('older version of %s loaded with "npmi" scoring parameter', cls.__name__)
                    logger.warning('setting scoring method to npmi_scorer for compatibility')
                    model.scoring = npmi_scorer
                else:
                    raise ValueError(f'failed to load {cls.__name__} model, unknown scoring "{model.scoring}"')
        if not hasattr(model, 'connector_words'):
            if hasattr(model, 'common_terms'):
                model.connector_words = model.common_terms
                del model.common_terms
            else:
                logger.warning('loaded older version of %s, setting connector_words to an empty set', cls.__name__)
                model.connector_words = frozenset()
        if not hasattr(model, 'corpus_word_count'):
            logger.warning('older version of %s loaded without corpus_word_count', cls.__name__)
            logger.warning('setting corpus_word_count to 0, do not use it in your scoring function')
            model.corpus_word_count = 0
        if getattr(model, 'vocab', None):
            word = next(iter(model.vocab))
            if not isinstance(word, str):
                logger.info('old version of %s loaded, upgrading %i words in memory', cls.__name__, len(model.vocab))
                logger.info('re-save the loaded model to avoid this upgrade in the future')
                vocab = {}
                for (key, value) in model.vocab.items():
                    vocab[str(key, encoding='utf8')] = value
                model.vocab = vocab
        if not isinstance(model.delimiter, str):
            model.delimiter = str(model.delimiter, encoding='utf8')
        return model

class Phrases(_PhrasesTransformation):
    """Detect phrases based on collocation counts."""

    def __init__(self, sentences=None, min_count=5, threshold=10.0, max_vocab_size=40000000, delimiter='_', progress_per=10000, scoring='default', connector_words=frozenset()):
        if False:
            while True:
                i = 10
        '\n\n        Parameters\n        ----------\n        sentences : iterable of list of str, optional\n            The `sentences` iterable can be simply a list, but for larger corpora, consider a generator that streams\n            the sentences directly from disk/network, See :class:`~gensim.models.word2vec.BrownCorpus`,\n            :class:`~gensim.models.word2vec.Text8Corpus` or :class:`~gensim.models.word2vec.LineSentence`\n            for such examples.\n        min_count : float, optional\n            Ignore all words and bigrams with total collected count lower than this value.\n        threshold : float, optional\n            Represent a score threshold for forming the phrases (higher means fewer phrases).\n            A phrase of words `a` followed by `b` is accepted if the score of the phrase is greater than threshold.\n            Heavily depends on concrete scoring-function, see the `scoring` parameter.\n        max_vocab_size : int, optional\n            Maximum size (number of tokens) of the vocabulary. Used to control pruning of less common words,\n            to keep memory under control. The default of 40M needs about 3.6GB of RAM. Increase/decrease\n            `max_vocab_size` depending on how much available memory you have.\n        delimiter : str, optional\n            Glue character used to join collocation tokens.\n        scoring : {\'default\', \'npmi\', function}, optional\n            Specify how potential phrases are scored. `scoring` can be set with either a string that refers to a\n            built-in scoring function, or with a function with the expected parameter names.\n            Two built-in scoring functions are available by setting `scoring` to a string:\n\n            #. "default" - :func:`~gensim.models.phrases.original_scorer`.\n            #. "npmi" - :func:`~gensim.models.phrases.npmi_scorer`.\n        connector_words : set of str, optional\n            Set of words that may be included within a phrase, without affecting its scoring.\n            No phrase can start nor end with a connector word; a phrase may contain any number of\n            connector words in the middle.\n\n            **If your texts are in English, set** ``connector_words=phrases.ENGLISH_CONNECTOR_WORDS``.\n\n            This will cause phrases to include common English articles, prepositions and\n            conjuctions, such as `bank_of_america` or `eye_of_the_beholder`.\n\n            For other languages or specific applications domains, use custom ``connector_words``\n            that make sense there: ``connector_words=frozenset("der die das".split())`` etc.\n\n        Examples\n        --------\n        .. sourcecode:: pycon\n\n            >>> from gensim.test.utils import datapath\n            >>> from gensim.models.word2vec import Text8Corpus\n            >>> from gensim.models.phrases import Phrases, ENGLISH_CONNECTOR_WORDS\n            >>>\n            >>> # Load corpus and train a model.\n            >>> sentences = Text8Corpus(datapath(\'testcorpus.txt\'))\n            >>> phrases = Phrases(sentences, min_count=1, threshold=1, connector_words=ENGLISH_CONNECTOR_WORDS)\n            >>>\n            >>> # Use the model to detect phrases in a new sentence.\n            >>> sent = [u\'trees\', u\'graph\', u\'minors\']\n            >>> print(phrases[sent])\n            [u\'trees_graph\', u\'minors\']\n            >>>\n            >>> # Or transform multiple sentences at once.\n            >>> sents = [[u\'trees\', u\'graph\', u\'minors\'], [u\'graph\', u\'minors\']]\n            >>> for phrase in phrases[sents]:\n            ...     print(phrase)\n            [u\'trees_graph\', u\'minors\']\n            [u\'graph_minors\']\n            >>>\n            >>> # Export a FrozenPhrases object that is more efficient but doesn\'t allow any more training.\n            >>> frozen_phrases = phrases.freeze()\n            >>> print(frozen_phrases[sent])\n            [u\'trees_graph\', u\'minors\']\n\n        Notes\n        -----\n\n        The ``scoring="npmi"`` is more robust when dealing with common words that form part of common bigrams, and\n        ranges from -1 to 1, but is slower to calculate than the default ``scoring="default"``.\n        The default is the PMI-like scoring as described in `Mikolov, et. al: "Distributed\n        Representations of Words and Phrases and their Compositionality" <https://arxiv.org/abs/1310.4546>`_.\n\n        To use your own custom ``scoring`` function, pass in a function with the following signature:\n\n        * ``worda_count`` - number of corpus occurrences in `sentences` of the first token in the bigram being scored\n        * ``wordb_count`` - number of corpus occurrences in `sentences` of the second token in the bigram being scored\n        * ``bigram_count`` - number of occurrences in `sentences` of the whole bigram\n        * ``len_vocab`` - the number of unique tokens in `sentences`\n        * ``min_count`` - the `min_count` setting of the Phrases class\n        * ``corpus_word_count`` - the total number of tokens (non-unique) in `sentences`\n\n        The scoring function must accept all these parameters, even if it doesn\'t use them in its scoring.\n\n        The scoring function **must be pickleable**.\n\n        '
        super().__init__(connector_words=connector_words)
        if min_count <= 0:
            raise ValueError('min_count should be at least 1')
        if threshold <= 0 and scoring == 'default':
            raise ValueError('threshold should be positive for default scoring')
        if scoring == 'npmi' and (threshold < -1 or threshold > 1):
            raise ValueError('threshold should be between -1 and 1 for npmi scoring')
        if isinstance(scoring, str):
            if scoring == 'default':
                scoring = original_scorer
            elif scoring == 'npmi':
                scoring = npmi_scorer
            else:
                raise ValueError(f'unknown scoring method string {scoring} specified')
        scoring_params = ['worda_count', 'wordb_count', 'bigram_count', 'len_vocab', 'min_count', 'corpus_word_count']
        if callable(scoring):
            missing = [param for param in scoring_params if param not in getargspec(scoring)[0]]
            if not missing:
                self.scoring = scoring
            else:
                raise ValueError(f'scoring function missing expected parameters {missing}')
        self.min_count = min_count
        self.threshold = threshold
        self.max_vocab_size = max_vocab_size
        self.vocab = {}
        self.min_reduce = 1
        self.delimiter = delimiter
        self.progress_per = progress_per
        self.corpus_word_count = 0
        try:
            pickle.loads(pickle.dumps(self.scoring))
        except pickle.PickleError:
            raise pickle.PickleError(f'Custom scoring function in {self.__class__.__name__} must be pickle-able')
        if sentences is not None:
            start = time.time()
            self.add_vocab(sentences)
            self.add_lifecycle_event('created', msg=f'built {self} in {time.time() - start:.2f}s')

    def __str__(self):
        if False:
            while True:
                i = 10
        return '%s<%i vocab, min_count=%s, threshold=%s, max_vocab_size=%s>' % (self.__class__.__name__, len(self.vocab), self.min_count, self.threshold, self.max_vocab_size)

    @staticmethod
    def _learn_vocab(sentences, max_vocab_size, delimiter, connector_words, progress_per):
        if False:
            while True:
                i = 10
        'Collect unigram and bigram counts from the `sentences` iterable.'
        (sentence_no, total_words, min_reduce) = (-1, 0, 1)
        vocab = {}
        logger.info('collecting all words and their counts')
        for (sentence_no, sentence) in enumerate(sentences):
            if sentence_no % progress_per == 0:
                logger.info('PROGRESS: at sentence #%i, processed %i words and %i word types', sentence_no, total_words, len(vocab))
            (start_token, in_between) = (None, [])
            for word in sentence:
                if word not in connector_words:
                    vocab[word] = vocab.get(word, 0) + 1
                    if start_token is not None:
                        phrase_tokens = itertools.chain([start_token], in_between, [word])
                        joined_phrase_token = delimiter.join(phrase_tokens)
                        vocab[joined_phrase_token] = vocab.get(joined_phrase_token, 0) + 1
                    (start_token, in_between) = (word, [])
                elif start_token is not None:
                    in_between.append(word)
                total_words += 1
            if len(vocab) > max_vocab_size:
                utils.prune_vocab(vocab, min_reduce)
                min_reduce += 1
        logger.info('collected %i token types (unigram + bigrams) from a corpus of %i words and %i sentences', len(vocab), total_words, sentence_no + 1)
        return (min_reduce, vocab, total_words)

    def add_vocab(self, sentences):
        if False:
            while True:
                i = 10
        "Update model parameters with new `sentences`.\n\n        Parameters\n        ----------\n        sentences : iterable of list of str\n            Text corpus to update this model's parameters from.\n\n        Example\n        -------\n        .. sourcecode:: pycon\n\n            >>> from gensim.test.utils import datapath\n            >>> from gensim.models.word2vec import Text8Corpus\n            >>> from gensim.models.phrases import Phrases, ENGLISH_CONNECTOR_WORDS\n            >>>\n            >>> # Train a phrase detector from a text corpus.\n            >>> sentences = Text8Corpus(datapath('testcorpus.txt'))\n            >>> phrases = Phrases(sentences, connector_words=ENGLISH_CONNECTOR_WORDS)  # train model\n            >>> assert len(phrases.vocab) == 37\n            >>>\n            >>> more_sentences = [\n            ...     [u'the', u'mayor', u'of', u'new', u'york', u'was', u'there'],\n            ...     [u'machine', u'learning', u'can', u'be', u'new', u'york', u'sometimes'],\n            ... ]\n            >>>\n            >>> phrases.add_vocab(more_sentences)  # add new sentences to model\n            >>> assert len(phrases.vocab) == 60\n\n        "
        (min_reduce, vocab, total_words) = self._learn_vocab(sentences, max_vocab_size=self.max_vocab_size, delimiter=self.delimiter, progress_per=self.progress_per, connector_words=self.connector_words)
        self.corpus_word_count += total_words
        if self.vocab:
            logger.info('merging %i counts into %s', len(vocab), self)
            self.min_reduce = max(self.min_reduce, min_reduce)
            for (word, count) in vocab.items():
                self.vocab[word] = self.vocab.get(word, 0) + count
            if len(self.vocab) > self.max_vocab_size:
                utils.prune_vocab(self.vocab, self.min_reduce)
                self.min_reduce += 1
        else:
            self.vocab = vocab
        logger.info('merged %s', self)

    def score_candidate(self, word_a, word_b, in_between):
        if False:
            print('Hello World!')
        word_a_cnt = self.vocab.get(word_a, 0)
        if word_a_cnt <= 0:
            return (None, None)
        word_b_cnt = self.vocab.get(word_b, 0)
        if word_b_cnt <= 0:
            return (None, None)
        phrase = self.delimiter.join([word_a] + in_between + [word_b])
        phrase_cnt = self.vocab.get(phrase, 0)
        if phrase_cnt <= 0:
            return (None, None)
        score = self.scoring(worda_count=word_a_cnt, wordb_count=word_b_cnt, bigram_count=phrase_cnt, len_vocab=len(self.vocab), min_count=self.min_count, corpus_word_count=self.corpus_word_count)
        if score <= self.threshold:
            return (None, None)
        return (phrase, score)

    def freeze(self):
        if False:
            return 10
        '\n        Return an object that contains the bare minimum of information while still allowing\n        phrase detection. See :class:`~gensim.models.phrases.FrozenPhrases`.\n\n        Use this "frozen model" to dramatically reduce RAM footprint if you don\'t plan to\n        make any further changes to your `Phrases` model.\n\n        Returns\n        -------\n        :class:`~gensim.models.phrases.FrozenPhrases`\n            Exported object that\'s smaller, faster, but doesn\'t support model updates.\n\n        '
        return FrozenPhrases(self)

    def export_phrases(self):
        if False:
            while True:
                i = 10
        'Extract all found phrases.\n\n        Returns\n        ------\n        dict(str, float)\n            Mapping between phrases and their scores.\n\n        '
        (result, source_vocab) = ({}, self.vocab)
        for token in source_vocab:
            unigrams = token.split(self.delimiter)
            if len(unigrams) < 2:
                continue
            (phrase, score) = self.score_candidate(unigrams[0], unigrams[-1], unigrams[1:-1])
            if score is not None:
                result[phrase] = score
        return result

class FrozenPhrases(_PhrasesTransformation):
    """Minimal state & functionality exported from a trained :class:`~gensim.models.phrases.Phrases` model.

    The goal of this class is to cut down memory consumption of `Phrases`, by discarding model state
    not strictly needed for the phrase detection task.

    Use this instead of `Phrases` if you do not need to update the bigram statistics with new documents any more.

    """

    def __init__(self, phrases_model):
        if False:
            return 10
        "\n\n        Parameters\n        ----------\n        phrases_model : :class:`~gensim.models.phrases.Phrases`\n            Trained phrases instance, to extract all phrases from.\n\n        Notes\n        -----\n        After the one-time initialization, a :class:`~gensim.models.phrases.FrozenPhrases` will be much\n        smaller and faster than using the full :class:`~gensim.models.phrases.Phrases` model.\n\n        Examples\n        ----------\n        .. sourcecode:: pycon\n\n            >>> from gensim.test.utils import datapath\n            >>> from gensim.models.word2vec import Text8Corpus\n            >>> from gensim.models.phrases import Phrases, ENGLISH_CONNECTOR_WORDS\n            >>>\n            >>> # Load corpus and train a model.\n            >>> sentences = Text8Corpus(datapath('testcorpus.txt'))\n            >>> phrases = Phrases(sentences, min_count=1, threshold=1, connector_words=ENGLISH_CONNECTOR_WORDS)\n            >>>\n            >>> # Export a FrozenPhrases object that is more efficient but doesn't allow further training.\n            >>> frozen_phrases = phrases.freeze()\n            >>> print(frozen_phrases[sent])\n            [u'trees_graph', u'minors']\n\n        "
        self.threshold = phrases_model.threshold
        self.min_count = phrases_model.min_count
        self.delimiter = phrases_model.delimiter
        self.scoring = phrases_model.scoring
        self.connector_words = phrases_model.connector_words
        logger.info('exporting phrases from %s', phrases_model)
        start = time.time()
        self.phrasegrams = phrases_model.export_phrases()
        self.add_lifecycle_event('created', msg=f'exported {self} from {phrases_model} in {time.time() - start:.2f}s')

    def __str__(self):
        if False:
            return 10
        return '%s<%i phrases, min_count=%s, threshold=%s>' % (self.__class__.__name__, len(self.phrasegrams), self.min_count, self.threshold)

    def score_candidate(self, word_a, word_b, in_between):
        if False:
            for i in range(10):
                print('nop')
        phrase = self.delimiter.join([word_a] + in_between + [word_b])
        score = self.phrasegrams.get(phrase, NEGATIVE_INFINITY)
        if score > self.threshold:
            return (phrase, score)
        return (None, None)
Phraser = FrozenPhrases