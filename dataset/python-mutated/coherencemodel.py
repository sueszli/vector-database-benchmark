"""Calculate topic coherence for topic models. This is the implementation of the four stage topic coherence pipeline
from the paper `Michael Roeder, Andreas Both and Alexander Hinneburg: "Exploring the space of topic coherence measures"
<http://svn.aksw.org/papers/2015/WSDM_Topic_Evaluation/public.pdf>`_.
Typically, :class:`~gensim.models.coherencemodel.CoherenceModel` used for evaluation of topic models.

The four stage pipeline is basically:

    * Segmentation
    * Probability Estimation
    * Confirmation Measure
    * Aggregation

Implementation of this pipeline allows for the user to in essence "make" a coherence measure of his/her choice
by choosing a method in each of the pipelines.

See Also
--------
:mod:`gensim.topic_coherence`
    Internal functions for pipelines.

"""
import logging
import multiprocessing as mp
from collections import namedtuple
import numpy as np
from gensim import interfaces, matutils
from gensim import utils
from gensim.topic_coherence import segmentation, probability_estimation, direct_confirmation_measure, indirect_confirmation_measure, aggregation
from gensim.topic_coherence.probability_estimation import unique_ids_from_segments
logger = logging.getLogger(__name__)
BOOLEAN_DOCUMENT_BASED = {'u_mass'}
SLIDING_WINDOW_BASED = {'c_v', 'c_uci', 'c_npmi', 'c_w2v'}
_make_pipeline = namedtuple('Coherence_Measure', 'seg, prob, conf, aggr')
COHERENCE_MEASURES = {'u_mass': _make_pipeline(segmentation.s_one_pre, probability_estimation.p_boolean_document, direct_confirmation_measure.log_conditional_probability, aggregation.arithmetic_mean), 'c_v': _make_pipeline(segmentation.s_one_set, probability_estimation.p_boolean_sliding_window, indirect_confirmation_measure.cosine_similarity, aggregation.arithmetic_mean), 'c_w2v': _make_pipeline(segmentation.s_one_set, probability_estimation.p_word2vec, indirect_confirmation_measure.word2vec_similarity, aggregation.arithmetic_mean), 'c_uci': _make_pipeline(segmentation.s_one_one, probability_estimation.p_boolean_sliding_window, direct_confirmation_measure.log_ratio_measure, aggregation.arithmetic_mean), 'c_npmi': _make_pipeline(segmentation.s_one_one, probability_estimation.p_boolean_sliding_window, direct_confirmation_measure.log_ratio_measure, aggregation.arithmetic_mean)}
SLIDING_WINDOW_SIZES = {'c_v': 110, 'c_w2v': 5, 'c_uci': 10, 'c_npmi': 10, 'u_mass': None}

class CoherenceModel(interfaces.TransformationABC):
    """Objects of this class allow for building and maintaining a model for topic coherence.

    Examples
    ---------
    One way of using this feature is through providing a trained topic model. A dictionary has to be explicitly provided
    if the model does not contain a dictionary already

    .. sourcecode:: pycon

        >>> from gensim.test.utils import common_corpus, common_dictionary
        >>> from gensim.models.ldamodel import LdaModel
        >>> from gensim.models.coherencemodel import CoherenceModel
        >>>
        >>> model = LdaModel(common_corpus, 5, common_dictionary)
        >>>
        >>> cm = CoherenceModel(model=model, corpus=common_corpus, coherence='u_mass')
        >>> coherence = cm.get_coherence()  # get coherence value

    Another way of using this feature is through providing tokenized topics such as:

    .. sourcecode:: pycon

        >>> from gensim.test.utils import common_corpus, common_dictionary
        >>> from gensim.models.coherencemodel import CoherenceModel
        >>> topics = [
        ...     ['human', 'computer', 'system', 'interface'],
        ...     ['graph', 'minors', 'trees', 'eps']
        ... ]
        >>>
        >>> cm = CoherenceModel(topics=topics, corpus=common_corpus, dictionary=common_dictionary, coherence='u_mass')
        >>> coherence = cm.get_coherence()  # get coherence value

    """

    def __init__(self, model=None, topics=None, texts=None, corpus=None, dictionary=None, window_size=None, keyed_vectors=None, coherence='c_v', topn=20, processes=-1):
        if False:
            for i in range(10):
                print('nop')
        "\n\n        Parameters\n        ----------\n        model : :class:`~gensim.models.basemodel.BaseTopicModel`, optional\n            Pre-trained topic model, should be provided if topics is not provided.\n            Currently supports :class:`~gensim.models.ldamodel.LdaModel`,\n            :class:`~gensim.models.ldamulticore.LdaMulticore`.\n            Use `topics` parameter to plug in an as yet unsupported model.\n        topics : list of list of str, optional\n            List of tokenized topics, if this is preferred over model - dictionary should be provided.\n        texts : list of list of str, optional\n            Tokenized texts, needed for coherence models that use sliding window based (i.e. coherence=`c_something`)\n            probability estimator .\n        corpus : iterable of list of (int, number), optional\n            Corpus in BoW format.\n        dictionary : :class:`~gensim.corpora.dictionary.Dictionary`, optional\n            Gensim dictionary mapping of id word to create corpus.\n            If `model.id2word` is present, this is not needed. If both are provided, passed `dictionary` will be used.\n        window_size : int, optional\n            Is the size of the window to be used for coherence measures using boolean sliding window as their\n            probability estimator. For 'u_mass' this doesn't matter.\n            If None - the default window sizes are used which are: 'c_v' - 110, 'c_uci' - 10, 'c_npmi' - 10.\n        coherence : {'u_mass', 'c_v', 'c_uci', 'c_npmi'}, optional\n            Coherence measure to be used.\n            Fastest method - 'u_mass', 'c_uci' also known as `c_pmi`.\n            For 'u_mass' corpus should be provided, if texts is provided, it will be converted to corpus\n            using the dictionary. For 'c_v', 'c_uci' and 'c_npmi' `texts` should be provided (`corpus` isn't needed)\n        topn : int, optional\n            Integer corresponding to the number of top words to be extracted from each topic.\n        processes : int, optional\n            Number of processes to use for probability estimation phase, any value less than 1 will be interpreted as\n            num_cpus - 1.\n\n        "
        if model is None and topics is None:
            raise ValueError('One of model or topics has to be provided.')
        elif topics is not None and dictionary is None:
            raise ValueError('dictionary has to be provided if topics are to be used.')
        self.keyed_vectors = keyed_vectors
        if keyed_vectors is None and texts is None and (corpus is None):
            raise ValueError('One of texts or corpus has to be provided.')
        if dictionary is None:
            if isinstance(model.id2word, utils.FakeDict):
                raise ValueError("The associated dictionary should be provided with the corpus or 'id2word' for topic model should be set as the associated dictionary.")
            else:
                self.dictionary = model.id2word
        else:
            self.dictionary = dictionary
        self.coherence = coherence
        self.window_size = window_size
        if self.window_size is None:
            self.window_size = SLIDING_WINDOW_SIZES[self.coherence]
        self.texts = texts
        self.corpus = corpus
        if coherence in BOOLEAN_DOCUMENT_BASED:
            if utils.is_corpus(corpus)[0]:
                self.corpus = corpus
            elif self.texts is not None:
                self.corpus = [self.dictionary.doc2bow(text) for text in self.texts]
            else:
                raise ValueError("Either 'corpus' with 'dictionary' or 'texts' should be provided for %s coherence.", coherence)
        elif coherence == 'c_w2v' and keyed_vectors is not None:
            pass
        elif coherence in SLIDING_WINDOW_BASED:
            if self.texts is None:
                raise ValueError("'texts' should be provided for %s coherence.", coherence)
        else:
            raise ValueError('%s coherence is not currently supported.', coherence)
        self._topn = topn
        self._model = model
        self._accumulator = None
        self._topics = None
        self.topics = topics
        self.processes = processes if processes >= 1 else max(1, mp.cpu_count() - 1)

    @classmethod
    def for_models(cls, models, dictionary, topn=20, **kwargs):
        if False:
            i = 10
            return i + 15
        "Initialize a CoherenceModel with estimated probabilities for all of the given models.\n        Use :meth:`~gensim.models.coherencemodel.CoherenceModel.for_topics` method.\n\n        Parameters\n        ----------\n        models : list of :class:`~gensim.models.basemodel.BaseTopicModel`\n            List of models to evaluate coherence of, each of it should implements\n            :meth:`~gensim.models.basemodel.BaseTopicModel.get_topics` method.\n        dictionary : :class:`~gensim.corpora.dictionary.Dictionary`\n            Gensim dictionary mapping of id word.\n        topn : int, optional\n            Integer corresponding to the number of top words to be extracted from each topic.\n        kwargs : object\n            Sequence of arguments, see :meth:`~gensim.models.coherencemodel.CoherenceModel.for_topics`.\n\n        Return\n        ------\n        :class:`~gensim.models.coherencemodel.CoherenceModel`\n            CoherenceModel with estimated probabilities for all of the given models.\n\n        Example\n        -------\n        .. sourcecode:: pycon\n\n            >>> from gensim.test.utils import common_corpus, common_dictionary\n            >>> from gensim.models.ldamodel import LdaModel\n            >>> from gensim.models.coherencemodel import CoherenceModel\n            >>>\n            >>> m1 = LdaModel(common_corpus, 3, common_dictionary)\n            >>> m2 = LdaModel(common_corpus, 5, common_dictionary)\n            >>>\n            >>> cm = CoherenceModel.for_models([m1, m2], common_dictionary, corpus=common_corpus, coherence='u_mass')\n        "
        topics = [cls.top_topics_as_word_lists(model, dictionary, topn) for model in models]
        kwargs['dictionary'] = dictionary
        kwargs['topn'] = topn
        return cls.for_topics(topics, **kwargs)

    @staticmethod
    def top_topics_as_word_lists(model, dictionary, topn=20):
        if False:
            while True:
                i = 10
        'Get `topn` topics as list of words.\n\n        Parameters\n        ----------\n        model : :class:`~gensim.models.basemodel.BaseTopicModel`\n            Pre-trained topic model.\n        dictionary : :class:`~gensim.corpora.dictionary.Dictionary`\n            Gensim dictionary mapping of id word.\n        topn : int, optional\n            Integer corresponding to the number of top words to be extracted from each topic.\n\n        Return\n        ------\n        list of list of str\n            Top topics in list-of-list-of-words format.\n\n        '
        if not dictionary.id2token:
            dictionary.id2token = {v: k for (k, v) in dictionary.token2id.items()}
        str_topics = []
        for topic in model.get_topics():
            bestn = matutils.argsort(topic, topn=topn, reverse=True)
            beststr = [dictionary.id2token[_id] for _id in bestn]
            str_topics.append(beststr)
        return str_topics

    @classmethod
    def for_topics(cls, topics_as_topn_terms, **kwargs):
        if False:
            i = 10
            return i + 15
        'Initialize a CoherenceModel with estimated probabilities for all of the given topics.\n\n        Parameters\n        ----------\n        topics_as_topn_terms : list of list of str\n            Each element in the top-level list should be the list of topics for a model.\n            The topics for the model should be a list of top-N words, one per topic.\n\n        Return\n        ------\n        :class:`~gensim.models.coherencemodel.CoherenceModel`\n            CoherenceModel with estimated probabilities for all of the given models.\n\n        '
        if not topics_as_topn_terms:
            raise ValueError('len(topics) must be > 0.')
        if any((len(topic_lists) == 0 for topic_lists in topics_as_topn_terms)):
            raise ValueError('found empty topic listing in `topics`')
        topn = 0
        for topic_list in topics_as_topn_terms:
            for topic in topic_list:
                topn = max(topn, len(topic))
        topn = min(kwargs.pop('topn', topn), topn)
        super_topic = utils.flatten(topics_as_topn_terms)
        logging.info('Number of relevant terms for all %d models: %d', len(topics_as_topn_terms), len(super_topic))
        cm = CoherenceModel(topics=[super_topic], topn=len(super_topic), **kwargs)
        cm.estimate_probabilities()
        cm.topn = topn
        return cm

    def __str__(self):
        if False:
            return 10
        return str(self.measure)

    @property
    def model(self):
        if False:
            while True:
                i = 10
        'Get `self._model` field.\n\n        Return\n        ------\n        :class:`~gensim.models.basemodel.BaseTopicModel`\n            Used model.\n\n        '
        return self._model

    @model.setter
    def model(self, model):
        if False:
            print('Hello World!')
        'Set `self._model` field.\n\n        Parameters\n        ----------\n        model : :class:`~gensim.models.basemodel.BaseTopicModel`\n            Input model.\n\n        '
        self._model = model
        if model is not None:
            new_topics = self._get_topics()
            self._update_accumulator(new_topics)
            self._topics = new_topics

    @property
    def topn(self):
        if False:
            print('Hello World!')
        'Get number of top words `self._topn`.\n\n        Return\n        ------\n        int\n            Integer corresponding to the number of top words.\n\n        '
        return self._topn

    @topn.setter
    def topn(self, topn):
        if False:
            while True:
                i = 10
        'Set number of top words `self._topn`.\n\n        Parameters\n        ----------\n        topn : int\n            Number of top words.\n\n        '
        current_topic_length = len(self._topics[0])
        requires_expansion = current_topic_length < topn
        if self.model is not None:
            self._topn = topn
            if requires_expansion:
                self.model = self._model
        else:
            if requires_expansion:
                raise ValueError('Model unavailable and topic sizes are less than topn=%d' % topn)
            self._topn = topn

    @property
    def measure(self):
        if False:
            i = 10
            return i + 15
        'Make pipeline, according to `coherence` parameter value.\n\n        Return\n        ------\n        namedtuple\n            Pipeline that contains needed functions/method for calculated coherence.\n\n        '
        return COHERENCE_MEASURES[self.coherence]

    @property
    def topics(self):
        if False:
            i = 10
            return i + 15
        'Get topics `self._topics`.\n\n        Return\n        ------\n        list of list of str\n            Topics as list of tokens.\n\n        '
        if len(self._topics[0]) > self._topn:
            return [topic[:self._topn] for topic in self._topics]
        else:
            return self._topics

    @topics.setter
    def topics(self, topics):
        if False:
            print('Hello World!')
        'Set topics `self._topics`.\n\n        Parameters\n        ----------\n        topics : list of list of str\n            Topics.\n\n        '
        if topics is not None:
            new_topics = []
            for topic in topics:
                topic_token_ids = self._ensure_elements_are_ids(topic)
                new_topics.append(topic_token_ids)
            if self.model is not None:
                logger.warning("The currently set model '%s' may be inconsistent with the newly set topics", self.model)
        elif self.model is not None:
            new_topics = self._get_topics()
            logger.debug('Setting topics to those of the model: %s', self.model)
        else:
            new_topics = None
        self._update_accumulator(new_topics)
        self._topics = new_topics

    def _ensure_elements_are_ids(self, topic):
        if False:
            i = 10
            return i + 15
        ids_from_tokens = [self.dictionary.token2id[t] for t in topic if t in self.dictionary.token2id]
        ids_from_ids = [i for i in topic if i in self.dictionary]
        if len(ids_from_tokens) > len(ids_from_ids):
            return np.array(ids_from_tokens)
        elif len(ids_from_ids) > len(ids_from_tokens):
            return np.array(ids_from_ids)
        else:
            raise ValueError('unable to interpret topic as either a list of tokens or a list of ids')

    def _update_accumulator(self, new_topics):
        if False:
            return 10
        if self._relevant_ids_will_differ(new_topics):
            logger.debug('Wiping cached accumulator since it does not contain all relevant ids.')
            self._accumulator = None

    def _relevant_ids_will_differ(self, new_topics):
        if False:
            while True:
                i = 10
        if self._accumulator is None or not self._topics_differ(new_topics):
            return False
        new_set = unique_ids_from_segments(self.measure.seg(new_topics))
        return not self._accumulator.relevant_ids.issuperset(new_set)

    def _topics_differ(self, new_topics):
        if False:
            return 10
        return new_topics is not None and self._topics is not None and (not np.array_equal(new_topics, self._topics))

    def _get_topics(self):
        if False:
            return 10
        'Internal helper function to return topics from a trained topic model.'
        return self._get_topics_from_model(self.model, self.topn)

    @staticmethod
    def _get_topics_from_model(model, topn):
        if False:
            return 10
        'Internal helper function to return topics from a trained topic model.\n\n        Parameters\n        ----------\n        model : :class:`~gensim.models.basemodel.BaseTopicModel`\n            Pre-trained topic model.\n        topn : int\n            Integer corresponding to the number of top words.\n\n        Return\n        ------\n        list of :class:`numpy.ndarray`\n            Topics matrix\n\n        '
        try:
            return [matutils.argsort(topic, topn=topn, reverse=True) for topic in model.get_topics()]
        except AttributeError:
            raise ValueError('This topic model is not currently supported. Supported topic models should implement the `get_topics` method.')

    def segment_topics(self):
        if False:
            for i in range(10):
                print('nop')
        'Segment topic, alias for `self.measure.seg(self.topics)`.\n\n        Return\n        ------\n        list of list of pair\n            Segmented topics.\n\n        '
        return self.measure.seg(self.topics)

    def estimate_probabilities(self, segmented_topics=None):
        if False:
            return 10
        'Accumulate word occurrences and co-occurrences from texts or corpus using the optimal method for the chosen\n        coherence metric.\n\n        Notes\n        -----\n        This operation may take quite some time for the sliding window based coherence methods.\n\n        Parameters\n        ----------\n        segmented_topics : list of list of pair, optional\n            Segmented topics, typically produced by :meth:`~gensim.models.coherencemodel.CoherenceModel.segment_topics`.\n\n        Return\n        ------\n        :class:`~gensim.topic_coherence.text_analysis.CorpusAccumulator`\n            Corpus accumulator.\n\n        '
        if segmented_topics is None:
            segmented_topics = self.segment_topics()
        if self.coherence in BOOLEAN_DOCUMENT_BASED:
            self._accumulator = self.measure.prob(self.corpus, segmented_topics)
        else:
            kwargs = dict(texts=self.texts, segmented_topics=segmented_topics, dictionary=self.dictionary, window_size=self.window_size, processes=self.processes)
            if self.coherence == 'c_w2v':
                kwargs['model'] = self.keyed_vectors
            self._accumulator = self.measure.prob(**kwargs)
        return self._accumulator

    def get_coherence_per_topic(self, segmented_topics=None, with_std=False, with_support=False):
        if False:
            print('Hello World!')
        'Get list of coherence values for each topic based on pipeline parameters.\n\n        Parameters\n        ----------\n        segmented_topics : list of list of (int, number)\n            Topics.\n        with_std : bool, optional\n            True to also include standard deviation across topic segment sets in addition to the mean coherence\n            for each topic.\n        with_support : bool, optional\n            True to also include support across topic segments. The support is defined as the number of pairwise\n            similarity comparisons were used to compute the overall topic coherence.\n\n        Return\n        ------\n        list of float\n            Sequence of similarity measure for each topic.\n\n        '
        measure = self.measure
        if segmented_topics is None:
            segmented_topics = measure.seg(self.topics)
        if self._accumulator is None:
            self.estimate_probabilities(segmented_topics)
        kwargs = dict(with_std=with_std, with_support=with_support)
        if self.coherence in BOOLEAN_DOCUMENT_BASED or self.coherence == 'c_w2v':
            pass
        elif self.coherence == 'c_v':
            kwargs['topics'] = self.topics
            kwargs['measure'] = 'nlr'
            kwargs['gamma'] = 1
        else:
            kwargs['normalize'] = self.coherence == 'c_npmi'
        return measure.conf(segmented_topics, self._accumulator, **kwargs)

    def aggregate_measures(self, topic_coherences):
        if False:
            for i in range(10):
                print('nop')
        "Aggregate the individual topic coherence measures using the pipeline's aggregation function.\n        Use `self.measure.aggr(topic_coherences)`.\n\n        Parameters\n        ----------\n        topic_coherences : list of float\n            List of calculated confirmation measure on each set in the segmented topics.\n\n        Returns\n        -------\n        float\n            Arithmetic mean of all the values contained in confirmation measures.\n\n        "
        return self.measure.aggr(topic_coherences)

    def get_coherence(self):
        if False:
            while True:
                i = 10
        'Get coherence value based on pipeline parameters.\n\n        Returns\n        -------\n        float\n            Value of coherence.\n\n        '
        confirmed_measures = self.get_coherence_per_topic()
        return self.aggregate_measures(confirmed_measures)

    def compare_models(self, models):
        if False:
            print('Hello World!')
        'Compare topic models by coherence value.\n\n        Parameters\n        ----------\n        models : :class:`~gensim.models.basemodel.BaseTopicModel`\n            Sequence of topic models.\n\n        Returns\n        -------\n        list of (float, float)\n            Sequence of pairs of average topic coherence and average coherence.\n\n        '
        model_topics = [self._get_topics_from_model(model, self.topn) for model in models]
        return self.compare_model_topics(model_topics)

    def compare_model_topics(self, model_topics):
        if False:
            return 10
        'Perform the coherence evaluation for each of the models.\n\n        Parameters\n        ----------\n        model_topics : list of list of str\n            list of list of words for the model trained with that number of topics.\n\n        Returns\n        -------\n        list of (float, float)\n            Sequence of pairs of average topic coherence and average coherence.\n\n        Notes\n        -----\n        This first precomputes the probabilities once, then evaluates coherence for each model.\n\n        Since we have already precomputed the probabilities, this simply involves using the accumulated stats in the\n        :class:`~gensim.models.coherencemodel.CoherenceModel` to perform the evaluations, which should be pretty quick.\n\n        '
        orig_topics = self._topics
        orig_topn = self.topn
        try:
            coherences = self._compare_model_topics(model_topics)
        finally:
            self.topics = orig_topics
            self.topn = orig_topn
        return coherences

    def _compare_model_topics(self, model_topics):
        if False:
            print('Hello World!')
        'Get average topic and model coherences.\n\n        Parameters\n        ----------\n        model_topics : list of list of str\n            Topics from the model.\n\n        Returns\n        -------\n        list of (float, float)\n            Sequence of pairs of average topic coherence and average coherence.\n\n        '
        coherences = []
        last_topn_value = min(self.topn - 1, 4)
        topn_grid = list(range(self.topn, last_topn_value, -5))
        for (model_num, topics) in enumerate(model_topics):
            self.topics = topics
            coherence_at_n = {}
            for n in topn_grid:
                self.topn = n
                topic_coherences = self.get_coherence_per_topic()
                filled_coherences = np.array(topic_coherences)
                filled_coherences[np.isnan(filled_coherences)] = np.nanmean(filled_coherences)
                coherence_at_n[n] = (topic_coherences, self.aggregate_measures(filled_coherences))
            (topic_coherences, avg_coherences) = zip(*coherence_at_n.values())
            avg_topic_coherences = np.vstack(topic_coherences).mean(0)
            model_coherence = np.mean(avg_coherences)
            logging.info('Avg coherence for model %d: %.5f' % (model_num, model_coherence))
            coherences.append((avg_topic_coherences, model_coherence))
        return coherences