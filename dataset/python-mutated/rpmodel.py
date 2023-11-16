"""Random Projections (also known as Random Indexing).

For theoretical background on Random Projections, see [1]_.


Examples
--------
.. sourcecode:: pycon

    >>> from gensim.models import RpModel
    >>> from gensim.corpora import Dictionary
    >>> from gensim.test.utils import common_texts, temporary_file
    >>>
    >>> dictionary = Dictionary(common_texts)  # fit dictionary
    >>> corpus = [dictionary.doc2bow(text) for text in common_texts]  # convert texts to BoW format
    >>>
    >>> model = RpModel(corpus, id2word=dictionary)  # fit model
    >>> result = model[corpus[3]]  # apply model to document, result is vector in BoW format
    >>>
    >>> with temporary_file("model_file") as fname:
    ...     model.save(fname)  # save model to file
    ...     loaded_model = RpModel.load(fname)  # load model


References
----------
.. [1] Kanerva et al., 2000, Random indexing of text samples for Latent Semantic Analysis,
       https://cloudfront.escholarship.org/dist/prd/content/qt5644k0w6/qt5644k0w6.pdf

"""
import logging
import numpy as np
from gensim import interfaces, matutils, utils
logger = logging.getLogger(__name__)

class RpModel(interfaces.TransformationABC):

    def __init__(self, corpus, id2word=None, num_topics=300):
        if False:
            while True:
                i = 10
        '\n\n        Parameters\n        ----------\n        corpus : iterable of iterable of (int, int)\n            Input corpus.\n\n        id2word : {dict of (int, str), :class:`~gensim.corpora.dictionary.Dictionary`}, optional\n            Mapping `token_id` -> `token`, will be determine from corpus if `id2word == None`.\n\n        num_topics : int, optional\n            Number of topics.\n\n        '
        self.id2word = id2word
        self.num_topics = num_topics
        if corpus is not None:
            self.initialize(corpus)
            self.add_lifecycle_event('created', msg=f'created {self}')

    def __str__(self):
        if False:
            print('Hello World!')
        return '%s<num_terms=%s, num_topics=%s>' % (self.__class__.__name__, self.num_terms, self.num_topics)

    def initialize(self, corpus):
        if False:
            for i in range(10):
                print('nop')
        'Initialize the random projection matrix.\n\n        Parameters\n        ----------\n        corpus : iterable of iterable of (int, int)\n          Input corpus.\n\n        '
        if self.id2word is None:
            logger.info('no word id mapping provided; initializing from corpus, assuming identity')
            self.id2word = utils.dict_from_corpus(corpus)
            self.num_terms = len(self.id2word)
        elif self.id2word:
            self.num_terms = 1 + max(self.id2word)
        else:
            self.num_terms = 0
        shape = (self.num_topics, self.num_terms)
        logger.info('constructing %s random matrix', str(shape))
        randmat = 1 - 2 * np.random.binomial(1, 0.5, shape)
        self.projection = np.asfortranarray(randmat, dtype=np.float32)

    def __getitem__(self, bow):
        if False:
            print('Hello World!')
        'Get random-projection representation of the input vector or corpus.\n\n        Parameters\n        ----------\n        bow : {list of (int, int), iterable of list of (int, int)}\n            Input document or corpus.\n\n        Returns\n        -------\n        list of (int, float)\n            if `bow` is document OR\n        :class:`~gensim.interfaces.TransformedCorpus`\n            if `bow` is corpus.\n\n        Examples\n        ----------\n        .. sourcecode:: pycon\n\n            >>> from gensim.models import RpModel\n            >>> from gensim.corpora import Dictionary\n            >>> from gensim.test.utils import common_texts\n            >>>\n            >>> dictionary = Dictionary(common_texts)  # fit dictionary\n            >>> corpus = [dictionary.doc2bow(text) for text in common_texts]  # convert texts to BoW format\n            >>>\n            >>> model = RpModel(corpus, id2word=dictionary)  # fit model\n            >>>\n            >>> # apply model to document, result is vector in BoW format, i.e. [(1, 0.3), ... ]\n            >>> result = model[corpus[0]]\n\n        '
        (is_corpus, bow) = utils.is_corpus(bow)
        if is_corpus:
            return self._apply(bow)
        if getattr(self, 'freshly_loaded', False):
            self.freshly_loaded = False
            self.projection = self.projection.copy('F')
        vec = matutils.sparse2full(bow, self.num_terms).reshape(self.num_terms, 1) / np.sqrt(self.num_topics)
        vec = np.asfortranarray(vec, dtype=np.float32)
        topic_dist = np.dot(self.projection, vec)
        return [(topicid, float(topicvalue)) for (topicid, topicvalue) in enumerate(topic_dist.flat) if np.isfinite(topicvalue) and (not np.allclose(topicvalue, 0.0))]

    def __setstate__(self, state):
        if False:
            return 10
        'Sets the internal state and updates freshly_loaded to True, called when unpicked.\n\n        Parameters\n        ----------\n        state : dict\n           State of the class.\n\n        '
        self.__dict__ = state
        self.freshly_loaded = True