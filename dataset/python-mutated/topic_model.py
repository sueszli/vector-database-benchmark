"""
Methods for creating a topic model and predicting the topics of new documents.
"""
from __future__ import print_function as _
from __future__ import division as _
from __future__ import absolute_import as _
import turicreate as _turicreate
from turicreate.toolkits._model import Model as _Model
from turicreate.data_structures.sframe import SFrame as _SFrame
from turicreate.data_structures.sarray import SArray as _SArray
from turicreate.toolkits.text_analytics._util import _check_input
from turicreate.toolkits.text_analytics._util import random_split as _random_split
from turicreate.toolkits._internal_utils import _check_categorical_option_type, _precomputed_field, _toolkit_repr_print
import sys as _sys
if _sys.version_info.major == 3:
    _izip = zip
    _xrange = range
else:
    from itertools import izip as _izip
    _xrange = xrange
import operator as _operator
import array as _array

def create(dataset, num_topics=10, initial_topics=None, alpha=None, beta=0.1, num_iterations=10, num_burnin=5, associations=None, verbose=False, print_interval=10, validation_set=None, method='auto'):
    if False:
        return 10
    '\n    Create a topic model from the given data set. A topic model assumes each\n    document is a mixture of a set of topics, where for each topic some words\n    are more likely than others. One statistical approach to do this is called a\n    "topic model". This method learns a topic model for the given document\n    collection.\n\n    Parameters\n    ----------\n    dataset : SArray of type dict or SFrame with a single column of type dict\n        A bag of words representation of a document corpus.\n        Each element is a dictionary representing a single document, where\n        the keys are words and the values are the number of times that word\n        occurs in that document.\n\n    num_topics : int, optional\n        The number of topics to learn.\n\n    initial_topics : SFrame, optional\n        An SFrame with a column of unique words representing the vocabulary\n        and a column of dense vectors representing\n        probability of that word given each topic. When provided,\n        these values are used to initialize the algorithm.\n\n    alpha : float, optional\n        Hyperparameter that controls the diversity of topics in a document.\n        Smaller values encourage fewer topics per document.\n        Provided value must be positive. Default value is 50/num_topics.\n\n    beta : float, optional\n        Hyperparameter that controls the diversity of words in a topic.\n        Smaller values encourage fewer words per topic. Provided value\n        must be positive.\n\n    num_iterations : int, optional\n        The number of iterations to perform.\n\n    num_burnin : int, optional\n        The number of iterations to perform when inferring the topics for\n        documents at prediction time.\n\n    verbose : bool, optional\n        When True, print most probable words for each topic while printing\n        progress.\n\n    print_interval : int, optional\n        The number of iterations to wait between progress reports.\n\n    associations : SFrame, optional\n        An SFrame with two columns named "word" and "topic" containing words\n        and the topic id that the word should be associated with. These words\n        are not considered during learning.\n\n    validation_set : SArray of type dict or SFrame with a single column\n        A bag of words representation of a document corpus, similar to the\n        format required for `dataset`. This will be used to monitor model\n        performance during training. Each document in the provided validation\n        set is randomly split: the first portion is used estimate which topic\n        each document belongs to, and the second portion is used to estimate\n        the model\'s performance at predicting the unseen words in the test data.\n\n    method : {\'cgs\', \'alias\'}, optional\n        The algorithm used for learning the model.\n\n        - *cgs:* Collapsed Gibbs sampling\n        - *alias:* AliasLDA method.\n\n    Returns\n    -------\n    out : TopicModel\n        A fitted topic model. This can be used with\n        :py:func:`~TopicModel.get_topics()` and\n        :py:func:`~TopicModel.predict()`. While fitting is in progress, several\n        metrics are shown, including:\n\n        +------------------+---------------------------------------------------+\n        |      Field       | Description                                       |\n        +==================+===================================================+\n        | Elapsed Time     | The number of elapsed seconds.                    |\n        +------------------+---------------------------------------------------+\n        | Tokens/second    | The number of unique words processed per second   |\n        +------------------+---------------------------------------------------+\n        | Est. Perplexity  | An estimate of the model\'s ability to model the   |\n        |                  | training data. See the documentation on evaluate. |\n        +------------------+---------------------------------------------------+\n\n    See Also\n    --------\n    TopicModel, TopicModel.get_topics, TopicModel.predict,\n    turicreate.SArray.dict_trim_by_keys, TopicModel.evaluate\n\n    References\n    ----------\n    - `Wikipedia - Latent Dirichlet allocation\n      <http://en.wikipedia.org/wiki/Latent_Dirichlet_allocation>`_\n\n    - Alias method: Li, A. et al. (2014) `Reducing the Sampling Complexity of\n      Topic Models. <http://www.sravi.org/pubs/fastlda-kdd2014.pdf>`_.\n      KDD 2014.\n\n    Examples\n    --------\n    The following example includes an SArray of documents, where\n    each element represents a document in "bag of words" representation\n    -- a dictionary with word keys and whose values are the number of times\n    that word occurred in the document:\n\n    >>> docs = turicreate.SArray(\'https://static.turi.com/datasets/nytimes\')\n\n    Once in this form, it is straightforward to learn a topic model.\n\n    >>> m = turicreate.topic_model.create(docs)\n\n    It is also easy to create a new topic model from an old one  -- whether\n    it was created using Turi Create or another package.\n\n    >>> m2 = turicreate.topic_model.create(docs, initial_topics=m[\'topics\'])\n\n    To manually fix several words to always be assigned to a topic, use\n    the `associations` argument. The following will ensure that topic 0\n    has the most probability for each of the provided words:\n\n    >>> from turicreate import SFrame\n    >>> associations = SFrame({\'word\':[\'hurricane\', \'wind\', \'storm\'],\n                               \'topic\': [0, 0, 0]})\n    >>> m = turicreate.topic_model.create(docs,\n                                        associations=associations)\n\n    More advanced usage allows you  to control aspects of the model and the\n    learning method.\n\n    >>> import turicreate as tc\n    >>> m = tc.topic_model.create(docs,\n                                  num_topics=20,       # number of topics\n                                  num_iterations=10,   # algorithm parameters\n                                  alpha=.01, beta=.1)  # hyperparameters\n\n    To evaluate the model\'s ability to generalize, we can create a train/test\n    split where a portion of the words in each document are held out from\n    training.\n\n    >>> train, test = tc.text_analytics.random_split(.8)\n    >>> m = tc.topic_model.create(train)\n    >>> results = m.evaluate(test)\n    >>> print results[\'perplexity\']\n\n    '
    dataset = _check_input(dataset)
    _check_categorical_option_type('method', method, ['auto', 'cgs', 'alias'])
    if method == 'cgs' or method == 'auto':
        model_name = 'cgs_topic_model'
    else:
        model_name = 'alias_topic_model'
    if associations is None:
        associations = _turicreate.SFrame({'word': [], 'topic': []})
    if isinstance(associations, _turicreate.SFrame) and associations.num_rows() > 0:
        assert set(associations.column_names()) == set(['word', 'topic']), 'Provided associations must be an SFrame containing a word column             and a topic column.'
        assert associations['word'].dtype == str, 'Words must be strings.'
        assert associations['topic'].dtype == int, 'Topic ids must be of int type.'
    if alpha is None:
        alpha = float(50) / num_topics
    if validation_set is not None:
        _check_input(validation_set)
        if isinstance(validation_set, _turicreate.SFrame):
            column_name = validation_set.column_names()[0]
            validation_set = validation_set[column_name]
        (validation_train, validation_test) = _random_split(validation_set)
    else:
        validation_train = _SArray()
        validation_test = _SArray()
    opts = {'model_name': model_name, 'data': dataset, 'num_topics': num_topics, 'num_iterations': num_iterations, 'print_interval': print_interval, 'alpha': alpha, 'beta': beta, 'num_burnin': num_burnin, 'associations': associations}
    response = _turicreate.extensions._text.topicmodel_init(opts)
    m = TopicModel(response['model'])
    if isinstance(initial_topics, _turicreate.SFrame):
        assert set(['vocabulary', 'topic_probabilities']) == set(initial_topics.column_names()), 'The provided initial_topics does not have the proper format,              e.g. wrong column names.'
        observed_topics = initial_topics['topic_probabilities'].apply(lambda x: len(x))
        assert all(observed_topics == num_topics), 'Provided num_topics value does not match the number of provided initial_topics.'
        weight = len(dataset) * 1000
        opts = {'model': m.__proxy__, 'topics': initial_topics['topic_probabilities'], 'vocabulary': initial_topics['vocabulary'], 'weight': weight}
        response = _turicreate.extensions._text.topicmodel_set_topics(opts)
        m = TopicModel(response['model'])
    opts = {'model': m.__proxy__, 'data': dataset, 'verbose': verbose, 'validation_train': validation_train, 'validation_test': validation_test}
    response = _turicreate.extensions._text.topicmodel_train(opts)
    m = TopicModel(response['model'])
    return m

class TopicModel(_Model):
    """
    TopicModel objects can be used to predict the underlying topic of a
    document.

    This model cannot be constructed directly.  Instead, use
    :func:`turicreate.topic_model.create` to create an instance
    of this model. A detailed list of parameter options and code samples
    are available in the documentation for the create function.
    """

    def __init__(self, model_proxy):
        if False:
            print('Hello World!')
        self.__proxy__ = model_proxy

    @classmethod
    def _native_name(cls):
        if False:
            for i in range(10):
                print('nop')
        return ['cgs_topic_model', 'alias_topic_model']

    def __str__(self):
        if False:
            while True:
                i = 10
        '\n        Return a string description of the model to the ``print`` method.\n\n        Returns\n        -------\n        out : string\n            A description of the model.\n        '
        return self.__repr__()

    def _get_summary_struct(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Returns a structured description of the model, including (where relevant)\n        the schema of the training data, description of the training data,\n        training statistics, and model hyperparameters.\n\n        Returns\n        -------\n        sections : list (of list of tuples)\n            A list of summary sections.\n              Each section is a list.\n                Each item in a section list is a tuple of the form:\n                  ('<label>','<field>')\n        section_titles: list\n            A list of section titles.\n              The order matches that of the 'sections' object.\n        "
        section_titles = ['Schema', 'Settings']
        vocab_length = len(self.vocabulary)
        verbose = self.verbose == 1
        sections = [[('Vocabulary Size', _precomputed_field(vocab_length))], [('Number of Topics', 'num_topics'), ('alpha', 'alpha'), ('beta', 'beta'), ('Iterations', 'num_iterations'), ('Training time', 'training_time'), ('Verbose', _precomputed_field(verbose))]]
        return (sections, section_titles)

    def __repr__(self):
        if False:
            while True:
                i = 10
        '\n        Print a string description of the model when the model name is entered\n        in the terminal.\n        '
        key_str = '{:<{}}: {}'
        width = 30
        (sections, section_titles) = self._get_summary_struct()
        out = _toolkit_repr_print(self, sections, section_titles, width=width)
        extra = []
        extra.append(key_str.format('Accessible fields', width, ''))
        extra.append(key_str.format('m.topics', width, 'An SFrame containing the topics.'))
        extra.append(key_str.format('m.vocabulary', width, 'An SArray containing the words in the vocabulary.'))
        extra.append(key_str.format('Useful methods', width, ''))
        extra.append(key_str.format('m.get_topics()', width, 'Get the most probable words per topic.'))
        extra.append(key_str.format('m.predict(new_docs)', width, 'Make predictions for new documents.'))
        return out + '\n' + '\n'.join(extra)

    def _get(self, field):
        if False:
            while True:
                i = 10
        '\n        Return the value of a given field. The list of all queryable fields is\n        detailed below, and can be obtained with the\n        :py:func:`~TopicModel._list_fields` method.\n\n        +-----------------------+----------------------------------------------+\n        |      Field            | Description                                  |\n        +=======================+==============================================+\n        | topics                | An SFrame containing a column with the unique|\n        |                       | words observed during training, and a column |\n        |                       | of arrays containing the probability values  |\n        |                       | for each word given each of the topics.      |\n        +-----------------------+----------------------------------------------+\n        | vocabulary            | An SArray containing the words used. This is |\n        |                       | same as the vocabulary column in the topics  |\n        |                       | field above.                                 |\n        +-----------------------+----------------------------------------------+\n\n        Parameters\n        ----------\n        field : string\n            Name of the field to be retrieved.\n\n        Returns\n        -------\n        out\n            Value of the requested field.\n        '
        opts = {'model': self.__proxy__, 'field': field}
        response = _turicreate.extensions._text.topicmodel_get_value(opts)
        return response['value']

    def _training_stats(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Return a dictionary of statistics collected during creation of the\n        model. These statistics are also available with the ``get`` method and\n        are described in more detail in that method's documentation.\n\n        Returns\n        -------\n        out : dict\n            Dictionary of statistics compiled during creation of the\n            TopicModel.\n\n        See Also\n        --------\n        summary\n\n        Examples\n        --------\n        >>> docs = turicreate.SArray('https://static.turi.com/datasets/nips-text')\n        >>> m = turicreate.topic_model.create(docs)\n        >>> m._training_stats()\n        {'training_iterations': 20,\n         'training_time': 20.5034}\n        "
        fields = self._list_fields()
        stat_fields = ['training_time', 'training_iterations']
        if 'validation_perplexity' in fields:
            stat_fields.append('validation_perplexity')
        ret = {k: self._get(k) for k in stat_fields}
        return ret

    def get_topics(self, topic_ids=None, num_words=5, cdf_cutoff=1.0, output_type='topic_probabilities'):
        if False:
            i = 10
            return i + 15
        "\n        Get the words associated with a given topic. The score column is the\n        probability of choosing that word given that you have chosen a\n        particular topic.\n\n        Parameters\n        ----------\n        topic_ids : list of int, optional\n            The topics to retrieve words. Topic ids are zero-based.\n            Throws an error if greater than or equal to m['num_topics'], or\n            if the requested topic name is not present.\n\n        num_words : int, optional\n            The number of words to show.\n\n        cdf_cutoff : float, optional\n            Allows one to only show the most probable words whose cumulative\n            probability is below this cutoff. For example if there exist\n            three words where\n\n            .. math::\n               p(word_1 | topic_k) = .1\n\n               p(word_2 | topic_k) = .2\n\n               p(word_3 | topic_k) = .05\n\n            then setting :math:`cdf_{cutoff}=.3` would return only\n            :math:`word_1` and :math:`word_2` since\n            :math:`p(word_1 | topic_k) + p(word_2 | topic_k) <= cdf_{cutoff}`\n\n        output_type : {'topic_probabilities' | 'topic_words'}, optional\n            Determine the type of desired output. See below.\n\n        Returns\n        -------\n        out : SFrame\n            If output_type is 'topic_probabilities', then the returned value is\n            an SFrame with a column of words ranked by a column of scores for\n            each topic. Otherwise, the returned value is a SArray where\n            each element is a list of the most probable words for each topic.\n\n        Examples\n        --------\n        Get the highest ranked words for all topics.\n\n        >>> docs = turicreate.SArray('https://static.turi.com/datasets/nips-text')\n        >>> m = turicreate.topic_model.create(docs,\n                                            num_iterations=50)\n        >>> m.get_topics()\n        +-------+----------+-----------------+\n        | topic |   word   |      score      |\n        +-------+----------+-----------------+\n        |   0   |   cell   |  0.028974400831 |\n        |   0   |  input   | 0.0259470208503 |\n        |   0   |  image   | 0.0215721599763 |\n        |   0   |  visual  | 0.0173635081992 |\n        |   0   |  object  | 0.0172447874156 |\n        |   1   | function | 0.0482834508265 |\n        |   1   |  input   | 0.0456270024091 |\n        |   1   |  point   | 0.0302662839454 |\n        |   1   |  result  | 0.0239474934631 |\n        |   1   | problem  | 0.0231750116011 |\n        |  ...  |   ...    |       ...       |\n        +-------+----------+-----------------+\n\n        Get the highest ranked words for topics 0 and 1 and show 15 words per\n        topic.\n\n        >>> m.get_topics([0, 1], num_words=15)\n        +-------+----------+------------------+\n        | topic |   word   |      score       |\n        +-------+----------+------------------+\n        |   0   |   cell   |  0.028974400831  |\n        |   0   |  input   | 0.0259470208503  |\n        |   0   |  image   | 0.0215721599763  |\n        |   0   |  visual  | 0.0173635081992  |\n        |   0   |  object  | 0.0172447874156  |\n        |   0   | response | 0.0139740298286  |\n        |   0   |  layer   | 0.0122585145062  |\n        |   0   | features | 0.0115343177265  |\n        |   0   | feature  | 0.0103530459301  |\n        |   0   | spatial  | 0.00823387994361 |\n        |  ...  |   ...    |       ...        |\n        +-------+----------+------------------+\n\n        If one wants to instead just get the top words per topic, one may\n        change the format of the output as follows.\n\n        >>> topics = m.get_topics(output_type='topic_words')\n        dtype: list\n        Rows: 10\n        [['cell', 'image', 'input', 'object', 'visual'],\n         ['algorithm', 'data', 'learning', 'method', 'set'],\n         ['function', 'input', 'point', 'problem', 'result'],\n         ['model', 'output', 'pattern', 'set', 'unit'],\n         ['action', 'learning', 'net', 'problem', 'system'],\n         ['error', 'function', 'network', 'parameter', 'weight'],\n         ['information', 'level', 'neural', 'threshold', 'weight'],\n         ['control', 'field', 'model', 'network', 'neuron'],\n         ['hidden', 'layer', 'system', 'training', 'vector'],\n         ['component', 'distribution', 'local', 'model', 'optimal']]\n        "
        _check_categorical_option_type('output_type', output_type, ['topic_probabilities', 'topic_words'])
        if topic_ids is None:
            topic_ids = list(range(self._get('num_topics')))
        assert isinstance(topic_ids, list), 'The provided topic_ids is not a list.'
        if any([type(x) == str for x in topic_ids]):
            raise ValueError('Only integer topic_ids can be used at this point in time.')
        if not all([x >= 0 and x < self.num_topics for x in topic_ids]):
            raise ValueError('Topic id values must be non-negative and less than the ' + 'number of topics used to fit the model.')
        opts = {'model': self.__proxy__, 'topic_ids': topic_ids, 'num_words': num_words, 'cdf_cutoff': cdf_cutoff}
        response = _turicreate.extensions._text.topicmodel_get_topic(opts)
        ret = response['top_words']

        def sort_wordlist_by_prob(z):
            if False:
                for i in range(10):
                    print('nop')
            words = sorted(z.items(), key=_operator.itemgetter(1), reverse=True)
            return [word for (word, prob) in words]
        if output_type != 'topic_probabilities':
            ret = ret.groupby('topic', {'word': _turicreate.aggregate.CONCAT('word', 'score')})
            words = ret.sort('topic')['word'].apply(sort_wordlist_by_prob)
            ret = _SFrame({'words': words})
        return ret

    def predict(self, dataset, output_type='assignment', num_burnin=None):
        if False:
            print('Hello World!')
        "\n        Use the model to predict topics for each document. The provided\n        `dataset` should be an SArray object where each element is a dict\n        representing a single document in bag-of-words format, where keys\n        are words and values are their corresponding counts. If `dataset` is\n        an SFrame, then it must contain a single column of dict type.\n\n        The current implementation will make inferences about each document\n        given its estimates of the topics learned when creating the model.\n        This is done via Gibbs sampling.\n\n        Parameters\n        ----------\n        dataset : SArray, SFrame of type dict\n            A set of documents to use for making predictions.\n\n        output_type : str, optional\n            The type of output desired. This can either be\n\n            - assignment: the returned values are integers in [0, num_topics)\n            - probability: each returned prediction is a vector with length\n              num_topics, where element k represents the probability that\n              document belongs to topic k.\n\n        num_burnin : int, optional\n            The number of iterations of Gibbs sampling to perform when\n            inferring the topics for documents at prediction time.\n            If provided this will override the burnin value set during\n            training.\n\n        Returns\n        -------\n        out : SArray\n\n        See Also\n        --------\n        evaluate\n\n        Examples\n        --------\n        Make predictions about which topic each document belongs to.\n\n        >>> docs = turicreate.SArray('https://static.turi.com/datasets/nips-text')\n        >>> m = turicreate.topic_model.create(docs)\n        >>> pred = m.predict(docs)\n\n        If one is interested in the probability of each topic\n\n        >>> pred = m.predict(docs, output_type='probability')\n\n        Notes\n        -----\n        For each unique word w in a document d, we sample an assignment to\n        topic k with probability proportional to\n\n        .. math::\n            p(z_{dw} = k) \\propto (n_{d,k} + \\alpha) * \\Phi_{w,k}\n\n        where\n\n        - :math:`W` is the size of the vocabulary,\n        - :math:`n_{d,k}` is the number of other times we have assigned a word in\n          document to d to topic :math:`k`,\n        - :math:`\\Phi_{w,k}` is the probability under the model of choosing word\n          :math:`w` given the word is of topic :math:`k`. This is the matrix\n          returned by calling `m['topics']`.\n\n        This represents a collapsed Gibbs sampler for the document assignments\n        while we keep the topics learned during training fixed.\n        This process is done in parallel across all documents, five times per\n        document.\n\n        "
        dataset = _check_input(dataset)
        if num_burnin is None:
            num_burnin = self.num_burnin
        opts = {'model': self.__proxy__, 'data': dataset, 'num_burnin': num_burnin}
        response = _turicreate.extensions._text.topicmodel_predict(opts)
        preds = response['predictions']
        if output_type not in ['probability', 'probabilities', 'prob']:
            preds = preds.apply(lambda x: max(_izip(x, _xrange(len(x))))[1])
        return preds

    def evaluate(self, train_data, test_data=None, metric='perplexity'):
        if False:
            return 10
        "\n        Estimate the model's ability to predict new data. Imagine you have a\n        corpus of books. One common approach to evaluating topic models is to\n        train on the first half of all of the books and see how well the model\n        predicts the second half of each book.\n\n        This method returns a metric called perplexity, which  is related to the\n        likelihood of observing these words under the given model. See\n        :py:func:`~turicreate.topic_model.perplexity` for more details.\n\n        The provided `train_data` and `test_data` must have the same length,\n        i.e., both data sets must have the same number of documents; the model\n        will use train_data to estimate which topic the document belongs to, and\n        this is used to estimate the model's performance at predicting the\n        unseen words in the test data.\n\n        See :py:func:`~turicreate.topic_model.TopicModel.predict` for details\n        on how these predictions are made, and see\n        :py:func:`~turicreate.text_analytics.random_split` for a helper function\n        that can be used for making train/test splits.\n\n        Parameters\n        ----------\n        train_data : SArray or SFrame\n            A set of documents to predict topics for.\n\n        test_data : SArray or SFrame, optional\n            A set of documents to evaluate performance on.\n            By default this will set to be the same as train_data.\n\n        metric : str\n            The chosen metric to use for evaluating the topic model.\n            Currently only 'perplexity' is supported.\n\n        Returns\n        -------\n        out : dict\n            The set of estimated evaluation metrics.\n\n        See Also\n        --------\n        predict, turicreate.toolkits.text_analytics.random_split\n\n        Examples\n        --------\n        >>> docs = turicreate.SArray('https://static.turi.com/datasets/nips-text')\n        >>> train_data, test_data = turicreate.text_analytics.random_split(docs)\n        >>> m = turicreate.topic_model.create(train_data)\n        >>> m.evaluate(train_data, test_data)\n        {'perplexity': 2467.530370396021}\n\n        "
        train_data = _check_input(train_data)
        if test_data is None:
            test_data = train_data
        else:
            test_data = _check_input(test_data)
        predictions = self.predict(train_data, output_type='probability')
        topics = self.topics
        ret = {}
        ret['perplexity'] = perplexity(test_data, predictions, topics['topic_probabilities'], topics['vocabulary'])
        return ret

def perplexity(test_data, predictions, topics, vocabulary):
    if False:
        for i in range(10):
            print('nop')
    '\n    Compute the perplexity of a set of test documents given a set\n    of predicted topics.\n\n    Let theta be the matrix of document-topic probabilities, where\n    theta_ik = p(topic k | document i). Let Phi be the matrix of term-topic\n    probabilities, where phi_jk = p(word j | topic k).\n\n    Then for each word in each document, we compute for a given word w\n    and document d\n\n    .. math::\n        p(word | \theta[doc_id,:], \\phi[word_id,:]) =\n       \\sum_k \theta[doc_id, k] * \\phi[word_id, k]\n\n    We compute loglikelihood to be:\n\n    .. math::\n        l(D) = \\sum_{i \\in D} \\sum_{j in D_i} count_{i,j} * log Pr(word_{i,j} | \theta, \\phi)\n\n    and perplexity to be\n\n    .. math::\n        \\exp \\{ - l(D) / \\sum_i \\sum_j count_{i,j} \\}\n\n    Parameters\n    ----------\n    test_data : SArray of type dict or SFrame with a single column of type dict\n        Documents in bag-of-words format.\n\n    predictions : SArray\n        An SArray of vector type, where each vector contains estimates of the\n        probability that this document belongs to each of the topics.\n        This must have the same size as test_data; otherwise an exception\n        occurs. This can be the output of\n        :py:func:`~turicreate.topic_model.TopicModel.predict`, for example.\n\n    topics : SFrame\n        An SFrame containing two columns: \'vocabulary\' and \'topic_probabilities\'.\n        The value returned by m[\'topics\'] is a valid input for this argument,\n        where m is a trained :py:class:`~turicreate.topic_model.TopicModel`.\n\n    vocabulary : SArray\n        An SArray of words to use. All words in test_data that are not in this\n        vocabulary will be ignored.\n\n    Notes\n    -----\n    For more details, see equations 13-16 of [PattersonTeh2013].\n\n\n    References\n    ----------\n    .. [PERP] `Wikipedia - perplexity <http://en.wikipedia.org/wiki/Perplexity>`_\n\n    .. [PattersonTeh2013] Patterson, Teh. `"Stochastic Gradient Riemannian\n       Langevin Dynamics on the Probability Simplex"\n       <http://www.stats.ox.ac.uk/~teh/research/compstats/PatTeh2013a.pdf>`_\n       NIPS, 2013.\n\n    Examples\n    --------\n    >>> from turicreate import topic_model\n    >>> train_data, test_data = turicreate.text_analytics.random_split(docs)\n    >>> m = topic_model.create(train_data)\n    >>> pred = m.predict(train_data)\n    >>> topics = m[\'topics\']\n    >>> p = topic_model.perplexity(test_data, pred,\n                                   topics[\'topic_probabilities\'],\n                                   topics[\'vocabulary\'])\n    >>> p\n    1720.7  # lower values are better\n    '
    test_data = _check_input(test_data)
    assert isinstance(predictions, _SArray), 'Predictions must be an SArray of vector type.'
    assert predictions.dtype == _array.array, 'Predictions must be probabilities. Try using m.predict() with ' + "output_type='probability'."
    opts = {'test_data': test_data, 'predictions': predictions, 'topics': topics, 'vocabulary': vocabulary}
    response = _turicreate.extensions._text.topicmodel_get_perplexity(opts)
    return response['perplexity']