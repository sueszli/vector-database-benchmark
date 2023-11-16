from __future__ import print_function as _
from __future__ import division as _
from __future__ import absolute_import as _
import turicreate as _tc
from turicreate.toolkits._internal_utils import _raise_error_if_not_sframe
from turicreate.toolkits._model import CustomModel as _CustomModel
from turicreate.toolkits._model import PythonProxy as _PythonProxy
from turicreate.toolkits._internal_utils import _toolkit_repr_print
from turicreate.toolkits import text_analytics as _text_analytics

def _BOW_FEATURE_EXTRACTOR(sf, target=None):
    if False:
        i = 10
        return i + 15
    '\n    Return an SFrame containing a bag of words representation of each column.\n    '
    if isinstance(sf, dict):
        out = _tc.SArray([sf]).unpack('')
    elif isinstance(sf, _tc.SFrame):
        out = sf.__copy__()
    else:
        raise ValueError('Unrecognized input to feature extractor.')
    for f in _get_str_columns(out):
        if target != f:
            out[f] = _tc.text_analytics.count_words(out[f])
    return out

def create(dataset, target, features=None, drop_stop_words=True, word_count_threshold=2, method='auto', validation_set='auto', max_iterations=10, l2_penalty=0.2):
    if False:
        for i in range(10):
            print('nop')
    '\n    Create a model that trains a classifier to classify text from a\n    collection of documents. The model is a\n    :class:`~turicreate.logistic_classifier.LogisticClassifier` model trained\n    using a bag-of-words representation of the text dataset.\n\n    Parameters\n    ----------\n    dataset : SFrame\n      Contains one or more columns of text data. This can be unstructured text\n      dataset, such as that appearing in forums, user-generated reviews, etc.\n\n    target : str\n      The column name containing class labels for each document.\n\n    features : list[str], optional\n      The column names of interest containing text dataset. Each provided column\n      must be str type. Defaults to using all columns of type str.\n\n    drop_stop_words : bool, optional\n        Ignore very common words, eg: "the", "a", "is".\n        For the complete list of stop words, see: `text_classifier.drop_words()`.\n\n    word_count_threshold : int, optional\n        Words which occur less than this often, in the entire dataset, will be\n        ignored.\n\n    method: str, optional\n      Method to use for feature engineering and modeling. Currently only\n      bag-of-words and logistic classifier (\'bow-logistic\') is available.\n\n    validation_set : SFrame, optional\n      A dataset for monitoring the model\'s generalization performance.\n      For each row of the progress table, the chosen metrics are computed\n      for both the provided training dataset and the validation_set. The\n      format of this SFrame must be the same as the training set.\n      By default this argument is set to \'auto\' and a validation set is\n      automatically sampled and used for progress printing. If\n      validation_set is set to None, then no additional metrics\n      are computed. The default value is \'auto\'.\n\n    max_iterations : int, optional\n      The maximum number of allowed passes through the data. More passes over\n      the data can result in a more accurately trained model. Consider\n      increasing this (the default value is 10) if the training accuracy is\n      low and the *Grad-Norm* in the display is large.\n\n    l2_penalty : float, optional\n      Weight on l2 regularization of the model. The larger this weight, the\n      more the model coefficients shrink toward 0. This introduces bias into\n      the model but decreases variance, potentially leading to better\n      predictions. The default value is 0.2; setting this parameter to 0\n      corresponds to unregularized logistic regression. See the ridge\n      regression reference for more detail.\n\n    Returns\n    -------\n    out : :class:`~TextClassifier`\n\n    Examples\n    --------\n    >>> import turicreate as tc\n    >>> dataset = tc.SFrame({\'rating\': [1, 5], \'text\': [\'hate it\', \'love it\']})\n\n    >>> m = tc.text_classifier.create(dataset, \'rating\', features=[\'text\'])\n    >>> m.predict(dataset)\n\n    You may also evaluate predictions against known text scores.\n\n    >>> metrics = m.evaluate(dataset)\n\n    See Also\n    --------\n    text_classifier.stop_words, text_classifier.drop_words\n\n\n    '
    _raise_error_if_not_sframe(dataset, 'dataset')
    if method == 'auto':
        method = 'bow-logistic'
    if method not in ['bow-logistic']:
        raise ValueError('Unsupported method provided.')
    if features is None:
        features = dataset.column_names()
    features = [f for f in features if f != target]
    feature_extractor = _BOW_FEATURE_EXTRACTOR
    train = feature_extractor(dataset, target)
    stop_words = None
    if drop_stop_words:
        stop_words = _text_analytics.stop_words()
    for cur_feature in features:
        train[cur_feature] = _text_analytics.drop_words(train[cur_feature], threshold=word_count_threshold, stop_words=stop_words)
    if isinstance(validation_set, _tc.SFrame):
        validation_set = feature_extractor(validation_set, target)
    m = _tc.logistic_classifier.create(train, target=target, features=features, l2_penalty=l2_penalty, max_iterations=max_iterations, validation_set=validation_set)
    num_examples = len(dataset)
    model = TextClassifier()
    model.__proxy__.update({'target': target, 'features': features, 'method': method, 'num_examples': num_examples, 'num_features': len(features), 'classifier': m})
    return model

class TextClassifier(_CustomModel):
    _PYTHON_TEXT_CLASSIFIER_MODEL_VERSION = 1

    def __init__(self, state=None):
        if False:
            print('Hello World!')
        if state is None:
            state = {}
        self.__proxy__ = _PythonProxy(state)

    @classmethod
    def _native_name(cls):
        if False:
            i = 10
            return i + 15
        return 'text_classifier'

    def _get_version(self):
        if False:
            return 10
        return self._PYTHON_TEXT_CLASSIFIER_MODEL_VERSION

    def _get_native_state(self):
        if False:
            print('Hello World!')
        import copy
        retstate = copy.copy(self.__proxy__.state)
        retstate['classifier'] = retstate['classifier'].__proxy__
        return retstate

    @classmethod
    def _load_version(self, state, version):
        if False:
            for i in range(10):
                print('nop')
        from turicreate.toolkits.classifier.logistic_classifier import LogisticClassifier
        state['classifier'] = LogisticClassifier(state['classifier'])
        state = _PythonProxy(state)
        return TextClassifier(state)

    def predict(self, dataset, output_type='class'):
        if False:
            while True:
                i = 10
        "\n        Return predictions for ``dataset``, using the trained model.\n\n        Parameters\n        ----------\n        dataset : SFrame\n            dataset of new observations. Must include columns with the same\n            names as the features used for model training, but does not require\n            a target column. Additional columns are ignored.\n\n        output_type : {'class', 'probability_vector'}, optional\n            Form of the predictions which are one of:\n\n            - 'probability_vector': Prediction probability associated with each\n              class as a vector. The probability of the first class (sorted\n              alphanumerically by name of the class in the training set) is in\n              position 0 of the vector, the second in position 1 and so on.\n            - 'class': Class prediction. For multi-class classification, this\n              returns the class with maximum probability.\n\n        Returns\n        -------\n        out : SArray\n            An SArray with model predictions.\n\n        See Also\n        ----------\n        create, evaluate, classify\n\n\n        Examples\n        --------\n        >>> import turicreate as tc\n        >>> dataset = tc.SFrame({'rating': [1, 5], 'text': ['hate it', 'love it']})\n        >>> m = tc.text_classifier.create(dataset, 'rating', features=['text'])\n        >>> m.predict(dataset)\n\n        "
        m = self.__proxy__['classifier']
        target = self.__proxy__['target']
        f = _BOW_FEATURE_EXTRACTOR
        return m.predict(f(dataset, target), output_type=output_type)

    def classify(self, dataset):
        if False:
            while True:
                i = 10
        "\n        Return a classification, for each example in the ``dataset``, using the\n        trained model. The output SFrame contains predictions as both class\n        labels as well as probabilities that the predicted value is the\n        associated label.\n\n        Parameters\n        ----------\n        dataset : SFrame\n            dataset of new observations. Must include columns with the same\n            names as the features used for model training, but does not require\n            a target column. Additional columns are ignored.\n\n        Returns\n        -------\n        out : SFrame\n            An SFrame with model predictions i.e class labels and probabilities.\n\n        See Also\n        ----------\n        create, evaluate, predict\n\n        Examples\n        --------\n        >>> import turicreate as tc\n        >>> dataset = tc.SFrame({'rating': [1, 5], 'text': ['hate it', 'love it']})\n        >>> m = tc.text_classifier.create(dataset, 'rating', features=['text'])\n        >>> output = m.classify(dataset)\n\n        "
        m = self.__proxy__['classifier']
        target = self.__proxy__['target']
        f = _BOW_FEATURE_EXTRACTOR
        return m.classify(f(dataset, target))

    def __str__(self):
        if False:
            while True:
                i = 10
        '\n        Return a string description of the model to the ``print`` method.\n\n        Returns\n        -------\n        out : string\n            A description of the NearestNeighborsModel.\n        '
        return self.__repr__()

    def _get_summary_struct(self):
        if False:
            i = 10
            return i + 15
        dataset_fields = [('Number of examples', 'num_examples')]
        model_fields = [('Target column', 'target'), ('Features', 'features'), ('Method', 'method')]
        sections = [dataset_fields, model_fields]
        section_titles = ['dataset', 'Model']
        return (sections, section_titles)

    def __repr__(self):
        if False:
            print('Hello World!')
        width = 32
        (sections, section_titles) = self._get_summary_struct()
        out = _toolkit_repr_print(self, sections, section_titles, width=width)
        return out

    def evaluate(self, dataset, metric='auto', **kwargs):
        if False:
            return 10
        "\n        Evaluate the model by making predictions of target values and comparing\n        these to actual values.\n\n        Parameters\n        ----------\n        dataset : SFrame\n            An SFrame having the same feature columns as provided when creating\n            the model.\n\n        metric : str, optional\n            Name of the evaluation metric.  Possible values are:\n\n            - 'auto'             : Returns all available metrics.\n            - 'accuracy'         : Classification accuracy (micro average).\n            - 'auc'              : Area under the ROC curve (macro average)\n            - 'precision'        : Precision score (macro average)\n            - 'recall'           : Recall score (macro average)\n            - 'f1_score'         : F1 score (macro average)\n            - 'log_loss'         : Log loss\n            - 'confusion_matrix' : An SFrame with counts of possible prediction/true label combinations.\n            - 'roc_curve'        : An SFrame containing information needed for an ROC curve\n\n            For more flexibility in calculating evaluation metrics, use the\n            :class:`~turicreate.evaluation` module.\n\n        Returns\n        -------\n        out : dict\n            Dictionary of evaluation results where the key is the name of the\n            evaluation metric (e.g. `accuracy`) and the value is the evaluation\n            score.\n\n        See Also\n        ----------\n        create, predict, classify\n\n        "
        m = self.__proxy__['classifier']
        target = self.__proxy__['target']
        f = _BOW_FEATURE_EXTRACTOR
        test = f(dataset, target)
        return m.evaluate(test, metric, **kwargs)

    def summary(self):
        if False:
            while True:
                i = 10
        '\n        Get a summary for the underlying classifier.\n        '
        return self.__proxy__['classifier'].summary()

    def export_coreml(self, filename):
        if False:
            print('Hello World!')
        '\n        Export the model in Core ML format.\n\n        Parameters\n        ----------\n        filename: str\n          A valid filename where the model can be saved.\n\n        Examples\n        --------\n        >>> model.export_coreml("MyTextMessageClassifier.mlmodel")\n        >>>\n        >>> from coremltools.models import MLModel\n        >>> coreml_model = MLModel("MyTextMessageClassifier.mlmodel")\n        >>>\n        >>> test_input = tc.SArray(["Hi! How are you?"])\n        >>> bag_of_words = tc.text_analytics.count_words(test_input)\n        >>>\n        >>> # "text" is the input column name\n        >>> coreml_model.predict({"text": bag_of_words[0]})\n        '
        from turicreate.extensions import _logistic_classifier_export_as_model_asset
        from turicreate.toolkits import _coreml_utils
        display_name = 'text classifier'
        short_description = _coreml_utils._mlmodel_short_description(display_name)
        context = {'class': self.__class__.__name__, 'short_description': short_description}
        context['user_defined'] = _coreml_utils._get_model_metadata(self.__class__.__name__, None)
        model = self.__proxy__['classifier'].__proxy__
        _logistic_classifier_export_as_model_asset(model, filename, context)

def _get_str_columns(sf):
    if False:
        print('Hello World!')
    '\n    Returns a list of names of columns that are string type.\n    '
    return [name for name in sf.column_names() if sf[name].dtype == str]