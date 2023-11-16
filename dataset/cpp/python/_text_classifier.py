# -*- coding: utf-8 -*-
# Copyright © 2017 Apple Inc. All rights reserved.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause
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
    """
    Return an SFrame containing a bag of words representation of each column.
    """
    if isinstance(sf, dict):
        out = _tc.SArray([sf]).unpack("")
    elif isinstance(sf, _tc.SFrame):
        out = sf.__copy__()
    else:
        raise ValueError("Unrecognized input to feature extractor.")
    for f in _get_str_columns(out):
        if target != f:
            out[f] = _tc.text_analytics.count_words(out[f])
    return out


def create(
    dataset,
    target,
    features=None,
    drop_stop_words=True,
    word_count_threshold=2,
    method="auto",
    validation_set="auto",
    max_iterations=10,
    l2_penalty=0.2,
):
    """
    Create a model that trains a classifier to classify text from a
    collection of documents. The model is a
    :class:`~turicreate.logistic_classifier.LogisticClassifier` model trained
    using a bag-of-words representation of the text dataset.

    Parameters
    ----------
    dataset : SFrame
      Contains one or more columns of text data. This can be unstructured text
      dataset, such as that appearing in forums, user-generated reviews, etc.

    target : str
      The column name containing class labels for each document.

    features : list[str], optional
      The column names of interest containing text dataset. Each provided column
      must be str type. Defaults to using all columns of type str.

    drop_stop_words : bool, optional
        Ignore very common words, eg: "the", "a", "is".
        For the complete list of stop words, see: `text_classifier.drop_words()`.

    word_count_threshold : int, optional
        Words which occur less than this often, in the entire dataset, will be
        ignored.

    method: str, optional
      Method to use for feature engineering and modeling. Currently only
      bag-of-words and logistic classifier ('bow-logistic') is available.

    validation_set : SFrame, optional
      A dataset for monitoring the model's generalization performance.
      For each row of the progress table, the chosen metrics are computed
      for both the provided training dataset and the validation_set. The
      format of this SFrame must be the same as the training set.
      By default this argument is set to 'auto' and a validation set is
      automatically sampled and used for progress printing. If
      validation_set is set to None, then no additional metrics
      are computed. The default value is 'auto'.

    max_iterations : int, optional
      The maximum number of allowed passes through the data. More passes over
      the data can result in a more accurately trained model. Consider
      increasing this (the default value is 10) if the training accuracy is
      low and the *Grad-Norm* in the display is large.

    l2_penalty : float, optional
      Weight on l2 regularization of the model. The larger this weight, the
      more the model coefficients shrink toward 0. This introduces bias into
      the model but decreases variance, potentially leading to better
      predictions. The default value is 0.2; setting this parameter to 0
      corresponds to unregularized logistic regression. See the ridge
      regression reference for more detail.

    Returns
    -------
    out : :class:`~TextClassifier`

    Examples
    --------
    >>> import turicreate as tc
    >>> dataset = tc.SFrame({'rating': [1, 5], 'text': ['hate it', 'love it']})

    >>> m = tc.text_classifier.create(dataset, 'rating', features=['text'])
    >>> m.predict(dataset)

    You may also evaluate predictions against known text scores.

    >>> metrics = m.evaluate(dataset)

    See Also
    --------
    text_classifier.stop_words, text_classifier.drop_words


    """
    _raise_error_if_not_sframe(dataset, "dataset")

    # Validate method.
    if method == "auto":
        method = "bow-logistic"
    if method not in ["bow-logistic"]:
        raise ValueError("Unsupported method provided.")

    # Validate dataset
    if features is None:
        features = dataset.column_names()

    # Remove target column from list of feature columns.
    features = [f for f in features if f != target]

    # Process training set using the default feature extractor.
    feature_extractor = _BOW_FEATURE_EXTRACTOR
    train = feature_extractor(dataset, target)

    stop_words = None
    if drop_stop_words:
        stop_words = _text_analytics.stop_words()
    for cur_feature in features:
        train[cur_feature] = _text_analytics.drop_words(
            train[cur_feature], threshold=word_count_threshold, stop_words=stop_words
        )

    # Check for a validation set.
    if isinstance(validation_set, _tc.SFrame):
        validation_set = feature_extractor(validation_set, target)

    m = _tc.logistic_classifier.create(
        train,
        target=target,
        features=features,
        l2_penalty=l2_penalty,
        max_iterations=max_iterations,
        validation_set=validation_set,
    )
    num_examples = len(dataset)
    model = TextClassifier()
    model.__proxy__.update(
        {
            "target": target,
            "features": features,
            "method": method,
            "num_examples": num_examples,
            "num_features": len(features),
            "classifier": m,
        }
    )
    return model


class TextClassifier(_CustomModel):
    _PYTHON_TEXT_CLASSIFIER_MODEL_VERSION = 1

    def __init__(self, state=None):
        if state is None:
            state = {}

        self.__proxy__ = _PythonProxy(state)

    @classmethod
    def _native_name(cls):
        return "text_classifier"

    def _get_version(self):
        return self._PYTHON_TEXT_CLASSIFIER_MODEL_VERSION

    def _get_native_state(self):
        import copy

        retstate = copy.copy(self.__proxy__.state)
        retstate["classifier"] = retstate["classifier"].__proxy__
        return retstate

    @classmethod
    def _load_version(self, state, version):
        from turicreate.toolkits.classifier.logistic_classifier import (
            LogisticClassifier,
        )

        state["classifier"] = LogisticClassifier(state["classifier"])
        state = _PythonProxy(state)
        return TextClassifier(state)

    def predict(self, dataset, output_type="class"):
        """
        Return predictions for ``dataset``, using the trained model.

        Parameters
        ----------
        dataset : SFrame
            dataset of new observations. Must include columns with the same
            names as the features used for model training, but does not require
            a target column. Additional columns are ignored.

        output_type : {'class', 'probability_vector'}, optional
            Form of the predictions which are one of:

            - 'probability_vector': Prediction probability associated with each
              class as a vector. The probability of the first class (sorted
              alphanumerically by name of the class in the training set) is in
              position 0 of the vector, the second in position 1 and so on.
            - 'class': Class prediction. For multi-class classification, this
              returns the class with maximum probability.

        Returns
        -------
        out : SArray
            An SArray with model predictions.

        See Also
        ----------
        create, evaluate, classify


        Examples
        --------
        >>> import turicreate as tc
        >>> dataset = tc.SFrame({'rating': [1, 5], 'text': ['hate it', 'love it']})
        >>> m = tc.text_classifier.create(dataset, 'rating', features=['text'])
        >>> m.predict(dataset)

        """
        m = self.__proxy__["classifier"]
        target = self.__proxy__["target"]
        f = _BOW_FEATURE_EXTRACTOR
        return m.predict(f(dataset, target), output_type=output_type)

    def classify(self, dataset):
        """
        Return a classification, for each example in the ``dataset``, using the
        trained model. The output SFrame contains predictions as both class
        labels as well as probabilities that the predicted value is the
        associated label.

        Parameters
        ----------
        dataset : SFrame
            dataset of new observations. Must include columns with the same
            names as the features used for model training, but does not require
            a target column. Additional columns are ignored.

        Returns
        -------
        out : SFrame
            An SFrame with model predictions i.e class labels and probabilities.

        See Also
        ----------
        create, evaluate, predict

        Examples
        --------
        >>> import turicreate as tc
        >>> dataset = tc.SFrame({'rating': [1, 5], 'text': ['hate it', 'love it']})
        >>> m = tc.text_classifier.create(dataset, 'rating', features=['text'])
        >>> output = m.classify(dataset)

        """
        m = self.__proxy__["classifier"]
        target = self.__proxy__["target"]
        f = _BOW_FEATURE_EXTRACTOR
        return m.classify(f(dataset, target))

    def __str__(self):
        """
        Return a string description of the model to the ``print`` method.

        Returns
        -------
        out : string
            A description of the NearestNeighborsModel.
        """
        return self.__repr__()

    def _get_summary_struct(self):

        dataset_fields = [("Number of examples", "num_examples")]
        model_fields = [
            ("Target column", "target"),
            ("Features", "features"),
            ("Method", "method"),
        ]
        sections = [dataset_fields, model_fields]
        section_titles = ["dataset", "Model"]
        return sections, section_titles

    def __repr__(self):
        width = 32
        (sections, section_titles) = self._get_summary_struct()
        out = _toolkit_repr_print(self, sections, section_titles, width=width)
        return out

    def evaluate(self, dataset, metric="auto", **kwargs):
        """
        Evaluate the model by making predictions of target values and comparing
        these to actual values.

        Parameters
        ----------
        dataset : SFrame
            An SFrame having the same feature columns as provided when creating
            the model.

        metric : str, optional
            Name of the evaluation metric.  Possible values are:

            - 'auto'             : Returns all available metrics.
            - 'accuracy'         : Classification accuracy (micro average).
            - 'auc'              : Area under the ROC curve (macro average)
            - 'precision'        : Precision score (macro average)
            - 'recall'           : Recall score (macro average)
            - 'f1_score'         : F1 score (macro average)
            - 'log_loss'         : Log loss
            - 'confusion_matrix' : An SFrame with counts of possible prediction/true label combinations.
            - 'roc_curve'        : An SFrame containing information needed for an ROC curve

            For more flexibility in calculating evaluation metrics, use the
            :class:`~turicreate.evaluation` module.

        Returns
        -------
        out : dict
            Dictionary of evaluation results where the key is the name of the
            evaluation metric (e.g. `accuracy`) and the value is the evaluation
            score.

        See Also
        ----------
        create, predict, classify

        """
        m = self.__proxy__["classifier"]
        target = self.__proxy__["target"]
        f = _BOW_FEATURE_EXTRACTOR
        test = f(dataset, target)
        return m.evaluate(test, metric, **kwargs)

    def summary(self):
        """
        Get a summary for the underlying classifier.
        """
        return self.__proxy__["classifier"].summary()

    def export_coreml(self, filename):
        """
        Export the model in Core ML format.

        Parameters
        ----------
        filename: str
          A valid filename where the model can be saved.

        Examples
        --------
        >>> model.export_coreml("MyTextMessageClassifier.mlmodel")
        >>>
        >>> from coremltools.models import MLModel
        >>> coreml_model = MLModel("MyTextMessageClassifier.mlmodel")
        >>>
        >>> test_input = tc.SArray(["Hi! How are you?"])
        >>> bag_of_words = tc.text_analytics.count_words(test_input)
        >>>
        >>> # "text" is the input column name
        >>> coreml_model.predict({"text": bag_of_words[0]})
        """
        from turicreate.extensions import _logistic_classifier_export_as_model_asset
        from turicreate.toolkits import _coreml_utils

        display_name = "text classifier"
        short_description = _coreml_utils._mlmodel_short_description(display_name)
        context = {
            "class": self.__class__.__name__,
            "short_description": short_description,
        }
        context["user_defined"] = _coreml_utils._get_model_metadata(
            self.__class__.__name__, None
        )

        model = self.__proxy__["classifier"].__proxy__
        _logistic_classifier_export_as_model_asset(model, filename, context)


def _get_str_columns(sf):
    """
    Returns a list of names of columns that are string type.
    """
    return [name for name in sf.column_names() if sf[name].dtype == str]
