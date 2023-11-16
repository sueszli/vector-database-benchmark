"""Utils module containing functionalities to infer the metadata from the supplied TextData."""
import warnings
from typing import List, Tuple
import numpy as np
import pandas as pd
from seqeval.metrics.sequence_labeling import get_entities
from sklearn.base import BaseEstimator
from deepchecks.core.errors import DeepchecksValueError
from deepchecks.nlp.task_type import TaskType
__all__ = ['infer_observed_and_model_labels']

def infer_observed_and_model_labels(train_dataset=None, test_dataset=None, model: BaseEstimator=None, y_pred_train: np.ndarray=None, y_pred_test: np.ndarray=None, model_classes: list=None, task_type: TaskType=None) -> Tuple[List, List]:
    if False:
        while True:
            i = 10
    '\n    Infer the observed labels from the given datasets and predictions.\n\n    Parameters\n    ----------\n    train_dataset : Union[TextData, None], default None\n        TextData object, representing data an estimator was fitted on\n    test_dataset : Union[TextData, None], default None\n        TextData object, representing data an estimator predicts on\n    model : Union[BaseEstimator, None], default None\n        A fitted estimator instance\n    y_pred_train : np.array\n        Predictions on train_dataset\n    y_pred_test : np.array\n        Predictions on test_dataset\n    model_classes : Optional[List], default None\n        list of classes known to the model\n    task_type : Union[TaskType, None], default None\n        The task type of the model\n\n    Returns\n    -------\n        observed_classes : list\n            List of observed label values. For multi-label, returns number of observed labels.\n        model_classes : list\n            List of the user-given model classes. For multi-label, if not given by the user, returns a range of\n            len(label)\n    '
    train_labels = []
    test_labels = []
    have_model = model is not None
    if train_dataset:
        if train_dataset.has_label():
            train_labels += list(train_dataset.label)
        if have_model:
            train_labels += list(model.predict(train_dataset))
    if test_dataset:
        if test_dataset.has_label():
            test_labels += list(test_dataset.label)
        if have_model:
            test_labels += list(model.predict(test_dataset))
    if task_type == TaskType.TOKEN_CLASSIFICATION:
        train_labels = [token_label for sentence in train_labels for token_label in sentence]
        test_labels = [token_label for sentence in test_labels for token_label in sentence]
        if model_classes and 'O' in model_classes:
            model_classes = [c for c in model_classes if c != 'O']
            warnings.warn('"O" label was removed from model_classes as it is ignored by metrics for token classification', UserWarning)
    observed_classes = np.array(test_labels + train_labels, dtype=object)
    if len(observed_classes.shape) == 2:
        len_observed_label = observed_classes.shape[1]
        if not model_classes:
            model_classes = list(range(len_observed_label))
            observed_classes = list(range(len_observed_label))
        else:
            if len(model_classes) != len_observed_label:
                raise DeepchecksValueError(f'Received model_classes of length {len(model_classes)}, but data indicates labels of length {len_observed_label}')
            observed_classes = model_classes
    else:
        observed_classes = observed_classes[~pd.isnull(observed_classes)]
        observed_classes = sorted(np.unique(observed_classes))
    if task_type == TaskType.TOKEN_CLASSIFICATION:
        observed_classes = [c for c in observed_classes if c != 'O']
        observed_classes = sorted({tag for (tag, _, _) in get_entities(observed_classes)})
    return (observed_classes, model_classes)