"""Common metrics to calculate performance on single samples."""
from typing import List, Optional
import numpy as np
import pandas as pd
from deepchecks.core.errors import DeepchecksValueError

def calculate_neg_mse_per_sample(labels, predictions, index=None) -> pd.Series:
    if False:
        i = 10
        return i + 15
    'Calculate negative mean squared error per sample.'
    if index is None and isinstance(labels, pd.Series):
        index = labels.index
    return pd.Series([-(y - y_pred) ** 2 for (y, y_pred) in zip(labels, predictions)], index=index)

def calculate_neg_cross_entropy_per_sample(labels, probas: np.ndarray, model_classes: Optional[List]=None, index=None, is_multilabel: bool=False, eps=1e-15) -> pd.Series:
    if False:
        for i in range(10):
            print('nop')
    'Calculate negative cross entropy per sample.'
    if not is_multilabel:
        if index is None and isinstance(labels, pd.Series):
            index = labels.index
        if model_classes is not None:
            if any((x not in model_classes for x in labels)):
                raise DeepchecksValueError(f'Label observed values {sorted(np.unique(labels))} contain values that are not found in the model classes: {model_classes}.')
            if probas.shape[1] != len(model_classes):
                raise DeepchecksValueError(f'Predicted probabilities shape {probas.shape} does not match the number of classes found in the labels: {model_classes}.')
            labels = pd.Series(labels).apply(list(model_classes).index)
        (num_samples, num_classes) = probas.shape
        one_hot_labels = np.zeros((num_samples, num_classes))
        one_hot_labels[list(np.arange(num_samples)), list(labels)] = 1
    else:
        one_hot_labels = labels
    return pd.Series(np.sum(one_hot_labels * np.log(probas + eps), axis=1), index=index)