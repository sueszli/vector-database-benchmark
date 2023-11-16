"""
Helper functions used internally for object detection tasks.
"""
from typing import Any, Dict, List, Optional
import numpy as np
from cleanlab.internal.numerics import softmax

def bbox_xyxy_to_xywh(bbox: List[float]) -> Optional[List[float]]:
    if False:
        for i in range(10):
            print('nop')
    'Converts bounding box coodrinate types from x1y1,x2y2 to x,y,w,h'
    if len(bbox) == 4:
        (x1, y1, x2, y2) = bbox
        w = x2 - x1
        h = y2 - y1
        return [x1, y1, w, h]
    else:
        print('Wrong bbox shape', len(bbox))
        return None

def softmin1d(scores: np.ndarray, temperature: float=0.99, axis: int=0) -> float:
    if False:
        return 10
    'Returns softmin of passed in scores.'
    scores = np.array(scores)
    softmax_scores = softmax(x=-1 * scores, temperature=temperature, axis=axis, shift=True)
    return np.dot(softmax_scores, scores)

def assert_valid_aggregation_weights(aggregation_weights: Dict[str, Any]) -> None:
    if False:
        i = 10
        return i + 15
    'assert aggregation weights are in the proper format'
    weights = np.array(list(aggregation_weights.values()))
    if not np.isclose(np.sum(weights), 1.0) or np.min(weights) < 0.0:
        raise ValueError(f'Aggregation weights should be non-negative and must sum to 1.0\n                ')

def assert_valid_inputs(labels: List[Dict[str, Any]], predictions, method: Optional[str]=None, threshold: Optional[float]=None):
    if False:
        print('Hello World!')
    'Asserts proper input format.'
    if len(labels) != len(predictions):
        raise ValueError(f'labels and predictions length needs to match. len(labels) == {len(labels)} while len(predictions) == {len(predictions)}.')
    if not isinstance(labels[0], dict):
        raise ValueError(f'Labels has to be a list of dicts. Instead it is list of {type(labels[0])}.')
    if not isinstance(predictions[0], (list, np.ndarray)):
        raise ValueError(f'Prediction has to be a list or np.ndarray. Instead it is type {type(predictions[0])}.')
    if not predictions[0][0].shape[1] == 5:
        raise ValueError(f'Prediction values have to be of format [x1,y1,x2,y2,pred_prob]. Please refer to the documentation for predicted probabilities under object_detection.rank.get_label_quality_scores for details')
    valid_methods = ['objectlab']
    if method is not None and method not in valid_methods:
        raise ValueError(f'\n            {method} is not a valid object detection scoring method!\n            Please choose a valid scoring_method: {valid_methods}\n            ')
    if threshold is not None and threshold > 1.0:
        raise ValueError(f'\n            Threshold is a cutoff of predicted probabilities and therefore should be <= 1.\n            ')