import logging
import pickle
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from snorkel.analysis import Scorer
from snorkel.utils import probs_to_preds

class BaseLabeler(ABC):
    """Abstract baseline label voter class."""

    def __init__(self, cardinality: int=2, **kwargs: Any) -> None:
        if False:
            while True:
                i = 10
        self.cardinality = cardinality

    @abstractmethod
    def predict_proba(self, L: np.ndarray) -> np.ndarray:
        if False:
            while True:
                i = 10
        'Abstract method for predicting probabilistic labels given a label matrix.\n\n        Parameters\n        ----------\n        L\n            An [n,m] matrix with values in {-1,0,1,...,k-1}f\n\n        Returns\n        -------\n        np.ndarray\n            An [n,k] array of probabilistic labels\n        '
        pass

    def predict(self, L: np.ndarray, return_probs: Optional[bool]=False, tie_break_policy: str='abstain') -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        if False:
            for i in range(10):
                print('nop')
        'Return predicted labels, with ties broken according to policy.\n\n        Policies to break ties include:\n        "abstain": return an abstain vote (-1)\n        "true-random": randomly choose among the tied options\n        "random": randomly choose among tied option using deterministic hash\n\n        NOTE: if tie_break_policy="true-random", repeated runs may have slightly different\n        results due to difference in broken ties\n\n\n        Parameters\n        ----------\n        L\n            An [n,m] matrix with values in {-1,0,1,...,k-1}\n        return_probs\n            Whether to return probs along with preds\n        tie_break_policy\n            Policy to break ties when converting probabilistic labels to predictions\n\n        Returns\n        -------\n        np.ndarray\n            An [n,1] array of integer labels\n\n        (np.ndarray, np.ndarray)\n            An [n,1] array of integer labels and an [n,k] array of probabilistic labels\n        '
        Y_probs = self.predict_proba(L)
        Y_p = probs_to_preds(Y_probs, tie_break_policy)
        if return_probs:
            return (Y_p, Y_probs)
        return Y_p

    def score(self, L: np.ndarray, Y: np.ndarray, metrics: Optional[List[str]]=['accuracy'], tie_break_policy: str='abstain') -> Dict[str, float]:
        if False:
            print('Hello World!')
        'Calculate one or more scores from user-specified and/or user-defined metrics.\n\n        Parameters\n        ----------\n        L\n            An [n,m] matrix with values in {-1,0,1,...,k-1}\n        Y\n            Gold labels associated with data points in L\n        metrics\n            A list of metric names\n        tie_break_policy\n            Policy to break ties when converting probabilistic labels to predictions\n\n\n        Returns\n        -------\n        Dict[str, float]\n            A dictionary mapping metric names to metric scores\n        '
        if tie_break_policy == 'abstain':
            logging.warning('Metrics calculated over data points with non-abstain labels only')
        (Y_pred, Y_prob) = self.predict(L, return_probs=True, tie_break_policy=tie_break_policy)
        scorer = Scorer(metrics=metrics)
        results = scorer.score(Y, Y_pred, Y_prob)
        return results

    def save(self, destination: str) -> None:
        if False:
            i = 10
            return i + 15
        "Save label model.\n\n        Parameters\n        ----------\n        destination\n            Filename for saving model\n\n        Example\n        -------\n        >>> label_model.save('./saved_label_model.pkl')  # doctest: +SKIP\n        "
        f = open(destination, 'wb')
        pickle.dump(self.__dict__, f)
        f.close()

    def load(self, source: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        "Load existing label model.\n\n        Parameters\n        ----------\n        source\n            Filename to load model from\n\n        Example\n        -------\n        Load parameters saved in ``saved_label_model``\n\n        >>> label_model.load('./saved_label_model.pkl')  # doctest: +SKIP\n        "
        f = open(source, 'rb')
        tmp_dict = pickle.load(f)
        f.close()
        self.__dict__.update(tmp_dict)