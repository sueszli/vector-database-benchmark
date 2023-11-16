"""The following is set of default rllib reduction methods for ResultDicts"""
from typing import List
import numpy as np
import tree
from ray.rllib.utils.typing import ResultDict

def _reduce_mean_results(results: List[ResultDict]) -> ResultDict:
    if False:
        return 10
    'Takes the average of all the leaves in the result dict\n\n    Args:\n        results: list of result dicts to average\n\n    Returns:\n        Averaged result dict\n    '
    return tree.map_structure(lambda *x: np.mean(x), *results)