from typing import Union
import pandas as pd
from numpy import log2
from scipy.stats import entropy

def column_imbalance_score(value_counts: pd.Series, n_classes: int) -> Union[float, int]:
    if False:
        return 10
    "column_imbalance_score\n\n    The class balance score for categorical and boolean variables uses entropy to calculate a  bounded score between 0 and 1.\n    A perfectly uniform distribution would return a score of 0, and a perfectly imbalanced distribution would return a score of 1.\n\n    When dealing with probabilities with finite values (e.g categorical), entropy is maximised the ‘flatter’ the distribution is. (Jaynes: Probability Theory, The Logic of Science)\n    To calculate the class imbalance, we calculate the entropy of that distribution and the maximum possible entropy for that number of classes.\n    To calculate the entropy of the 'distribution' we use value counts (e.g frequency of classes) and we can determine the maximum entropy as log2(number of classes).\n    We then divide the entropy by the maximum possible entropy to get a value between 0 and 1 which we then subtract from 1.\n\n    Args:\n        value_counts (pd.Series): frequency of each category\n        n_classes (int): number of classes\n\n    Returns:\n        Union[float, int]: float or integer bounded between 0 and 1 inclusively\n    "
    if n_classes > 1:
        value_counts = value_counts.to_numpy(dtype=float)
        return 1 - entropy(value_counts, base=2) / log2(n_classes)
    return 0