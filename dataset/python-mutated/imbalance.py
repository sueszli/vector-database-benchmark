from __future__ import annotations
from typing import TYPE_CHECKING, ClassVar
import numpy as np
import pandas as pd
from cleanlab.datalab.internal.issue_manager import IssueManager
if TYPE_CHECKING:
    from cleanlab.datalab.datalab import Datalab

class ClassImbalanceIssueManager(IssueManager):
    """Manages issues related to imbalance class examples.

    Parameters
    ----------
    datalab:
        The Datalab instance that this issue manager searches for issues in.

    threshold:
        Minimum fraction of samples of each class that are present in a dataset without class imbalance.

    """
    description: ClassVar[str] = 'Examples belonging to the most under-represented class in the dataset.'
    issue_name: ClassVar[str] = 'class_imbalance'
    verbosity_levels = {0: [], 1: [], 2: []}

    def __init__(self, datalab: Datalab, threshold: float=0.1):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(datalab)
        self.threshold = threshold

    def find_issues(self, **kwargs) -> None:
        if False:
            return 10
        labels = self.datalab.labels
        K = len(self.datalab.class_names)
        class_probs = np.bincount(labels) / len(labels)
        rarest_class_idx = int(np.argmin(class_probs))
        imbalance_exists = class_probs[rarest_class_idx] < self.threshold * (1 / K)
        rarest_class = rarest_class_idx if imbalance_exists else -1
        is_issue_column = labels == rarest_class
        scores = np.where(is_issue_column, class_probs[rarest_class], 1)
        self.issues = pd.DataFrame({f'is_{self.issue_name}_issue': is_issue_column, self.issue_score_key: scores})
        self.summary = self.make_summary(score=class_probs[rarest_class_idx])
        self.info = self.collect_info()

    def collect_info(self) -> dict:
        if False:
            i = 10
            return i + 15
        params_dict = {'threshold': self.threshold}
        info_dict = {**params_dict}
        return info_dict