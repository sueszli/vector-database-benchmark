from __future__ import annotations
from typing import TYPE_CHECKING, Any, ClassVar, Dict, List, Optional, Union
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from cleanlab.datalab.internal.issue_manager import IssueManager
if TYPE_CHECKING:
    import pandas as pd
    from cleanlab.datalab.datalab import Datalab

class DataValuationIssueManager(IssueManager):
    """Manages data sample with low valuation."""
    description: ClassVar[str] = "\n    Examples that contribute minimally to a model's training\n    receive lower valuation scores.\n    "
    issue_name: ClassVar[str] = 'data_valuation'
    issue_score_key: ClassVar[str]
    verbosity_levels: ClassVar[Dict[int, List[str]]] = {0: [], 1: [], 2: [], 3: ['average_data_valuation']}
    DEFAULT_THRESHOLDS = 1e-06

    def __init__(self, datalab: Datalab, threshold: Optional[float]=None, k: int=10, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__(datalab)
        self.k = k
        self.threshold = threshold if threshold is not None else self.DEFAULT_THRESHOLDS

    def find_issues(self, **kwargs) -> None:
        if False:
            return 10
        "Calculate the data valuation score with a provided or existing knn graph.\n        Based on KNN-Shapley value described in https://arxiv.org/abs/1911.07128\n        The larger the score, the more valuable the data point is, the more contribution it will make to the model's training.\n        "
        knn_graph = self._process_knn_graph_from_inputs(kwargs)
        labels = self.datalab.labels.reshape(-1, 1)
        assert knn_graph is not None, 'knn_graph must be already calculated by other issue managers'
        assert labels is not None, 'labels must be provided'
        scores = _knn_shapley_score(knn_graph, labels)
        self.issues = pd.DataFrame({f'is_{self.issue_name}_issue': scores < self.threshold, self.issue_score_key: scores})
        self.summary = self.make_summary(score=scores.mean())
        self.info = self.collect_info(self.issues)

    def _process_knn_graph_from_inputs(self, kwargs: Dict[str, Any]) -> Union[csr_matrix, None]:
        if False:
            while True:
                i = 10
        'Determine if a knn_graph is provided in the kwargs or if one is already stored in the associated Datalab instance.'
        knn_graph_kwargs: Optional[csr_matrix] = kwargs.get('knn_graph', None)
        knn_graph_stats = self.datalab.get_info('statistics').get('weighted_knn_graph', None)
        knn_graph: Optional[csr_matrix] = None
        if knn_graph_kwargs is not None:
            knn_graph = knn_graph_kwargs
        elif knn_graph_stats is not None:
            knn_graph = knn_graph_stats
        if isinstance(knn_graph, csr_matrix) and kwargs.get('k', 0) > knn_graph.nnz // knn_graph.shape[0]:
            knn_graph = None
        return knn_graph

    def collect_info(self, issues: pd.DataFrame) -> dict:
        if False:
            print('Hello World!')
        issues_info = {'num_low_valuation_issues': sum(issues[f'is_{self.issue_name}_issue']), 'average_data_valuation': issues[self.issue_score_key].mean()}
        info_dict = {**issues_info}
        return info_dict

def _knn_shapley_score(knn_graph: csr_matrix, labels: np.ndarray) -> np.ndarray:
    if False:
        print('Hello World!')
    'Compute the Shapley values of data points based on a knn graph.'
    N = labels.shape[0]
    scores = np.zeros((N, N))
    dist = knn_graph.indices.reshape(N, -1)
    k = dist.shape[1]
    for (i, y) in enumerate(labels):
        idx = dist[i][::-1]
        ans = labels[idx]
        scores[idx[k - 1]][i] = float(ans[k - 1] == y) / k
        cur = k - 2
        for j in range(k - 1):
            scores[idx[cur]][i] = scores[idx[cur + 1]][i] + float(int(ans[cur] == y) - int(ans[cur + 1] == y)) / k * (min(cur, k - 1) + 1) / (cur + 1)
            cur -= 1
    return 0.5 * (np.mean(scores, axis=1) + 1)