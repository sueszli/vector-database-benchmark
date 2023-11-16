"""
Module for the :py:class:`DataIssues` class, which serves as a central repository for storing
information and statistics about issues found in a dataset.

It collects information from various
:py:class:`IssueManager <cleanlab.datalab.internal.issue_manager.issue_manager.IssueManager>`
instances and keeps track of each issue, a summary for each type of issue,
related information and statistics about the issues.

The collected information can be accessed using the
:py:meth:`get_info <cleanlab.datalab.internal.data_issues.DataIssues.get_info>` method.
We recommend using that method instead of this module, which is just intended for internal use.
"""
from __future__ import annotations
import warnings
from typing import TYPE_CHECKING, Any, Dict, Optional, Union
import numpy as np
import pandas as pd
if TYPE_CHECKING:
    from cleanlab.datalab.internal.data import Data
    from cleanlab.datalab.internal.issue_manager import IssueManager
    from cleanvision import Imagelab

class DataIssues:
    """
    Class that collects and stores information and statistics on issues found in a dataset.

    Parameters
    ----------
    data :
        The data object for which the issues are being collected.

    Parameters
    ----------
    issues : pd.DataFrame
        Stores information about each individual issue found in the data,
        on a per-example basis.
    issue_summary : pd.DataFrame
        Summarizes the overall statistics for each issue type.
    info : dict
        A dictionary that contains information and statistics about the data and each issue type.
    """

    def __init__(self, data: Data) -> None:
        if False:
            return 10
        self.issues: pd.DataFrame = pd.DataFrame(index=range(len(data)))
        self.issue_summary: pd.DataFrame = pd.DataFrame(columns=['issue_type', 'score', 'num_issues']).astype({'score': np.float64, 'num_issues': np.int64})
        self.info: Dict[str, Dict[str, Any]] = {'statistics': get_data_statistics(data)}
        self._label_map = data.labels.label_map

    @property
    def statistics(self) -> Dict[str, Any]:
        if False:
            return 10
        'Returns the statistics dictionary.\n\n        Shorthand for self.info["statistics"].\n        '
        return self.info['statistics']

    def get_issues(self, issue_name: Optional[str]=None) -> pd.DataFrame:
        if False:
            i = 10
            return i + 15
        '\n        Use this after finding issues to see which examples suffer from which types of issues.\n\n        Parameters\n        ----------\n        issue_name : str or None\n            The type of issue to focus on. If `None`, returns full DataFrame summarizing all of the types of issues detected in each example from the dataset.\n\n        Raises\n        ------\n        ValueError\n            If `issue_name` is not a type of issue previously considered in the audit.\n\n        Returns\n        -------\n        specific_issues :\n            A DataFrame where each row corresponds to an example from the dataset and columns specify:\n            whether this example exhibits a particular type of issue and how severely (via a numeric quality score where lower values indicate more severe instances of the issue).\n\n            Additional columns may be present in the DataFrame depending on the type of issue specified.\n        '
        if issue_name is None:
            return self.issues
        columns = [col for col in self.issues.columns if issue_name in col]
        if not columns:
            raise ValueError(f"No columns found for issue type '{issue_name}'.")
        specific_issues = self.issues[columns]
        info = self.get_info(issue_name=issue_name)
        if issue_name == 'label':
            specific_issues = specific_issues.assign(given_label=info['given_label'], predicted_label=info['predicted_label'])
        if issue_name == 'near_duplicate':
            column_dict = {k: info.get(k) for k in ['near_duplicate_sets', 'distance_to_nearest_neighbor'] if info.get(k) is not None}
            specific_issues = specific_issues.assign(**column_dict)
        return specific_issues

    def get_issue_summary(self, issue_name: Optional[str]=None) -> pd.DataFrame:
        if False:
            return 10
        'Summarize the issues found in dataset of a particular type,\n        including how severe this type of issue is overall across the dataset.\n\n        Parameters\n        ----------\n        issue_name :\n            Name of the issue type to summarize. If `None`, summarizes each of the different issue types previously considered in the audit.\n\n        Returns\n        -------\n        issue_summary :\n            DataFrame where each row corresponds to a type of issue, and columns quantify:\n            the number of examples in the dataset estimated to exhibit this type of issue,\n            and the overall severity of the issue across the dataset (via a numeric quality score where lower values indicate that the issue is overall more severe).\n        '
        if self.issue_summary.empty:
            raise ValueError('No issues found in the dataset. Call `find_issues` before calling `get_issue_summary`.')
        if issue_name is None:
            return self.issue_summary
        row_mask = self.issue_summary['issue_type'] == issue_name
        if not any(row_mask):
            raise ValueError(f'Issue type {issue_name} not found in the summary.')
        return self.issue_summary[row_mask].reset_index(drop=True)

    def get_info(self, issue_name: Optional[str]=None) -> Dict[str, Any]:
        if False:
            i = 10
            return i + 15
        'Get the info for the issue_name key.\n\n        This function is used to get the info for a specific issue_name. If the info is not computed yet, it will raise an error.\n\n        Parameters\n        ----------\n        issue_name :\n            The issue name for which the info is required.\n\n        Returns\n        -------\n        info:\n            The info for the issue_name.\n        '
        info = self.info.get(issue_name, None) if issue_name else self.info
        if info is None:
            raise ValueError(f'issue_name {issue_name} not found in self.info. These have not been computed yet.')
        info = info.copy()
        if issue_name == 'label':
            if self._label_map is None:
                raise ValueError('The label map is not available. Most likely, no label column was provided when creating the Data object.')
            for key in ['given_label', 'predicted_label']:
                labels = info.get(key, None)
                if labels is not None:
                    info[key] = np.vectorize(self._label_map.get)(labels)
            info['class_names'] = self.statistics['class_names']
        return info

    def collect_statistics(self, issue_manager: Union[IssueManager, 'Imagelab']) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Update the statistics in the info dictionary.\n\n        Parameters\n        ----------\n        statistics :\n            A dictionary of statistics to add/update in the info dictionary.\n\n        Examples\n        --------\n\n        A common use case is to reuse the KNN-graph across multiple issue managers.\n        To avoid recomputing the KNN-graph for each issue manager,\n        we can pass it as a statistic to the issue managers.\n\n        >>> from scipy.sparse import csr_matrix\n        >>> weighted_knn_graph = csr_matrix(...)\n        >>> issue_manager_that_computes_knn_graph = ...\n\n        '
        key = 'statistics'
        statistics: Dict[str, Any] = issue_manager.info.get(key, {})
        if statistics:
            self.info[key].update(statistics)

    def _update_issues(self, issue_manager):
        if False:
            while True:
                i = 10
        overlapping_columns = list(set(self.issues.columns) & set(issue_manager.issues.columns))
        if overlapping_columns:
            warnings.warn(f'Overwriting columns {overlapping_columns} in self.issues with columns from issue manager {issue_manager}.')
            self.issues.drop(columns=overlapping_columns, inplace=True)
        self.issues = self.issues.join(issue_manager.issues, how='outer')

    def _update_issue_info(self, issue_name, new_info):
        if False:
            while True:
                i = 10
        if issue_name in self.info:
            warnings.warn(f'Overwriting key {issue_name} in self.info')
        self.info[issue_name] = new_info

    def collect_issues_from_issue_manager(self, issue_manager: IssueManager) -> None:
        if False:
            while True:
                i = 10
        '\n        Collects results from an IssueManager and update the corresponding\n        attributes of the Datalab object.\n\n        This includes:\n        - self.issues\n        - self.issue_summary\n        - self.info\n\n        Parameters\n        ----------\n        issue_manager :\n            IssueManager object to collect results from.\n        '
        self._update_issues(issue_manager)
        if issue_manager.issue_name in self.issue_summary['issue_type'].values:
            warnings.warn(f'Overwriting row in self.issue_summary with row from issue manager {issue_manager}.')
            self.issue_summary = self.issue_summary[self.issue_summary['issue_type'] != issue_manager.issue_name]
        issue_column_name: str = f'is_{issue_manager.issue_name}_issue'
        num_issues: int = int(issue_manager.issues[issue_column_name].sum())
        self.issue_summary = pd.concat([self.issue_summary, issue_manager.summary.assign(num_issues=num_issues)], axis=0, ignore_index=True)
        self._update_issue_info(issue_manager.issue_name, issue_manager.info)

    def set_health_score(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Set the health score for the dataset based on the issue summary.\n\n        Currently, the health score is the mean of the scores for each issue type.\n        '
        self.info['statistics']['health_score'] = self.issue_summary['score'].mean()

def get_data_statistics(data: Data) -> Dict[str, Any]:
    if False:
        i = 10
        return i + 15
    'Get statistics about a dataset.\n\n    This function is called to initialize the "statistics" info in all `Datalab` objects.\n\n    Parameters\n    ----------\n    data : Data\n        Data object containing the dataset.\n    '
    statistics: Dict[str, Any] = {'num_examples': len(data), 'multi_label': False, 'health_score': None}
    if data.labels.is_available:
        class_names = data.class_names
        statistics['class_names'] = class_names
        statistics['num_classes'] = len(class_names)
    return statistics