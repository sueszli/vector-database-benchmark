"""
Module for the :class:`IssueFinder` class, which is responsible for configuring,
creating and running issue managers.

It determines which types of issues to look for, instatiates the IssueManagers
via a factory, run the issue managers
(:py:meth:`IssueManager.find_issues <cleanlab.datalab.internal.issue_manager.issue_manager.IssueManager.find_issues>`),
and collects the results to :py:class:`DataIssues <cleanlab.datalab.internal.data_issues.DataIssues>`.

.. note::

    This module is not intended to be used directly. Instead, use the public-facing
    :py:meth:`Datalab.find_issues <cleanlab.datalab.datalab.Datalab.find_issues>` method.
"""
from __future__ import annotations
import warnings
from typing import Any, List, Optional, Dict, TYPE_CHECKING
import numpy as np
from scipy.sparse import csr_matrix
from cleanlab.datalab.internal.issue_manager_factory import _IssueManagerFactory, REGISTRY
if TYPE_CHECKING:
    import numpy.typing as npt
    from cleanlab.datalab.datalab import Datalab

class IssueFinder:
    """
    The IssueFinder class is responsible for managing the process of identifying
    issues in the dataset by handling the creation and execution of relevant
    IssueManagers. It serves as a coordinator or helper class for the Datalab class
    to encapsulate the specific behavior of the issue finding process.

    At a high level, the IssueFinder is responsible for:

    - Determining which types of issues to look for.
    - Instantiating the appropriate IssueManagers using a factory.
    - Running the IssueManagers' `find_issues` methods.
    - Collecting the results into a DataIssues instance.

    Parameters
    ----------
    datalab : Datalab
        The Datalab instance associated with this IssueFinder.

    verbosity : int
        Controls the verbosity of the output during the issue finding process.

    Note
    ----
    This class is not intended to be used directly. Instead, use the
    `Datalab.find_issues` method which internally utilizes an IssueFinder instance.
    """

    def __init__(self, datalab: 'Datalab', verbosity=1):
        if False:
            i = 10
            return i + 15
        self.datalab = datalab
        self.verbosity = verbosity

    def find_issues(self, *, pred_probs: Optional[np.ndarray]=None, features: Optional[npt.NDArray]=None, knn_graph: Optional[csr_matrix]=None, issue_types: Optional[Dict[str, Any]]=None) -> None:
        if False:
            return 10
        '\n        Checks the dataset for all sorts of common issues in real-world data (in both labels and feature values).\n\n        You can use Datalab to find issues in your data, utilizing *any* model you have already trained.\n        This method only interacts with your model via its predictions or embeddings (and other functions thereof).\n        The more of these inputs you provide, the more types of issues Datalab can detect in your dataset/labels.\n        If you provide a subset of these inputs, Datalab will output what insights it can based on the limited information from your model.\n\n        Note\n        ----\n        This method is not intended to be used directly. Instead, use the\n        :py:meth:`Datalab.find_issues <cleanlab.datalab.datalab.Datalab.find_issues>` method.\n\n        Note\n        ----\n        The issues are saved in the ``self.datalab.data_issues.issues`` attribute, but are not returned.\n\n        Parameters\n        ----------\n        pred_probs :\n            Out-of-sample predicted class probabilities made by the model for every example in the dataset.\n            To best detect label issues, provide this input obtained from the most accurate model you can produce.\n\n            If provided, this must be a 2D array with shape (num_examples, K) where K is the number of classes in the dataset.\n\n        features : Optional[np.ndarray]\n            Feature embeddings (vector representations) of every example in the dataset.\n\n            If provided, this must be a 2D array with shape (num_examples, num_features).\n\n        knn_graph :\n            Sparse matrix representing distances between examples in the dataset in a k nearest neighbor graph.\n\n            If provided, this must be a square CSR matrix with shape (num_examples, num_examples) and (k*num_examples) non-zero entries (k is the number of nearest neighbors considered for each example)\n            evenly distributed across the rows.\n            The non-zero entries must be the distances between the corresponding examples. Self-distances must be omitted\n            (i.e. the diagonal must be all zeros and the k nearest neighbors of each example must not include itself).\n\n            For any duplicated examples i,j whose distance is 0, there should be an *explicit* zero stored in the matrix, i.e. ``knn_graph[i,j] = 0``.\n\n            If both `knn_graph` and `features` are provided, the `knn_graph` will take precendence.\n            If `knn_graph` is not provided, it is constructed based on the provided `features`.\n            If neither `knn_graph` nor `features` are provided, certain issue types like (near) duplicates will not be considered.\n\n        issue_types :\n            Collection specifying which types of issues to consider in audit and any non-default parameter settings to use.\n            If unspecified, a default set of issue types and recommended parameter settings is considered.\n\n            This is a dictionary of dictionaries, where the keys are the issue types of interest\n            and the values are dictionaries of parameter values that control how each type of issue is detected (only for advanced users).\n            More specifically, the values are constructor keyword arguments passed to the corresponding ``IssueManager``,\n            which is responsible for detecting the particular issue type.\n\n            .. seealso::\n                :py:class:`IssueManager <cleanlab.datalab.internal.issue_manager.issue_manager.IssueManager>`\n        '
        issue_types_copy = self.get_available_issue_types(pred_probs=pred_probs, features=features, knn_graph=knn_graph, issue_types=issue_types)
        if not issue_types_copy:
            return None
        new_issue_managers = [factory(datalab=self.datalab, **issue_types_copy.get(factory.issue_name, {})) for factory in _IssueManagerFactory.from_list(list(issue_types_copy.keys()))]
        failed_managers = []
        data_issues = self.datalab.data_issues
        for (issue_manager, arg_dict) in zip(new_issue_managers, issue_types_copy.values()):
            try:
                if self.verbosity:
                    print(f'Finding {issue_manager.issue_name} issues ...')
                issue_manager.find_issues(**arg_dict)
                data_issues.collect_statistics(issue_manager)
                data_issues.collect_issues_from_issue_manager(issue_manager)
            except Exception as e:
                print(f'Error in {issue_manager.issue_name}: {e}')
                failed_managers.append(issue_manager)
        if failed_managers:
            print(f'Failed to check for these issue types: {failed_managers}')
        data_issues.set_health_score()

    def _resolve_required_args(self, pred_probs, features, knn_graph):
        if False:
            print('Hello World!')
        'Resolves the required arguments for each issue type.\n\n        This is a helper function that filters out any issue manager\n        that does not have the required arguments.\n\n        This does not consider custom hyperparameters for each issue type.\n\n\n        Parameters\n        ----------\n        pred_probs :\n            Out-of-sample predicted probabilities made on the data.\n\n        features :\n            Name of column containing precomputed embeddings.\n\n        knn_graph :\n            Sparse matrix representing distances between examples in the dataset in a k nearest neighbor graph.\n\n        Returns\n        -------\n        args_dict :\n            Dictionary of required arguments for each issue type, if available.\n        '
        args_dict = {'label': {'pred_probs': pred_probs, 'features': features}, 'outlier': {'pred_probs': pred_probs, 'features': features, 'knn_graph': knn_graph}, 'near_duplicate': {'features': features, 'knn_graph': knn_graph}, 'non_iid': {'pred_probs': pred_probs, 'features': features, 'knn_graph': knn_graph}, 'data_valuation': {'knn_graph': knn_graph}}
        args_dict = {k: {k2: v2 for (k2, v2) in v.items() if v2 is not None} for (k, v) in args_dict.items() if v}
        for v in args_dict.values():
            if 'knn_graph' in v and 'features' in v:
                warnings.warn('Both `features` and `knn_graph` were provided. Most issue managers will likely prefer using `knn_graph` instead of `features` for efficiency.')
        args_dict = {k: v for (k, v) in args_dict.items() if v}
        return args_dict

    def _set_issue_types(self, issue_types: Optional[Dict[str, Any]], required_defaults_dict: Dict[str, Any]) -> Dict[str, Any]:
        if False:
            for i in range(10):
                print('nop')
        'Set necessary configuration for each IssueManager in a dictionary.\n\n        While each IssueManager defines default values for its arguments,\n        the Datalab class needs to organize the calls to each IssueManager\n        with different arguments, some of which may be user-provided.\n\n        Parameters\n        ----------\n        issue_types :\n            Dictionary of issue types and argument configuration for their respective IssueManagers.\n            If None, then the `required_defaults_dict` is used.\n\n        required_defaults_dict :\n            Dictionary of default parameter configuration for each issue type.\n\n        Returns\n        -------\n        issue_types_copy :\n            Dictionary of issue types and their parameter configuration.\n            The input `issue_types` is copied and updated with the necessary default values.\n        '
        if issue_types is not None:
            issue_types_copy = issue_types.copy()
            self._check_missing_args(required_defaults_dict, issue_types_copy)
        else:
            issue_types_copy = required_defaults_dict.copy()
            issue_types_copy = {issue: issue_types_copy[issue] for issue in self.list_default_issue_types() if issue in issue_types_copy}
        self._validate_issue_types_dict(issue_types_copy, required_defaults_dict)
        for (key, value) in issue_types_copy.items():
            issue_types_copy[key] = {k: v for (k, v) in value.items() if v is not None}
        return issue_types_copy

    @staticmethod
    def _check_missing_args(required_defaults_dict, issue_types):
        if False:
            for i in range(10):
                print('nop')
        for (key, issue_type_value) in issue_types.items():
            missing_args = set(required_defaults_dict.get(key, {})) - set(issue_type_value.keys())
            missing_dict = {missing_arg: required_defaults_dict[key][missing_arg] for missing_arg in missing_args}
            issue_types[key].update(missing_dict)

    @staticmethod
    def _validate_issue_types_dict(issue_types: Dict[str, Any], required_defaults_dict: Dict[str, Any]) -> None:
        if False:
            while True:
                i = 10
        missing_required_args_dict = {}
        for (issue_name, required_args) in required_defaults_dict.items():
            if issue_name in issue_types:
                missing_args = set(required_args.keys()) - set(issue_types[issue_name].keys())
                if missing_args:
                    missing_required_args_dict[issue_name] = missing_args
        if any(missing_required_args_dict.values()):
            error_message = ''
            for (issue_name, missing_required_args) in missing_required_args_dict.items():
                error_message += f'Required argument {missing_required_args} for issue type {issue_name} was not provided.\n'
            raise ValueError(error_message)

    @staticmethod
    def list_possible_issue_types() -> List[str]:
        if False:
            for i in range(10):
                print('nop')
        'Returns a list of all registered issue types.\n\n        Any issue type that is not in this list cannot be used in the :py:meth:`find_issues` method.\n\n        See Also\n        --------\n        :py:class:`REGISTRY <cleanlab.datalab.internal.issue_manager_factory.REGISTRY>` : All available issue types and their corresponding issue managers can be found here.\n        '
        return list(REGISTRY.keys())

    @staticmethod
    def list_default_issue_types() -> List[str]:
        if False:
            return 10
        'Returns a list of the issue types that are run by default\n        when :py:meth:`find_issues` is called without specifying `issue_types`.\n\n        See Also\n        --------\n        :py:class:`REGISTRY <cleanlab.datalab.internal.issue_manager_factory.REGISTRY>` : All available issue types and their corresponding issue managers can be found here.\n        '
        return ['label', 'outlier', 'near_duplicate', 'non_iid']

    def get_available_issue_types(self, **kwargs):
        if False:
            i = 10
            return i + 15
        'Returns a dictionary of issue types that can be used in :py:meth:`Datalab.find_issues\n        <cleanlab.datalab.datalab.Datalab.find_issues>` method.'
        pred_probs = kwargs.get('pred_probs', None)
        features = kwargs.get('features', None)
        knn_graph = kwargs.get('knn_graph', None)
        issue_types = kwargs.get('issue_types', None)
        required_args_per_issue_type = self._resolve_required_args(pred_probs, features, knn_graph)
        issue_types_copy = self._set_issue_types(issue_types, required_args_per_issue_type)
        if issue_types is None:
            issue_types_copy = {issue: issue_types_copy[issue] for issue in self.list_default_issue_types() if issue in issue_types_copy}
        drop_label_check = 'label' in issue_types_copy and (not self.datalab.has_labels)
        if drop_label_check:
            warnings.warn("No labels were provided. The 'label' issue type will not be run.")
            issue_types_copy.pop('label')
        outlier_check_needs_features = 'outlier' in issue_types_copy and (not self.datalab.has_labels)
        if outlier_check_needs_features:
            no_features = features is None
            no_knn_graph = knn_graph is None
            pred_probs_given = issue_types_copy['outlier'].get('pred_probs', None) is not None
            only_pred_probs_given = pred_probs_given and no_features and no_knn_graph
            if only_pred_probs_given:
                warnings.warn("No labels were provided. The 'outlier' issue type will not be run.")
                issue_types_copy.pop('outlier')
        return issue_types_copy