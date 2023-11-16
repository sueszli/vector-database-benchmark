"""
Datalab offers a unified audit to detect all kinds of issues in data and labels.

.. note::
    .. include:: optional_dependencies.rst
"""
from __future__ import annotations
import warnings
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
import cleanlab
from cleanlab.datalab.internal.adapter.imagelab import create_imagelab
from cleanlab.datalab.internal.data import Data
from cleanlab.datalab.internal.display import _Displayer
from cleanlab.datalab.internal.helper_factory import data_issues_factory, issue_finder_factory, report_factory
from cleanlab.datalab.internal.issue_finder import IssueFinder
from cleanlab.datalab.internal.serialize import _Serializer
if TYPE_CHECKING:
    import numpy.typing as npt
    from datasets.arrow_dataset import Dataset
    from scipy.sparse import csr_matrix
    DatasetLike = Union[Dataset, pd.DataFrame, Dict[str, Any], List[Dict[str, Any]], str]
__all__ = ['Datalab']

class Datalab:
    """
    A single object to automatically detect all kinds of issues in datasets.
    This is how we recommend you interface with the cleanlab library if you want to audit the quality of your data and detect issues within it.
    If you have other specific goals (or are doing a less standard ML task not supported by Datalab), then consider using the other methods across the library.
    Datalab tracks intermediate state (e.g. data statistics) from certain cleanlab functions that can be re-used across other cleanlab functions for better efficiency.

    Parameters
    ----------
    data : Union[Dataset, pd.DataFrame, dict, list, str]
        Dataset-like object that can be converted to a Hugging Face Dataset object.

        It should contain the labels for all examples, identified by a
        `label_name` column in the Dataset object.

        Supported formats:
          - datasets.Dataset
          - pandas.DataFrame
          - dict (keys are strings, values are arrays/lists of length ``N``)
          - list (list of dictionaries that each have the same keys)
          - str

            - path to a local file: Text (.txt), CSV (.csv), JSON (.json)
            - or a dataset identifier on the Hugging Face Hub

    label_name : str, optional
        The name of the label column in the dataset.

    image_key : str, optional
        Optional key that can be specified for image datasets to point to the field containing the actual images themselves.
        If specified, additional image-specific issue types can be detected in the dataset.
        See the CleanVision package `documentation <https://cleanvision.readthedocs.io/en/latest/>`_ for descriptions of these image-specific issue types.

    verbosity : int, optional
        The higher the verbosity level, the more information
        Datalab prints when auditing a dataset.
        Valid values are 0 through 4. Default is 1.

    Examples
    --------
    >>> import datasets
    >>> from cleanlab import Datalab
    >>> data = datasets.load_dataset("glue", "sst2", split="train")
    >>> datalab = Datalab(data, label_name="label")
    """

    def __init__(self, data: 'DatasetLike', label_name: Optional[str]=None, image_key: Optional[str]=None, verbosity: int=1) -> None:
        if False:
            while True:
                i = 10
        self._data = Data(data, label_name)
        self.data = self._data._data
        self._labels = self._data.labels
        self._label_map = self._labels.label_map
        self.label_name = self._labels.label_name
        self._data_hash = self._data._data_hash
        self.cleanlab_version = cleanlab.version.__version__
        self.verbosity = verbosity
        self._imagelab = create_imagelab(dataset=self.data, image_key=image_key)
        self.data_issues = data_issues_factory(self._imagelab)(self._data)

    def __repr__(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return _Displayer(data_issues=self.data_issues).__repr__()

    def __str__(self) -> str:
        if False:
            return 10
        return _Displayer(data_issues=self.data_issues).__str__()

    @property
    def labels(self) -> np.ndarray:
        if False:
            for i in range(10):
                print('nop')
        'Labels of the dataset, in a [0, 1, ..., K-1] format.'
        return self._labels.labels

    @property
    def has_labels(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        'Whether the dataset has labels.'
        return self._labels.is_available

    @property
    def class_names(self) -> List[str]:
        if False:
            return 10
        'Names of the classes in the dataset.\n\n        If the dataset has no labels, returns an empty list.\n        '
        return self._labels.class_names

    def find_issues(self, *, pred_probs: Optional[np.ndarray]=None, features: Optional[npt.NDArray]=None, knn_graph: Optional[csr_matrix]=None, issue_types: Optional[Dict[str, Any]]=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Checks the dataset for all sorts of common issues in real-world data (in both labels and feature values).\n\n        You can use Datalab to find issues in your data, utilizing *any* model you have already trained.\n        This method only interacts with your model via its predictions or embeddings (and other functions thereof).\n        The more of these inputs you provide, the more types of issues Datalab can detect in your dataset/labels.\n        If you provide a subset of these inputs, Datalab will output what insights it can based on the limited information from your model.\n\n        Note\n        ----\n        This method acts as a wrapper around the :py:meth:`IssueFinder.find_issues <cleanlab.datalab.internal.issue_finder.IssueFinder.find_issues>` method,\n        where the core logic for issue detection is implemented.\n\n        Note\n        ----\n        The issues are saved in the ``self.issues`` attribute, but are not returned.\n\n        Parameters\n        ----------\n        pred_probs :\n            Out-of-sample predicted class probabilities made by the model for every example in the dataset.\n            To best detect label issues, provide this input obtained from the most accurate model you can produce.\n\n            If provided, this must be a 2D array with shape (num_examples, K) where K is the number of classes in the dataset.\n\n        features : Optional[np.ndarray]\n            Feature embeddings (vector representations) of every example in the dataset.\n\n            If provided, this must be a 2D array with shape (num_examples, num_features).\n\n        knn_graph :\n            Sparse matrix representing distances between examples in the dataset in a k nearest neighbor graph.\n\n            If provided, this must be a square CSR matrix with shape (num_examples, num_examples) and (k*num_examples) non-zero entries (k is the number of nearest neighbors considered for each example)\n            evenly distributed across the rows.\n            The non-zero entries must be the distances between the corresponding examples. Self-distances must be omitted\n            (i.e. the diagonal must be all zeros and the k nearest neighbors of each example must not include itself).\n\n            For any duplicated examples i,j whose distance is 0, there should be an *explicit* zero stored in the matrix, i.e. ``knn_graph[i,j] = 0``.\n\n            If both `knn_graph` and `features` are provided, the `knn_graph` will take precendence.\n            If `knn_graph` is not provided, it is constructed based on the provided `features`.\n            If neither `knn_graph` nor `features` are provided, certain issue types like (near) duplicates will not be considered.\n\n        issue_types :\n            Collection specifying which types of issues to consider in audit and any non-default parameter settings to use.\n            If unspecified, a default set of issue types and recommended parameter settings is considered.\n\n            This is a dictionary of dictionaries, where the keys are the issue types of interest\n            and the values are dictionaries of parameter values that control how each type of issue is detected (only for advanced users).\n            More specifically, the values are constructor keyword arguments passed to the corresponding ``IssueManager``,\n            which is responsible for detecting the particular issue type.\n\n            .. seealso::\n                :py:class:`IssueManager <cleanlab.datalab.internal.issue_manager.issue_manager.IssueManager>`\n\n        Examples\n        --------\n\n        Here are some ways to provide inputs to :py:meth:`find_issues`:\n\n        - Passing ``pred_probs``:\n            .. code-block:: python\n\n                >>> from sklearn.linear_model import LogisticRegression\n                >>> import numpy as np\n                >>> from cleanlab import Datalab\n                >>> X = np.array([[0, 1], [1, 1], [2, 2], [2, 0]])\n                >>> y = np.array([0, 1, 1, 0])\n                >>> clf = LogisticRegression(random_state=0).fit(X, y)\n                >>> pred_probs = clf.predict_proba(X)\n                >>> lab = Datalab(data={"X": X, "y": y}, label_name="y")\n                >>> lab.find_issues(pred_probs=pred_probs)\n\n\n        - Passing ``features``:\n            .. code-block:: python\n\n                >>> from sklearn.linear_model import LogisticRegression\n                >>> from sklearn.neighbors import NearestNeighbors\n                >>> import numpy as np\n                >>> from cleanlab import Datalab\n                >>> X = np.array([[0, 1], [1, 1], [2, 2], [2, 0]])\n                >>> y = np.array([0, 1, 1, 0])\n                >>> lab = Datalab(data={"X": X, "y": y}, label_name="y")\n                >>> lab.find_issues(features=X)\n\n        .. note::\n\n            You can pass both ``pred_probs`` and ``features`` to :py:meth:`find_issues` for a more comprehensive audit.\n\n        - Passing a ``knn_graph``:\n            .. code-block:: python\n\n                >>> from sklearn.neighbors import NearestNeighbors\n                >>> import numpy as np\n                >>> from cleanlab import Datalab\n                >>> X = np.array([[0, 1], [1, 1], [2, 2], [2, 0]])\n                >>> y = np.array([0, 1, 1, 0])\n                >>> nbrs = NearestNeighbors(n_neighbors=2, metric="euclidean").fit(X)\n                >>> knn_graph = nbrs.kneighbors_graph(mode="distance")\n                >>> knn_graph # Pass this to Datalab\n                <4x4 sparse matrix of type \'<class \'numpy.float64\'>\'\n                        with 8 stored elements in Compressed Sparse Row format>\n                >>> knn_graph.toarray()  # DO NOT PASS knn_graph.toarray() to Datalab, only pass the sparse matrix itself\n                array([[0.        , 1.        , 2.23606798, 0.        ],\n                        [1.        , 0.        , 1.41421356, 0.        ],\n                        [0.        , 1.41421356, 0.        , 2.        ],\n                        [0.        , 1.41421356, 2.        , 0.        ]])\n                >>> lab = Datalab(data={"X": X, "y": y}, label_name="y")\n                >>> lab.find_issues(knn_graph=knn_graph)\n\n        - Configuring issue types:\n            Suppose you want to only consider label issues. Just pass a dictionary with the key "label" and an empty dictionary as the value (to use default label issue parameters).\n\n            .. code-block:: python\n\n                >>> issue_types = {"label": {}}\n                >>> # lab.find_issues(pred_probs=pred_probs, issue_types=issue_types)\n\n            If you are advanced user who wants greater control, you can pass keyword arguments to the issue manager that handles the label issues.\n            For example, if you want to pass the keyword argument "clean_learning_kwargs"\n            to the constructor of the :py:class:`LabelIssueManager <cleanlab.datalab.internal.issue_manager.label.LabelIssueManager>`, you would pass:\n\n\n            .. code-block:: python\n\n                >>> issue_types = {\n                ...     "label": {\n                ...         "clean_learning_kwargs": {\n                ...             "prune_method": "prune_by_noise_rate",\n                ...         },\n                ...     },\n                ... }\n                >>> # lab.find_issues(pred_probs=pred_probs, issue_types=issue_types)\n\n        '
        if issue_types is not None and (not issue_types):
            warnings.warn('No issue types were specified so no issues will be found in the dataset. Set `issue_types` as None to consider a default set of issues.')
            return None
        issue_finder = issue_finder_factory(self._imagelab)(datalab=self, verbosity=self.verbosity)
        issue_finder.find_issues(pred_probs=pred_probs, features=features, knn_graph=knn_graph, issue_types=issue_types)
        if self.verbosity:
            print(f"\nAudit complete. {self.data_issues.issue_summary['num_issues'].sum()} issues found in the dataset.")

    def report(self, *, num_examples: int=5, verbosity: Optional[int]=None, include_description: bool=True, show_summary_score: bool=False) -> None:
        if False:
            print('Hello World!')
        "Prints informative summary of all issues.\n\n        Parameters\n        ----------\n        num_examples :\n            Number of examples to show for each type of issue.\n            The report shows the top `num_examples` instances in the dataset that suffer the most from each type of issue.\n\n        verbosity :\n            Higher verbosity levels add more information to the report.\n\n        include_description :\n            Whether or not to include a description of each issue type in the report.\n            Consider setting this to ``False`` once you're familiar with how each issue type is defined.\n\n        See Also\n        --------\n        For advanced usage, see documentation for the\n        :py:class:`Reporter <cleanlab.datalab.internal.report.Reporter>` class.\n        "
        if verbosity is None:
            verbosity = self.verbosity
        if self.data_issues.issue_summary.empty:
            print('Please specify some `issue_types` in datalab.find_issues() to see a report.\n')
            return
        reporter = report_factory(self._imagelab)(data_issues=self.data_issues, verbosity=verbosity, include_description=include_description, show_summary_score=show_summary_score, imagelab=self._imagelab)
        reporter.report(num_examples=num_examples)

    @property
    def issues(self) -> pd.DataFrame:
        if False:
            i = 10
            return i + 15
        'Issues found in each example from the dataset.'
        return self.data_issues.issues

    @issues.setter
    def issues(self, issues: pd.DataFrame) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.data_issues.issues = issues

    @property
    def issue_summary(self) -> pd.DataFrame:
        if False:
            print('Hello World!')
        'Summary of issues found in the dataset and the overall severity of each type of issue.\n\n        This is a wrapper around the ``DataIssues.issue_summary`` attribute.\n\n        Examples\n        -------\n\n        If checks for "label" and "outlier" issues were run,\n        then the issue summary will look something like this:\n\n        >>> datalab.issue_summary\n        issue_type  score\n        outlier     0.123\n        label       0.456\n        '
        return self.data_issues.issue_summary

    @issue_summary.setter
    def issue_summary(self, issue_summary: pd.DataFrame) -> None:
        if False:
            while True:
                i = 10
        self.data_issues.issue_summary = issue_summary

    @property
    def info(self) -> Dict[str, Dict[str, Any]]:
        if False:
            return 10
        'Information and statistics about the dataset issues found.\n\n        This is a wrapper around the ``DataIssues.info`` attribute.\n\n        Examples\n        -------\n\n        If checks for "label" and "outlier" issues were run,\n        then the info will look something like this:\n\n        >>> datalab.info\n        {\n            "label": {\n                "given_labels": [0, 1, 0, 1, 1, 1, 1, 1, 0, 1, ...],\n                "predicted_label": [0, 0, 0, 1, 0, 1, 0, 1, 0, 1, ...],\n                ...,\n            },\n            "outlier": {\n                "nearest_neighbor": [3, 7, 1, 2, 8, 4, 5, 9, 6, 0, ...],\n                "distance_to_nearest_neighbor": [0.123, 0.789, 0.456, ...],\n                ...,\n            },\n        }\n        '
        return self.data_issues.info

    @info.setter
    def info(self, info: Dict[str, Dict[str, Any]]) -> None:
        if False:
            print('Hello World!')
        self.data_issues.info = info

    def get_issues(self, issue_name: Optional[str]=None) -> pd.DataFrame:
        if False:
            return 10
        '\n        Use this after finding issues to see which examples suffer from which types of issues.\n\n        NOTE\n        ----\n        This is a wrapper around the :py:meth:`DataIssues.get_issues <cleanlab.datalab.internal.data_issues.DataIssues.get_issues>` method.\n\n        Parameters\n        ----------\n        issue_name : str or None\n            The type of issue to focus on. If `None`, returns full DataFrame summarizing all of the types of issues detected in each example from the dataset.\n\n        Raises\n        ------\n        ValueError\n            If `issue_name` is not a type of issue previously considered in the audit.\n\n        Returns\n        -------\n        specific_issues :\n            A DataFrame where each row corresponds to an example from the dataset and columns specify:\n            whether this example exhibits a particular type of issue, and how severely (via a numeric quality score where lower values indicate more severe instances of the issue).\n            The quality scores lie between 0-1 and are directly comparable between examples (for the same issue type), but not across different issue types.\n\n            Additional columns may be present in the DataFrame depending on the type of issue specified.\n        '
        return self.data_issues.get_issues(issue_name=issue_name)

    def get_issue_summary(self, issue_name: Optional[str]=None) -> pd.DataFrame:
        if False:
            i = 10
            return i + 15
        'Summarize the issues found in dataset of a particular type,\n        including how severe this type of issue is overall across the dataset.\n\n        NOTE\n        ----\n        This is a wrapper around the\n        :py:meth:`DataIssues.get_issue_summary <cleanlab.datalab.internal.data_issues.DataIssues.get_issue_summary>` method.\n\n        Parameters\n        ----------\n        issue_name :\n            Name of the issue type to summarize. If `None`, summarizes each of the different issue types previously considered in the audit.\n\n        Returns\n        -------\n        issue_summary :\n            DataFrame where each row corresponds to a type of issue, and columns quantify:\n            the number of examples in the dataset estimated to exhibit this type of issue,\n            and the overall severity of the issue across the dataset (via a numeric quality score where lower values indicate that the issue is overall more severe).\n            The quality scores lie between 0-1 and are directly comparable between multiple datasets (for the same issue type), but not across different issue types.\n        '
        return self.data_issues.get_issue_summary(issue_name=issue_name)

    def get_info(self, issue_name: Optional[str]=None) -> Dict[str, Any]:
        if False:
            while True:
                i = 10
        'Get the info for the issue_name key.\n\n        This function is used to get the info for a specific issue_name. If the info is not computed yet, it will raise an error.\n\n        NOTE\n        ----\n        This is a wrapper around the\n        :py:meth:`DataIssues.get_info <cleanlab.datalab.internal.data_issues.DataIssues.get_info>` method.\n\n        Parameters\n        ----------\n        issue_name :\n            The issue name for which the info is required.\n\n        Returns\n        -------\n        :py:meth:`info <cleanlab.datalab.internal.data_issues.DataIssues.get_info>` :\n            The info for the issue_name.\n        '
        return self.data_issues.get_info(issue_name)

    @staticmethod
    def list_possible_issue_types() -> List[str]:
        if False:
            print('Hello World!')
        'Returns a list of all registered issue types.\n\n        Any issue type that is not in this list cannot be used in the :py:meth:`find_issues` method.\n\n        Note\n        ----\n        This method is a wrapper around :py:meth:`IssueFinder.list_possible_issue_types <cleanlab.datalab.internal.issue_finder.IssueFinder.list_possible_issue_types>`.\n\n        See Also\n        --------\n        :py:class:`REGISTRY <cleanlab.datalab.internal.issue_manager_factory.REGISTRY>` : All available issue types and their corresponding issue managers can be found here.\n        '
        return IssueFinder.list_possible_issue_types()

    @staticmethod
    def list_default_issue_types() -> List[str]:
        if False:
            i = 10
            return i + 15
        'Returns a list of the issue types that are run by default\n        when :py:meth:`find_issues` is called without specifying `issue_types`.\n\n        Note\n        ----\n        This method is a wrapper around :py:meth:`IssueFinder.list_default_issue_types <cleanlab.datalab.internal.issue_finder.IssueFinder.list_default_issue_types>`.\n\n        See Also\n        --------\n        :py:class:`REGISTRY <cleanlab.datalab.internal.issue_manager_factory.REGISTRY>` : All available issue types and their corresponding issue managers can be found here.\n        '
        return IssueFinder.list_default_issue_types()

    def save(self, path: str, force: bool=False) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Saves this Datalab\xa0object to file (all files are in folder at `path/`).\n        We do not guarantee saved Datalab can be loaded from future versions of cleanlab.\n\n        Parameters\n        ----------\n        path :\n            Folder in which all information about this Datalab should be saved.\n\n        force :\n            If ``True``, overwrites any existing files in the folder at `path`. Use this with caution!\n\n        Note\n        ----\n        You have to save the Dataset yourself separately if you want it saved to file.\n        '
        _Serializer.serialize(path=path, datalab=self, force=force)
        save_message = f'Saved Datalab to folder: {path}'
        print(save_message)

    @staticmethod
    def load(path: str, data: Optional[Dataset]=None) -> 'Datalab':
        if False:
            print('Hello World!')
        'Loads Datalab object from a previously saved folder.\n\n        Parameters\n        ----------\n        `path` :\n            Path to the folder previously specified in ``Datalab.save()``.\n\n        `data` :\n            The dataset used to originally construct the Datalab.\n            Remember the dataset is not saved as part of the Datalab,\n            you must save/load the data separately.\n\n        Returns\n        -------\n        `datalab` :\n            A Datalab object that is identical to the one originally saved.\n        '
        datalab = _Serializer.deserialize(path=path, data=data)
        load_message = f'Datalab loaded from folder: {path}'
        print(load_message)
        return datalab