"""
Implementation of :py:func:`filter.find_label_issues <cleanlab.filter.find_label_issues>`
that does not need much memory by operating in mini-batches.
You can also use this approach to estimate label quality scores or the number of label issues
for big datasets with limited memory.

With default settings, the results returned from this approach closely approximate those returned from:
``cleanlab.filter.find_label_issues(..., filter_by="low_self_confidence", return_indices_ranked_by="self_confidence")``

To run this approach, either use the ``find_label_issues_batched()`` convenience function defined in this module,
or follow the examples script for the ``LabelInspector`` class if you require greater customization.
"""
import numpy as np
from typing import Optional, List, Tuple, Any
from cleanlab.count import get_confident_thresholds
from cleanlab.rank import find_top_issues, _compute_label_quality_scores
from cleanlab.typing import LabelLike
from cleanlab.internal.util import value_counts_fill_missing_classes
from cleanlab.internal.constants import CONFIDENT_THRESHOLDS_LOWER_BOUND, FLOATING_POINT_COMPARISON, CLIPPING_LOWER_BOUND
import platform
import multiprocessing as mp
try:
    import psutil
    PSUTIL_EXISTS = True
except ImportError:
    PSUTIL_EXISTS = False
adj_confident_thresholds_shared: np.ndarray
labels_shared: LabelLike
pred_probs_shared: np.ndarray

def find_label_issues_batched(labels: Optional[LabelLike]=None, pred_probs: Optional[np.ndarray]=None, *, labels_file: Optional[str]=None, pred_probs_file: Optional[str]=None, batch_size: int=10000, n_jobs: Optional[int]=1, verbose: bool=True, quality_score_kwargs: Optional[dict]=None, num_issue_kwargs: Optional[dict]=None, return_mask: bool=False) -> np.ndarray:
    if False:
        i = 10
        return i + 15
    '\n    Variant of :py:func:`filter.find_label_issues <cleanlab.filter.find_label_issues>`\n    that requires less memory by reading from `pred_probs`, `labels` in mini-batches.\n    To avoid loading big `pred_probs`, `labels` arrays into memory,\n    provide these as memory-mapped objects like Zarr arrays or memmap arrays instead of regular numpy arrays.\n    See: https://pythonspeed.com/articles/mmap-vs-zarr-hdf5/\n\n    With default settings, the results returned from this method closely approximate those returned from:\n    ``cleanlab.filter.find_label_issues(..., filter_by="low_self_confidence", return_indices_ranked_by="self_confidence")``\n\n    This function internally implements the example usage script of the ``LabelInspector`` class,\n    but you can further customize that script by running it yourself instead of this function.\n    See the documentation of ``LabelInspector`` to learn more about how this method works internally.\n\n    Parameters\n    ----------\n    labels: np.ndarray-like object, optional\n      1D array of given class labels for each example in the dataset, (int) values in ``0,1,2,...,K-1``.\n      To avoid loading big objects into memory, you should pass this as a memory-mapped object like:\n      Zarr array loaded with ``zarr.convenience.open(YOURFILE.zarr, mode="r")``,\n      or memmap array loaded with ``np.load(YOURFILE.npy, mmap_mode="r")``.\n\n      Tip: You can save an existing numpy array to Zarr via: ``zarr.convenience.save_array(YOURFILE.zarr, your_array)``,\n      or to .npy file that can be loaded with mmap via: ``np.save(YOURFILE.npy, your_array)``.\n\n    pred_probs: np.ndarray-like object, optional\n      2D array of model-predicted class probabilities (floats) for each example in the dataset.\n      To avoid loading big objects into memory, you should pass this as a memory-mapped object like:\n      Zarr array loaded with ``zarr.convenience.open(YOURFILE.zarr, mode="r")``\n      or memmap array loaded with ``np.load(YOURFILE.npy, mmap_mode="r")``.\n\n    labels_file: str, optional\n      Specify this instead of `labels` if you want this method to load from file for you into a memmap array.\n      Path to .npy file where the entire 1D `labels` numpy array is stored on disk (list format is not supported).\n      This is loaded using: ``np.load(labels_file, mmap_mode="r")``\n      so make sure this file was created via: ``np.save()`` or other compatible methods (.npz not supported).\n\n    pred_probs_file: str, optional\n      Specify this instead of `pred_probs` if you want this method to load from file for you into a memmap array.\n      Path to .npy file where the entire `pred_probs` numpy array is stored on disk.\n      This is loaded using: ``np.load(pred_probs_file, mmap_mode="r")``\n      so make sure this file was created via: ``np.save()`` or other compatible methods (.npz not supported).\n\n    batch_size : int, optional\n      Size of mini-batches to use for estimating the label issues.\n      To maximize efficiency, try to use the largest `batch_size` your memory allows.\n\n    n_jobs: int, optional\n      Number of processes for multiprocessing (default value = 1). Only used on Linux.\n      If `n_jobs=None`, will use either the number of: physical cores if psutil is installed, or logical cores otherwise.\n\n    verbose : bool, optional\n      Whether to suppress print statements or not.\n\n    quality_score_kwargs : dict, optional\n      Keyword arguments to pass into :py:func:`rank.get_label_quality_scores <cleanlab.rank.get_label_quality_scores>`.\n\n    num_issue_kwargs : dict, optional\n      Keyword arguments to :py:func:`count.num_label_issues <cleanlab.count.num_label_issues>`\n      to control estimation of the number of label issues.\n      The only supported kwarg here for now is: `estimation_method`.\n    return_mask : bool, optional\n       Determines what is returned by this method: If `return_mask=True`, return a boolean mask.\n       If `False`, return a list of indices specifying examples with label issues, sorted by label quality score.\n\n    Returns\n    -------\n    label_issues : np.ndarray\n      If `return_mask` is `True`, returns a boolean **mask** for the entire dataset\n      where ``True`` represents a label issue and ``False`` represents an example that is\n      accurately labeled with high confidence.\n      If `return_mask` is `False`, returns an array containing **indices** of examples identified to have\n      label issues (i.e. those indices where the mask would be ``True``), sorted by likelihood that the corresponding label is correct.\n    --------\n    >>> batch_size = 10000  # for efficiency, set this to as large of a value as your memory can handle\n    >>> # Just demonstrating how to save your existing numpy labels, pred_probs arrays to compatible .npy files:\n    >>> np.save("LABELS.npy", labels_array)\n    >>> np.save("PREDPROBS.npy", pred_probs_array)\n    >>> # You can load these back into memmap arrays via: labels = np.load("LABELS.npy", mmap_mode="r")\n    >>> # and then run this method on the memmap arrays, or just run it directly on the .npy files like this:\n    >>> issues = find_label_issues_batched(labels_file="LABELS.npy", pred_probs_file="PREDPROBS.npy", batch_size=batch_size)\n    >>> # This method also works with Zarr arrays:\n    >>> import zarr\n    >>> # Just demonstrating how to save your existing numpy labels, pred_probs arrays to compatible .zarr files:\n    >>> zarr.convenience.save_array("LABELS.zarr", labels_array)\n    >>> zarr.convenience.save_array("PREDPROBS.zarr", pred_probs_array)\n    >>> # You can load from such files into Zarr arrays:\n    >>> labels = zarr.convenience.open("LABELS.zarr", mode="r")\n    >>> pred_probs = zarr.convenience.open("PREDPROBS.zarr", mode="r")\n    >>> # This method can be directly run on Zarr arrays, memmap arrays, or regular numpy arrays:\n    >>> issues = find_label_issues_batched(labels=labels, pred_probs=pred_probs, batch_size=batch_size)\n    '
    if labels_file is not None:
        if labels is not None:
            raise ValueError('only specify one of: `labels` or `labels_file`')
        if not isinstance(labels_file, str):
            raise ValueError('labels_file must be str specifying path to .npy file containing the array of labels')
        labels = np.load(labels_file, mmap_mode='r')
        assert isinstance(labels, np.ndarray)
    if pred_probs_file is not None:
        if pred_probs is not None:
            raise ValueError('only specify one of: `pred_probs` or `pred_probs_file`')
        if not isinstance(pred_probs_file, str):
            raise ValueError('pred_probs_file must be str specifying path to .npy file containing 2D array of pred_probs')
        pred_probs = np.load(pred_probs_file, mmap_mode='r')
        assert isinstance(pred_probs, np.ndarray)
        if verbose:
            print(f'mmap-loaded numpy arrays have: {len(pred_probs)} examples, {pred_probs.shape[1]} classes')
    if labels is None:
        raise ValueError('must provide one of: `labels` or `labels_file`')
    if pred_probs is None:
        raise ValueError('must provide one of: `pred_probs` or `pred_probs_file`')
    assert pred_probs is not None
    if len(labels) != len(pred_probs):
        raise ValueError(f'len(labels)={len(labels)} does not match len(pred_probs)={len(pred_probs)}. Perhaps an issue loading mmap numpy arrays from file.')
    lab = LabelInspector(num_class=pred_probs.shape[1], verbose=verbose, n_jobs=n_jobs, quality_score_kwargs=quality_score_kwargs, num_issue_kwargs=num_issue_kwargs)
    n = len(labels)
    if verbose:
        from tqdm.auto import tqdm
        pbar = tqdm(desc='number of examples processed for estimating thresholds', total=n)
    i = 0
    while i < n:
        end_index = i + batch_size
        labels_batch = labels[i:end_index]
        pred_probs_batch = pred_probs[i:end_index, :]
        i = end_index
        lab.update_confident_thresholds(labels_batch, pred_probs_batch)
        if verbose:
            pbar.update(batch_size)
    if verbose:
        pbar.close()
        pbar = tqdm(desc='number of examples processed for checking labels', total=n)
    i = 0
    while i < n:
        end_index = i + batch_size
        labels_batch = labels[i:end_index]
        pred_probs_batch = pred_probs[i:end_index, :]
        i = end_index
        _ = lab.score_label_quality(labels_batch, pred_probs_batch)
        if verbose:
            pbar.update(batch_size)
    if verbose:
        pbar.close()
    label_issues_indices = lab.get_label_issues()
    if return_mask:
        label_issues_mask = np.zeros(len(labels), dtype=bool)
        label_issues_mask[label_issues_indices] = True
        return label_issues_mask
    return label_issues_indices

class LabelInspector:
    """
    Class for finding label issues in big datasets where memory becomes a problem for other cleanlab methods.
    Only create one such object per dataset and do not try to use the same ``LabelInspector`` across 2 datasets.
    For efficiency, this class does little input checking.
    You can first run :py:func:`filter.find_label_issues <cleanlab.filter.find_label_issues>`
    on a small subset of your data to verify your inputs are properly formatted.
    Do NOT modify any of the attributes of this class yourself!
    Multi-label classification is not supported by this class, it is only for multi-class classification.

    The recommended usage demonstrated in the examples script below involves two passes over your data:
    one pass to compute `confident_thresholds`, another to evaluate each label.
    To maximize efficiency, try to use the largest batch_size your memory allows.
    To reduce runtime further, you can run the first pass on a subset of your dataset
    as long as it contains enough data from each class to estimate `confident_thresholds` accurately.

    In the examples script below:
    - `labels` is a (big) 1D ``np.ndarray`` of class labels represented as integers in ``0,1,...,K-1``.
    - ``pred_probs`` = is a (big) 2D ``np.ndarray`` of predicted class probabilities,
    where each row is an example, each column represents a class.

    `labels` and `pred_probs` can be stored in a file instead where you load chunks of them at a time.
    Methods to load arrays in chunks include: ``np.load(...,mmap_mode='r')``, ``numpy.memmap()``,
    HDF5 or Zarr files, see: https://pythonspeed.com/articles/mmap-vs-zarr-hdf5/

    Examples
    --------
    >>> n = len(labels)
    >>> batch_size = 10000  # you can change this in between batches, set as big as your RAM allows
    >>> lab = LabelInspector(num_class = pred_probs.shape[1])
    >>> # First compute confident thresholds (for faster results, can also do this on a random subset of your data):
    >>> i = 0
    >>> while i < n:
    >>>     end_index = i + batch_size
    >>>     labels_batch = labels[i:end_index]
    >>>     pred_probs_batch = pred_probs[i:end_index,:]
    >>>     i = end_index
    >>>     lab.update_confident_thresholds(labels_batch, pred_probs_batch)
    >>> # See what we calculated:
    >>> confident_thresholds = lab.get_confident_thresholds()
    >>> # Evaluate the quality of the labels (run this on full dataset you want to evaluate):
    >>> i = 0
    >>> while i < n:
    >>>     end_index = i + batch_size
    >>>     labels_batch = labels[i:end_index]
    >>>     pred_probs_batch = pred_probs[i:end_index,:]
    >>>     i = end_index
    >>>     batch_results = lab.score_label_quality(labels_batch, pred_probs_batch)
    >>> # Indices of examples with label issues, sorted by label quality score (most severe to least severe):
    >>> indices_of_examples_with_issues = lab.get_label_issues()
    >>> # If your `pred_probs` and `labels` are arrays already in memory,
    >>> # then you can use this shortcut for all of the above:
    >>> indices_of_examples_with_issues = find_label_issues_batched(labels, pred_probs, batch_size=10000)

    Parameters
    ----------
    num_class : int
      The number of classes in your multi-class classification task.

    store_results : bool, optional
      Whether this object will store all label quality scores, a 1D array of shape ``(N,)``
      where ``N`` is the total number of examples in your dataset.
      Set this to False if you encounter memory problems even for small batch sizes (~1000).
      If ``False``, you can still identify the label issues yourself by aggregating
      the label quality scores for each batch, sorting them across all batches, and returning the top ``T`` indices
      with ``T = self.get_num_issues()``.

    verbose : bool, optional
      Whether to suppress print statements or not.

    n_jobs: int, optional
      Number of processes for multiprocessing (default value = 1). Only used on Linux.
      If `n_jobs=None`, will use either the number of: physical cores if psutil is installed, or logical cores otherwise.

    quality_score_kwargs : dict, optional
      Keyword arguments to pass into :py:func:`rank.get_label_quality_scores <cleanlab.rank.get_label_quality_scores>`.

    num_issue_kwargs : dict, optional
      Keyword arguments to :py:func:`count.num_label_issues <cleanlab.count.num_label_issues>`
      to control estimation of the number of label issues.
      The only supported kwarg here for now is: `estimation_method`.
    """

    def __init__(self, *, num_class: int, store_results: bool=True, verbose: bool=True, quality_score_kwargs: Optional[dict]=None, num_issue_kwargs: Optional[dict]=None, n_jobs: Optional[int]=1):
        if False:
            i = 10
            return i + 15
        if quality_score_kwargs is None:
            quality_score_kwargs = {}
        if num_issue_kwargs is None:
            num_issue_kwargs = {}
        self.num_class = num_class
        self.store_results = store_results
        self.verbose = verbose
        self.quality_score_kwargs = quality_score_kwargs
        self.num_issue_kwargs = num_issue_kwargs
        self.off_diagonal_calibrated = False
        if num_issue_kwargs.get('estimation_method') == 'off_diagonal_calibrated':
            self.off_diagonal_calibrated = True
            self.prune_counts = np.zeros(self.num_class)
            self.class_counts = np.zeros(self.num_class)
            self.normalization = np.zeros(self.num_class)
        else:
            self.prune_count = 0
        if self.store_results:
            self.label_quality_scores: List[float] = []
        self.confident_thresholds = np.zeros((num_class,))
        self.examples_per_class = np.zeros((num_class,))
        self.examples_processed_thresh = 0
        self.examples_processed_quality = 0
        self.n_jobs: Optional[int] = None
        os_name = platform.system()
        if os_name != 'Linux':
            self.n_jobs = 1
            if n_jobs is not None and n_jobs != 1 and self.verbose:
                print('n_jobs is overridden to 1 because multiprocessing is only supported for Linux.')
        elif n_jobs is not None:
            self.n_jobs = n_jobs
        else:
            if PSUTIL_EXISTS:
                self.n_jobs = psutil.cpu_count(logical=False)
            if not self.n_jobs:
                self.n_jobs = mp.cpu_count()
                if self.verbose:
                    print(f'Multiprocessing will default to using the number of logical cores ({self.n_jobs}). To default to number of physical cores: pip install psutil')

    def get_confident_thresholds(self, silent: bool=False) -> np.ndarray:
        if False:
            i = 10
            return i + 15
        '\n        Fetches already-computed confident thresholds from the data seen so far\n        in same format as: :py:func:`count.get_confident_thresholds <cleanlab.count.get_confident_thresholds>`.\n\n\n        Returns\n        -------\n        confident_thresholds : np.ndarray\n          An array of shape ``(K, )`` where ``K`` is the number of classes.\n        '
        if self.examples_processed_thresh < 1:
            raise ValueError('Have not computed any confident_thresholds yet. Call `update_confident_thresholds()` first.')
        else:
            if self.verbose and (not silent):
                print(f'Total number of examples used to estimate confident thresholds: {self.examples_processed_thresh}')
            return self.confident_thresholds

    def get_num_issues(self, silent: bool=False) -> int:
        if False:
            while True:
                i = 10
        '\n        Fetches already-computed estimate of the number of label issues in the data seen so far\n        in the same format as: :py:func:`count.num_label_issues <cleanlab.count.num_label_issues>`.\n\n        Note: The estimated number of issues may differ from :py:func:`count.num_label_issues <cleanlab.count.num_label_issues>`\n        by 1 due to rounding differences.\n\n        Returns\n        -------\n        num_issues : int\n          The estimated number of examples with label issues in the data seen so far.\n        '
        if self.examples_processed_quality < 1:
            raise ValueError('Have not evaluated any labels yet. Call `score_label_quality()` first.')
        else:
            if self.verbose and (not silent):
                print(f'Total number of examples whose labels have been evaluated: {self.examples_processed_quality}')
            if self.off_diagonal_calibrated:
                calibrated_prune_counts = self.prune_counts * self.class_counts / np.clip(self.normalization, a_min=CLIPPING_LOWER_BOUND, a_max=None)
                return np.rint(np.sum(calibrated_prune_counts)).astype('int')
            else:
                return self.prune_count

    def get_quality_scores(self) -> np.ndarray:
        if False:
            while True:
                i = 10
        '\n        Fetches already-computed estimate of the label quality of each example seen so far\n        in the same format as: :py:func:`rank.get_label_quality_scores <cleanlab.rank.get_label_quality_scores>`.\n\n        Returns\n        -------\n        label_quality_scores : np.ndarray\n          Contains one score (between 0 and 1) per example seen so far.\n          Lower scores indicate more likely mislabeled examples.\n        '
        if not self.store_results:
            raise ValueError('Must initialize the LabelInspector with `store_results` == True. Otherwise you can assemble the label quality scores yourself based on the scores returned for each batch of data from `score_label_quality()`')
        else:
            return np.asarray(self.label_quality_scores)

    def get_label_issues(self) -> np.ndarray:
        if False:
            for i in range(10):
                print('nop')
        '\n        Fetches already-computed estimate of indices of examples with label issues in the data seen so far,\n        in the same format as: :py:func:`filter.find_label_issues <cleanlab.filter.find_label_issues>`\n        with its `return_indices_ranked_by` argument specified.\n\n        Note: this method corresponds to ``filter.find_label_issues(..., filter_by=METHOD1, return_indices_ranked_by=METHOD2)``\n        where by default: ``METHOD1="low_self_confidence"``, ``METHOD2="self_confidence"``\n        or if this object was instantiated with ``quality_score_kwargs = {"method": "normalized_margin"}`` then we instead have:\n        ``METHOD1="low_normalized_margin"``, ``METHOD2="normalized_margin"``.\n\n        Note: The estimated number of issues may differ from :py:func:`filter.find_label_issues <cleanlab.filter.find_label_issues>`\n        by 1 due to rounding differences.\n\n        Returns\n        -------\n        issue_indices : np.ndarray\n          Indices of examples with label issues, sorted by label quality score.\n        '
        if not self.store_results:
            raise ValueError('Must initialize the LabelInspector with `store_results` == True. Otherwise you can identify label issues yourself based on the scores from all the batches of data and the total number of issues returned by `get_num_issues()`')
        if self.examples_processed_quality < 1:
            raise ValueError('Have not evaluated any labels yet. Call `score_label_quality()` first.')
        if self.verbose:
            print(f'Total number of examples whose labels have been evaluated: {self.examples_processed_quality}')
        return find_top_issues(self.get_quality_scores(), top=self.get_num_issues(silent=True))

    def update_confident_thresholds(self, labels: LabelLike, pred_probs: np.ndarray):
        if False:
            i = 10
            return i + 15
        '\n        Updates the estimate of confident_thresholds stored in this class using a new batch of data.\n        Inputs should be in same format as for: :py:func:`count.get_confident_thresholds <cleanlab.count.get_confident_thresholds>`.\n\n        Parameters\n        ----------\n        labels: np.ndarray or list\n          Given class labels for each example in the batch, values in ``0,1,2,...,K-1``.\n\n        pred_probs: np.ndarray\n          2D array of model-predicted class probabilities for each example in the batch.\n        '
        labels = _batch_check(labels, pred_probs, self.num_class)
        batch_size = len(labels)
        batch_thresholds = get_confident_thresholds(labels, pred_probs)
        batch_class_counts = value_counts_fill_missing_classes(labels, num_classes=self.num_class)
        self.confident_thresholds = (self.examples_per_class * self.confident_thresholds + batch_class_counts * batch_thresholds) / np.clip(self.examples_per_class + batch_class_counts, a_min=1, a_max=None)
        self.confident_thresholds = np.clip(self.confident_thresholds, a_min=CONFIDENT_THRESHOLDS_LOWER_BOUND, a_max=None)
        self.examples_per_class += batch_class_counts
        self.examples_processed_thresh += batch_size

    def score_label_quality(self, labels: LabelLike, pred_probs: np.ndarray, *, update_num_issues: bool=True) -> np.ndarray:
        if False:
            for i in range(10):
                print('nop')
        '\n        Scores the label quality of each example in the provided batch of data,\n        and also updates the number of label issues stored in this class.\n        Inputs should be in same format as for: :py:func:`rank.get_label_quality_scores <cleanlab.rank.get_label_quality_scores>`.\n\n        Parameters\n        ----------\n        labels: np.ndarray\n          Given class labels for each example in the batch, values in ``0,1,2,...,K-1``.\n\n        pred_probs: np.ndarray\n          2D array of model-predicted class probabilities for each example in the batch of data.\n\n        update_num_issues: bool, optional\n          Whether or not to update the number of label issues or only compute label quality scores.\n          For lower runtimes, set this to ``False`` if you only want to score label quality and not find label issues.\n\n        Returns\n        -------\n        label_quality_scores : np.ndarray\n          Contains one score (between 0 and 1) for each example in the batch of data.\n        '
        labels = _batch_check(labels, pred_probs, self.num_class)
        batch_size = len(labels)
        scores = _compute_label_quality_scores(labels, pred_probs, confident_thresholds=self.get_confident_thresholds(silent=True), **self.quality_score_kwargs)
        class_counts = value_counts_fill_missing_classes(labels, num_classes=self.num_class)
        if update_num_issues:
            self._update_num_label_issues(labels, pred_probs, **self.num_issue_kwargs)
        self.examples_processed_quality += batch_size
        if self.store_results:
            self.label_quality_scores += list(scores)
        return scores

    def _update_num_label_issues(self, labels: LabelLike, pred_probs: np.ndarray, **kwargs):
        if False:
            return 10
        '\n        Update the estimate of num_label_issues stored in this class using a new batch of data.\n        Kwargs are ignored here for now (included for forwards compatibility).\n        Instead of being specified here, `estimation_method` should be declared when this class is initialized.\n        '
        thorough = False
        if self.examples_processed_thresh < 1:
            raise ValueError('Have not computed any confident_thresholds yet. Call `update_confident_thresholds()` first.')
        if self.n_jobs == 1:
            adj_confident_thresholds = self.confident_thresholds - FLOATING_POINT_COMPARISON
            pred_class = np.argmax(pred_probs, axis=1)
            batch_size = len(labels)
            if thorough:
                pred_gt_thresholds = pred_probs >= adj_confident_thresholds
                max_ind = np.argmax(pred_probs * pred_gt_thresholds, axis=1)
                if not self.off_diagonal_calibrated:
                    mask = (max_ind != labels) & (pred_class != labels)
                else:
                    mask = pred_class != labels
            else:
                max_ind = pred_class
                mask = pred_class != labels
            if not self.off_diagonal_calibrated:
                prune_count_batch = np.sum((pred_probs[np.arange(batch_size), max_ind] >= adj_confident_thresholds[max_ind]) & mask)
                self.prune_count += prune_count_batch
            else:
                self.class_counts += value_counts_fill_missing_classes(labels, num_classes=self.num_class)
                to_increment = pred_probs[np.arange(batch_size), max_ind] >= adj_confident_thresholds[max_ind]
                for class_label in range(self.num_class):
                    labels_equal_to_class = labels == class_label
                    self.normalization[class_label] += np.sum(labels_equal_to_class & to_increment)
                    self.prune_counts[class_label] += np.sum(labels_equal_to_class & to_increment & (max_ind != labels))
        else:
            global adj_confident_thresholds_shared
            adj_confident_thresholds_shared = self.confident_thresholds - FLOATING_POINT_COMPARISON
            global labels_shared, pred_probs_shared
            labels_shared = labels
            pred_probs_shared = pred_probs
            processes = 5000
            if len(labels) <= processes:
                chunksize = 1
            else:
                chunksize = len(labels) // processes
            inds = split_arr(np.arange(len(labels)), chunksize)
            if thorough:
                use_thorough = np.ones(len(inds), dtype=bool)
            else:
                use_thorough = np.zeros(len(inds), dtype=bool)
            args = zip(inds, use_thorough)
            with mp.Pool(self.n_jobs) as pool:
                if not self.off_diagonal_calibrated:
                    prune_count_batch = np.sum(np.asarray(list(pool.imap_unordered(_compute_num_issues, args))))
                    self.prune_count += prune_count_batch
                else:
                    results = list(pool.imap_unordered(_compute_num_issues_calibrated, args))
                    for result in results:
                        class_label = result[0]
                        self.class_counts[class_label] += 1
                        self.normalization[class_label] += result[1]
                        self.prune_counts[class_label] += result[2]

def split_arr(arr: np.ndarray, chunksize: int) -> List[np.ndarray]:
    if False:
        for i in range(10):
            print('nop')
    '\n    Helper function to split array into chunks for multiprocessing.\n    '
    return np.split(arr, np.arange(chunksize, arr.shape[0], chunksize), axis=0)

def _compute_num_issues(arg: Tuple[np.ndarray, bool]) -> int:
    if False:
        for i in range(10):
            print('nop')
    '\n    Helper function for `_update_num_label_issues` multiprocessing without calibration.\n    '
    ind = arg[0]
    thorough = arg[1]
    label = labels_shared[ind]
    pred_prob = pred_probs_shared[ind, :]
    pred_class = np.argmax(pred_prob, axis=-1)
    batch_size = len(label)
    if thorough:
        pred_gt_thresholds = pred_prob >= adj_confident_thresholds_shared
        max_ind = np.argmax(pred_prob * pred_gt_thresholds, axis=-1)
        prune_count_batch = np.sum((pred_prob[np.arange(batch_size), max_ind] >= adj_confident_thresholds_shared[max_ind]) & (max_ind != label) & (pred_class != label))
    else:
        prune_count_batch = np.sum((pred_prob[np.arange(batch_size), pred_class] >= adj_confident_thresholds_shared[pred_class]) & (pred_class != label))
    return prune_count_batch

def _compute_num_issues_calibrated(arg: Tuple[np.ndarray, bool]) -> Tuple[Any, int, int]:
    if False:
        for i in range(10):
            print('nop')
    '\n    Helper function for `_update_num_label_issues` multiprocessing with calibration.\n    '
    ind = arg[0]
    thorough = arg[1]
    label = labels_shared[ind]
    pred_prob = pred_probs_shared[ind, :]
    batch_size = len(label)
    pred_class = np.argmax(pred_prob, axis=-1)
    if thorough:
        pred_gt_thresholds = pred_prob >= adj_confident_thresholds_shared
        max_ind = np.argmax(pred_prob * pred_gt_thresholds, axis=-1)
        to_inc = pred_prob[np.arange(batch_size), max_ind] >= adj_confident_thresholds_shared[max_ind]
        prune_count_batch = to_inc & (max_ind != label)
        normalization_batch = to_inc
    else:
        to_inc = pred_prob[np.arange(batch_size), pred_class] >= adj_confident_thresholds_shared[pred_class]
        normalization_batch = to_inc
        prune_count_batch = to_inc & (pred_class != label)
    return (label, normalization_batch, prune_count_batch)

def _batch_check(labels: LabelLike, pred_probs: np.ndarray, num_class: int) -> np.ndarray:
    if False:
        print('Hello World!')
    '\n    Basic checks to ensure batch of data looks ok. For efficiency, this check is quite minimal.\n\n    Returns\n    -------\n    labels : np.ndarray\n      `labels` formatted as a 1D array.\n    '
    batch_size = pred_probs.shape[0]
    labels = np.asarray(labels)
    if len(labels) != batch_size:
        raise ValueError('labels and pred_probs must have same length')
    if pred_probs.shape[1] != num_class:
        raise ValueError('num_class must equal pred_probs.shape[1]')
    return labels