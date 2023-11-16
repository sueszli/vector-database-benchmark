"""
Methods for analysis of classification data labeled by multiple annotators.

To analyze a fixed dataset labeled by multiple annotators, use the
:py:func:`get_label_quality_multiannotator <cleanlab.multiannotator.get_label_quality_multiannotator>` function which estimates:

* A consensus label for each example that aggregates the individual annotations more accurately than alternative aggregation via majority-vote or other algorithms used in crowdsourcing like Dawid-Skene.
* A quality score for each consensus label which measures our confidence that this label is correct.
* An analogous label quality score for each individual label chosen by one annotator for a particular example.
* An overall quality score for each annotator which measures our confidence in the overall correctness of labels obtained from this annotator.

The algorithms to compute these estimates are described in `the CROWDLAB paper <https://arxiv.org/abs/2210.06812>`_.

If you have some labeled and unlabeled data (with multiple annotators for some labeled examples) and want to decide what data to collect additional labels for,
use the :py:func:`get_active_learning_scores <cleanlab.multiannotator.get_active_learning_scores>` function, which is intended for active learning. 
This function estimates an ActiveLab quality score for each example,
which can be used to prioritize which examples are most informative to collect additional labels for.
This function is effective for settings where some examples have been labeled by one or more annotators and other examples can have no labels at all so far,
as well as settings where new labels are collected either in batches of examples or one at a time. 
Here is an `example notebook <https://github.com/cleanlab/examples/blob/master/active_learning_multiannotator/active_learning.ipynb>`_ showcasing the use of this ActiveLab method for active learning with data re-labeling.

The algorithms to compute these active learning scores are described in `the ActiveLab paper <https://arxiv.org/abs/2301.11856>`_.

Each of the main functions in this module utilizes any trained classifier model.
Variants of these functions are provided for settings where you have trained an ensemble of multiple models.
"""
import warnings
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Union, Tuple, Optional
from cleanlab.rank import get_label_quality_scores
from cleanlab.internal.util import get_num_classes, value_counts
from cleanlab.internal.constants import CLIPPING_LOWER_BOUND
from cleanlab.internal.multiannotator_utils import assert_valid_inputs_multiannotator, assert_valid_pred_probs, check_consensus_label_classes, find_best_temp_scaler, temp_scale_pred_probs

def get_label_quality_multiannotator(labels_multiannotator: Union[pd.DataFrame, np.ndarray], pred_probs: np.ndarray, *, consensus_method: Union[str, List[str]]='best_quality', quality_method: str='crowdlab', calibrate_probs: bool=False, return_detailed_quality: bool=True, return_annotator_stats: bool=True, return_weights: bool=False, verbose: bool=True, label_quality_score_kwargs: dict={}) -> Dict[str, Any]:
    if False:
        return 10
    'Returns label quality scores for each example and for each annotator in a dataset labeled by multiple annotators.\n\n    This function is for multiclass classification datasets where examples have been labeled by\n    multiple annotators (not necessarily the same number of annotators per example).\n\n    It computes one consensus label for each example that best accounts for the labels chosen by each\n    annotator (and their quality), as well as a consensus quality score for how confident we are that this consensus label is actually correct.\n    It also computes similar quality scores for each annotator\'s individual labels, and the quality of each annotator.\n    Scores are between 0 and 1 (estimated via methods like CROWDLAB); lower scores indicate labels/annotators less likely to be correct.\n\n    To decide what data to collect additional labels for, try the :py:func:`get_active_learning_scores <cleanlab.multiannotator.get_active_learning_scores>`\n    (ActiveLab) function, which is intended for active learning with multiple annotators.\n\n    Parameters\n    ----------\n    labels_multiannotator : pd.DataFrame or np.ndarray\n        2D pandas DataFrame or array of multiple given labels for each example with shape ``(N, M)``,\n        where N is the number of examples and M is the number of annotators.\n        ``labels_multiannotator[n][m]`` = label for n-th example given by m-th annotator.\n\n        For a dataset with K classes, each given label must be an integer in 0, 1, ..., K-1 or ``NaN`` if this annotator did not label a particular example.\n        If you have string or other differently formatted labels, you can convert them to the proper format using :py:func:`format_multiannotator_labels <cleanlab.internal.multiannotator_utils.format_multiannotator_labels>`.\n        If pd.DataFrame, column names should correspond to each annotator\'s ID.\n    pred_probs : np.ndarray\n        An array of shape ``(N, K)`` of predicted class probabilities from a trained classifier model.\n        Predicted probabilities in the same format expected by the :py:func:`get_label_quality_scores <cleanlab.rank.get_label_quality_scores>`.\n    consensus_method : str or List[str], default = "majority_vote"\n        Specifies the method used to aggregate labels from multiple annotators into a single consensus label.\n        Options include:\n\n        * ``majority_vote``: consensus obtained using a simple majority vote among annotators, with ties broken via ``pred_probs``.\n        * ``best_quality``: consensus obtained by selecting the label with highest label quality (quality determined by method specified in ``quality_method``).\n\n        A List may be passed if you want to consider multiple methods for producing consensus labels.\n        If a List is passed, then the 0th element of the list is the method used to produce columns `consensus_label`, `consensus_quality_score`, `annotator_agreement` in the returned DataFrame.\n        The remaning (1st, 2nd, 3rd, etc.) elements of this list are output as extra columns in the returned pandas DataFrame with names formatted as:\n        `consensus_label_SUFFIX`, `consensus_quality_score_SUFFIX` where `SUFFIX` = each element of this\n        list, which must correspond to a valid method for computing consensus labels.\n    quality_method : str, default = "crowdlab"\n        Specifies the method used to calculate the quality of the consensus label.\n        Options include:\n\n        * ``crowdlab``: an emsemble method that weighs both the annotators\' labels as well as the model\'s prediction.\n        * ``agreement``: the fraction of annotators that agree with the consensus label.\n    calibrate_probs : bool, default = False\n        Boolean value that specifies whether the provided `pred_probs` should be re-calibrated to better match the annotators\' empirical label distribution.\n        We recommend setting this to True in active learning applications, in order to prevent overconfident models from suggesting the wrong examples to collect labels for.\n    return_detailed_quality: bool, default = True\n        Boolean to specify if `detailed_label_quality` is returned.\n    return_annotator_stats : bool, default = True\n        Boolean to specify if `annotator_stats` is returned.\n    return_weights : bool, default = False\n        Boolean to specify if `model_weight` and `annotator_weight` is returned.\n        Model and annotator weights are applicable for ``quality_method == crowdlab``, will return ``None`` for any other quality methods.\n    verbose : bool, default = True\n        Important warnings and other printed statements may be suppressed if ``verbose`` is set to ``False``.\n    label_quality_score_kwargs : dict, optional\n        Keyword arguments to pass into :py:func:`get_label_quality_scores <cleanlab.rank.get_label_quality_scores>`.\n\n    Returns\n    -------\n    labels_info : dict\n        Dictionary containing up to 5 pandas DataFrame with keys as below:\n\n        ``label_quality`` : pandas.DataFrame\n            pandas DataFrame in which each row corresponds to one example, with columns:\n\n            * ``num_annotations``: the number of annotators that have labeled each example.\n            * ``consensus_label``: the single label that is best for each example (you can control how it is derived from all annotators\' labels via the argument: ``consensus_method``).\n            * ``annotator_agreement``: the fraction of annotators that agree with the consensus label (only consider the annotators that labeled that particular example).\n            * ``consensus_quality_score``: label quality score for consensus label, calculated by the method specified in ``quality_method``.\n\n        ``detailed_label_quality`` : pandas.DataFrame\n            Only returned if `return_detailed_quality=True`.\n            Returns a pandas DataFrame with columns `quality_annotator_1`, `quality_annotator_2`, ..., `quality_annotator_M` where each entry is\n            the label quality score for the labels provided by each annotator (is ``NaN`` for examples which this annotator did not label).\n\n        ``annotator_stats`` : pandas.DataFrame\n            Only returned if `return_annotator_stats=True`.\n            Returns overall statistics about each annotator, sorted by lowest annotator_quality first.\n            pandas DataFrame in which each row corresponds to one annotator (the row IDs correspond to annotator IDs), with columns:\n\n            * ``annotator_quality``: overall quality of a given annotator\'s labels, calculated by the method specified in ``quality_method``.\n            * ``num_examples_labeled``: number of examples annotated by a given annotator.\n            * ``agreement_with_consensus``: fraction of examples where a given annotator agrees with the consensus label.\n            * ``worst_class``: the class that is most frequently mislabeled by a given annotator.\n\n        ``model_weight`` : float\n            Only returned if `return_weights=True`. It is only applicable for ``quality_method == crowdlab``.\n            The model weight specifies the weight of classifier model in weighted averages used to estimate label quality\n            This number is an estimate of how trustworthy the model is relative the annotators.\n\n        ``annotator_weight`` : np.ndarray\n            Only returned if `return_weights=True`. It is only applicable for ``quality_method == crowdlab``.\n            An array of shape ``(M,)`` where M is the number of annotators, specifying the weight of each annotator in weighted averages used to estimate label quality.\n            These weights are estimates of how trustworthy each annotator is relative to the other annotators.\n\n    '
    if isinstance(labels_multiannotator, pd.DataFrame):
        annotator_ids = labels_multiannotator.columns
        index_col = labels_multiannotator.index
        labels_multiannotator = labels_multiannotator.replace({pd.NA: np.NaN}).astype(float).to_numpy()
    elif isinstance(labels_multiannotator, np.ndarray):
        annotator_ids = None
        index_col = None
    else:
        raise ValueError('labels_multiannotator must be either a NumPy array or Pandas DataFrame.')
    if return_weights == True and quality_method != 'crowdlab':
        raise ValueError("Model and annotator weights are only applicable to the crowdlab quality method. Either set return_weights=False or quality_method='crowdlab'.")
    assert_valid_inputs_multiannotator(labels_multiannotator, pred_probs, annotator_ids=annotator_ids)
    num_annotations = np.sum(~np.isnan(labels_multiannotator), axis=1)
    if calibrate_probs:
        optimal_temp = find_best_temp_scaler(labels_multiannotator, pred_probs)
        pred_probs = temp_scale_pred_probs(pred_probs, optimal_temp)
    if not isinstance(consensus_method, list):
        consensus_method = [consensus_method]
    if 'best_quality' in consensus_method or 'majority_vote' in consensus_method:
        majority_vote_label = get_majority_vote_label(labels_multiannotator=labels_multiannotator, pred_probs=pred_probs, verbose=False)
        (MV_annotator_agreement, MV_consensus_quality_score, MV_post_pred_probs, MV_model_weight, MV_annotator_weight) = _get_consensus_stats(labels_multiannotator=labels_multiannotator, pred_probs=pred_probs, num_annotations=num_annotations, consensus_label=majority_vote_label, quality_method=quality_method, verbose=verbose, label_quality_score_kwargs=label_quality_score_kwargs)
    label_quality = pd.DataFrame({'num_annotations': num_annotations}, index=index_col)
    valid_methods = ['majority_vote', 'best_quality']
    main_method = True
    for curr_method in consensus_method:
        if curr_method == 'majority_vote':
            consensus_label = majority_vote_label
            annotator_agreement = MV_annotator_agreement
            consensus_quality_score = MV_consensus_quality_score
            post_pred_probs = MV_post_pred_probs
            model_weight = MV_model_weight
            annotator_weight = MV_annotator_weight
        elif curr_method == 'best_quality':
            consensus_label = np.full(len(majority_vote_label), np.nan)
            for i in range(len(consensus_label)):
                max_pred_probs_ind = np.where(MV_post_pred_probs[i] == np.max(MV_post_pred_probs[i]))[0]
                if len(max_pred_probs_ind) == 1:
                    consensus_label[i] = max_pred_probs_ind[0]
                else:
                    consensus_label[i] = majority_vote_label[i]
            consensus_label = consensus_label.astype(int)
            (annotator_agreement, consensus_quality_score, post_pred_probs, model_weight, annotator_weight) = _get_consensus_stats(labels_multiannotator=labels_multiannotator, pred_probs=pred_probs, num_annotations=num_annotations, consensus_label=consensus_label, quality_method=quality_method, verbose=verbose, label_quality_score_kwargs=label_quality_score_kwargs)
        else:
            raise ValueError(f'\n                {curr_method} is not a valid consensus method!\n                Please choose a valid consensus_method: {valid_methods}\n                ')
        if verbose:
            check_consensus_label_classes(labels_multiannotator=labels_multiannotator, consensus_label=consensus_label, consensus_method=curr_method)
        if main_method:
            (label_quality['consensus_label'], label_quality['consensus_quality_score'], label_quality['annotator_agreement']) = (consensus_label, consensus_quality_score, annotator_agreement)
            label_quality = label_quality.reindex(columns=['consensus_label', 'consensus_quality_score', 'annotator_agreement', 'num_annotations'])
            detailed_label_quality = None
            if return_detailed_quality:
                detailed_label_quality = np.apply_along_axis(_get_annotator_label_quality_score, axis=0, arr=labels_multiannotator, pred_probs=post_pred_probs, label_quality_score_kwargs=label_quality_score_kwargs)
                detailed_label_quality_df = pd.DataFrame(detailed_label_quality, index=index_col, columns=annotator_ids).add_prefix('quality_annotator_')
            if return_annotator_stats:
                annotator_stats = _get_annotator_stats(labels_multiannotator=labels_multiannotator, pred_probs=post_pred_probs, consensus_label=consensus_label, num_annotations=num_annotations, annotator_agreement=annotator_agreement, model_weight=model_weight, annotator_weight=annotator_weight, consensus_quality_score=consensus_quality_score, detailed_label_quality=detailed_label_quality, annotator_ids=annotator_ids, quality_method=quality_method)
            main_method = False
        else:
            (label_quality[f'consensus_label_{curr_method}'], label_quality[f'consensus_quality_score_{curr_method}'], label_quality[f'annotator_agreement_{curr_method}']) = (consensus_label, consensus_quality_score, annotator_agreement)
    labels_info = {'label_quality': label_quality}
    if return_detailed_quality:
        labels_info['detailed_label_quality'] = detailed_label_quality_df
    if return_annotator_stats:
        labels_info['annotator_stats'] = annotator_stats
    if return_weights:
        labels_info['model_weight'] = model_weight
        labels_info['annotator_weight'] = annotator_weight
    return labels_info

def get_label_quality_multiannotator_ensemble(labels_multiannotator: Union[pd.DataFrame, np.ndarray], pred_probs: np.ndarray, *, calibrate_probs: bool=False, return_detailed_quality: bool=True, return_annotator_stats: bool=True, return_weights: bool=False, verbose: bool=True, label_quality_score_kwargs: dict={}) -> Dict[str, Any]:
    if False:
        return 10
    'Returns label quality scores for each example and for each annotator, based on predictions from an ensemble of models.\n\n    This function is similar to :py:func:`get_label_quality_multiannotator <cleanlab.multiannotator.get_label_quality_multiannotator>` but for settings where\n    you have trained an ensemble of multiple classifier models rather than a single model.\n\n    Parameters\n    ----------\n    labels_multiannotator : pd.DataFrame or np.ndarray\n        Multiannotator labels in the same format expected by :py:func:`get_label_quality_multiannotator <cleanlab.multiannotator.get_label_quality_multiannotator>`.\n    pred_probs : np.ndarray\n        An array of shape ``(P, N, K)`` where P is the number of models, consisting of predicted class probabilities from the ensemble models.\n        Each set of predicted probabilities with shape ``(N, K)`` is in the same format expected by the :py:func:`get_label_quality_scores <cleanlab.rank.get_label_quality_scores>`.\n    calibrate_probs : bool, default = False\n        Boolean value as expected by :py:func:`get_label_quality_multiannotator <cleanlab.multiannotator.get_label_quality_multiannotator>`.\n    return_detailed_quality: bool, default = True\n        Boolean value as expected by :py:func:`get_label_quality_multiannotator <cleanlab.multiannotator.get_label_quality_multiannotator>`.\n    return_annotator_stats : bool, default = True\n        Boolean value as expected by :py:func:`get_label_quality_multiannotator <cleanlab.multiannotator.get_label_quality_multiannotator>`.\n    return_weights : bool, default = False\n        Boolean value as expected by :py:func:`get_label_quality_multiannotator <cleanlab.multiannotator.get_label_quality_multiannotator>`.\n    verbose : bool, default = True\n        Boolean value as expected by :py:func:`get_label_quality_multiannotator <cleanlab.multiannotator.get_label_quality_multiannotator>`.\n    label_quality_score_kwargs : dict, optional\n        Keyword arguments in the same format expected by py:func:`get_label_quality_multiannotator <cleanlab.multiannotator.get_label_quality_multiannotator>`.\n\n    Returns\n    -------\n    labels_info : dict\n        Dictionary containing up to 5 pandas DataFrame with keys as below:\n\n        ``label_quality`` : pandas.DataFrame\n            Similar to output as :py:func:`get_label_quality_multiannotator <cleanlab.multiannotator.get_label_quality_multiannotator>`.\n\n        ``detailed_label_quality`` : pandas.DataFrame\n            Similar to output as :py:func:`get_label_quality_multiannotator <cleanlab.multiannotator.get_label_quality_multiannotator>`.\n\n        ``annotator_stats`` : pandas.DataFrame\n            Similar to output as :py:func:`get_label_quality_multiannotator <cleanlab.multiannotator.get_label_quality_multiannotator>`.\n\n        ``model_weight`` : np.ndarray\n            Only returned if `return_weights=True`.\n            An array of shape ``(P,)`` where is the number of models in the ensemble, specifying the weight of each classifier model in weighted averages used to estimate label quality.\n            These weigthts is an estimate of how trustworthy the model is relative the annotators.\n            An array of shape ``(P,)`` where is the number of models in the ensemble, specifying the model weight used in weighted averages.\n\n        ``annotator_weight`` : np.ndarray\n            Only returned if `return_weights=True`.\n            Similar to output as :py:func:`get_label_quality_multiannotator <cleanlab.multiannotator.get_label_quality_multiannotator>`.\n\n    See Also\n    --------\n    get_label_quality_multiannotator\n    '
    if isinstance(labels_multiannotator, pd.DataFrame):
        annotator_ids = labels_multiannotator.columns
        index_col = labels_multiannotator.index
        labels_multiannotator = labels_multiannotator.replace({pd.NA: np.NaN}).astype(float).to_numpy()
    elif isinstance(labels_multiannotator, np.ndarray):
        annotator_ids = None
        index_col = None
    else:
        raise ValueError('labels_multiannotator must be either a NumPy array or Pandas DataFrame.')
    assert_valid_inputs_multiannotator(labels_multiannotator, pred_probs, ensemble=True, annotator_ids=annotator_ids)
    num_annotations = np.sum(~np.isnan(labels_multiannotator), axis=1)
    if calibrate_probs:
        for i in range(len(pred_probs)):
            curr_pred_probs = pred_probs[i]
            optimal_temp = find_best_temp_scaler(labels_multiannotator, curr_pred_probs)
            pred_probs[i] = temp_scale_pred_probs(curr_pred_probs, optimal_temp)
    label_quality = pd.DataFrame({'num_annotations': num_annotations}, index=index_col)
    avg_pred_probs = np.mean(pred_probs, axis=0)
    majority_vote_label = get_majority_vote_label(labels_multiannotator=labels_multiannotator, pred_probs=avg_pred_probs, verbose=False)
    (MV_annotator_agreement, MV_consensus_quality_score, MV_post_pred_probs, MV_model_weight, MV_annotator_weight) = _get_consensus_stats(labels_multiannotator=labels_multiannotator, pred_probs=pred_probs, num_annotations=num_annotations, consensus_label=majority_vote_label, verbose=verbose, ensemble=True, **label_quality_score_kwargs)
    consensus_label = np.full(len(majority_vote_label), np.nan)
    for i in range(len(consensus_label)):
        max_pred_probs_ind = np.where(MV_post_pred_probs[i] == np.max(MV_post_pred_probs[i]))[0]
        if len(max_pred_probs_ind) == 1:
            consensus_label[i] = max_pred_probs_ind[0]
        else:
            consensus_label[i] = majority_vote_label[i]
    consensus_label = consensus_label.astype(int)
    (annotator_agreement, consensus_quality_score, post_pred_probs, model_weight, annotator_weight) = _get_consensus_stats(labels_multiannotator=labels_multiannotator, pred_probs=pred_probs, num_annotations=num_annotations, consensus_label=consensus_label, verbose=verbose, ensemble=True, **label_quality_score_kwargs)
    if verbose:
        check_consensus_label_classes(labels_multiannotator=labels_multiannotator, consensus_label=consensus_label, consensus_method='crowdlab')
    (label_quality['consensus_label'], label_quality['consensus_quality_score'], label_quality['annotator_agreement']) = (consensus_label, consensus_quality_score, annotator_agreement)
    label_quality = label_quality.reindex(columns=['consensus_label', 'consensus_quality_score', 'annotator_agreement', 'num_annotations'])
    detailed_label_quality = None
    if return_detailed_quality:
        detailed_label_quality = np.apply_along_axis(_get_annotator_label_quality_score, axis=0, arr=labels_multiannotator, pred_probs=post_pred_probs, label_quality_score_kwargs=label_quality_score_kwargs)
        detailed_label_quality_df = pd.DataFrame(detailed_label_quality, index=index_col, columns=annotator_ids).add_prefix('quality_annotator_')
    if return_annotator_stats:
        annotator_stats = _get_annotator_stats(labels_multiannotator=labels_multiannotator, pred_probs=post_pred_probs, consensus_label=consensus_label, num_annotations=num_annotations, annotator_agreement=annotator_agreement, model_weight=np.mean(model_weight), annotator_weight=annotator_weight, consensus_quality_score=consensus_quality_score, detailed_label_quality=detailed_label_quality, annotator_ids=annotator_ids)
    labels_info = {'label_quality': label_quality}
    if return_detailed_quality:
        labels_info['detailed_label_quality'] = detailed_label_quality_df
    if return_annotator_stats:
        labels_info['annotator_stats'] = annotator_stats
    if return_weights:
        labels_info['model_weight'] = model_weight
        labels_info['annotator_weight'] = annotator_weight
    return labels_info

def get_active_learning_scores(labels_multiannotator: Optional[Union[pd.DataFrame, np.ndarray]]=None, pred_probs: Optional[np.ndarray]=None, pred_probs_unlabeled: Optional[np.ndarray]=None) -> Tuple[np.ndarray, np.ndarray]:
    if False:
        while True:
            i = 10
    'Returns an ActiveLab quality score for each example in the dataset, to estimate which examples are most informative to (re)label next in active learning.\n\n    We consider settings where one example can be labeled by one or more annotators and some examples have no labels at all so far.\n\n    The score is in between 0 and 1, and can be used to prioritize what data to collect additional labels for.\n    Lower scores indicate examples whose true label we are least confident about based on the current data;\n    collecting additional labels for these low-scoring examples will be more informative than collecting labels for other examples.\n    To use an annotation budget most efficiently, select a batch of examples with the lowest scores and collect one additional label for each example,\n    and repeat this process after retraining your classifier.\n\n    You can use this function to get active learning scores for: examples that already have one or more labels (specify ``labels_multiannotator`` and ``pred_probs``\n    as arguments), or for unlabeled examples (specify ``pred_probs_unlabeled``), or for both types of examples (specify all of the above arguments).\n\n    To analyze a fixed dataset labeled by multiple annotators rather than collecting additional labels, try the\n    :py:func:`get_label_quality_multiannotator <cleanlab.multiannotator.get_label_quality_multiannotator>` (CROWDLAB) function instead.\n\n    Parameters\n    ----------\n    labels_multiannotator : pd.DataFrame or np.ndarray, optional\n        2D pandas DataFrame or array of multiple given labels for each example with shape ``(N, M)``,\n        where N is the number of examples and M is the number of annotators. Note that this function also works with\n        datasets where there is only one annotator (M=1).\n        For more details, labels in the same format expected by the :py:func:`get_label_quality_multiannotator <cleanlab.multiannotator.get_label_quality_multiannotator>`.\n        Note that examples that have no annotator labels should not be included in this DataFrame/array.\n        This argument is optional if ``pred_probs`` is not provided (you might only provide ``pred_probs_unlabeled`` to only get active learning scores for the unlabeled examples).\n    pred_probs : np.ndarray, optional\n        An array of shape ``(N, K)`` of predicted class probabilities from a trained classifier model.\n        Predicted probabilities in the same format expected by the :py:func:`get_label_quality_scores <cleanlab.rank.get_label_quality_scores>`.\n        This argument is optional if you only want to get active learning scores for unlabeled examples (specify only ``pred_probs_unlabeled`` instead).\n    pred_probs_unlabeled : np.ndarray, optional\n        An array of shape ``(N, K)`` of predicted class probabilities from a trained classifier model for examples that have no annotator labels.\n        Predicted probabilities in the same format expected by the :py:func:`get_label_quality_scores <cleanlab.rank.get_label_quality_scores>`.\n        This argument is optional if you only want to get active learning scores for already-labeled examples (specify only ``pred_probs`` instead).\n\n    Returns\n    -------\n    active_learning_scores : np.ndarray\n        Array of shape ``(N,)`` indicating the ActiveLab quality scores for each example.\n        This array is empty if no already-labeled data was provided via ``labels_multiannotator``.\n        Examples with the lowest scores are those we should label next in order to maximally improve our classifier model.\n\n    active_learning_scores_unlabeled : np.ndarray\n        Array of shape ``(N,)`` indicating the active learning quality scores for each unlabeled example.\n        Returns an empty array if no unlabeled data is provided.\n        Examples with the lowest scores are those we should label next in order to maximally improve our classifier model\n        (scores for unlabeled data are directly comparable with the `active_learning_scores` for labeled data).\n    '
    assert_valid_pred_probs(pred_probs=pred_probs, pred_probs_unlabeled=pred_probs_unlabeled)
    if pred_probs is not None:
        if labels_multiannotator is None:
            raise ValueError('labels_multiannotator cannot be None when passing in pred_probs. ', 'Either provide labels_multiannotator to obtain active learning scores for the labeled examples, or just pass in pred_probs_unlabeled to get active learning scores for unlabeled examples.')
        if isinstance(labels_multiannotator, pd.DataFrame):
            labels_multiannotator = labels_multiannotator.replace({pd.NA: np.NaN}).astype(float).to_numpy()
        elif not isinstance(labels_multiannotator, np.ndarray):
            raise ValueError('labels_multiannotator must be either a NumPy array or Pandas DataFrame.')
        if labels_multiannotator.ndim != 2:
            raise ValueError('labels_multiannotator must be a 2D array or dataframe, each row represents an example and each column represents an annotator.')
        num_classes = get_num_classes(pred_probs=pred_probs)
        if (np.sum(~np.isnan(labels_multiannotator), axis=1) == 1).all():
            optimal_temp = 1.0
            assert_valid_inputs_multiannotator(labels_multiannotator, pred_probs, allow_single_label=True)
            consensus_label = get_majority_vote_label(labels_multiannotator=labels_multiannotator, pred_probs=pred_probs, verbose=False)
            quality_of_consensus_labeled = get_label_quality_scores(consensus_label, pred_probs)
            model_weight = 1
            annotator_weight = np.full(labels_multiannotator.shape[1], 1)
            avg_annotator_weight = np.mean(annotator_weight)
        else:
            optimal_temp = find_best_temp_scaler(labels_multiannotator, pred_probs)
            pred_probs = temp_scale_pred_probs(pred_probs, optimal_temp)
            multiannotator_info = get_label_quality_multiannotator(labels_multiannotator, pred_probs, return_annotator_stats=False, return_detailed_quality=False, return_weights=True)
            quality_of_consensus_labeled = multiannotator_info['label_quality']['consensus_quality_score']
            model_weight = multiannotator_info['model_weight']
            annotator_weight = multiannotator_info['annotator_weight']
            avg_annotator_weight = np.mean(annotator_weight)
        active_learning_scores = np.full(len(labels_multiannotator), np.nan)
        for (i, annotator_labels) in enumerate(labels_multiannotator):
            active_learning_scores[i] = np.average((quality_of_consensus_labeled[i], 1 / num_classes), weights=(np.sum(annotator_weight[~np.isnan(annotator_labels)]) + model_weight, avg_annotator_weight))
    elif pred_probs_unlabeled is not None:
        num_classes = get_num_classes(pred_probs=pred_probs_unlabeled)
        optimal_temp = 1
        model_weight = 1
        avg_annotator_weight = 1
        active_learning_scores = np.array([])
    else:
        raise ValueError('pred_probs and pred_probs_unlabeled cannot both be None, specify at least one of the two.')
    if pred_probs_unlabeled is not None:
        pred_probs_unlabeled = temp_scale_pred_probs(pred_probs_unlabeled, optimal_temp)
        quality_of_consensus_unlabeled = np.max(pred_probs_unlabeled, axis=1)
        active_learning_scores_unlabeled = np.average(np.stack([quality_of_consensus_unlabeled, np.full(len(quality_of_consensus_unlabeled), 1 / num_classes)]), weights=[model_weight, avg_annotator_weight], axis=0)
    else:
        active_learning_scores_unlabeled = np.array([])
    return (active_learning_scores, active_learning_scores_unlabeled)

def get_active_learning_scores_ensemble(labels_multiannotator: Optional[Union[pd.DataFrame, np.ndarray]]=None, pred_probs: Optional[np.ndarray]=None, pred_probs_unlabeled: Optional[np.ndarray]=None) -> Tuple[np.ndarray, np.ndarray]:
    if False:
        i = 10
        return i + 15
    'Returns an ActiveLab quality score for each example in the dataset, based on predictions from an ensemble of models.\n\n    This function is similar to :py:func:`get_active_learning_scores <cleanlab.multiannotator.get_active_learning_scores>` but allows for an\n    ensemble of multiple classifier models to be trained and will aggregate predictions from the models to compute the ActiveLab quality score.\n\n    Parameters\n    ----------\n    labels_multiannotator : pd.DataFrame or np.ndarray\n        Multiannotator labels in the same format expected by :py:func:`get_active_learning_scores <cleanlab.multiannotator.get_active_learning_scores>`.\n        This argument is optional if ``pred_probs`` is not provided (in cases where you only provide ``pred_probs_unlabeled`` to get active learning scores for unlabeled examples).\n    pred_probs : np.ndarray\n        An array of shape ``(P, N, K)`` where P is the number of models, consisting of predicted class probabilities from the ensemble models.\n        Note that this function also works with datasets where there is only one annotator (M=1).\n        Each set of predicted probabilities with shape ``(N, K)`` is in the same format expected by the :py:func:`get_label_quality_scores <cleanlab.rank.get_label_quality_scores>`.\n        This argument is optional if you only want to get active learning scores for unlabeled examples (pass in ``pred_probs_unlabeled`` instead).\n    pred_probs_unlabeled : np.ndarray, optional\n        An array of shape ``(P, N, K)`` where P is the number of models, consisting of predicted class probabilities from a trained classifier model\n        for examples that have no annotated labels so far (but which we may want to label in the future, and hence compute active learning quality scores for).\n        Each set of predicted probabilities with shape ``(N, K)`` is in the same format expected by the :py:func:`get_label_quality_scores <cleanlab.rank.get_label_quality_scores>`.\n        This argument is optional if you only want to get active learning scores for labeled examples (pass in ``pred_probs`` instead).\n\n    Returns\n    -------\n    active_learning_scores : np.ndarray\n        Similar to output as :py:func:`get_label_quality_scores <cleanlab.multiannotator.get_label_quality_scores>`.\n    active_learning_scores_unlabeled : np.ndarray\n        Similar to output as :py:func:`get_label_quality_scores <cleanlab.multiannotator.get_label_quality_scores>`.\n\n    See Also\n    --------\n    get_active_learning_scores\n    '
    assert_valid_pred_probs(pred_probs=pred_probs, pred_probs_unlabeled=pred_probs_unlabeled, ensemble=True)
    if pred_probs is not None:
        if labels_multiannotator is None:
            raise ValueError('labels_multiannotator cannot be None when passing in pred_probs. ', 'You can either provide labels_multiannotator to obtain active learning scores for the labeled examples, or just pass in pred_probs_unlabeled to get active learning scores for unlabeled examples.')
        if isinstance(labels_multiannotator, pd.DataFrame):
            labels_multiannotator = labels_multiannotator.replace({pd.NA: np.NaN}).astype(float).to_numpy()
        elif not isinstance(labels_multiannotator, np.ndarray):
            raise ValueError('labels_multiannotator must be either a NumPy array or Pandas DataFrame.')
        if labels_multiannotator.ndim != 2:
            raise ValueError('labels_multiannotator must be a 2D array or dataframe, each row represents an example and each column represents an annotator.')
        num_classes = get_num_classes(pred_probs=pred_probs[0])
        if (np.sum(~np.isnan(labels_multiannotator), axis=1) == 1).all():
            optimal_temp = np.full(len(pred_probs), 1.0)
            assert_valid_inputs_multiannotator(labels_multiannotator, pred_probs, ensemble=True, allow_single_label=True)
            avg_pred_probs = np.mean(pred_probs, axis=0)
            consensus_label = get_majority_vote_label(labels_multiannotator=labels_multiannotator, pred_probs=avg_pred_probs, verbose=False)
            quality_of_consensus_labeled = get_label_quality_scores(consensus_label, avg_pred_probs)
            model_weight = np.full(len(pred_probs), 1)
            annotator_weight = np.full(labels_multiannotator.shape[1], 1)
            avg_annotator_weight = np.mean(annotator_weight)
        else:
            optimal_temp = np.full(len(pred_probs), np.NaN)
            for (i, curr_pred_probs) in enumerate(pred_probs):
                curr_optimal_temp = find_best_temp_scaler(labels_multiannotator, curr_pred_probs)
                pred_probs[i] = temp_scale_pred_probs(curr_pred_probs, curr_optimal_temp)
                optimal_temp[i] = curr_optimal_temp
            multiannotator_info = get_label_quality_multiannotator_ensemble(labels_multiannotator, pred_probs, return_annotator_stats=False, return_detailed_quality=False, return_weights=True)
            quality_of_consensus_labeled = multiannotator_info['label_quality']['consensus_quality_score']
            model_weight = multiannotator_info['model_weight']
            annotator_weight = multiannotator_info['annotator_weight']
            avg_annotator_weight = np.mean(annotator_weight)
        active_learning_scores = np.full(len(labels_multiannotator), np.nan)
        for (i, annotator_labels) in enumerate(labels_multiannotator):
            active_learning_scores[i] = np.average((quality_of_consensus_labeled[i], 1 / num_classes), weights=(np.sum(annotator_weight[~np.isnan(annotator_labels)]) + np.sum(model_weight), avg_annotator_weight))
    elif pred_probs_unlabeled is not None:
        num_classes = get_num_classes(pred_probs=pred_probs_unlabeled[0])
        optimal_temp = np.full(len(pred_probs_unlabeled), 1.0)
        model_weight = np.full(len(pred_probs_unlabeled), 1)
        avg_annotator_weight = 1
        active_learning_scores = np.array([])
    else:
        raise ValueError('pred_probs and pred_probs_unlabeled cannot both be None, specify at least one of the two.')
    if pred_probs_unlabeled is not None:
        for i in range(len(pred_probs_unlabeled)):
            pred_probs_unlabeled[i] = temp_scale_pred_probs(pred_probs_unlabeled[i], optimal_temp[i])
        avg_pred_probs_unlabeled = np.mean(pred_probs_unlabeled, axis=0)
        consensus_label_unlabeled = get_majority_vote_label(np.argmax(pred_probs_unlabeled, axis=2).T, avg_pred_probs_unlabeled)
        modified_pred_probs_unlabeled = np.average(np.concatenate((pred_probs_unlabeled, np.full(pred_probs_unlabeled.shape[1:], 1 / num_classes)[np.newaxis, :, :])), weights=np.concatenate((model_weight, np.array([avg_annotator_weight]))), axis=0)
        active_learning_scores_unlabeled = get_label_quality_scores(consensus_label_unlabeled, modified_pred_probs_unlabeled)
    else:
        active_learning_scores_unlabeled = np.array([])
    return (active_learning_scores, active_learning_scores_unlabeled)

def get_majority_vote_label(labels_multiannotator: Union[pd.DataFrame, np.ndarray], pred_probs: Optional[np.ndarray]=None, verbose: bool=True) -> np.ndarray:
    if False:
        return 10
    'Returns the majority vote label for each example, aggregated from the labels given by multiple annotators.\n\n    Parameters\n    ----------\n    labels_multiannotator : pd.DataFrame or np.ndarray\n        2D pandas DataFrame or array of multiple given labels for each example with shape ``(N, M)``,\n        where N is the number of examples and M is the number of annotators.\n        For more details, labels in the same format expected by the :py:func:`get_label_quality_multiannotator <cleanlab.multiannotator.get_label_quality_multiannotator>`.\n    pred_probs : np.ndarray, optional\n        An array of shape ``(N, K)`` of model-predicted probabilities, ``P(label=k|x)``.\n        For details, predicted probabilities in the same format expected by :py:func:`get_label_quality_multiannotator <cleanlab.multiannotator.get_label_quality_multiannotator>`.\n    verbose : bool, optional\n        Important warnings and other printed statements may be suppressed if ``verbose`` is set to ``False``.\n    Returns\n    -------\n    consensus_label: np.ndarray\n        An array of shape ``(N,)`` with the majority vote label aggregated from all annotators.\n\n        In the event of majority vote ties, ties are broken in the following order:\n        using the model ``pred_probs`` (if provided) and selecting the class with highest predicted probability,\n        using the empirical class frequencies and selecting the class with highest frequency,\n        using an initial annotator quality score and selecting the class that has been labeled by annotators with higher quality,\n        and lastly by random selection.\n    '
    if isinstance(labels_multiannotator, pd.DataFrame):
        annotator_ids = labels_multiannotator.columns
        labels_multiannotator = labels_multiannotator.replace({pd.NA: np.NaN}).astype(float).to_numpy()
    elif isinstance(labels_multiannotator, np.ndarray):
        annotator_ids = None
    else:
        raise ValueError('labels_multiannotator must be either a NumPy array or Pandas DataFrame.')
    if verbose:
        assert_valid_inputs_multiannotator(labels_multiannotator, pred_probs, annotator_ids=annotator_ids)
    if pred_probs is not None:
        num_classes = pred_probs.shape[1]
    else:
        num_classes = int(np.nanmax(labels_multiannotator) + 1)

    def get_labels_mode(label_count, num_classes):
        if False:
            i = 10
            return i + 15
        max_count_idx = np.where(label_count == np.nanmax(label_count))[0].astype(float)
        return np.pad(max_count_idx, (0, num_classes - len(max_count_idx)), 'constant', constant_values=np.NaN)
    majority_vote_label = np.full(len(labels_multiannotator), np.nan)
    label_count = np.apply_along_axis(lambda s: np.bincount(s[~np.isnan(s)].astype(int), minlength=num_classes), axis=1, arr=labels_multiannotator)
    mode_labels_multiannotator = np.apply_along_axis(get_labels_mode, axis=1, arr=label_count, num_classes=num_classes)
    nontied_idx = []
    tied_idx = dict()
    for (idx, label_mode) in enumerate(mode_labels_multiannotator):
        label_mode = label_mode[~np.isnan(label_mode)].astype(int)
        if len(label_mode) == 1:
            majority_vote_label[idx] = label_mode[0]
            nontied_idx.append(idx)
        else:
            tied_idx[idx] = label_mode
    if pred_probs is not None and len(tied_idx) > 0:
        for (idx, label_mode) in tied_idx.copy().items():
            max_pred_probs = np.where(pred_probs[idx, label_mode] == np.max(pred_probs[idx, label_mode]))[0]
            if len(max_pred_probs) == 1:
                majority_vote_label[idx] = label_mode[max_pred_probs[0]]
                del tied_idx[idx]
            else:
                tied_idx[idx] = label_mode[max_pred_probs]
    if len(tied_idx) > 0:
        class_frequencies = label_count.sum(axis=0)
        for (idx, label_mode) in tied_idx.copy().items():
            min_frequency = np.where(class_frequencies[label_mode] == np.min(class_frequencies[label_mode]))[0]
            if len(min_frequency) == 1:
                majority_vote_label[idx] = label_mode[min_frequency[0]]
                del tied_idx[idx]
            else:
                tied_idx[idx] = label_mode[min_frequency]
    if len(tied_idx) > 0:
        nontied_majority_vote_label = majority_vote_label[nontied_idx]
        nontied_labels_multiannotator = labels_multiannotator[nontied_idx]
        annotator_agreement_with_consensus = np.zeros(nontied_labels_multiannotator.shape[1])
        for i in range(len(annotator_agreement_with_consensus)):
            labels = nontied_labels_multiannotator[:, i]
            labels_mask = ~np.isnan(labels)
            if np.sum(labels_mask) == 0:
                annotator_agreement_with_consensus[i] = np.NaN
            else:
                annotator_agreement_with_consensus[i] = np.mean(labels[labels_mask] == nontied_majority_vote_label[labels_mask])
        nan_mask = np.isnan(annotator_agreement_with_consensus)
        avg_annotator_agreement = np.mean(annotator_agreement_with_consensus[~nan_mask])
        annotator_agreement_with_consensus[nan_mask] = avg_annotator_agreement
        for (idx, label_mode) in tied_idx.copy().items():
            label_quality_score = np.array([np.mean(annotator_agreement_with_consensus[np.where(labels_multiannotator[idx] == label)[0]]) for label in label_mode])
            max_score = np.where(label_quality_score == label_quality_score.max())[0]
            if len(max_score) == 1:
                majority_vote_label[idx] = label_mode[max_score[0]]
                del tied_idx[idx]
            else:
                tied_idx[idx] = label_mode[max_score]
    if len(tied_idx) > 0:
        warnings.warn(f'breaking ties of examples {list(tied_idx.keys())} by random selection, you may want to set seed for reproducability')
        for (idx, label_mode) in tied_idx.items():
            majority_vote_label[idx] = np.random.choice(label_mode)
    if verbose:
        check_consensus_label_classes(labels_multiannotator=labels_multiannotator, consensus_label=majority_vote_label, consensus_method='majority_vote')
    return majority_vote_label.astype(int)

def convert_long_to_wide_dataset(labels_multiannotator_long: pd.DataFrame) -> pd.DataFrame:
    if False:
        for i in range(10):
            print('nop')
    'Converts a long format dataset to wide format which is suitable for passing into\n    :py:func:`get_label_quality_multiannotator <cleanlab.multiannotator.get_label_quality_multiannotator>`.\n\n    Dataframe must contain three columns named:\n\n    #. ``task`` representing each example labeled by the annotators\n    #. ``annotator`` representing each annotator\n    #. ``label`` representing the label given by an annotator for the corresponding task (i.e. example)\n\n    Parameters\n    ----------\n    labels_multiannotator_long : pd.DataFrame\n        pandas DataFrame in long format with three columns named ``task``, ``annotator`` and ``label``\n\n    Returns\n    -------\n    labels_multiannotator_wide : pd.DataFrame\n        pandas DataFrame of the proper format to be passed as ``labels_multiannotator`` for the other ``cleanlab.multiannotator`` functions.\n    '
    labels_multiannotator_wide = labels_multiannotator_long.pivot(index='task', columns='annotator', values='label')
    labels_multiannotator_wide.index.name = None
    labels_multiannotator_wide.columns.name = None
    return labels_multiannotator_wide

def _get_consensus_stats(labels_multiannotator: np.ndarray, pred_probs: np.ndarray, num_annotations: np.ndarray, consensus_label: np.ndarray, quality_method: str='crowdlab', verbose: bool=True, ensemble: bool=False, label_quality_score_kwargs: dict={}) -> tuple:
    if False:
        while True:
            i = 10
    'Returns a tuple containing the consensus labels, annotator agreement scores, and quality of consensus\n\n    Parameters\n    ----------\n    labels_multiannotator : np.ndarray\n        2D numpy array of multiple given labels for each example with shape ``(N, M)``,\n        where N is the number of examples and M is the number of annotators.\n        For more details, labels in the same format expected by the :py:func:`get_label_quality_multiannotator <cleanlab.multiannotator.get_label_quality_multiannotator>`.\n    pred_probs : np.ndarray\n        An array of shape ``(N, K)`` of model-predicted probabilities, ``P(label=k|x)``.\n        For details, predicted probabilities in the same format expected by the :py:func:`get_label_quality_multiannotator <cleanlab.multiannotator.get_label_quality_multiannotator>`.\n    num_annotations : np.ndarray\n        An array of shape ``(N,)`` with the number of annotators that have labeled each example.\n    consensus_label : np.ndarray\n        An array of shape ``(N,)`` with the consensus labels aggregated from all annotators.\n    quality_method : str, default = "crowdlab" (Options: ["crowdlab", "agreement"])\n        Specifies the method used to calculate the quality of the consensus label.\n        For valid quality methods, view :py:func:`get_label_quality_multiannotator <cleanlab.multiannotator.get_label_quality_multiannotator>`\n    label_quality_score_kwargs : dict, optional\n        Keyword arguments to pass into ``get_label_quality_scores()``.\n    verbose : bool, default = True\n        Certain warnings and notes will be printed if ``verbose`` is set to ``True``.\n    ensemble : bool, default = False\n        Boolean flag to indicate whether the pred_probs passed are from ensemble models.\n\n    Returns\n    ------\n    stats : tuple\n        A tuple of (consensus_label, annotator_agreement, consensus_quality_score, post_pred_probs).\n    '
    annotator_agreement = _get_annotator_agreement_with_consensus(labels_multiannotator=labels_multiannotator, consensus_label=consensus_label)
    if ensemble:
        (post_pred_probs, model_weight, annotator_weight) = _get_post_pred_probs_and_weights_ensemble(labels_multiannotator=labels_multiannotator, consensus_label=consensus_label, prior_pred_probs=pred_probs, num_annotations=num_annotations, annotator_agreement=annotator_agreement, quality_method=quality_method, verbose=verbose)
    else:
        (post_pred_probs, model_weight, annotator_weight) = _get_post_pred_probs_and_weights(labels_multiannotator=labels_multiannotator, consensus_label=consensus_label, prior_pred_probs=pred_probs, num_annotations=num_annotations, annotator_agreement=annotator_agreement, quality_method=quality_method, verbose=verbose)
    consensus_quality_score = _get_consensus_quality_score(consensus_label=consensus_label, pred_probs=post_pred_probs, num_annotations=num_annotations, annotator_agreement=annotator_agreement, quality_method=quality_method, label_quality_score_kwargs=label_quality_score_kwargs)
    return (annotator_agreement, consensus_quality_score, post_pred_probs, model_weight, annotator_weight)

def _get_annotator_stats(labels_multiannotator: np.ndarray, pred_probs: np.ndarray, consensus_label: np.ndarray, num_annotations: np.ndarray, annotator_agreement: np.ndarray, model_weight: np.ndarray, annotator_weight: np.ndarray, consensus_quality_score: np.ndarray, detailed_label_quality: Optional[np.ndarray]=None, annotator_ids: Optional[pd.Index]=None, quality_method: str='crowdlab') -> pd.DataFrame:
    if False:
        return 10
    'Returns a dictionary containing overall statistics about each annotator.\n\n    Parameters\n    ----------\n    labels_multiannotator : np.ndarray\n        2D numpy array of multiple given labels for each example with shape ``(N, M)``,\n        where N is the number of examples and M is the number of annotators.\n        For more details, labels in the same format expected by the :py:func:`get_label_quality_multiannotator <cleanlab.multiannotator.get_label_quality_multiannotator>`.\n    pred_probs : np.ndarray\n        An array of shape ``(N, K)`` of model-predicted probabilities, ``P(label=k|x)``.\n        For details, predicted probabilities in the same format expected by the :py:func:`get_label_quality_multiannotator <cleanlab.multiannotator.get_label_quality_multiannotator>`.\n    consensus_label : np.ndarray\n        An array of shape ``(N,)`` with the consensus labels aggregated from all annotators.\n    num_annotations : np.ndarray\n        An array of shape ``(N,)`` with the number of annotators that have labeled each example.\n    annotator_agreement : np.ndarray\n        An array of shape ``(N,)`` with the fraction of annotators that agree with each consensus label.\n    model_weight : float\n        float specifying the model weight used in weighted averages,\n        None if model weight is not used to compute quality scores\n    annotator_weight : np.ndarray\n        An array of shape ``(M,)`` where M is the number of annotators, specifying the annotator weights used in weighted averages,\n        None if annotator weights are not used to compute quality scores\n    consensus_quality_score : np.ndarray\n        An array of shape ``(N,)`` with the quality score of the consensus.\n    detailed_label_quality :\n        pandas DataFrame containing the detailed label quality scores for all examples and annotators\n    quality_method : str, default = "crowdlab" (Options: ["crowdlab", "agreement"])\n        Specifies the method used to calculate the quality of the consensus label.\n        For valid quality methods, view :py:func:`get_label_quality_multiannotator <cleanlab.multiannotator.get_label_quality_multiannotator>`\n\n    Returns\n    -------\n    annotator_stats : pd.DataFrame\n        Overall statistics about each annotator.\n        For details, see the documentation of :py:func:`get_label_quality_multiannotator <cleanlab.multiannotator.get_label_quality_multiannotator>`.\n    '
    annotator_quality = _get_annotator_quality(labels_multiannotator=labels_multiannotator, pred_probs=pred_probs, consensus_label=consensus_label, num_annotations=num_annotations, annotator_agreement=annotator_agreement, model_weight=model_weight, annotator_weight=annotator_weight, detailed_label_quality=detailed_label_quality, quality_method=quality_method)
    num_examples_labeled = np.sum(~np.isnan(labels_multiannotator), axis=0)
    agreement_with_consensus = np.zeros(labels_multiannotator.shape[1])
    for i in range(len(agreement_with_consensus)):
        labels = labels_multiannotator[:, i]
        labels_mask = ~np.isnan(labels)
        agreement_with_consensus[i] = np.mean(labels[labels_mask] == consensus_label[labels_mask])
    worst_class = _get_annotator_worst_class(labels_multiannotator=labels_multiannotator, consensus_label=consensus_label, consensus_quality_score=consensus_quality_score)
    annotator_stats = pd.DataFrame({'annotator_quality': annotator_quality, 'agreement_with_consensus': agreement_with_consensus, 'worst_class': worst_class, 'num_examples_labeled': num_examples_labeled}, index=annotator_ids)
    return annotator_stats.sort_values(by=['annotator_quality', 'agreement_with_consensus'])

def _get_annotator_agreement_with_consensus(labels_multiannotator: np.ndarray, consensus_label: np.ndarray) -> np.ndarray:
    if False:
        print('Hello World!')
    'Returns the fractions of annotators that agree with the consensus label per example. Note that the\n    fraction for each example only considers the annotators that labeled that particular example.\n\n    Parameters\n    ----------\n    labels_multiannotator : np.ndarray\n        2D numpy array of multiple given labels for each example with shape ``(N, M)``,\n        where N is the number of examples and M is the number of annotators.\n        For more details, labels in the same format expected by the :py:func:`get_label_quality_multiannotator <cleanlab.multiannotator.get_label_quality_multiannotator>`.\n    consensus_label : np.ndarray\n        An array of shape ``(N,)`` with the consensus labels aggregated from all annotators.\n\n    Returns\n    -------\n    annotator_agreement : np.ndarray\n        An array of shape ``(N,)`` with the fraction of annotators that agree with each consensus label.\n    '
    annotator_agreement = np.zeros(len(labels_multiannotator))
    for (i, labels) in enumerate(labels_multiannotator):
        annotator_agreement[i] = np.mean(labels[~np.isnan(labels)] == consensus_label[i])
    return annotator_agreement

def _get_annotator_agreement_with_annotators(labels_multiannotator: np.ndarray, num_annotations: np.ndarray, verbose: bool=True) -> np.ndarray:
    if False:
        while True:
            i = 10
    'Returns the average agreement of each annotator with other annotators that label the same example.\n\n    Parameters\n    ----------\n    labels_multiannotator : np.ndarray\n        2D numpy array of multiple given labels for each example with shape ``(N, M)``,\n        where N is the number of examples and M is the number of annotators.\n        For more details, labels in the same format expected by the :py:func:`get_label_quality_multiannotator <cleanlab.multiannotator.get_label_quality_multiannotator>`.\n    consensus_label : np.ndarray\n        An array of shape ``(N,)`` with the consensus labels aggregated from all annotators.\n    verbose : bool, default = True\n        Certain warnings and notes will be printed if ``verbose`` is set to ``True``.\n\n    Returns\n    -------\n    annotator_agreement : np.ndarray\n        An array of shape ``(M,)`` where M is the number of annotators, with the agreement of each annotator with other\n        annotators that labeled the same examples.\n    '
    annotator_agreement_with_annotators = np.zeros(labels_multiannotator.shape[1])
    for i in range(len(annotator_agreement_with_annotators)):
        annotator_labels = labels_multiannotator[:, i]
        annotator_labels_mask = ~np.isnan(annotator_labels)
        annotator_agreement_with_annotators[i] = _get_single_annotator_agreement(labels_multiannotator[annotator_labels_mask], num_annotations[annotator_labels_mask], i)
    non_overlap_mask = np.isnan(annotator_agreement_with_annotators)
    if np.sum(non_overlap_mask) > 0:
        if verbose:
            print(f"Annotator(s) {list(np.where(non_overlap_mask)[0])} did not annotate any examples that overlap with other annotators,                 \nusing the average annotator agreeement among other annotators as this annotator's agreement.")
        avg_annotator_agreement = np.mean(annotator_agreement_with_annotators[~non_overlap_mask])
        annotator_agreement_with_annotators[non_overlap_mask] = avg_annotator_agreement
    return annotator_agreement_with_annotators

def _get_single_annotator_agreement(labels_multiannotator: np.ndarray, num_annotations: np.ndarray, annotator_idx: int) -> float:
    if False:
        for i in range(10):
            print('nop')
    'Returns the average agreement of a given annotator other annotators that label the same example.\n\n    Parameters\n    ----------\n    labels_multiannotator : np.ndarray\n        2D numpy array of multiple given labels for each example with shape ``(N, M)``,\n        where N is the number of examples and M is the number of annotators.\n        For more details, labels in the same format expected by the :py:func:`get_label_quality_multiannotator <cleanlab.multiannotator.get_label_quality_multiannotator>`.\n    num_annotations : np.ndarray\n        An array of shape ``(N,)`` with the number of annotators that have labeled each example.\n    annotator_idx : int\n        The index of the annotator we want to compute the annotator agreement for.\n\n    Returns\n    -------\n    annotator_agreement : float\n        An float repesenting the agreement of each annotator with other annotators that labeled the same examples.\n    '
    annotator_agreement_per_example = np.zeros(len(labels_multiannotator))
    for (i, labels) in enumerate(labels_multiannotator):
        labels_subset = labels[~np.isnan(labels)]
        examples_num_annotators = len(labels_subset)
        if examples_num_annotators > 1:
            annotator_agreement_per_example[i] = (np.sum(labels_subset == labels[annotator_idx]) - 1) / (examples_num_annotators - 1)
    adjusted_num_annotations = num_annotations - 1
    if np.sum(adjusted_num_annotations) == 0:
        annotator_agreement = np.NaN
    else:
        annotator_agreement = np.average(annotator_agreement_per_example, weights=num_annotations - 1)
    return annotator_agreement

def _get_post_pred_probs_and_weights(labels_multiannotator: np.ndarray, consensus_label: np.ndarray, prior_pred_probs: np.ndarray, num_annotations: np.ndarray, annotator_agreement: np.ndarray, quality_method: str='crowdlab', verbose: bool=True) -> Tuple[np.ndarray, Optional[float], Optional[np.ndarray]]:
    if False:
        while True:
            i = 10
    'Return the posterior predicted probabilities of each example given a specified quality method.\n\n    Parameters\n    ----------\n    labels_multiannotator : np.ndarray\n        2D numpy array of multiple given labels for each example with shape ``(N, M)``,\n        where N is the number of examples and M is the number of annotators.\n        For more details, labels in the same format expected by the :py:func:`get_label_quality_multiannotator <cleanlab.multiannotator.get_label_quality_multiannotator>`.\n    consensus_label : np.ndarray\n        An array of shape ``(N,)`` with the consensus labels aggregated from all annotators.\n    prior_pred_probs : np.ndarray\n        An array of shape ``(N, K)`` of prior predicted probabilities, ``P(label=k|x)``, usually the out-of-sample predicted probability computed by a model.\n        For details, predicted probabilities in the same format expected by the :py:func:`get_label_quality_multiannotator <cleanlab.multiannotator.get_label_quality_multiannotator>`.\n    num_annotations : np.ndarray\n        An array of shape ``(N,)`` with the number of annotators that have labeled each example.\n    annotator_agreement : np.ndarray\n        An array of shape ``(N,)`` with the fraction of annotators that agree with each consensus label.\n    quality_method : default = "crowdlab" (Options: ["crowdlab", "agreement"])\n        Specifies the method used to calculate the quality of the consensus label.\n        For valid quality methods, view :py:func:`get_label_quality_multiannotator <cleanlab.multiannotator.get_label_quality_multiannotator>`\n    verbose : default = True\n        Certain warnings and notes will be printed if ``verbose`` is set to ``True``.\n\n    Returns\n    -------\n    post_pred_probs : np.ndarray\n        An array of shape ``(N, K)`` with the posterior predicted probabilities.\n\n    model_weight : float\n        float specifying the model weight used in weighted averages,\n        None if model weight is not used to compute quality scores\n\n    annotator_weight : np.ndarray\n        An array of shape ``(M,)`` where M is the number of annotators, specifying the annotator weights used in weighted averages,\n        None if annotator weights are not used to compute quality scores\n\n    '
    valid_methods = ['crowdlab', 'agreement']
    return_model_weight = None
    return_annotator_weight = None
    if quality_method == 'crowdlab':
        num_classes = get_num_classes(pred_probs=prior_pred_probs)
        consensus_likelihood = np.mean(annotator_agreement[num_annotations != 1])
        non_consensus_likelihood = (1 - consensus_likelihood) / (num_classes - 1)
        mask = num_annotations != 1
        consensus_label_subset = consensus_label[mask]
        prior_pred_probs_subset = prior_pred_probs[mask]
        most_likely_class_error = np.clip(np.mean(consensus_label_subset != np.argmax(np.bincount(consensus_label_subset, minlength=num_classes))), a_min=CLIPPING_LOWER_BOUND, a_max=None)
        annotator_agreement_with_annotators = _get_annotator_agreement_with_annotators(labels_multiannotator, num_annotations, verbose)
        annotator_error = 1 - annotator_agreement_with_annotators
        adjusted_annotator_agreement = np.clip(1 - annotator_error / most_likely_class_error, a_min=CLIPPING_LOWER_BOUND, a_max=None)
        model_error = np.mean(np.argmax(prior_pred_probs_subset, axis=1) != consensus_label_subset)
        model_weight = np.max([1 - model_error / most_likely_class_error, CLIPPING_LOWER_BOUND]) * np.sqrt(np.mean(num_annotations))
        post_pred_probs = np.full(prior_pred_probs.shape, np.nan)
        for (i, labels) in enumerate(labels_multiannotator):
            labels_mask = ~np.isnan(labels)
            labels_subset = labels[labels_mask]
            post_pred_probs[i] = [np.average([prior_pred_probs[i, true_label]] + [consensus_likelihood if annotator_label == true_label else non_consensus_likelihood for annotator_label in labels_subset], weights=np.concatenate(([model_weight], adjusted_annotator_agreement[labels_mask]))) for true_label in range(num_classes)]
        return_model_weight = model_weight
        return_annotator_weight = adjusted_annotator_agreement
    elif quality_method == 'agreement':
        num_classes = get_num_classes(pred_probs=prior_pred_probs)
        label_counts = np.full((len(labels_multiannotator), num_classes), np.NaN)
        for (i, labels) in enumerate(labels_multiannotator):
            label_counts[i, :] = value_counts(labels[~np.isnan(labels)], num_classes=num_classes)
        post_pred_probs = label_counts / num_annotations.reshape(-1, 1)
    else:
        raise ValueError(f'\n            {quality_method} is not a valid quality method!\n            Please choose a valid quality_method: {valid_methods}\n            ')
    return (post_pred_probs, return_model_weight, return_annotator_weight)

def _get_post_pred_probs_and_weights_ensemble(labels_multiannotator: np.ndarray, consensus_label: np.ndarray, prior_pred_probs: np.ndarray, num_annotations: np.ndarray, annotator_agreement: np.ndarray, quality_method: str='crowdlab', verbose: bool=True) -> Tuple[np.ndarray, Any, Any]:
    if False:
        for i in range(10):
            print('nop')
    'Return the posterior predicted class probabilites of each example given a specified quality method and prior predicted class probabilities from an ensemble of multiple classifier models.\n\n    Parameters\n    ----------\n    labels_multiannotator : np.ndarray\n        2D numpy array of multiple given labels for each example with shape ``(N, M)``,\n        where N is the number of examples and M is the number of annotators.\n        For more details, labels in the same format expected by the :py:func:`get_label_quality_multiannotator <cleanlab.multiannotator.get_label_quality_multiannotator>`.\n    consensus_label : np.ndarray\n        An array of shape ``(P, N, K)`` where P is the number of models, consisting of predicted class probabilities from the ensemble models.\n        Each set of predicted probabilities with shape ``(N, K)`` is in the same format expected by the :py:func:`get_label_quality_scores <cleanlab.rank.get_label_quality_scores>`.\n    prior_pred_probs : np.ndarray\n        An array of shape ``(N, K)`` of prior predicted probabilities, ``P(label=k|x)``, usually the out-of-sample predicted probability computed by a model.\n        For details, predicted probabilities in the same format expected by the :py:func:`get_label_quality_multiannotator <cleanlab.multiannotator.get_label_quality_multiannotator>`.\n    num_annotations : np.ndarray\n        An array of shape ``(N,)`` with the number of annotators that have labeled each example.\n    annotator_agreement : np.ndarray\n        An array of shape ``(N,)`` with the fraction of annotators that agree with each consensus label.\n    quality_method : str, default = "crowdlab" (Options: ["crowdlab", "agreement"])\n        Specifies the method used to calculate the quality of the consensus label.\n        For valid quality methods, view :py:func:`get_label_quality_multiannotator <cleanlab.multiannotator.get_label_quality_multiannotator>`\n    verbose : bool, default = True\n        Certain warnings and notes will be printed if ``verbose`` is set to ``True``.\n\n    Returns\n    -------\n    post_pred_probs : np.ndarray\n        An array of shape ``(N, K)`` with the posterior predicted probabilities.\n\n    model_weight : np.ndarray\n        An array of shape ``(P,)`` where P is the number of models in this ensemble, specifying the model weight used in weighted averages,\n        ``None`` if model weight is not used to compute quality scores\n\n    annotator_weight : np.ndarray\n        An array of shape ``(M,)`` where M is the number of annotators, specifying the annotator weights used in weighted averages,\n        ``None`` if annotator weights are not used to compute quality scores\n\n    '
    num_classes = get_num_classes(pred_probs=prior_pred_probs[0])
    consensus_likelihood = np.mean(annotator_agreement[num_annotations != 1])
    non_consensus_likelihood = (1 - consensus_likelihood) / (num_classes - 1)
    mask = num_annotations != 1
    consensus_label_subset = consensus_label[mask]
    most_likely_class_error = np.clip(np.mean(consensus_label_subset != np.argmax(np.bincount(consensus_label_subset, minlength=num_classes))), a_min=CLIPPING_LOWER_BOUND, a_max=None)
    annotator_agreement_with_annotators = _get_annotator_agreement_with_annotators(labels_multiannotator, num_annotations, verbose)
    annotator_error = 1 - annotator_agreement_with_annotators
    adjusted_annotator_agreement = np.clip(1 - annotator_error / most_likely_class_error, a_min=CLIPPING_LOWER_BOUND, a_max=None)
    model_weight = np.full(prior_pred_probs.shape[0], np.nan)
    for idx in range(prior_pred_probs.shape[0]):
        prior_pred_probs_subset = prior_pred_probs[idx][mask]
        model_error = np.mean(np.argmax(prior_pred_probs_subset, axis=1) != consensus_label_subset)
        model_weight[idx] = np.max([1 - model_error / most_likely_class_error, CLIPPING_LOWER_BOUND]) * np.sqrt(np.mean(num_annotations))
    post_pred_probs = np.full(prior_pred_probs[0].shape, np.nan)
    for (i, labels) in enumerate(labels_multiannotator):
        labels_mask = ~np.isnan(labels)
        labels_subset = labels[labels_mask]
        post_pred_probs[i] = [np.average([prior_pred_probs[ind][i, true_label] for ind in range(prior_pred_probs.shape[0])] + [consensus_likelihood if annotator_label == true_label else non_consensus_likelihood for annotator_label in labels_subset], weights=np.concatenate((model_weight, adjusted_annotator_agreement[labels_mask]))) for true_label in range(num_classes)]
    return_model_weight = model_weight
    return_annotator_weight = adjusted_annotator_agreement
    return (post_pred_probs, return_model_weight, return_annotator_weight)

def _get_consensus_quality_score(consensus_label: np.ndarray, pred_probs: np.ndarray, num_annotations: np.ndarray, annotator_agreement: np.ndarray, quality_method: str='crowdlab', label_quality_score_kwargs: dict={}) -> np.ndarray:
    if False:
        while True:
            i = 10
    'Return scores representing quality of the consensus label for each example.\n\n    Parameters\n    ----------\n    labels_multiannotator : np.ndarray\n        2D numpy array of multiple given labels for each example with shape ``(N, M)``,\n        where N is the number of examples and M is the number of annotators.\n        For more details, labels in the same format expected by the :py:func:`get_label_quality_multiannotator <cleanlab.multiannotator.get_label_quality_multiannotator>`.\n    consensus_label : np.ndarray\n        An array of shape ``(N,)`` with the consensus labels aggregated from all annotators.\n    pred_probs : np.ndarray\n        An array of shape ``(N, K)`` of posterior predicted probabilities, ``P(label=k|x)``.\n        For details, predicted probabilities in the same format expected by the :py:func:`get_label_quality_multiannotator <cleanlab.multiannotator.get_label_quality_multiannotator>`.\n    num_annotations : np.ndarray\n        An array of shape ``(N,)`` with the number of annotators that have labeled each example.\n    annotator_agreement : np.ndarray\n        An array of shape ``(N,)`` with the fraction of annotators that agree with each consensus label.\n    quality_method : str, default = "crowdlab" (Options: ["crowdlab", "agreement"])\n        Specifies the method used to calculate the quality of the consensus label.\n        For valid quality methods, view :py:func:`get_label_quality_multiannotator <cleanlab.multiannotator.get_label_quality_multiannotator>`\n\n    Returns\n    -------\n    consensus_quality_score : np.ndarray\n        An array of shape ``(N,)`` with the quality score of the consensus.\n    '
    valid_methods = ['crowdlab', 'agreement']
    if quality_method == 'crowdlab':
        consensus_quality_score = get_label_quality_scores(consensus_label, pred_probs, **label_quality_score_kwargs)
    elif quality_method == 'agreement':
        consensus_quality_score = annotator_agreement
    else:
        raise ValueError(f'\n            {quality_method} is not a valid consensus quality method!\n            Please choose a valid quality_method: {valid_methods}\n            ')
    return consensus_quality_score

def _get_annotator_label_quality_score(annotator_label: np.ndarray, pred_probs: np.ndarray, label_quality_score_kwargs: dict={}) -> np.ndarray:
    if False:
        while True:
            i = 10
    'Returns quality scores for each datapoint.\n    Very similar functionality as ``_get_consensus_quality_score`` with additional support for annotator labels that contain NaN values.\n    For more info about parameters and returns, see the docstring of :py:func:`_get_consensus_quality_score <cleanlab.multiannotator._get_consensus_quality_score>`.\n    '
    mask = ~np.isnan(annotator_label)
    annotator_label_quality_score_subset = get_label_quality_scores(labels=annotator_label[mask].astype(int), pred_probs=pred_probs[mask], **label_quality_score_kwargs)
    annotator_label_quality_score = np.full(len(annotator_label), np.nan)
    annotator_label_quality_score[mask] = annotator_label_quality_score_subset
    return annotator_label_quality_score

def _get_annotator_quality(labels_multiannotator: np.ndarray, pred_probs: np.ndarray, consensus_label: np.ndarray, num_annotations: np.ndarray, annotator_agreement: np.ndarray, model_weight: np.ndarray, annotator_weight: np.ndarray, detailed_label_quality: Optional[np.ndarray]=None, quality_method: str='crowdlab') -> pd.DataFrame:
    if False:
        return 10
    'Returns annotator quality score for each annotator.\n\n    Parameters\n    ----------\n    labels_multiannotator : np.ndarray\n        2D numpy array of multiple given labels for each example with shape ``(N, M)``,\n        where N is the number of examples and M is the number of annotators.\n        For more details, labels in the same format expected by the :py:func:`get_label_quality_multiannotator <cleanlab.multiannotator.get_label_quality_multiannotator>`.\n    pred_probs : np.ndarray\n        An array of shape ``(N, K)`` of model-predicted probabilities, ``P(label=k|x)``.\n        For details, predicted probabilities in the same format expected by the :py:func:`get_label_quality_multiannotator <cleanlab.multiannotator.get_label_quality_multiannotator>`.\n    consensus_label : np.ndarray\n        An array of shape ``(N,)`` with the consensus labels aggregated from all annotators.\n    num_annotations : np.ndarray\n        An array of shape ``(N,)`` with the number of annotators that have labeled each example.\n    annotator_agreement : np.ndarray\n        An array of shape ``(N,)`` with the fraction of annotators that agree with each consensus label.\n    model_weight : float\n        An array of shape ``(P,)`` where P is the number of models in this ensemble, specifying the model weight used in weighted averages,\n        ``None`` if model weight is not used to compute quality scores\n    annotator_weight : np.ndarray\n        An array of shape ``(M,)`` where M is the number of annotators, specifying the annotator weights used in weighted averages,\n        ``None`` if annotator weights are not used to compute quality scores\n    detailed_label_quality :\n        pandas DataFrame containing the detailed label quality scores for all examples and annotators\n    quality_method : str, default = "crowdlab" (Options: ["crowdlab", "agreement"])\n        Specifies the method used to calculate the quality of the annotators.\n        For valid quality methods, view :py:func:`get_label_quality_multiannotator <cleanlab.multiannotator.get_label_quality_multiannotator>`\n\n    Returns\n    -------\n    annotator_quality : np.ndarray\n        Quality scores of a given annotator\'s labels\n    '
    valid_methods = ['crowdlab', 'agreement']
    if quality_method == 'crowdlab':
        if detailed_label_quality is None:
            annotator_lqs = np.zeros(labels_multiannotator.shape[1])
            for i in range(len(annotator_lqs)):
                labels = labels_multiannotator[:, i]
                labels_mask = ~np.isnan(labels)
                annotator_lqs[i] = np.mean(get_label_quality_scores(labels[labels_mask].astype(int), pred_probs[labels_mask]))
        else:
            annotator_lqs = np.nanmean(detailed_label_quality, axis=0)
        mask = num_annotations != 1
        labels_multiannotator_subset = labels_multiannotator[mask]
        consensus_label_subset = consensus_label[mask]
        annotator_agreement = np.zeros(labels_multiannotator_subset.shape[1])
        for i in range(len(annotator_agreement)):
            labels = labels_multiannotator_subset[:, i]
            labels_mask = ~np.isnan(labels)
            if np.sum(labels_mask) == 0:
                annotator_agreement[i] = np.NaN
            else:
                annotator_agreement[i] = np.mean(labels[labels_mask] == consensus_label_subset[labels_mask])
        avg_num_annotations_frac = np.mean(num_annotations) / len(annotator_weight)
        annotator_weight_adjusted = np.sum(annotator_weight) * avg_num_annotations_frac
        w = model_weight / (model_weight + annotator_weight_adjusted)
        annotator_quality = w * annotator_lqs + (1 - w) * annotator_agreement
    elif quality_method == 'agreement':
        mask = num_annotations != 1
        labels_multiannotator_subset = labels_multiannotator[mask]
        consensus_label_subset = consensus_label[mask]
        annotator_quality = np.zeros(labels_multiannotator_subset.shape[1])
        for i in range(len(annotator_quality)):
            labels = labels_multiannotator_subset[:, i]
            labels_mask = ~np.isnan(labels)
            if np.sum(labels_mask) == 0:
                annotator_quality[i] = np.NaN
            else:
                annotator_quality[i] = np.mean(labels[labels_mask] == consensus_label_subset[labels_mask])
    else:
        raise ValueError(f'\n            {quality_method} is not a valid annotator quality method!\n            Please choose a valid quality_method: {valid_methods}\n            ')
    return annotator_quality

def _get_annotator_worst_class(labels_multiannotator: np.ndarray, consensus_label: np.ndarray, consensus_quality_score: np.ndarray) -> np.ndarray:
    if False:
        while True:
            i = 10
    'Returns the class which each annotator makes the most errors in.\n\n    Parameters\n    ----------\n    labels_multiannotator : np.ndarray\n        2D pandas DataFrame of multiple given labels for each example with shape ``(N, M)``,\n        where N is the number of examples and M is the number of annotators.\n        For more details, labels in the same format expected by the :py:func:`get_label_quality_multiannotator <cleanlab.multiannotator.get_label_quality_multiannotator>`.\n    consensus_label : np.ndarray\n        An array of shape ``(N,)`` with the consensus labels aggregated from all annotators.\n    consensus_quality_score : np.ndarray\n        An array of shape ``(N,)`` with the quality score of the consensus.\n\n    Returns\n    -------\n    worst_class : np.ndarray\n        The class that is most frequently mislabeled by a given annotator.\n    '
    worst_class = np.apply_along_axis(_get_single_annotator_worst_class, axis=0, arr=labels_multiannotator, consensus_label=consensus_label, consensus_quality_score=consensus_quality_score).astype(int)
    return worst_class

def _get_single_annotator_worst_class(labels: np.ndarray, consensus_label: np.ndarray, consensus_quality_score: np.ndarray) -> int:
    if False:
        print('Hello World!')
    'Returns the class a given annotator makes the most errors in.\n\n    Parameters\n    ----------\n    labels : np.ndarray\n        An array of shape ``(N,)`` with the labels from the annotator we want to evaluate.\n    consensus_label : np.ndarray\n        An array of shape ``(N,)`` with the consensus labels aggregated from all annotators.\n    consensus_quality_score : np.ndarray\n        An array of shape ``(N,)`` with the quality score of the consensus.\n\n    Returns\n    -------\n    worst_class : int\n        The class that is most frequently mislabeled by the given annotator.\n    '
    labels = pd.Series(labels)
    labels_mask = pd.notna(labels)
    class_accuracies = (labels[labels_mask] == consensus_label[labels_mask]).groupby(labels).mean()
    accuracy_min_idx = class_accuracies[class_accuracies == class_accuracies.min()].index.values
    if len(accuracy_min_idx) == 1:
        return accuracy_min_idx[0]
    class_count = labels[labels_mask].groupby(labels).count()[accuracy_min_idx]
    count_max_idx = class_count[class_count == class_count.max()].index.values
    if len(count_max_idx) == 1:
        return count_max_idx[0]
    avg_consensus_quality = pd.DataFrame({'annotator_label': labels, 'consensus_quality_score': consensus_quality_score})[labels_mask].groupby('annotator_label').mean()['consensus_quality_score'][count_max_idx]
    quality_max_idx = avg_consensus_quality[avg_consensus_quality == avg_consensus_quality.max()].index.values
    return quality_max_idx[0]