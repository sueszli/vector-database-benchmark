import numpy as np
import pytest
from copy import deepcopy
from cleanlab.benchmarking.noise_generation import generate_noise_matrix_from_trace
from cleanlab.benchmarking.noise_generation import generate_noisy_labels
from cleanlab import count
from cleanlab.multiannotator import get_label_quality_multiannotator, get_label_quality_multiannotator_ensemble, get_active_learning_scores, get_active_learning_scores_ensemble, get_majority_vote_label, convert_long_to_wide_dataset
from cleanlab.internal.multiannotator_utils import format_multiannotator_labels
import pandas as pd
from sklearn.linear_model import LogisticRegression

def make_data(means=[[3, 2], [7, 7], [0, 8]], covs=[[[5, -1.5], [-1.5, 1]], [[1, 0.5], [0.5, 4]], [[5, 1], [1, 5]]], labeled_sizes=[80, 40, 40], unlabeled_sizes=[20, 10, 10], avg_trace=0.8, num_annotators=50, seed=1):
    if False:
        print('Hello World!')
    np.random.seed(seed=seed)
    m = len(means)
    n = sum(labeled_sizes)
    local_data = []
    labels = []
    unlabeled_data = []
    unlabeled_labels = []
    for idx in range(m):
        local_data.append(np.random.multivariate_normal(mean=means[idx], cov=covs[idx], size=labeled_sizes[idx]))
        unlabeled_data.append(np.random.multivariate_normal(mean=means[idx], cov=covs[idx], size=unlabeled_sizes[idx]))
        labels.append(np.array([idx for i in range(labeled_sizes[idx])]))
        unlabeled_labels.append(np.array([idx for i in range(unlabeled_sizes[idx])]))
    X_train = np.vstack(local_data)
    X_train_unlabeled = np.vstack(unlabeled_data)
    true_labels_train = np.hstack(labels)
    true_labels_train_unlabeled = np.hstack(unlabeled_labels)
    py = np.bincount(true_labels_train) / float(len(true_labels_train))
    noise_matrix = generate_noise_matrix_from_trace(m, trace=avg_trace * m, py=py, valid_noise_matrix=True, seed=seed)
    s = pd.DataFrame(np.vstack([generate_noisy_labels(true_labels_train, noise_matrix) for _ in range(num_annotators)]).transpose())
    complete_labels = deepcopy(s)
    s = s.apply(lambda x: x.mask(np.random.random(n) < 0.8))
    s.dropna(axis=1, how='all', inplace=True)
    latent = count.estimate_py_noise_matrices_and_cv_pred_proba(X=X_train, labels=true_labels_train, cv_n_folds=3)
    latent_unlabeled = count.estimate_py_noise_matrices_and_cv_pred_proba(X=X_train_unlabeled, labels=true_labels_train_unlabeled, cv_n_folds=3)
    row_NA_check = pd.notna(s).any(axis=1)
    return {'X_train': X_train[row_NA_check], 'X_train_unlabeled': X_train_unlabeled, 'X_train_complete': X_train, 'true_labels_train': true_labels_train[row_NA_check], 'true_labels_train_unlabeled': true_labels_train_unlabeled, 'labels': s[row_NA_check].reset_index(drop=True), 'labels_unlabeled': pd.DataFrame(np.full((len(true_labels_train_unlabeled), num_annotators), np.NaN)), 'complete_labels': complete_labels, 'pred_probs': latent[4][row_NA_check], 'pred_probs_unlabeled': latent_unlabeled[4], 'pred_probs_complete': latent[4], 'noise_matrix': noise_matrix}

def make_ensemble_data(means=[[3, 2], [7, 7], [0, 8]], covs=[[[5, -1.5], [-1.5, 1]], [[1, 0.5], [0.5, 4]], [[5, 1], [1, 5]]], unlabeled_sizes=[20, 10, 10], avg_trace=0.8, num_annotators=50, seed=1):
    if False:
        print('Hello World!')
    np.random.seed(seed=seed)
    data = make_data()
    X_train = data['X_train']
    true_labels_train = data['true_labels_train']
    X_train_unlabeled = data['X_train_unlabeled']
    true_labels_train_unlabeled = data['true_labels_train_unlabeled']
    pred_probs_extra = count.estimate_py_noise_matrices_and_cv_pred_proba(X=X_train, labels=true_labels_train, cv_n_folds=3, clf=LogisticRegression())[4]
    pred_probs_labeled = np.array([data['pred_probs'], pred_probs_extra])
    pred_probs_extra_unlabeled = count.estimate_py_noise_matrices_and_cv_pred_proba(X=X_train_unlabeled, labels=true_labels_train_unlabeled, cv_n_folds=3, clf=LogisticRegression())[4]
    pred_probs_unlabeled = np.array([data['pred_probs_unlabeled'], pred_probs_extra_unlabeled])
    return {'X_train': data['X_train'], 'X_train_unlabeled': data['X_train_unlabeled'], 'true_labels_train': data['true_labels_train'], 'true_labels_train_unlabeled': data['true_labels_train_unlabeled'], 'labels': data['labels'], 'labels_unlabeled': data['labels_unlabeled'], 'complete_labels': data['complete_labels'], 'pred_probs': pred_probs_labeled, 'pred_probs_unlabeled': pred_probs_unlabeled, 'noise_matrix': data['noise_matrix']}

def make_data_long(data):
    if False:
        while True:
            i = 10
    data_long = data.stack().reset_index()
    data_long.columns = ['task', 'annotator', 'label']
    return data_long
data = make_data()
ensemble_data = make_ensemble_data()
small_data = make_data(labeled_sizes=[5, 5, 5], unlabeled_sizes=[5, 5, 5], num_annotators=1)

def test_convert_long_to_wide():
    if False:
        i = 10
        return i + 15
    labels_long = make_data_long(data['labels'])
    labels_wide = convert_long_to_wide_dataset(labels_long)
    assert isinstance(labels_wide, pd.DataFrame)
    assert labels_wide.count(axis=1).sum() == len(labels_long)
    example_long = labels_long[labels_long['task'] == 0].sort_values('annotator')
    example_wide = labels_wide.iloc[0].dropna()
    assert all(example_long['annotator'] == example_wide.index)
    assert all(example_long['label'].reset_index(drop=True) == example_wide.reset_index(drop=True))

def test_label_quality_scores_multiannotator():
    if False:
        while True:
            i = 10
    labels = data['labels']
    pred_probs = data['pred_probs']
    multiannotator_dict = get_label_quality_multiannotator(labels, pred_probs)
    assert isinstance(multiannotator_dict, dict)
    assert len(multiannotator_dict) == 3
    label_quality_multiannotator = multiannotator_dict['label_quality']
    assert isinstance(label_quality_multiannotator, pd.DataFrame)
    assert len(label_quality_multiannotator) == len(labels)
    assert all(label_quality_multiannotator['num_annotations'] > 0)
    assert set(label_quality_multiannotator['consensus_label']).issubset(np.unique(labels))
    assert all((label_quality_multiannotator['annotator_agreement'] >= 0) & (label_quality_multiannotator['annotator_agreement'] <= 1))
    assert all((label_quality_multiannotator['consensus_quality_score'] >= 0) & (label_quality_multiannotator['consensus_quality_score'] <= 1))
    annotator_stats = multiannotator_dict['annotator_stats']
    assert isinstance(annotator_stats, pd.DataFrame)
    assert len(annotator_stats) == labels.shape[1]
    assert all((annotator_stats['annotator_quality'] >= 0) & (annotator_stats['annotator_quality'] <= 1))
    assert all(annotator_stats['num_examples_labeled'] > 0)
    assert all((annotator_stats['agreement_with_consensus'] >= 0) & (annotator_stats['agreement_with_consensus'] <= 1))
    assert set(annotator_stats['worst_class']).issubset(np.unique(labels))
    detailed_label_quality = multiannotator_dict['detailed_label_quality']
    assert detailed_label_quality.shape == labels.shape
    multiannotator_dict = get_label_quality_multiannotator(labels, pred_probs, verbose=False)
    multiannotator_dict = get_label_quality_multiannotator(labels, pred_probs, consensus_method=['majority_vote', 'best_quality'])
    multiannotator_dict = get_label_quality_multiannotator(labels, pred_probs, label_quality_score_kwargs={'method': 'normalized_margin'})
    multiannotator_dict = get_label_quality_multiannotator(np.array(labels), pred_probs, quality_method='agreement')
    multiannotator_dict = get_label_quality_multiannotator(labels, pred_probs, return_annotator_stats=False)
    assert isinstance(multiannotator_dict, dict)
    assert len(multiannotator_dict) == 2
    assert isinstance(multiannotator_dict['label_quality'], pd.DataFrame)
    assert isinstance(multiannotator_dict['detailed_label_quality'], pd.DataFrame)
    multiannotator_dict = get_label_quality_multiannotator(labels, pred_probs, return_detailed_quality=False)
    assert isinstance(multiannotator_dict, dict)
    assert len(multiannotator_dict) == 2
    assert isinstance(multiannotator_dict['label_quality'], pd.DataFrame)
    assert isinstance(multiannotator_dict['annotator_stats'], pd.DataFrame)
    multiannotator_dict = get_label_quality_multiannotator(labels, pred_probs, return_detailed_quality=False, return_annotator_stats=False)
    assert isinstance(multiannotator_dict, dict)
    assert len(multiannotator_dict) == 1
    assert isinstance(multiannotator_dict['label_quality'], pd.DataFrame)
    multiannotator_dict = get_label_quality_multiannotator(labels, pred_probs, return_weights=True)
    assert len(multiannotator_dict) == 5
    assert isinstance(multiannotator_dict['model_weight'], float)
    assert isinstance(multiannotator_dict['annotator_weight'], np.ndarray)
    labels_string_names = labels.add_prefix('anno_')
    multiannotator_dict = get_label_quality_multiannotator(labels_string_names, pred_probs, return_detailed_quality=False)
    multiannotator_dict = get_label_quality_multiannotator(labels, pred_probs, calibrate_probs=True)
    try:
        multiannotator_dict = get_label_quality_multiannotator(labels, pred_probs, consensus_method='fake_method')
    except ValueError as e:
        assert 'not a valid consensus method' in str(e)
    try:
        multiannotator_dict = get_label_quality_multiannotator(labels, pred_probs, return_weights=True, quality_method='agreement')
    except ValueError as e:
        assert 'Model and annotator weights are only applicable to the crowdlab quality method' in str(e)
    labels_NA = deepcopy(labels_string_names)
    labels_NA['anno_0'] = pd.NA
    try:
        multiannotator_dict = get_label_quality_multiannotator(labels_NA, pred_probs)
    except ValueError as e:
        assert 'cannot have columns with all NaN' in str(e)
        assert "Annotators ['anno_0'] did not label any examples." in str(e)
    labels_nan = deepcopy(labels).values.astype(float)
    labels_nan[:, 1] = np.NaN
    try:
        multiannotator_dict = get_label_quality_multiannotator(labels_nan, pred_probs)
    except ValueError as e:
        assert 'cannot have columns with all NaN' in str(e)
        assert 'Annotators [1] did not label any examples.' in str(e)
    labels_nan = pd.DataFrame([[0, np.NaN, np.NaN], [np.NaN, 1, np.NaN], [np.NaN, np.NaN, 2], [np.NaN, np.NaN, np.NaN], [np.NaN, np.NaN, 2]])
    pred_probs = np.random.random((5, 3))
    try:
        multiannotator_dict = get_label_quality_multiannotator(labels_nan, pred_probs)
    except ValueError as e:
        assert 'cannot have rows with all NaN' in str(e)
        assert 'Examples [3] do not have any labels.' in str(e)
    try:
        multiannotator_dict = get_label_quality_multiannotator(labels, np.array([pred_probs, pred_probs]), return_weights=True)
    except ValueError as e:
        assert 'use the ensemble version of this function' in str(e)
    labels_flat = labels.values[:, 0].flatten()
    print(labels_flat.ndim)
    print(labels_flat)
    try:
        multiannotator_dict = get_label_quality_multiannotator(labels_flat, pred_probs)
    except ValueError as e:
        assert 'labels_multiannotator must be a 2D array or dataframe' in str(e)

@pytest.mark.filterwarnings('ignore::UserWarning')
def test_label_quality_scores_multiannotator_ensemble():
    if False:
        for i in range(10):
            print('nop')
    labels = ensemble_data['labels']
    pred_probs = ensemble_data['pred_probs']
    multiannotator_dict = get_label_quality_multiannotator_ensemble(labels, pred_probs, return_weights=True)
    assert isinstance(multiannotator_dict, dict)
    assert len(multiannotator_dict) == 5
    assert isinstance(multiannotator_dict['label_quality'], pd.DataFrame)
    assert isinstance(multiannotator_dict['annotator_stats'], pd.DataFrame)
    assert isinstance(multiannotator_dict['detailed_label_quality'], pd.DataFrame)
    assert isinstance(multiannotator_dict['model_weight'], np.ndarray)
    assert isinstance(multiannotator_dict['annotator_weight'], np.ndarray)
    labels_string_names = labels.add_prefix('anno_')
    multiannotator_dict = get_label_quality_multiannotator_ensemble(labels_string_names, pred_probs, return_detailed_quality=False)
    multiannotator_dict = get_label_quality_multiannotator_ensemble(labels, pred_probs, return_weights=True)
    assert len(multiannotator_dict) == 5
    assert isinstance(multiannotator_dict['model_weight'], np.ndarray)
    assert isinstance(multiannotator_dict['annotator_weight'], np.ndarray)
    multiannotator_dict = get_label_quality_multiannotator_ensemble(np.array(labels), pred_probs, calibrate_probs=True)
    labels_tiebreaks = np.array([[1, 2, 0], [1, 1, 0], [1, 0, 0], [2, 2, 2], [1, 2, 0], [1, 2, 0]])
    pred_probs_tiebreaks = np.array([[0.4, 0.4, 0.2], [0.3, 0.6, 0.1], [0.75, 0.2, 0.05], [0.1, 0.4, 0.5], [0.2, 0.4, 0.4], [0.2, 0.4, 0.4]])
    pred_probs_tiebreaks_ensemble = np.array([pred_probs_tiebreaks, pred_probs_tiebreaks, pred_probs_tiebreaks])
    consensus_label = get_label_quality_multiannotator_ensemble(labels_tiebreaks, pred_probs_tiebreaks_ensemble)
    try:
        multiannotator_dict = get_label_quality_multiannotator_ensemble(labels, pred_probs[0], return_weights=True)
    except ValueError as e:
        assert 'use the non-ensemble version of this function' in str(e)

def test_get_active_learning_scores():
    if False:
        return 10
    labels = data['labels']
    pred_probs = data['pred_probs']
    pred_probs_unlabeled = data['pred_probs_unlabeled']
    (active_learning_scores, active_learning_scores_unlabeled) = get_active_learning_scores(labels, pred_probs, pred_probs_unlabeled)
    assert isinstance(active_learning_scores, np.ndarray)
    assert len(active_learning_scores) == len(pred_probs)
    assert len(active_learning_scores_unlabeled) == len(pred_probs_unlabeled)
    (active_learning_scores, active_learning_scores_unlabeled) = get_active_learning_scores(np.array(labels), pred_probs)
    assert isinstance(active_learning_scores, np.ndarray)
    assert len(active_learning_scores) == len(pred_probs)
    assert len(active_learning_scores_unlabeled) == 0
    (active_learning_scores, active_learning_scores_unlabeled) = get_active_learning_scores(pred_probs_unlabeled=pred_probs_unlabeled)
    assert len(active_learning_scores) == 0
    assert len(active_learning_scores_unlabeled) == len(pred_probs_unlabeled)
    try:
        (active_learning_scores, active_learning_scores_unlabeled) = get_active_learning_scores(labels, pred_probs, pred_probs_unlabeled[:, :-1])
    except ValueError as e:
        assert 'must have the same number of classes' in str(e)
    single_labels = data['complete_labels'].iloc[[0]]
    singe_pred_probs = pred_probs[[0]]
    singe_pred_probs_unlabeled = pred_probs_unlabeled[[0]]
    get_active_learning_scores(single_labels, singe_pred_probs, singe_pred_probs_unlabeled)
    labels = pd.DataFrame([[0, np.NaN, np.NaN], [np.NaN, 1, np.NaN], [np.NaN, np.NaN, 2], [np.NaN, 1, np.NaN], [np.NaN, np.NaN, 2]])
    pred_probs = np.random.random((5, 3))
    get_active_learning_scores(labels, pred_probs, pred_probs)

def test_get_active_learning_scores_ensemble():
    if False:
        i = 10
        return i + 15
    labels = ensemble_data['labels']
    pred_probs = ensemble_data['pred_probs']
    labels_unlabeled = ensemble_data['labels_unlabeled']
    pred_probs_unlabeled = ensemble_data['pred_probs_unlabeled']
    (active_learning_scores, active_learning_scores_unlabeled) = get_active_learning_scores_ensemble(labels, pred_probs, pred_probs_unlabeled)
    assert isinstance(active_learning_scores, np.ndarray)
    assert len(active_learning_scores) == len(labels)
    assert len(active_learning_scores_unlabeled) == pred_probs_unlabeled.shape[1]
    (active_learning_scores, active_learning_scores_unlabeled) = get_active_learning_scores_ensemble(np.array(labels), pred_probs)
    assert isinstance(active_learning_scores, np.ndarray)
    assert len(active_learning_scores) == len(labels)
    assert len(active_learning_scores_unlabeled) == 0
    (active_learning_scores, active_learning_scores_unlabeled) = get_active_learning_scores_ensemble(pred_probs_unlabeled=pred_probs_unlabeled)
    assert len(active_learning_scores) == 0
    assert len(active_learning_scores_unlabeled) == len(labels_unlabeled)
    try:
        (active_learning_scores, active_learning_scores_unlabeled) = get_active_learning_scores_ensemble(labels, pred_probs, pred_probs_unlabeled[:, :-1])
    except ValueError as e:
        assert 'must have the same number of classes' in str(e)
    single_labels = ensemble_data['complete_labels'].iloc[[0]]
    singe_pred_probs = pred_probs[:, [0]]
    singe_pred_probs_unlabeled = pred_probs_unlabeled[:, [0]]
    get_active_learning_scores_ensemble(single_labels, singe_pred_probs, singe_pred_probs_unlabeled)
    labels = pd.DataFrame([[0, np.NaN, np.NaN], [np.NaN, 1, np.NaN], [np.NaN, np.NaN, 2], [np.NaN, 1, np.NaN], [np.NaN, np.NaN, 2]])
    pred_probs = np.random.random((2, 5, 3))
    get_active_learning_scores_ensemble(labels, pred_probs)

def test_single_label_active_learning():
    if False:
        print('Hello World!')
    labels = np.array(small_data['complete_labels'])
    labels_unlabeled = small_data['true_labels_train_unlabeled']
    pred_probs = small_data['pred_probs_complete']
    pred_probs_unlabeled = small_data['pred_probs_unlabeled']
    assert len(labels) == 15
    for i in range(5):
        (active_learning_scores, active_learning_scores_unlabeled) = get_active_learning_scores(labels, pred_probs, pred_probs_unlabeled)
        min_ind = np.argmin(active_learning_scores_unlabeled)
        labels = np.append(labels, labels_unlabeled[min_ind]).reshape(-1, 1)
        pred_probs = np.append(pred_probs, pred_probs_unlabeled[min_ind].reshape(1, -1), axis=0)
        labels_unlabeled = np.delete(labels_unlabeled, min_ind)
        pred_probs_unlabeled = np.delete(pred_probs_unlabeled, min_ind, axis=0)
    assert len(labels) == 20
    labels_flat = np.array(small_data['complete_labels']).reshape(1, -1)
    try:
        (active_learning_scores, active_learning_scores_unlabeled) = get_active_learning_scores(labels, pred_probs, pred_probs_unlabeled)
    except ValueError as e:
        assert 'labels_multiannotator must be a 2D array or dataframe' in str(e)

def test_single_label_active_learning_ensemble():
    if False:
        i = 10
        return i + 15
    labels = np.array(small_data['complete_labels'])
    labels_unlabeled = small_data['true_labels_train_unlabeled']
    pred_probs = small_data['pred_probs_complete']
    pred_probs_unlabeled = small_data['pred_probs_unlabeled']
    assert len(labels) == 15
    for i in range(5):
        (active_learning_scores, active_learning_scores_unlabeled) = get_active_learning_scores_ensemble(labels, np.array([pred_probs, pred_probs]), np.array([pred_probs_unlabeled, pred_probs_unlabeled]))
        min_ind = np.argmin(active_learning_scores_unlabeled)
        labels = np.append(labels, labels_unlabeled[min_ind]).reshape(-1, 1)
        pred_probs = np.append(pred_probs, pred_probs_unlabeled[min_ind].reshape(1, -1), axis=0)
        labels_unlabeled = np.delete(labels_unlabeled, min_ind)
        pred_probs_unlabeled = np.delete(pred_probs_unlabeled, min_ind, axis=0)
    assert len(labels) == 20
    labels_flat = np.array(small_data['complete_labels']).reshape(1, -1)
    try:
        (active_learning_scores, active_learning_scores_unlabeled) = get_active_learning_scores_ensemble(labels, np.array([pred_probs, pred_probs]), np.array([pred_probs_unlabeled, pred_probs_unlabeled]))
    except ValueError as e:
        assert 'labels_multiannotator must be a 2D array or dataframe' in str(e)

def test_missing_class():
    if False:
        print('Hello World!')
    labels = np.array([[1, np.NaN, 2], [1, 1, 2], [2, 2, 1], [np.NaN, 2, 2], [np.NaN, 2, 1], [np.NaN, 2, 2]])
    pred_probs = np.array([[0.4, 0.4, 0.2], [0.3, 0.6, 0.1], [0.05, 0.2, 0.75], [0.1, 0.4, 0.5], [0.2, 0.4, 0.4], [0.2, 0.4, 0.4]])
    consensus_label = get_majority_vote_label(labels)
    consensus_label = get_majority_vote_label(labels, pred_probs)
    multiannotator_dict = get_label_quality_multiannotator(labels, pred_probs)
    multiannotator_dict = get_label_quality_multiannotator(labels, pred_probs, quality_method='agreement')
    multiannotator_dict = get_label_quality_multiannotator(labels, pred_probs, consensus_method='majority_vote')

@pytest.mark.filterwarnings('ignore::UserWarning')
def test_rare_class():
    if False:
        print('Hello World!')
    labels = np.array([[1, np.NaN, 2], [1, 1, 0], [2, 2, 0], [np.NaN, 2, 2], [np.NaN, 2, 1], [np.NaN, 2, 2]])
    pred_probs = np.array([[0.4, 0.4, 0.2], [0.3, 0.6, 0.1], [0.05, 0.2, 0.75], [0.1, 0.4, 0.5], [0.2, 0.4, 0.4], [0.2, 0.4, 0.4]])
    consensus_label = get_majority_vote_label(labels)
    multiannotator_dict = get_label_quality_multiannotator(labels, pred_probs)
    pred_probs_missing = np.array([[0.8, 0.2], [0.6, 0.14], [0.95, 0.05], [0.5, 0.5], [0.4, 0.6], [0.4, 0.6]])
    try:
        multiannotator_dict = get_label_quality_multiannotator(labels, pred_probs_missing)
    except ValueError as e:
        assert 'pred_probs must have at least 3 columns' in str(e)

@pytest.mark.filterwarnings('ignore::UserWarning')
def test_get_consensus_label():
    if False:
        print('Hello World!')
    labels = data['labels']
    consensus_label = get_majority_vote_label(labels)
    labels_tiebreaks = np.array([[1, 2, 0], [1, 1, 0], [1, 0, 0], [2, 2, 2], [1, 2, 0], [1, 2, 0]])
    pred_probs_tiebreaks = np.array([[0.4, 0.4, 0.2], [0.3, 0.6, 0.1], [0.75, 0.2, 0.05], [0.1, 0.4, 0.5], [0.2, 0.4, 0.4], [0.2, 0.4, 0.4]])
    consensus_label = get_majority_vote_label(labels_tiebreaks, pred_probs_tiebreaks)
    labels_tiebreaks = np.array([[1, np.NaN, np.NaN, 2, np.NaN], [np.NaN, 1, 0, np.NaN, np.NaN], [np.NaN, np.NaN, 0, np.NaN, np.NaN], [np.NaN, 2, np.NaN, np.NaN, np.NaN], [2, np.NaN, 0, 2, np.NaN], [np.NaN, np.NaN, np.NaN, 2, 1]])
    consensus_label = get_majority_vote_label(labels_tiebreaks)
    assert all(consensus_label == np.array([1, 1, 0, 2, 2, 1]))

def test_impute_nonoverlaping_annotators():
    if False:
        while True:
            i = 10
    labels = np.array([[1, np.NaN, np.NaN], [np.NaN, 1, 0], [np.NaN, 0, 0], [np.NaN, 2, 2], [np.NaN, 2, 0], [np.NaN, 2, 0]])
    pred_probs = np.array([[0.4, 0.4, 0.2], [0.3, 0.6, 0.1], [0.75, 0.2, 0.05], [0.1, 0.4, 0.5], [0.2, 0.4, 0.4], [0.2, 0.4, 0.4]])
    multiannotator_dict = get_label_quality_multiannotator(labels, pred_probs)
    multiannotator_dict = get_label_quality_multiannotator(labels, pred_probs, quality_method='agreement')

def test_format_multiannotator_labels():
    if False:
        return 10
    str_labels = np.array([['a', 'b', 'c'], ['b', 'b', np.NaN], ['z', np.NaN, 'c']])
    (labels, label_map) = format_multiannotator_labels(str_labels)
    assert isinstance(labels, pd.DataFrame)
    assert label_map[0] == 'a'
    assert label_map[3] == 'z'
    num_labels = pd.DataFrame([[3, 2, 1], [1, 2, np.NaN], [3, np.NaN, 3]])
    (labels, label_map) = format_multiannotator_labels(num_labels)