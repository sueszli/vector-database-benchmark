"""
============================================================
Custom refit strategy of a grid search with cross-validation
============================================================

This examples shows how a classifier is optimized by cross-validation,
which is done using the :class:`~sklearn.model_selection.GridSearchCV` object
on a development set that comprises only half of the available labeled data.

The performance of the selected hyper-parameters and trained model is
then measured on a dedicated evaluation set that was not used during
the model selection step.

More details on tools available for model selection can be found in the
sections on :ref:`cross_validation` and :ref:`grid_search`.
"""
from sklearn import datasets
digits = datasets.load_digits()
n_samples = len(digits.images)
X = digits.images.reshape((n_samples, -1))
y = digits.target == 8
print(f'The number of images is {X.shape[0]} and each image contains {X.shape[1]} pixels')
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.5, random_state=0)
scores = ['precision', 'recall']
import pandas as pd

def print_dataframe(filtered_cv_results):
    if False:
        while True:
            i = 10
    'Pretty print for filtered dataframe'
    for (mean_precision, std_precision, mean_recall, std_recall, params) in zip(filtered_cv_results['mean_test_precision'], filtered_cv_results['std_test_precision'], filtered_cv_results['mean_test_recall'], filtered_cv_results['std_test_recall'], filtered_cv_results['params']):
        print(f'precision: {mean_precision:0.3f} (±{std_precision:0.03f}), recall: {mean_recall:0.3f} (±{std_recall:0.03f}), for {params}')
    print()

def refit_strategy(cv_results):
    if False:
        i = 10
        return i + 15
    'Define the strategy to select the best estimator.\n\n    The strategy defined here is to filter-out all results below a precision threshold\n    of 0.98, rank the remaining by recall and keep all models with one standard\n    deviation of the best by recall. Once these models are selected, we can select the\n    fastest model to predict.\n\n    Parameters\n    ----------\n    cv_results : dict of numpy (masked) ndarrays\n        CV results as returned by the `GridSearchCV`.\n\n    Returns\n    -------\n    best_index : int\n        The index of the best estimator as it appears in `cv_results`.\n    '
    precision_threshold = 0.98
    cv_results_ = pd.DataFrame(cv_results)
    print('All grid-search results:')
    print_dataframe(cv_results_)
    high_precision_cv_results = cv_results_[cv_results_['mean_test_precision'] > precision_threshold]
    print(f'Models with a precision higher than {precision_threshold}:')
    print_dataframe(high_precision_cv_results)
    high_precision_cv_results = high_precision_cv_results[['mean_score_time', 'mean_test_recall', 'std_test_recall', 'mean_test_precision', 'std_test_precision', 'rank_test_recall', 'rank_test_precision', 'params']]
    best_recall_std = high_precision_cv_results['mean_test_recall'].std()
    best_recall = high_precision_cv_results['mean_test_recall'].max()
    best_recall_threshold = best_recall - best_recall_std
    high_recall_cv_results = high_precision_cv_results[high_precision_cv_results['mean_test_recall'] > best_recall_threshold]
    print('Out of the previously selected high precision models, we keep all the\nthe models within one standard deviation of the highest recall model:')
    print_dataframe(high_recall_cv_results)
    fastest_top_recall_high_precision_index = high_recall_cv_results['mean_score_time'].idxmin()
    print(f'\nThe selected final model is the fastest to predict out of the previously\nselected subset of best models based on precision and recall.\nIts scoring time is:\n\n{high_recall_cv_results.loc[fastest_top_recall_high_precision_index]}')
    return fastest_top_recall_high_precision_index
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [0.001, 0.0001], 'C': [1, 10, 100, 1000]}, {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
grid_search = GridSearchCV(SVC(), tuned_parameters, scoring=scores, refit=refit_strategy)
grid_search.fit(X_train, y_train)
grid_search.best_params_
from sklearn.metrics import classification_report
y_pred = grid_search.predict(X_test)
print(classification_report(y_test, y_pred))