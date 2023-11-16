"""
=================================
Bagging classifiers using sampler
=================================

In this example, we show how
:class:`~imblearn.ensemble.BalancedBaggingClassifier` can be used to create a
large variety of classifiers by giving different samplers.

We will give several examples that have been published in the passed year.
"""
print(__doc__)
from sklearn.datasets import make_classification
(X, y) = make_classification(n_samples=10000, n_features=10, weights=[0.1, 0.9], class_sep=0.5, random_state=0)
import pandas as pd
pd.Series(y).value_counts(normalize=True)
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import cross_validate
ebb = BaggingClassifier()
cv_results = cross_validate(ebb, X, y, scoring='balanced_accuracy')
print(f"{cv_results['test_score'].mean():.3f} +/- {cv_results['test_score'].std():.3f}")
from imblearn.ensemble import BalancedBaggingClassifier
from imblearn.under_sampling import RandomUnderSampler
ebb = BalancedBaggingClassifier(sampler=RandomUnderSampler())
cv_results = cross_validate(ebb, X, y, scoring='balanced_accuracy')
print(f"{cv_results['test_score'].mean():.3f} +/- {cv_results['test_score'].std():.3f}")
from imblearn.over_sampling import RandomOverSampler
over_bagging = BalancedBaggingClassifier(sampler=RandomOverSampler())
cv_results = cross_validate(over_bagging, X, y, scoring='balanced_accuracy')
print(f"{cv_results['test_score'].mean():.3f} +/- {cv_results['test_score'].std():.3f}")
from imblearn.over_sampling import SMOTE
smote_bagging = BalancedBaggingClassifier(sampler=SMOTE())
cv_results = cross_validate(smote_bagging, X, y, scoring='balanced_accuracy')
print(f"{cv_results['test_score'].mean():.3f} +/- {cv_results['test_score'].std():.3f}")
from collections import Counter
import numpy as np
from imblearn import FunctionSampler

def roughly_balanced_bagging(X, y, replace=False):
    if False:
        print('Hello World!')
    'Implementation of Roughly Balanced Bagging for binary problem.'
    class_counts = Counter(y)
    majority_class = max(class_counts, key=class_counts.get)
    minority_class = min(class_counts, key=class_counts.get)
    n_minority_class = class_counts[minority_class]
    n_majority_resampled = np.random.negative_binomial(n=n_minority_class, p=0.5)
    majority_indices = np.random.choice(np.flatnonzero(y == majority_class), size=n_majority_resampled, replace=replace)
    minority_indices = np.random.choice(np.flatnonzero(y == minority_class), size=n_minority_class, replace=replace)
    indices = np.hstack([majority_indices, minority_indices])
    return (X[indices], y[indices])
rbb = BalancedBaggingClassifier(sampler=FunctionSampler(func=roughly_balanced_bagging, kw_args={'replace': True}))
cv_results = cross_validate(rbb, X, y, scoring='balanced_accuracy')
print(f"{cv_results['test_score'].mean():.3f} +/- {cv_results['test_score'].std():.3f}")