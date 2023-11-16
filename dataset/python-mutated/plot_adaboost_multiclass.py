"""
=====================================
Multi-class AdaBoosted Decision Trees
=====================================

This example shows how boosting can improve the prediction accuracy on a
multi-label classification problem. It reproduces a similar experiment as
depicted by Figure 1 in Zhu et al [1]_.

The core principle of AdaBoost (Adaptive Boosting) is to fit a sequence of weak
learners (e.g. Decision Trees) on repeatedly re-sampled versions of the data.
Each sample carries a weight that is adjusted after each training step, such
that misclassified samples will be assigned higher weights. The re-sampling
process with replacement takes into account the weights assigned to each sample.
Samples with higher weights have a greater chance of being selected multiple
times in the new data set, while samples with lower weights are less likely to
be selected. This ensures that subsequent iterations of the algorithm focus on
the difficult-to-classify samples.

.. topic:: References:

    .. [1] :doi:`J. Zhu, H. Zou, S. Rosset, T. Hastie, "Multi-class adaboost."
           Statistics and its Interface 2.3 (2009): 349-360.
           <10.4310/SII.2009.v2.n3.a8>`

"""
from sklearn.datasets import make_gaussian_quantiles
(X, y) = make_gaussian_quantiles(n_samples=2000, n_features=10, n_classes=3, random_state=1)
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(X, y, train_size=0.7, random_state=42)
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
weak_learner = DecisionTreeClassifier(max_leaf_nodes=8)
n_estimators = 300
adaboost_clf = AdaBoostClassifier(estimator=weak_learner, n_estimators=n_estimators, algorithm='SAMME', random_state=42).fit(X_train, y_train)
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score
dummy_clf = DummyClassifier()

def misclassification_error(y_true, y_pred):
    if False:
        for i in range(10):
            print('nop')
    return 1 - accuracy_score(y_true, y_pred)
weak_learners_misclassification_error = misclassification_error(y_test, weak_learner.fit(X_train, y_train).predict(X_test))
dummy_classifiers_misclassification_error = misclassification_error(y_test, dummy_clf.fit(X_train, y_train).predict(X_test))
print(f"DecisionTreeClassifier's misclassification_error: {weak_learners_misclassification_error:.3f}")
print(f"DummyClassifier's misclassification_error: {dummy_classifiers_misclassification_error:.3f}")
import matplotlib.pyplot as plt
import pandas as pd
boosting_errors = pd.DataFrame({'Number of trees': range(1, n_estimators + 1), 'AdaBoost': [misclassification_error(y_test, y_pred) for y_pred in adaboost_clf.staged_predict(X_test)]}).set_index('Number of trees')
ax = boosting_errors.plot()
ax.set_ylabel('Misclassification error on test set')
ax.set_title('Convergence of AdaBoost algorithm')
plt.plot([boosting_errors.index.min(), boosting_errors.index.max()], [weak_learners_misclassification_error, weak_learners_misclassification_error], color='tab:orange', linestyle='dashed')
plt.plot([boosting_errors.index.min(), boosting_errors.index.max()], [dummy_classifiers_misclassification_error, dummy_classifiers_misclassification_error], color='c', linestyle='dotted')
plt.legend(['AdaBoost', 'DecisionTreeClassifier', 'DummyClassifier'], loc=1)
plt.show()
weak_learners_info = pd.DataFrame({'Number of trees': range(1, n_estimators + 1), 'Errors': adaboost_clf.estimator_errors_, 'Weights': adaboost_clf.estimator_weights_}).set_index('Number of trees')
axs = weak_learners_info.plot(subplots=True, layout=(1, 2), figsize=(10, 4), legend=False, color='tab:blue')
axs[0, 0].set_ylabel('Train error')
axs[0, 0].set_title("Weak learner's training error")
axs[0, 1].set_ylabel('Weight')
axs[0, 1].set_title("Weak learner's weight")
fig = axs[0, 0].get_figure()
fig.suptitle("Weak learner's errors and weights for the AdaBoostClassifier")
fig.tight_layout()