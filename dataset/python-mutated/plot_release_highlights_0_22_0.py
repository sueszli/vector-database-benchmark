"""
========================================
Release Highlights for scikit-learn 0.22
========================================

.. currentmodule:: sklearn

We are pleased to announce the release of scikit-learn 0.22, which comes
with many bug fixes and new features! We detail below a few of the major
features of this release. For an exhaustive list of all the changes, please
refer to the :ref:`release notes <changes_0_22>`.

To install the latest version (with pip)::

    pip install --upgrade scikit-learn

or with conda::

    conda install -c conda-forge scikit-learn

"""
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import RocCurveDisplay
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
(X, y) = make_classification(random_state=0)
(X_train, X_test, y_train, y_test) = train_test_split(X, y, random_state=42)
svc = SVC(random_state=42)
svc.fit(X_train, y_train)
rfc = RandomForestClassifier(random_state=42)
rfc.fit(X_train, y_train)
svc_disp = RocCurveDisplay.from_estimator(svc, X_test, y_test)
rfc_disp = RocCurveDisplay.from_estimator(rfc, X_test, y_test, ax=svc_disp.ax_)
rfc_disp.figure_.suptitle('ROC curve comparison')
plt.show()
from sklearn.datasets import load_iris
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
(X, y) = load_iris(return_X_y=True)
estimators = [('rf', RandomForestClassifier(n_estimators=10, random_state=42)), ('svr', make_pipeline(StandardScaler(), LinearSVC(dual='auto', random_state=42)))]
clf = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
(X_train, X_test, y_train, y_test) = train_test_split(X, y, stratify=y, random_state=42)
clf.fit(X_train, y_train).score(X_test, y_test)
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
(X, y) = make_classification(random_state=0, n_features=5, n_informative=3)
feature_names = np.array([f'x_{i}' for i in range(X.shape[1])])
rf = RandomForestClassifier(random_state=0).fit(X, y)
result = permutation_importance(rf, X, y, n_repeats=10, random_state=0, n_jobs=2)
(fig, ax) = plt.subplots()
sorted_idx = result.importances_mean.argsort()
ax.boxplot(result.importances[sorted_idx].T, vert=False, labels=feature_names[sorted_idx])
ax.set_title('Permutation Importance of each feature')
ax.set_ylabel('Features')
fig.tight_layout()
plt.show()
from sklearn.ensemble import HistGradientBoostingClassifier
X = np.array([0, 1, 2, np.nan]).reshape(-1, 1)
y = [0, 0, 1, 1]
gbdt = HistGradientBoostingClassifier(min_samples_leaf=1).fit(X, y)
print(gbdt.predict(X))
from tempfile import TemporaryDirectory
from sklearn.manifold import Isomap
from sklearn.neighbors import KNeighborsTransformer
from sklearn.pipeline import make_pipeline
(X, y) = make_classification(random_state=0)
with TemporaryDirectory(prefix='sklearn_cache_') as tmpdir:
    estimator = make_pipeline(KNeighborsTransformer(n_neighbors=10, mode='distance'), Isomap(n_neighbors=10, metric='precomputed'), memory=tmpdir)
    estimator.fit(X)
    estimator.set_params(isomap__n_neighbors=5)
    estimator.fit(X)
from sklearn.impute import KNNImputer
X = [[1, 2, np.nan], [3, 4, 3], [np.nan, 6, 5], [8, 8, 7]]
imputer = KNNImputer(n_neighbors=2)
print(imputer.fit_transform(X))
(X, y) = make_classification(random_state=0)
rf = RandomForestClassifier(random_state=0, ccp_alpha=0).fit(X, y)
print('Average number of nodes without pruning {:.1f}'.format(np.mean([e.tree_.node_count for e in rf.estimators_])))
rf = RandomForestClassifier(random_state=0, ccp_alpha=0.05).fit(X, y)
print('Average number of nodes with pruning {:.1f}'.format(np.mean([e.tree_.node_count for e in rf.estimators_])))
from sklearn.datasets import fetch_openml
titanic = fetch_openml('titanic', version=1, as_frame=True, parser='pandas')
print(titanic.data.head()[['pclass', 'embarked']])
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils.estimator_checks import parametrize_with_checks

@parametrize_with_checks([LogisticRegression(), DecisionTreeRegressor()])
def test_sklearn_compatible_estimator(estimator, check):
    if False:
        print('Hello World!')
    check(estimator)
from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC
(X, y) = make_classification(n_classes=4, n_informative=16)
clf = SVC(decision_function_shape='ovo', probability=True).fit(X, y)
print(roc_auc_score(y, clf.predict_proba(X), multi_class='ovo'))