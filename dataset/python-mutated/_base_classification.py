import numpy as np
from scipy import sparse
from sklearn.base import ClassifierMixin
from ..externals.estimator_checks import check_is_fitted

class _BaseStackingClassifier(ClassifierMixin):
    """Base class of stacking classifiers"""

    def _do_predict(self, X, predict_fn):
        if False:
            i = 10
            return i + 15
        meta_features = self.predict_meta_features(X)
        if not self.use_features_in_secondary:
            return predict_fn(meta_features)
        elif sparse.issparse(X):
            return predict_fn(sparse.hstack((X, meta_features)))
        else:
            return predict_fn(np.hstack((X, meta_features)))

    def predict(self, X):
        if False:
            while True:
                i = 10
        'Predict target values for X.\n\n        Parameters\n        ----------\n        X : numpy array, shape = [n_samples, n_features]\n            Training vectors, where n_samples is the number of samples and\n            n_features is the number of features.\n\n        Returns\n        ----------\n        labels : array-like, shape = [n_samples]\n            Predicted class labels.\n\n        '
        check_is_fitted(self, ['clfs_', 'meta_clf_'])
        return self._do_predict(X, self.meta_clf_.predict)

    def predict_proba(self, X):
        if False:
            while True:
                i = 10
        ' Predict class probabilities for X.\n\n        Parameters\n        ----------\n        X : {array-like, sparse matrix}, shape = [n_samples, n_features]\n            Training vectors, where n_samples is the number of samples and\n            n_features is the number of features.\n\n        Returns\n        ----------\n        proba : array-like, shape = [n_samples, n_classes] or a list of                 n_outputs of such arrays if n_outputs > 1.\n            Probability for each class per sample.\n\n        '
        check_is_fitted(self, ['clfs_', 'meta_clf_'])
        return self._do_predict(X, self.meta_clf_.predict_proba)

    def decision_function(self, X):
        if False:
            for i in range(10):
                print('nop')
        ' Predict class confidence scores for X.\n\n        Parameters\n        ----------\n        X : {array-like, sparse matrix}, shape = [n_samples, n_features]\n            Training vectors, where n_samples is the number of samples and\n            n_features is the number of features.\n\n        Returns\n        ----------\n        scores : shape=(n_samples,) if n_classes == 2 else             (n_samples, n_classes).\n            Confidence scores per (sample, class) combination. In the binary\n            case, confidence score for self.classes_[1] where >0 means this\n            class would be predicted.\n\n        '
        check_is_fitted(self, ['clfs_', 'meta_clf_'])
        return self._do_predict(X, self.meta_clf_.decision_function)