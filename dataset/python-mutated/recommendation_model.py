import mlflow.pyfunc
import random
import numpy as np
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.feature_selection import SequentialFeatureSelector as sfs
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier as knc

def select_features(X, y):
    if False:
        return 10
    '\n    Dimensional reduction of X using k-nearest neighbors and sequential feature selector.\n    Final dimension set to three features.\n    Params:\n        X: Array which will be reduced in dimension (batch_size, n_features).\n        y: Array of labels (batch_size,).\n    Output: function that reduces dimension of array.\n    '
    knn = knc(n_neighbors=3)
    selector = sfs(knn, n_features_to_select=3)
    X_transformed = selector.fit_transform(X, y)

    def transform(input):
        if False:
            print('Hello World!')
        return selector.transform(input)
    return (transform, X_transformed)

def sort_database(X, y):
    if False:
        while True:
            i = 10
    '\n    Random shuffle of training values with its respective labels.\n    Params:\n        X: Array of features.\n        y: Array of labels.\n    Output: Tuple (X_rand_sorted, y_rand_sorted).\n    '
    sort_list = list(range(len(y)))
    random.shuffle(sort_list)
    return (X[sort_list], y[sort_list])

def preprocess(X):
    if False:
        return 10
    '\n    Preprocessing of features (no dimensional reduction) using principal component analysis.\n    Params:\n        X: Array of features.\n    Output: Tuple (processed array of features function that reduces dimension of array).\n    '
    (_, n) = X.shape
    pca = PCA(n_components=n)
    x = pca.fit_transform(normalize(X))

    def transform(input):
        if False:
            while True:
                i = 10
        return pca.transform(normalize(input))
    return (x, transform)

class SVM_recommendation(mlflow.pyfunc.PythonModel):

    def __init__(self, test=False, **params):
        if False:
            return 10
        f'{SVC.__doc__}'
        params['probability'] = True
        self.svm = SVC(**params)
        self.transforms = []
        self.score = 0
        self.confusion_matrix = None
        if test:
            knn = knc(n_neighbors=3)
            self.transform = [PCA(n_components=3), sfs(knn, n_features_to_select=2)]

    def fit(self, X, y):
        if False:
            return 10
        '\n        Train preprocess function, feature selection and Support Vector Machine model\n        Params:\n            X: Array of features.\n            y: Array of labels.\n        '
        assert X.shape[0] == y.shape[0], 'X and y must have same length'
        assert len(X.shape) == 2, 'X must be a two dimension vector'
        (X, t1) = preprocess(X)
        (t2, X) = select_features(X, y)
        self.transforms = [t1, t2]
        self.svm.fit(X, y)
        pred = self.svm.predict(X)
        z = y + 2 * pred
        n = len(z)
        false_pos = np.count_nonzero(z == 1) / n
        false_neg = np.count_nonzero(z == 2) / n
        true_pos = np.count_nonzero(z == 3) / n
        true_neg = 1 - false_neg - false_pos - true_pos
        self.confusion_matrix = np.array([[true_neg, false_pos], [false_neg, true_pos]])
        self.score = true_pos + true_neg

    def predict(self, x):
        if False:
            i = 10
            return i + 15
        '\n            Transform and prediction of input features and sorting of each by probability\n            Params:\n                X: Array of features.\n            Output: prediction probability for True (1).\n            '
        for t in self.transforms:
            x = t(x)
        return self.svm.predict_proba(x)[:, 1]

    def recommendation_order(self, x):
        if False:
            for i in range(10):
                print('nop')
        '\n        Transform and prediction of input features and sorting of each by probability\n        Params:\n            X: Array of features.\n        Output: Tuple (sorted_features, predictions).\n        '
        for t in self.transforms:
            x = t(x)
        pred = self.svm.predict_proba(x)
        return (sorted(range(len(pred)), key=lambda k: pred[k][1], reverse=True), pred)

    def plots(self):
        if False:
            i = 10
            return i + 15
        "\n        Returns the plots in a dict format.\n            {\n                'confusion_matrix': confusion matrix figure,\n            }\n        "
        display = metrics.ConfusionMatrixDisplay(confusion_matrix=self.confusion_matrix, display_labels=[False, True])
        return {'confusion_matrix': display.plot().figure_}