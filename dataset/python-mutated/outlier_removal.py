from mage_ai.data_cleaner.estimators.base import BaseEstimator
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import numpy as np
MAX_CONTAMINATION_RATE = 0.15
MIN_CONTAMINATION_RATE = 0.025
SUPPORTED_METHODS = ['lof', 'auto', 'itree']

class OutlierRemover(BaseEstimator):
    """
    Automatically removes outliers from the given data.
    """

    def __init__(self, method: str='auto') -> None:
        if False:
            while True:
                i = 10
        "\n        Constructs an outlier remover.\n\n        Args:\n            method (str, optional): Specifies the outlier removal method to use. Defaults to\n            'auto'. There are three options:\n            - 'lof' - Local Outlier Factor\n            - 'itree' - Isolation Forest\n            - 'auto' - Automatically best method based on runtime comparisons\n        "
        if method not in SUPPORTED_METHODS:
            raise ValueError(f"The method specified '{method}' is not supported.")
        self.method = method

    def estimate_contamination_rate(self, X: np.ndarray) -> float:
        if False:
            return 10
        '\n        Guesses the outlier contamination rate from the data. Uses the IQR\n        rule to estimate outliers dimensionwise, and then aggregates the entire\n        count to provide an estimate on the number of outliers in the dataset. This\n        estimate is used to compute the contamination factor used in finding outliers with\n        more complex algorithms.\n\n        The exact algorithm used to estimate the contamination rate is as follows:\n        1. Calculate the first and third quantile for each dimension\n        2. Calculate the interquartile range for each dimension\n        3. Compute the per-dimension outlier region as x <= lower - 1.5*IQR and x >= upper + 1.5*IQR\n        4. Find all entries that exist in the outlier region per-dimension\n        5. Mark examples with at least 50% of its features as outliers to be potential\n        multidimensional outliers\n        6. Return the ratio of these potential outliers in the region\n\n        Args:\n            X (np.ndarray): Input array of shape (n_samples, n_dimensions)\n\n        Returns:\n            float: The guessed contamination rate\n        '
        first = np.quantile(X, 0.25, axis=0)
        third = np.quantile(X, 0.75, axis=0)
        iqr_diffs = 1.5 * (third - first)
        lower_bound = first - iqr_diffs
        upper_bound = third + iqr_diffs
        mask = (X < lower_bound) | (X > upper_bound)
        mask = mask.sum(axis=1) / mask.shape[1]
        mask = mask >= 0.5
        return mask.sum() / mask.size

    def fit(self, X: np.ndarray, y: np.ndarray=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Fits the outlier remover on the given data.\n\n        Args:\n            X (np.ndarray): Input array of shape (n_samples, n_dimensions)\n            y (np.ndarray, optional): Not used. Defaults to None.\n        '
        (count, ndim) = X.shape
        pca_transformer = PCA(n_components=20, random_state=42)
        contamination_rate = max(self.estimate_contamination_rate(X), MIN_CONTAMINATION_RATE)
        contamination_rate = min(contamination_rate, MAX_CONTAMINATION_RATE)
        if ndim > 20:
            X = pca_transformer.fit_transform(X)
        if self.method == 'auto':
            if ndim <= 5:
                self.method = 'lof'
            else:
                self.method = 'itree'
        if self.method == 'lof':
            if count < 10:
                n_neighbors = 2
            elif count < 500:
                n_neighbors = count // 10 + 1
            elif contamination_rate <= 0.1:
                n_neighbors = 20
            else:
                n_neighbors = 35
            self.outlier_remover = LocalOutlierFactor(n_neighbors=n_neighbors, n_jobs=-1, contamination=contamination_rate)
        elif self.method == 'itree':
            self.outlier_remover = IsolationForest(contamination=contamination_rate, n_estimators=100, n_jobs=-1, random_state=42)
            self.outlier_remover.fit(X, y)

    def transform(self, X: np.ndarray, **kwargs) -> np.ndarray:
        if False:
            return 10
        '\n        Determines the labels of outliers in the input data\n\n        Args:\n            X (np.ndarray): Input array of shape (n_samples, n_dimensions)\n\n        Returns:\n            np.ndarray: Array of shape (n_samples,) containing labels for the entries.\n            Label is True if outlier, else False.\n        '
        if self.method == 'lof':
            labels = self.outlier_remover.fit_predict(X, **kwargs)
        else:
            labels = self.outlier_remover.predict(X, **kwargs)
        return labels == -1