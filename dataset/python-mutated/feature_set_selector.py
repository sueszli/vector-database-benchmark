"""This file is part of the TPOT library.

TPOT was primarily developed at the University of Pennsylvania by:
    - Randal S. Olson (rso@randalolson.com)
    - Weixuan Fu (weixuanf@upenn.edu)
    - Daniel Angell (dpa34@drexel.edu)
    - and many more generous open source contributors

TPOT is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as
published by the Free Software Foundation, either version 3 of
the License, or (at your option) any later version.

TPOT is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with TPOT. If not, see <http://www.gnu.org/licenses/>.
"""
import numpy as np
import pandas as pd
import os, os.path
from sklearn.base import BaseEstimator
try:
    from sklearn.feature_selection._base import SelectorMixin
except ImportError:
    from sklearn.feature_selection.base import SelectorMixin

class FeatureSetSelector(BaseEstimator, SelectorMixin):
    """Select predefined feature subsets."""

    @property
    def __name__(self):
        if False:
            i = 10
            return i + 15
        'Instance name is the same as the class name.'
        return self.__class__.__name__

    def __init__(self, subset_list, sel_subset):
        if False:
            for i in range(10):
                print('nop')
        "Create a FeatureSetSelector object.\n\n        Parameters\n        ----------\n        subset_list: string, required\n            Path to a file that indicates all the subset lists. Currently,\n            this file needs to be a .csv with one header row.\n            There should be 3 columns on the table, including subset names (Subset),\n            number of features (Size) and features in the subset (Features).\n            The feature names or indexs of input features\n            should be seprated by ';' on the 3rd column of the file.\n            The feature names in the files must match those in the (training and\n            testing) dataset.\n        sel_subset: int or string or list or tuple\n            int: index of subset in subset file\n            string: subset name of subset\n            list or tuple: list of int or string for indexs or subset names\n        Returns\n        -------\n        None\n\n        "
        self.subset_list = subset_list
        self.sel_subset = sel_subset

    def fit(self, X, y=None):
        if False:
            for i in range(10):
                print('nop')
        'Fit FeatureSetSelector for feature selection\n\n        Parameters\n        ----------\n        X: array-like of shape (n_samples, n_features)\n            The training input samples.\n        y: array-like, shape (n_samples,)\n            The target values (integers that correspond to classes in classification, real numbers in regression).\n\n        Returns\n        -------\n        self: object\n            Returns a copy of the estimator\n        '
        subset_df = pd.read_csv(self.subset_list, header=0, index_col=0)
        if isinstance(self.sel_subset, int):
            self.sel_subset_name = subset_df.index[self.sel_subset]
        elif isinstance(self.sel_subset, str):
            self.sel_subset_name = self.sel_subset
        else:
            self.sel_subset_name = []
            for s in self.sel_subset:
                if isinstance(s, int):
                    self.sel_subset_name.append(subset_df.index[s])
                else:
                    self.sel_subset_name.append(s)
        sel_features = subset_df.loc[self.sel_subset_name, 'Features']
        if not isinstance(sel_features, str):
            sel_features = ';'.join(sel_features.tolist())
        sel_uniq_features = set(sel_features.split(';'))
        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns.values)
            self.feat_list = sorted(list(set(sel_uniq_features).intersection(set(self.feature_names))))
            self.feat_list_idx = [list(X.columns).index(feat_name) for feat_name in self.feat_list]
        elif isinstance(X, np.ndarray):
            self.feature_names = list(range(X.shape[1]))
            sel_uniq_features = [int(val) for val in sel_uniq_features]
            self.feat_list = sorted(list(set(sel_uniq_features).intersection(set(self.feature_names))))
            self.feat_list_idx = self.feat_list
        if not len(self.feat_list):
            raise ValueError('No feature is found on the subset list!')
        return self

    def transform(self, X):
        if False:
            i = 10
            return i + 15
        'Make subset after fit\n\n        Parameters\n        ----------\n        X: numpy ndarray, {n_samples, n_features}\n            New data, where n_samples is the number of samples and n_features is the number of features.\n\n        Returns\n        -------\n        X_transformed: array-like, shape (n_samples, n_features + 1) or (n_samples, n_features + 1 + n_classes) for classifier with predict_proba attribute\n            The transformed feature set.\n        '
        if isinstance(X, pd.DataFrame):
            X_transformed = X[self.feat_list].values
        elif isinstance(X, np.ndarray):
            X_transformed = X[:, self.feat_list_idx]
        return X_transformed.astype(np.float64)

    def _get_support_mask(self):
        if False:
            print('Hello World!')
        '\n        Get the boolean mask indicating which features are selected\n        Returns\n        -------\n        support : boolean array of shape [# input features]\n            An element is True iff its corresponding feature is selected for\n            retention.\n        '
        n_features = len(self.feature_names)
        mask = np.zeros(n_features, dtype=bool)
        mask[np.asarray(self.feat_list_idx)] = True
        return mask