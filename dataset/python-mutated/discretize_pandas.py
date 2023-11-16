from enum import Enum
from typing import List
import numpy as np
import pandas as pd

class DiscretizationType(Enum):
    UNIFORM = 'uniform'
    QUANTILE = 'quantile'

class Discretizer:
    """
    A class which enables the discretization of a pandas dataframe.
    Perform this action when you want to convert a continuous variable
    into a categorical variable.

    Attributes:

    method (DiscretizationType): this attribute controls how the buckets
    of your discretization are formed. A uniform discretization type forms
    the bins to be of equal width whereas a quantile discretization type
    forms the bins to be of equal size.

    n_bins (int): number of bins
    reset_index (bool): instruction to reset the index of
                        the dataframe after the discretization
    """

    def __init__(self, method: DiscretizationType, n_bins: int=10, reset_index: bool=False) -> None:
        if False:
            return 10
        self.discretization_type = method
        self.n_bins = n_bins
        self.reset_index = reset_index

    def discretize_dataframe(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        if False:
            return 10
        '_summary_\n\n        Args:\n            dataframe (pd.DataFrame): pandas dataframe\n\n        Returns:\n            pd.DataFrame: discretized dataframe\n        '
        discretized_df = dataframe.copy()
        all_columns = dataframe.columns
        num_columns = self._get_numerical_columns(dataframe)
        for column in num_columns:
            discretized_df.loc[:, column] = self._discretize_column(discretized_df[column])
        discretized_df = discretized_df[all_columns]
        return discretized_df.reset_index(drop=True) if self.reset_index else discretized_df

    def _discretize_column(self, column: pd.Series) -> pd.Series:
        if False:
            i = 10
            return i + 15
        if self.discretization_type == DiscretizationType.QUANTILE:
            return self._descritize_quantile(column)
        elif self.discretization_type == DiscretizationType.UNIFORM:
            return self._descritize_uniform(column)

    def _descritize_quantile(self, column: pd.Series) -> pd.Series:
        if False:
            print('Hello World!')
        return pd.qcut(column, q=self.n_bins, labels=False, retbins=False, duplicates='drop').values

    def _descritize_uniform(self, column: pd.Series) -> pd.Series:
        if False:
            print('Hello World!')
        return pd.cut(column, bins=self.n_bins, labels=False, retbins=True, duplicates='drop')[0].values

    def _get_numerical_columns(self, dataframe: pd.DataFrame) -> List[str]:
        if False:
            print('Hello World!')
        return dataframe.select_dtypes(include=np.number).columns.tolist()