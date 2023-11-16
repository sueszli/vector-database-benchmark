import numpy as np
import pandas as pd
import sklearn.metrics as metrics
from bigdl.chronos.utils import deprecated
from bigdl.chronos.autots.deprecated.preprocessing.impute.abstract import BaseImpute

@deprecated('Please use `bigdl.chronos.data.TSDataset` instead.')
class LastFill(BaseImpute):
    """
    Impute missing data with last seen value
    """

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Construct model for last filling method\n        '
        pass

    def impute(self, df):
        if False:
            return 10
        '\n        impute data\n        :params df: input dataframe\n        :return: imputed dataframe\n        '
        df.iloc[0] = df.iloc[0].fillna(0)
        return df.fillna(method='pad')