import numpy as np
import pandas as pd
from ydata_profiling.config import Settings
from ydata_profiling.model.missing import missing_bar, missing_heatmap, missing_matrix
from ydata_profiling.visualisation.missing import plot_missing_bar, plot_missing_heatmap, plot_missing_matrix

@missing_bar.register
def pandas_missing_bar(config: Settings, df: pd.DataFrame) -> str:
    if False:
        i = 10
        return i + 15
    notnull_counts = len(df) - df.isnull().sum()
    return plot_missing_bar(config, notnull_counts=notnull_counts, nrows=len(df), columns=list(df.columns))

@missing_matrix.register
def pandas_missing_matrix(config: Settings, df: pd.DataFrame) -> str:
    if False:
        return 10
    return plot_missing_matrix(config, columns=list(df.columns), notnull=df.notnull().values, nrows=len(df))

@missing_heatmap.register
def pandas_missing_heatmap(config: Settings, df: pd.DataFrame) -> str:
    if False:
        return 10
    columns = [i for (i, n) in enumerate(np.var(df.isnull(), axis='rows')) if n > 0]
    df = df.iloc[:, columns]
    corr_mat = df.isnull().corr()
    mask = np.zeros_like(corr_mat)
    mask[np.triu_indices_from(mask)] = True
    return plot_missing_heatmap(config, corr_mat=corr_mat, mask=mask, columns=list(df.columns))