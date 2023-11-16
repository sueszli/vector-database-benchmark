import pandas as pd

def deduplicate_timeseries_dataframe(df, dt_col):
    if False:
        while True:
            i = 10
    '\n    deduplicate and return a dataframe with no identical rows.\n    :param df: input dataframe.\n    :param dt_col: name of datetime colomn.\n    '
    from bigdl.nano.utils.common import invalidInputError
    invalidInputError(dt_col in df.columns, f'dt_col {dt_col} can not be found in df.')
    invalidInputError(pd.isna(df[dt_col]).sum() == 0, 'There is N/A in datetime col')
    res_df = df.drop_duplicates()
    return res_df