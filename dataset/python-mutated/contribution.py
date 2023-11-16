from decimal import Decimal
from typing import Optional
from flask_babel import gettext as _
from pandas import DataFrame
from superset.exceptions import InvalidPostProcessingError
from superset.utils.core import PostProcessingContributionOrientation
from superset.utils.pandas_postprocessing.utils import validate_column_args

@validate_column_args('columns')
def contribution(df: DataFrame, orientation: Optional[PostProcessingContributionOrientation]=PostProcessingContributionOrientation.COLUMN, columns: Optional[list[str]]=None, rename_columns: Optional[list[str]]=None) -> DataFrame:
    if False:
        i = 10
        return i + 15
    '\n    Calculate cell contribution to row/column total for numeric columns.\n    Non-numeric columns will be kept untouched.\n\n    If `columns` are specified, only calculate contributions on selected columns.\n\n    :param df: DataFrame containing all-numeric data (temporal column ignored)\n    :param columns: Columns to calculate values from.\n    :param rename_columns: The new labels for the calculated contribution columns.\n                           The original columns will not be removed.\n    :param orientation: calculate by dividing cell with row/column total\n    :return: DataFrame with contributions.\n    '
    contribution_df = df.copy()
    numeric_df = contribution_df.select_dtypes(include=['number', Decimal])
    numeric_df.fillna(0, inplace=True)
    if columns:
        numeric_columns = numeric_df.columns.tolist()
        for col in columns:
            if col not in numeric_columns:
                raise InvalidPostProcessingError(_('Column "%(column)s" is not numeric or does not exists in the query results.', column=col))
    columns = columns or numeric_df.columns
    rename_columns = rename_columns or columns
    if len(rename_columns) != len(columns):
        raise InvalidPostProcessingError(_('`rename_columns` must have the same length as `columns`.'))
    numeric_df = numeric_df[columns]
    axis = 0 if orientation == PostProcessingContributionOrientation.COLUMN else 1
    numeric_df = numeric_df / numeric_df.values.sum(axis=axis, keepdims=True)
    contribution_df[rename_columns] = numeric_df
    return contribution_df