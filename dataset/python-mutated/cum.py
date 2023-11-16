from flask_babel import gettext as _
from pandas import DataFrame
from superset.exceptions import InvalidPostProcessingError
from superset.utils.pandas_postprocessing.utils import _append_columns, ALLOWLIST_CUMULATIVE_FUNCTIONS, validate_column_args

@validate_column_args('columns')
def cum(df: DataFrame, operator: str, columns: dict[str, str]) -> DataFrame:
    if False:
        return 10
    "\n    Calculate cumulative sum/product/min/max for select columns.\n\n    :param df: DataFrame on which the cumulative operation will be based.\n    :param columns: columns on which to perform a cumulative operation, mapping source\n           column to target column. For instance, `{'y': 'y'}` will replace the column\n           `y` with the cumulative value in `y`, while `{'y': 'y2'}` will add a column\n           `y2` based on cumulative values calculated from `y`, leaving the original\n           column `y` unchanged.\n    :param operator: cumulative operator, e.g. `sum`, `prod`, `min`, `max`\n    :return: DataFrame with cumulated columns\n    "
    columns = columns or {}
    df_cum = df.loc[:, columns.keys()]
    operation = 'cum' + operator
    if operation not in ALLOWLIST_CUMULATIVE_FUNCTIONS or not hasattr(df_cum, operation):
        raise InvalidPostProcessingError(_('Invalid cumulative operator: %(operator)s', operator=operator))
    df_cum = _append_columns(df, getattr(df_cum, operation)(), columns)
    return df_cum