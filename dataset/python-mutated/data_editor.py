from __future__ import annotations
import json
from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Mapping, Optional, Set, Tuple, TypeVar, Union, cast, overload
import pandas as pd
import pyarrow as pa
from typing_extensions import Literal, TypeAlias, TypedDict
from streamlit import logger as _logger
from streamlit import type_util
from streamlit.deprecation_util import deprecate_func_name
from streamlit.elements.form import current_form_id
from streamlit.elements.lib.column_config_utils import INDEX_IDENTIFIER, ColumnConfigMapping, ColumnConfigMappingInput, ColumnDataKind, DataframeSchema, apply_data_specific_configs, determine_dataframe_schema, is_type_compatible, marshall_column_config, process_config_mapping, update_column_config
from streamlit.elements.lib.pandas_styler_utils import marshall_styler
from streamlit.elements.utils import check_callback_rules, check_session_state_rules
from streamlit.errors import StreamlitAPIException
from streamlit.proto.Arrow_pb2 import Arrow as ArrowProto
from streamlit.runtime.metrics_util import gather_metrics
from streamlit.runtime.scriptrunner import get_script_run_ctx
from streamlit.runtime.state import WidgetArgs, WidgetCallback, WidgetKwargs, register_widget
from streamlit.runtime.state.common import compute_widget_id
from streamlit.type_util import DataFormat, DataFrameGenericAlias, Key, is_type, to_key
from streamlit.util import calc_md5
if TYPE_CHECKING:
    import numpy as np
    from pandas.io.formats.style import Styler
    from streamlit.delta_generator import DeltaGenerator
_LOGGER = _logger.get_logger('root')
EditableData = TypeVar('EditableData', bound=Union[DataFrameGenericAlias[Any], Tuple[Any], List[Any], Set[Any], Dict[str, Any]])
DataTypes: TypeAlias = Union[pd.DataFrame, pd.Series, pd.Index, 'Styler', pa.Table, 'np.ndarray[Any, np.dtype[np.float64]]', Tuple[Any], List[Any], Set[Any], Dict[str, Any]]

class EditingState(TypedDict, total=False):
    """
    A dictionary representing the current state of the data editor.

    Attributes
    ----------
    edited_rows : Dict[int, Dict[str, str | int | float | bool | None]]
        An hierarchical mapping of edited cells based on: row position -> column name -> value.

    added_rows : List[Dict[str, str | int | float | bool | None]]
        A list of added rows, where each row is a mapping from column name to the cell value.

    deleted_rows : List[int]
        A list of deleted rows, where each row is the numerical position of the deleted row.
    """
    edited_rows: Dict[int, Dict[str, str | int | float | bool | None]]
    added_rows: List[Dict[str, str | int | float | bool | None]]
    deleted_rows: List[int]

@dataclass
class DataEditorSerde:
    """DataEditorSerde is used to serialize and deserialize the data editor state."""

    def deserialize(self, ui_value: Optional[str], widget_id: str='') -> EditingState:
        if False:
            print('Hello World!')
        data_editor_state: EditingState = {'edited_rows': {}, 'added_rows': [], 'deleted_rows': []} if ui_value is None else json.loads(ui_value)
        if 'edited_rows' not in data_editor_state:
            data_editor_state['edited_rows'] = {}
        if 'deleted_rows' not in data_editor_state:
            data_editor_state['deleted_rows'] = []
        if 'added_rows' not in data_editor_state:
            data_editor_state['added_rows'] = []
        data_editor_state['edited_rows'] = {int(k): v for (k, v) in data_editor_state['edited_rows'].items()}
        return data_editor_state

    def serialize(self, editing_state: EditingState) -> str:
        if False:
            i = 10
            return i + 15
        return json.dumps(editing_state, default=str)

def _parse_value(value: str | int | float | bool | None, column_data_kind: ColumnDataKind) -> Any:
    if False:
        i = 10
        return i + 15
    'Convert a value to the correct type.\n\n    Parameters\n    ----------\n    value : str | int | float | bool | None\n        The value to convert.\n\n    column_data_kind : ColumnDataKind\n        The determined data kind of the column. The column data kind refers to the\n        shared data type of the values in the column (e.g. int, float, str).\n\n    Returns\n    -------\n    The converted value.\n    '
    if value is None:
        return None
    try:
        if column_data_kind == ColumnDataKind.STRING:
            return str(value)
        if column_data_kind == ColumnDataKind.INTEGER:
            return int(value)
        if column_data_kind == ColumnDataKind.FLOAT:
            return float(value)
        if column_data_kind == ColumnDataKind.BOOLEAN:
            return bool(value)
        if column_data_kind == ColumnDataKind.DECIMAL:
            return Decimal(str(value))
        if column_data_kind == ColumnDataKind.TIMEDELTA:
            return pd.Timedelta(value)
        if column_data_kind in [ColumnDataKind.DATETIME, ColumnDataKind.DATE, ColumnDataKind.TIME]:
            datetime_value = pd.Timestamp(value)
            if datetime_value is pd.NaT:
                return None
            if column_data_kind == ColumnDataKind.DATETIME:
                return datetime_value
            if column_data_kind == ColumnDataKind.DATE:
                return datetime_value.date()
            if column_data_kind == ColumnDataKind.TIME:
                return datetime_value.time()
    except (ValueError, pd.errors.ParserError) as ex:
        _LOGGER.warning('Failed to parse value %s as %s. Exception: %s', value, column_data_kind, ex)
        return None
    return value

def _apply_cell_edits(df: pd.DataFrame, edited_rows: Mapping[int, Mapping[str, str | int | float | bool | None]], dataframe_schema: DataframeSchema) -> None:
    if False:
        while True:
            i = 10
    'Apply cell edits to the provided dataframe (inplace).\n\n    Parameters\n    ----------\n    df : pd.DataFrame\n        The dataframe to apply the cell edits to.\n\n    edited_rows : Mapping[int, Mapping[str, str | int | float | bool | None]]\n        A hierarchical mapping based on row position -> column name -> value\n\n    dataframe_schema: DataframeSchema\n        The schema of the dataframe.\n    '
    for (row_id, row_changes) in edited_rows.items():
        row_pos = int(row_id)
        for (col_name, value) in row_changes.items():
            if col_name == INDEX_IDENTIFIER:
                df.index.values[row_pos] = _parse_value(value, dataframe_schema[INDEX_IDENTIFIER])
            else:
                col_pos = df.columns.get_loc(col_name)
                df.iat[row_pos, col_pos] = _parse_value(value, dataframe_schema[col_name])

def _apply_row_additions(df: pd.DataFrame, added_rows: List[Dict[str, Any]], dataframe_schema: DataframeSchema) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Apply row additions to the provided dataframe (inplace).\n\n    Parameters\n    ----------\n    df : pd.DataFrame\n        The dataframe to apply the row additions to.\n\n    added_rows : List[Dict[str, Any]]\n        A list of row additions. Each row addition is a dictionary with the\n        column position as key and the new cell value as value.\n\n    dataframe_schema: DataframeSchema\n        The schema of the dataframe.\n    '
    if not added_rows:
        return
    range_index_stop = None
    range_index_step = None
    if isinstance(df.index, pd.RangeIndex):
        range_index_stop = df.index.stop
        range_index_step = df.index.step
    for added_row in added_rows:
        index_value = None
        new_row: List[Any] = [None for _ in range(df.shape[1])]
        for col_name in added_row.keys():
            value = added_row[col_name]
            if col_name == INDEX_IDENTIFIER:
                index_value = _parse_value(value, dataframe_schema[INDEX_IDENTIFIER])
            else:
                col_pos = df.columns.get_loc(col_name)
                new_row[col_pos] = _parse_value(value, dataframe_schema[col_name])
        if range_index_stop is not None:
            df.loc[range_index_stop, :] = new_row
            range_index_stop += range_index_step
        elif index_value is not None:
            df.loc[index_value, :] = new_row

def _apply_row_deletions(df: pd.DataFrame, deleted_rows: List[int]) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Apply row deletions to the provided dataframe (inplace).\n\n    Parameters\n    ----------\n    df : pd.DataFrame\n        The dataframe to apply the row deletions to.\n\n    deleted_rows : List[int]\n        A list of row numbers to delete.\n    '
    df.drop(df.index[deleted_rows], inplace=True)

def _apply_dataframe_edits(df: pd.DataFrame, data_editor_state: EditingState, dataframe_schema: DataframeSchema) -> None:
    if False:
        return 10
    'Apply edits to the provided dataframe (inplace).\n\n    This includes cell edits, row additions and row deletions.\n\n    Parameters\n    ----------\n    df : pd.DataFrame\n        The dataframe to apply the edits to.\n\n    data_editor_state : EditingState\n        The editing state of the data editor component.\n\n    dataframe_schema: DataframeSchema\n        The schema of the dataframe.\n    '
    if data_editor_state.get('edited_rows'):
        _apply_cell_edits(df, data_editor_state['edited_rows'], dataframe_schema)
    if data_editor_state.get('added_rows'):
        _apply_row_additions(df, data_editor_state['added_rows'], dataframe_schema)
    if data_editor_state.get('deleted_rows'):
        _apply_row_deletions(df, data_editor_state['deleted_rows'])

def _is_supported_index(df_index: pd.Index) -> bool:
    if False:
        return 10
    'Check if the index is supported by the data editor component.\n\n    Parameters\n    ----------\n\n    df_index : pd.Index\n        The index to check.\n\n    Returns\n    -------\n\n    bool\n        True if the index is supported, False otherwise.\n    '
    return type(df_index) in [pd.RangeIndex, pd.Index, pd.DatetimeIndex] or is_type(df_index, 'pandas.core.indexes.numeric.Int64Index') or is_type(df_index, 'pandas.core.indexes.numeric.Float64Index') or is_type(df_index, 'pandas.core.indexes.numeric.UInt64Index')

def _fix_column_headers(data_df: pd.DataFrame) -> None:
    if False:
        i = 10
        return i + 15
    'Fix the column headers of the provided dataframe inplace to work\n    correctly for data editing.'
    if isinstance(data_df.columns, pd.MultiIndex):
        data_df.columns = ['_'.join(map(str, header)) for header in data_df.columns.to_flat_index()]
    elif pd.api.types.infer_dtype(data_df.columns) != 'string':
        data_df.rename(columns={column: str(column) for column in data_df.columns}, inplace=True)

def _check_column_names(data_df: pd.DataFrame):
    if False:
        i = 10
        return i + 15
    "Check if the column names in the provided dataframe are valid.\n\n    It's not allowed to have duplicate column names or column names that are\n    named ``_index``. If the column names are not valid, a ``StreamlitAPIException``\n    is raised.\n    "
    if data_df.columns.empty:
        return
    duplicated_columns = data_df.columns[data_df.columns.duplicated()]
    if len(duplicated_columns) > 0:
        raise StreamlitAPIException(f'All column names are required to be unique for usage with data editor. The following column names are duplicated: {list(duplicated_columns)}. Please rename the duplicated columns in the provided data.')
    if INDEX_IDENTIFIER in data_df.columns:
        raise StreamlitAPIException(f"The column name '{INDEX_IDENTIFIER}' is reserved for the index column and can't be used for data columns. Please rename the column in the provided data.")

def _check_type_compatibilities(data_df: pd.DataFrame, columns_config: ColumnConfigMapping, dataframe_schema: DataframeSchema):
    if False:
        i = 10
        return i + 15
    'Check column type to data type compatibility.\n\n    Iterates the index and all columns of the dataframe to check if\n    the configured column types are compatible with the underlying data types.\n\n    Parameters\n    ----------\n    data_df : pd.DataFrame\n        The dataframe to check the type compatibilities for.\n\n    columns_config : ColumnConfigMapping\n        A mapping of column to column configurations.\n\n    dataframe_schema : DataframeSchema\n        The schema of the dataframe.\n\n    Raises\n    ------\n    StreamlitAPIException\n        If a configured column type is editable and not compatible with the\n        underlying data type.\n    '
    indices = [(INDEX_IDENTIFIER, data_df.index)]
    for column in indices + list(data_df.items()):
        (column_name, _) = column
        column_data_kind = dataframe_schema[column_name]
        if column_name in columns_config:
            column_config = columns_config[column_name]
            if column_config.get('disabled') is True:
                continue
            type_config = column_config.get('type_config')
            if type_config is None:
                continue
            configured_column_type = type_config.get('type')
            if configured_column_type is None:
                continue
            if is_type_compatible(configured_column_type, column_data_kind) is False:
                raise StreamlitAPIException(f'The configured column type `{configured_column_type}` for column `{column_name}` is not compatible for editing the underlying data type `{column_data_kind}`.\n\nYou have following options to fix this: 1) choose a compatible type 2) disable the column 3) convert the column into a compatible data type.')

class DataEditorMixin:

    @overload
    def data_editor(self, data: EditableData, *, width: int | None=None, height: int | None=None, use_container_width: bool=False, hide_index: bool | None=None, column_order: Iterable[str] | None=None, column_config: ColumnConfigMappingInput | None=None, num_rows: Literal['fixed', 'dynamic']='fixed', disabled: bool | Iterable[str]=False, key: Key | None=None, on_change: WidgetCallback | None=None, args: WidgetArgs | None=None, kwargs: WidgetKwargs | None=None) -> EditableData:
        if False:
            while True:
                i = 10
        pass

    @overload
    def data_editor(self, data: Any, *, width: int | None=None, height: int | None=None, use_container_width: bool=False, hide_index: bool | None=None, column_order: Iterable[str] | None=None, column_config: ColumnConfigMappingInput | None=None, num_rows: Literal['fixed', 'dynamic']='fixed', disabled: bool | Iterable[str]=False, key: Key | None=None, on_change: WidgetCallback | None=None, args: WidgetArgs | None=None, kwargs: WidgetKwargs | None=None) -> pd.DataFrame:
        if False:
            for i in range(10):
                print('nop')
        pass

    @gather_metrics('data_editor')
    def data_editor(self, data: DataTypes, *, width: int | None=None, height: int | None=None, use_container_width: bool=False, hide_index: bool | None=None, column_order: Iterable[str] | None=None, column_config: ColumnConfigMappingInput | None=None, num_rows: Literal['fixed', 'dynamic']='fixed', disabled: bool | Iterable[str]=False, key: Key | None=None, on_change: WidgetCallback | None=None, args: WidgetArgs | None=None, kwargs: WidgetKwargs | None=None) -> DataTypes:
        if False:
            for i in range(10):
                print('nop')
        'Display a data editor widget.\n\n        The data editor widget allows you to edit dataframes and many other data structures in a table-like UI.\n\n        .. warning::\n            When going from ``st.experimental_data_editor`` to ``st.data_editor`` in\n            1.23.0, the data editor\'s representation in ``st.session_state`` was changed.\n            The ``edited_cells`` dictionary is now called ``edited_rows`` and uses a\n            different format (``{0: {"column name": "edited value"}}`` instead of\n            ``{"0:1": "edited value"}``). You may need to adjust the code if your app uses\n            ``st.experimental_data_editor`` in combination with ``st.session_state``."\n\n        Parameters\n        ----------\n        data : pandas.DataFrame, pandas.Series, pandas.Styler, pandas.Index, pyarrow.Table, numpy.ndarray, pyspark.sql.DataFrame, snowflake.snowpark.DataFrame, list, set, tuple, dict, or None\n            The data to edit in the data editor.\n\n            .. note::\n                - Styles from ``pandas.Styler`` will only be applied to non-editable columns.\n                - Mixing data types within a column can make the column uneditable.\n                - Additionally, the following data types are not yet supported for editing:\n                  complex, list, tuple, bytes, bytearray, memoryview, dict, set, frozenset,\n                  datetime.timedelta, decimal.Decimal, fractions.Fraction, pandas.Interval,\n                  pandas.Period, pandas.Timedelta.\n\n        width : int or None\n            Desired width of the data editor expressed in pixels. If None, the width will\n            be automatically determined.\n\n        height : int or None\n            Desired height of the data editor expressed in pixels. If None, the height will\n            be automatically determined.\n\n        use_container_width : bool\n            If True, set the data editor width to the width of the parent container.\n            This takes precedence over the width argument. Defaults to False.\n\n        hide_index : bool or None\n            Whether to hide the index column(s). If None (default), the visibility of\n            index columns is automatically determined based on the data.\n\n        column_order : Iterable of str or None\n            Specifies the display order of columns. This also affects which columns are\n            visible. For example, ``column_order=("col2", "col1")`` will display \'col2\'\n            first, followed by \'col1\', and will hide all other non-index columns. If\n            None (default), the order is inherited from the original data structure.\n\n        column_config : dict or None\n            Configures how columns are displayed, e.g. their title, visibility, type, or\n            format, as well as editing properties such as min/max value or step.\n            This needs to be a dictionary where each key is a column name and the value\n            is one of:\n\n            * ``None`` to hide the column.\n\n            * A string to set the display label of the column.\n\n            * One of the column types defined under ``st.column_config``, e.g.\n              ``st.column_config.NumberColumn("Dollar values”, format=”$ %d")`` to show\n              a column as dollar amounts. See more info on the available column types\n              and config options `here <https://docs.streamlit.io/library/api-reference/data/st.column_config>`_.\n\n            To configure the index column(s), use ``_index`` as the column name.\n\n        num_rows : "fixed" or "dynamic"\n            Specifies if the user can add and delete rows in the data editor.\n            If "fixed", the user cannot add or delete rows. If "dynamic", the user can\n            add and delete rows in the data editor, but column sorting is disabled.\n            Defaults to "fixed".\n\n        disabled : bool or Iterable of str\n            Controls the editing of columns. If True, editing is disabled for all columns.\n            If an Iterable of column names is provided (e.g., ``disabled=("col1", "col2"))``,\n            only the specified columns will be disabled for editing. If False (default),\n            all columns that support editing are editable.\n\n        key : str\n            An optional string to use as the unique key for this widget. If this\n            is omitted, a key will be generated for the widget based on its\n            content. Multiple widgets of the same type may not share the same\n            key.\n\n        on_change : callable\n            An optional callback invoked when this data_editor\'s value changes.\n\n        args : tuple\n            An optional tuple of args to pass to the callback.\n\n        kwargs : dict\n            An optional dict of kwargs to pass to the callback.\n\n        Returns\n        -------\n        pandas.DataFrame, pandas.Series, pyarrow.Table, numpy.ndarray, list, set, tuple, or dict.\n            The edited data. The edited data is returned in its original data type if\n            it corresponds to any of the supported return types. All other data types\n            are returned as a ``pd.DataFrame``.\n\n        Examples\n        --------\n        >>> import streamlit as st\n        >>> import pandas as pd\n        >>>\n        >>> df = pd.DataFrame(\n        >>>     [\n        >>>        {"command": "st.selectbox", "rating": 4, "is_widget": True},\n        >>>        {"command": "st.balloons", "rating": 5, "is_widget": False},\n        >>>        {"command": "st.time_input", "rating": 3, "is_widget": True},\n        >>>    ]\n        >>> )\n        >>> edited_df = st.data_editor(df)\n        >>>\n        >>> favorite_command = edited_df.loc[edited_df["rating"].idxmax()]["command"]\n        >>> st.markdown(f"Your favorite command is **{favorite_command}** 🎈")\n\n        .. output::\n           https://doc-data-editor.streamlit.app/\n           height: 350px\n\n        You can also allow the user to add and delete rows by setting ``num_rows`` to "dynamic":\n\n        >>> import streamlit as st\n        >>> import pandas as pd\n        >>>\n        >>> df = pd.DataFrame(\n        >>>     [\n        >>>        {"command": "st.selectbox", "rating": 4, "is_widget": True},\n        >>>        {"command": "st.balloons", "rating": 5, "is_widget": False},\n        >>>        {"command": "st.time_input", "rating": 3, "is_widget": True},\n        >>>    ]\n        >>> )\n        >>> edited_df = st.data_editor(df, num_rows="dynamic")\n        >>>\n        >>> favorite_command = edited_df.loc[edited_df["rating"].idxmax()]["command"]\n        >>> st.markdown(f"Your favorite command is **{favorite_command}** 🎈")\n\n        .. output::\n           https://doc-data-editor1.streamlit.app/\n           height: 450px\n\n        Or you can customize the data editor via ``column_config``, ``hide_index``, ``column_order``, or ``disabled``:\n\n        >>> import pandas as pd\n        >>> import streamlit as st\n        >>>\n        >>> df = pd.DataFrame(\n        >>>     [\n        >>>         {"command": "st.selectbox", "rating": 4, "is_widget": True},\n        >>>         {"command": "st.balloons", "rating": 5, "is_widget": False},\n        >>>         {"command": "st.time_input", "rating": 3, "is_widget": True},\n        >>>     ]\n        >>> )\n        >>> edited_df = st.data_editor(\n        >>>     df,\n        >>>     column_config={\n        >>>         "command": "Streamlit Command",\n        >>>         "rating": st.column_config.NumberColumn(\n        >>>             "Your rating",\n        >>>             help="How much do you like this command (1-5)?",\n        >>>             min_value=1,\n        >>>             max_value=5,\n        >>>             step=1,\n        >>>             format="%d ⭐",\n        >>>         ),\n        >>>         "is_widget": "Widget ?",\n        >>>     },\n        >>>     disabled=["command", "is_widget"],\n        >>>     hide_index=True,\n        >>> )\n        >>>\n        >>> favorite_command = edited_df.loc[edited_df["rating"].idxmax()]["command"]\n        >>> st.markdown(f"Your favorite command is **{favorite_command}** 🎈")\n\n\n        .. output::\n           https://doc-data-editor-config.streamlit.app/\n           height: 350px\n\n        '
        key = to_key(key)
        check_callback_rules(self.dg, on_change)
        check_session_state_rules(default_value=None, key=key, writes_allowed=False)
        if column_order is not None:
            column_order = list(column_order)
        column_config_mapping: ColumnConfigMapping = {}
        data_format = type_util.determine_data_format(data)
        if data_format == DataFormat.UNKNOWN:
            raise StreamlitAPIException(f'The data type ({type(data).__name__}) or format is not supported by the data editor. Please convert your data into a Pandas Dataframe or another supported data format.')
        data_df = type_util.convert_anything_to_df(data, ensure_copy=True)
        if not _is_supported_index(data_df.index):
            raise StreamlitAPIException(f'The type of the dataframe index - {type(data_df.index).__name__} - is not yet supported by the data editor.')
        _check_column_names(data_df)
        column_config_mapping = process_config_mapping(column_config)
        apply_data_specific_configs(column_config_mapping, data_df, data_format, check_arrow_compatibility=True)
        _fix_column_headers(data_df)
        if isinstance(data_df.index, pd.RangeIndex) and num_rows == 'dynamic':
            update_column_config(column_config_mapping, INDEX_IDENTIFIER, {'hidden': True})
        if hide_index is not None:
            update_column_config(column_config_mapping, INDEX_IDENTIFIER, {'hidden': hide_index})
        if not isinstance(disabled, bool):
            for column in disabled:
                update_column_config(column_config_mapping, column, {'disabled': True})
        arrow_table = pa.Table.from_pandas(data_df)
        dataframe_schema = determine_dataframe_schema(data_df, arrow_table.schema)
        _check_type_compatibilities(data_df, column_config_mapping, dataframe_schema)
        arrow_bytes = type_util.pyarrow_table_to_bytes(arrow_table)
        ctx = get_script_run_ctx()
        id = compute_widget_id('data_editor', user_key=key, data=arrow_bytes, width=width, height=height, use_container_width=use_container_width, column_order=column_order, column_config_mapping=str(column_config_mapping), num_rows=num_rows, key=key, form_id=current_form_id(self.dg), page=ctx.page_script_hash if ctx else None)
        proto = ArrowProto()
        proto.id = id
        proto.use_container_width = use_container_width
        if width:
            proto.width = width
        if height:
            proto.height = height
        if column_order:
            proto.column_order[:] = column_order
        proto.disabled = disabled is True
        proto.editing_mode = ArrowProto.EditingMode.DYNAMIC if num_rows == 'dynamic' else ArrowProto.EditingMode.FIXED
        proto.form_id = current_form_id(self.dg)
        if type_util.is_pandas_styler(data):
            styler_uuid = calc_md5(key or self.dg._get_delta_path_str())[:10]
            data.set_uuid(styler_uuid)
            marshall_styler(proto, data, styler_uuid)
        proto.data = arrow_bytes
        marshall_column_config(proto, column_config_mapping)
        serde = DataEditorSerde()
        widget_state = register_widget('data_editor', proto, user_key=key, on_change_handler=on_change, args=args, kwargs=kwargs, deserializer=serde.deserialize, serializer=serde.serialize, ctx=ctx)
        _apply_dataframe_edits(data_df, widget_state.value, dataframe_schema)
        self.dg._enqueue('arrow_data_frame', proto)
        return type_util.convert_df_to_data_format(data_df, data_format)

    @property
    def dg(self) -> 'DeltaGenerator':
        if False:
            while True:
                i = 10
        'Get our DeltaGenerator.'
        return cast('DeltaGenerator', self)
    experimental_data_editor = deprecate_func_name(gather_metrics('experimental_data_editor', data_editor), 'experimental_data_editor', '2023-09-01', '\n**Breaking change:** The data editor\'s representation in `st.session_state` was changed. The `edited_cells` dictionary is now called `edited_rows` and uses a\ndifferent format (`{0: {"column name": "edited value"}}` instead of\n`{"0:1": "edited value"}`). You may need to adjust the code if your app uses\n`st.experimental_data_editor` in combination with `st.session_state`."\n')