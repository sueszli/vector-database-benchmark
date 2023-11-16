"""Data marshalling utilities for ArrowTable protobufs, which are used by
CustomComponent for dataframe serialization.
"""
from __future__ import annotations
from typing import Any
import pandas as pd
from streamlit import type_util
from streamlit.elements.lib import pandas_styler_utils
from streamlit.proto.Components_pb2 import ArrowTable as ArrowTableProto

def marshall(proto: ArrowTableProto, data: Any, default_uuid: str | None=None) -> None:
    if False:
        return 10
    'Marshall data into an ArrowTable proto.\n\n    Parameters\n    ----------\n    proto : proto.ArrowTable\n        Output. The protobuf for a Streamlit ArrowTable proto.\n\n    data : pandas.DataFrame, pandas.Styler, numpy.ndarray, Iterable, dict, or None\n        Something that is or can be converted to a dataframe.\n\n    '
    if type_util.is_pandas_styler(data):
        pandas_styler_utils.marshall_styler(proto, data, default_uuid)
    df = type_util.convert_anything_to_df(data)
    _marshall_index(proto, df.index)
    _marshall_columns(proto, df.columns)
    _marshall_data(proto, df)

def _marshall_index(proto: ArrowTableProto, index: pd.Index) -> None:
    if False:
        return 10
    'Marshall pandas.DataFrame index into an ArrowTable proto.\n\n    Parameters\n    ----------\n    proto : proto.ArrowTable\n        Output. The protobuf for a Streamlit ArrowTable proto.\n\n    index : pd.Index\n        Index to use for resulting frame.\n        Will default to RangeIndex (0, 1, 2, ..., n) if no index is provided.\n\n    '
    index = map(type_util.maybe_tuple_to_list, index.values)
    index_df = pd.DataFrame(index)
    proto.index = type_util.data_frame_to_bytes(index_df)

def _marshall_columns(proto: ArrowTableProto, columns: pd.Series) -> None:
    if False:
        print('Hello World!')
    'Marshall pandas.DataFrame columns into an ArrowTable proto.\n\n    Parameters\n    ----------\n    proto : proto.ArrowTable\n        Output. The protobuf for a Streamlit ArrowTable proto.\n\n    columns : Series\n        Column labels to use for resulting frame.\n        Will default to RangeIndex (0, 1, 2, ..., n) if no column labels are provided.\n\n    '
    columns = map(type_util.maybe_tuple_to_list, columns.values)
    columns_df = pd.DataFrame(columns)
    proto.columns = type_util.data_frame_to_bytes(columns_df)

def _marshall_data(proto: ArrowTableProto, df: pd.DataFrame) -> None:
    if False:
        while True:
            i = 10
    'Marshall pandas.DataFrame data into an ArrowTable proto.\n\n    Parameters\n    ----------\n    proto : proto.ArrowTable\n        Output. The protobuf for a Streamlit ArrowTable proto.\n\n    df : pandas.DataFrame\n        A dataframe to marshall.\n\n    '
    proto.data = type_util.data_frame_to_bytes(df)

def arrow_proto_to_dataframe(proto: ArrowTableProto) -> pd.DataFrame:
    if False:
        i = 10
        return i + 15
    'Convert ArrowTable proto to pandas.DataFrame.\n\n    Parameters\n    ----------\n    proto : proto.ArrowTable\n        Output. pandas.DataFrame\n\n    '
    if type_util.is_pyarrow_version_less_than('14.0.1'):
        raise RuntimeError('The installed pyarrow version is not compatible with this component. Please upgrade to 14.0.1 or higher: pip install -U pyarrow')
    data = type_util.bytes_to_data_frame(proto.data)
    index = type_util.bytes_to_data_frame(proto.index)
    columns = type_util.bytes_to_data_frame(proto.columns)
    return pd.DataFrame(data.values, index=index.values.T.tolist(), columns=columns.values.T.tolist())