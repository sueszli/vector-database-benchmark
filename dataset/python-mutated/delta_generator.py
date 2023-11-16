"""Allows us to create and absorb changes (aka Deltas) to elements."""
from __future__ import annotations
import sys
from typing import TYPE_CHECKING, Any, Callable, Hashable, Iterable, NoReturn, Optional, Type, TypeVar, cast, overload
import click
from typing_extensions import Final, Literal
from streamlit import config, cursor, env_util, logger, runtime, type_util, util
from streamlit.cursor import Cursor
from streamlit.elements.alert import AlertMixin
from streamlit.elements.altair_utils import AddRowsMetadata
from streamlit.elements.arrow import ArrowMixin
from streamlit.elements.arrow_altair import ArrowAltairMixin, prep_data
from streamlit.elements.arrow_vega_lite import ArrowVegaLiteMixin
from streamlit.elements.balloons import BalloonsMixin
from streamlit.elements.bokeh_chart import BokehMixin
from streamlit.elements.code import CodeMixin
from streamlit.elements.deck_gl_json_chart import PydeckMixin
from streamlit.elements.doc_string import HelpMixin
from streamlit.elements.empty import EmptyMixin
from streamlit.elements.exception import ExceptionMixin
from streamlit.elements.form import FormData, FormMixin, current_form_id
from streamlit.elements.graphviz_chart import GraphvizMixin
from streamlit.elements.heading import HeadingMixin
from streamlit.elements.iframe import IframeMixin
from streamlit.elements.image import ImageMixin
from streamlit.elements.json import JsonMixin
from streamlit.elements.layouts import LayoutsMixin
from streamlit.elements.map import MapMixin
from streamlit.elements.markdown import MarkdownMixin
from streamlit.elements.media import MediaMixin
from streamlit.elements.metric import MetricMixin
from streamlit.elements.plotly_chart import PlotlyMixin
from streamlit.elements.progress import ProgressMixin
from streamlit.elements.pyplot import PyplotMixin
from streamlit.elements.snow import SnowMixin
from streamlit.elements.text import TextMixin
from streamlit.elements.toast import ToastMixin
from streamlit.elements.widgets.button import ButtonMixin
from streamlit.elements.widgets.camera_input import CameraInputMixin
from streamlit.elements.widgets.chat import ChatMixin
from streamlit.elements.widgets.checkbox import CheckboxMixin
from streamlit.elements.widgets.color_picker import ColorPickerMixin
from streamlit.elements.widgets.data_editor import DataEditorMixin
from streamlit.elements.widgets.file_uploader import FileUploaderMixin
from streamlit.elements.widgets.multiselect import MultiSelectMixin
from streamlit.elements.widgets.number_input import NumberInputMixin
from streamlit.elements.widgets.radio import RadioMixin
from streamlit.elements.widgets.select_slider import SelectSliderMixin
from streamlit.elements.widgets.selectbox import SelectboxMixin
from streamlit.elements.widgets.slider import SliderMixin
from streamlit.elements.widgets.text_widgets import TextWidgetsMixin
from streamlit.elements.widgets.time_widgets import TimeWidgetsMixin
from streamlit.elements.write import WriteMixin
from streamlit.errors import NoSessionContext, StreamlitAPIException
from streamlit.logger import get_logger
from streamlit.proto import Block_pb2, ForwardMsg_pb2
from streamlit.proto.RootContainer_pb2 import RootContainer
from streamlit.runtime import caching, legacy_caching
from streamlit.runtime.scriptrunner import get_script_run_ctx
from streamlit.runtime.state import NoValue
if TYPE_CHECKING:
    from google.protobuf.message import Message
    from numpy import typing as npt
    from pandas import DataFrame, Series
    from streamlit.elements.arrow import Data
LOGGER: Final = get_logger(__name__)
MAX_DELTA_BYTES: Final[int] = 14 * 1024 * 1024
ARROW_DELTA_TYPES_THAT_MELT_DATAFRAMES: Final = ('arrow_line_chart', 'arrow_area_chart', 'arrow_bar_chart', 'arrow_scatter_chart')
Value = TypeVar('Value')
DG = TypeVar('DG', bound='DeltaGenerator')
BlockType = str
ParentBlockTypes = Iterable[BlockType]
_use_warning_has_been_displayed: bool = False

def _maybe_print_use_warning() -> None:
    if False:
        for i in range(10):
            print('nop')
    'Print a warning if Streamlit is imported but not being run with `streamlit run`.\n    The warning is printed only once, and is printed using the root logger.\n    '
    global _use_warning_has_been_displayed
    if not _use_warning_has_been_displayed:
        _use_warning_has_been_displayed = True
        warning = click.style('Warning:', bold=True, fg='yellow')
        if env_util.is_repl():
            logger.get_logger('root').warning(f'\n  {warning} to view a Streamlit app on a browser, use Streamlit in a file and\n  run it with the following command:\n\n    streamlit run [FILE_NAME] [ARGUMENTS]')
        elif not runtime.exists() and config.get_option('global.showWarningOnDirectExecution'):
            script_name = sys.argv[0]
            logger.get_logger('root').warning(f'\n  {warning} to view this Streamlit app on a browser, run it with the following\n  command:\n\n    streamlit run {script_name} [ARGUMENTS]')

class DeltaGenerator(AlertMixin, BalloonsMixin, BokehMixin, ButtonMixin, CameraInputMixin, ChatMixin, CheckboxMixin, CodeMixin, ColorPickerMixin, EmptyMixin, ExceptionMixin, FileUploaderMixin, FormMixin, GraphvizMixin, HeadingMixin, HelpMixin, IframeMixin, ImageMixin, LayoutsMixin, MarkdownMixin, MapMixin, MediaMixin, MetricMixin, MultiSelectMixin, NumberInputMixin, PlotlyMixin, ProgressMixin, PydeckMixin, PyplotMixin, RadioMixin, SelectboxMixin, SelectSliderMixin, SliderMixin, SnowMixin, JsonMixin, TextMixin, TextWidgetsMixin, TimeWidgetsMixin, ToastMixin, WriteMixin, ArrowMixin, ArrowAltairMixin, ArrowVegaLiteMixin, DataEditorMixin):
    """Creator of Delta protobuf messages.

    Parameters
    ----------
    root_container: BlockPath_pb2.BlockPath.ContainerValue or None
      The root container for this DeltaGenerator. If None, this is a null
      DeltaGenerator which doesn't print to the app at all (useful for
      testing).

    cursor: cursor.Cursor or None
      This is either:
      - None: if this is the running DeltaGenerator for a top-level
        container (MAIN or SIDEBAR)
      - RunningCursor: if this is the running DeltaGenerator for a
        non-top-level container (created with dg.container())
      - LockedCursor: if this is a locked DeltaGenerator returned by some
        other DeltaGenerator method. E.g. the dg returned in dg =
        st.text("foo").

    parent: DeltaGenerator
      To support the `with dg` notation, DGs are arranged as a tree. Each DG
      remembers its own parent, and the root of the tree is the main DG.

    block_type: None or "vertical" or "horizontal" or "column" or "expandable"
      If this is a block DG, we track its type to prevent nested columns/expanders

    """

    def __init__(self, root_container: int | None=RootContainer.MAIN, cursor: Cursor | None=None, parent: DeltaGenerator | None=None, block_type: str | None=None) -> None:
        if False:
            while True:
                i = 10
        'Inserts or updates elements in Streamlit apps.\n\n        As a user, you should never initialize this object by hand. Instead,\n        DeltaGenerator objects are initialized for you in two places:\n\n        1) When you call `dg = st.foo()` for some method "foo", sometimes `dg`\n        is a DeltaGenerator object. You can call methods on the `dg` object to\n        update the element `foo` that appears in the Streamlit app.\n\n        2) This is an internal detail, but `st.sidebar` itself is a\n        DeltaGenerator. That\'s why you can call `st.sidebar.foo()` to place\n        an element `foo` inside the sidebar.\n\n        '
        if root_container is not None and cursor is not None and (root_container != cursor.root_container):
            raise RuntimeError('DeltaGenerator root_container and cursor.root_container must be the same')
        self._root_container = root_container
        self._provided_cursor = cursor
        self._parent = parent
        self._block_type = block_type
        self._form_data: FormData | None = None
        for mixin in self.__class__.__bases__:
            for (name, func) in mixin.__dict__.items():
                if callable(func):
                    func.__module__ = self.__module__

    def __repr__(self) -> str:
        if False:
            while True:
                i = 10
        return util.repr_(self)

    def __enter__(self) -> None:
        if False:
            i = 10
            return i + 15
        ctx = get_script_run_ctx()
        if ctx:
            ctx.dg_stack.append(self)

    def __exit__(self, type: Any, value: Any, traceback: Any) -> Literal[False]:
        if False:
            while True:
                i = 10
        ctx = get_script_run_ctx()
        if ctx is not None:
            ctx.dg_stack.pop()
        return False

    @property
    def _active_dg(self) -> DeltaGenerator:
        if False:
            i = 10
            return i + 15
        "Return the DeltaGenerator that's currently 'active'.\n        If we are the main DeltaGenerator, and are inside a `with` block that\n        creates a container, our active_dg is that container. Otherwise,\n        our active_dg is self.\n        "
        if self == self._main_dg:
            ctx = get_script_run_ctx()
            if ctx and len(ctx.dg_stack) > 0:
                return ctx.dg_stack[-1]
        return self

    @property
    def _main_dg(self) -> DeltaGenerator:
        if False:
            print('Hello World!')
        "Return this DeltaGenerator's root - that is, the top-level ancestor\n        DeltaGenerator that we belong to (this generally means the st._main\n        DeltaGenerator).\n        "
        return self._parent._main_dg if self._parent else self

    def __getattr__(self, name: str) -> Callable[..., NoReturn]:
        if False:
            return 10
        import streamlit as st
        streamlit_methods = [method_name for method_name in dir(st) if callable(getattr(st, method_name))]

        def wrapper(*args: Any, **kwargs: Any) -> NoReturn:
            if False:
                for i in range(10):
                    print('nop')
            if name in streamlit_methods:
                if self._root_container == RootContainer.SIDEBAR:
                    message = 'Method `%(name)s()` does not exist for `st.sidebar`. Did you mean `st.%(name)s()`?' % {'name': name}
                else:
                    message = 'Method `%(name)s()` does not exist for `DeltaGenerator` objects. Did you mean `st.%(name)s()`?' % {'name': name}
            else:
                message = '`%(name)s()` is not a valid Streamlit command.' % {'name': name}
            raise StreamlitAPIException(message)
        return wrapper

    @property
    def _parent_block_types(self) -> ParentBlockTypes:
        if False:
            print('Hello World!')
        'Iterate all the block types used by this DeltaGenerator and all\n        its ancestor DeltaGenerators.\n        '
        current_dg: DeltaGenerator | None = self
        while current_dg is not None:
            if current_dg._block_type is not None:
                yield current_dg._block_type
            current_dg = current_dg._parent

    def _count_num_of_parent_columns(self, parent_block_types: ParentBlockTypes) -> int:
        if False:
            i = 10
            return i + 15
        return sum((1 for parent_block in parent_block_types if parent_block == 'column'))

    @property
    def _cursor(self) -> Cursor | None:
        if False:
            for i in range(10):
                print('nop')
        'Return our Cursor. This will be None if we\'re not running in a\n        ScriptThread - e.g., if we\'re running a "bare" script outside of\n        Streamlit.\n        '
        if self._provided_cursor is None:
            return cursor.get_container_cursor(self._root_container)
        else:
            return self._provided_cursor

    @property
    def _is_top_level(self) -> bool:
        if False:
            return 10
        return self._provided_cursor is None

    @property
    def id(self) -> str:
        if False:
            return 10
        return str(id(self))

    def _get_delta_path_str(self) -> str:
        if False:
            i = 10
            return i + 15
        'Returns the element\'s delta path as a string like "[0, 2, 3, 1]".\n\n        This uniquely identifies the element\'s position in the front-end,\n        which allows (among other potential uses) the MediaFileManager to maintain\n        session-specific maps of MediaFile objects placed with their "coordinates".\n\n        This way, users can (say) use st.image with a stream of different images,\n        and Streamlit will expire the older images and replace them in place.\n        '
        dg = self._active_dg
        return str(dg._cursor.delta_path) if dg._cursor is not None else '[]'

    @overload
    def _enqueue(self, delta_type: str, element_proto: Message, return_value: None, add_rows_metadata: Optional[AddRowsMetadata]=None, element_width: int | None=None, element_height: int | None=None) -> DeltaGenerator:
        if False:
            print('Hello World!')
        ...

    @overload
    def _enqueue(self, delta_type: str, element_proto: Message, return_value: Type[NoValue], add_rows_metadata: Optional[AddRowsMetadata]=None, element_width: int | None=None, element_height: int | None=None) -> None:
        if False:
            print('Hello World!')
        ...

    @overload
    def _enqueue(self, delta_type: str, element_proto: Message, return_value: Value, add_rows_metadata: Optional[AddRowsMetadata]=None, element_width: int | None=None, element_height: int | None=None) -> Value:
        if False:
            i = 10
            return i + 15
        ...

    @overload
    def _enqueue(self, delta_type: str, element_proto: Message, return_value: None=None, add_rows_metadata: Optional[AddRowsMetadata]=None, element_width: int | None=None, element_height: int | None=None) -> DeltaGenerator:
        if False:
            for i in range(10):
                print('nop')
        ...

    @overload
    def _enqueue(self, delta_type: str, element_proto: Message, return_value: Type[NoValue] | Value | None=None, add_rows_metadata: Optional[AddRowsMetadata]=None, element_width: int | None=None, element_height: int | None=None) -> DeltaGenerator | Value | None:
        if False:
            return 10
        ...

    def _enqueue(self, delta_type: str, element_proto: Message, return_value: Type[NoValue] | Value | None=None, add_rows_metadata: Optional[AddRowsMetadata]=None, element_width: int | None=None, element_height: int | None=None) -> DeltaGenerator | Value | None:
        if False:
            while True:
                i = 10
        'Create NewElement delta, fill it, and enqueue it.\n\n        Parameters\n        ----------\n        delta_type : str\n            The name of the streamlit method being called\n        element_proto : proto\n            The actual proto in the NewElement type e.g. Alert/Button/Slider\n        return_value : any or None\n            The value to return to the calling script (for widgets)\n        element_width : int or None\n            Desired width for the element\n        element_height : int or None\n            Desired height for the element\n\n        Returns\n        -------\n        DeltaGenerator or any\n            If this element is NOT an interactive widget, return a\n            DeltaGenerator that can be used to modify the newly-created\n            element. Otherwise, if the element IS a widget, return the\n            `return_value` parameter.\n\n        '
        dg = self._active_dg
        legacy_caching.maybe_show_cached_st_function_warning(dg, delta_type)
        caching.maybe_show_cached_st_function_warning(dg, delta_type)
        _maybe_print_use_warning()
        proto_type = delta_type
        if proto_type in ARROW_DELTA_TYPES_THAT_MELT_DATAFRAMES:
            proto_type = 'arrow_vega_lite_chart'
        msg = ForwardMsg_pb2.ForwardMsg()
        msg_el_proto = getattr(msg.delta.new_element, proto_type)
        msg_el_proto.CopyFrom(element_proto)
        msg_was_enqueued = False
        if dg._root_container is not None and dg._cursor is not None:
            msg.metadata.delta_path[:] = dg._cursor.delta_path
            if element_width is not None:
                msg.metadata.element_dimension_spec.width = element_width
            if element_height is not None:
                msg.metadata.element_dimension_spec.height = element_height
            _enqueue_message(msg)
            msg_was_enqueued = True
        if msg_was_enqueued:
            new_cursor = dg._cursor.get_locked_cursor(delta_type=delta_type, add_rows_metadata=add_rows_metadata) if dg._cursor is not None else None
            output_dg = DeltaGenerator(root_container=dg._root_container, cursor=new_cursor, parent=dg)
        else:
            output_dg = dg
        caching.save_element_message(delta_type, element_proto, invoked_dg_id=self.id, used_dg_id=dg.id, returned_dg_id=output_dg.id)
        return _value_or_dg(return_value, output_dg)

    def _block(self, block_proto: Block_pb2.Block=Block_pb2.Block(), dg_type: type | None=None) -> DeltaGenerator:
        if False:
            while True:
                i = 10
        dg = self._active_dg
        block_type = block_proto.WhichOneof('type')
        parent_block_types = list(dg._parent_block_types)
        if block_type == 'column':
            num_of_parent_columns = self._count_num_of_parent_columns(parent_block_types)
            if self._root_container == RootContainer.SIDEBAR and num_of_parent_columns > 0:
                raise StreamlitAPIException('Columns cannot be placed inside other columns in the sidebar. This is only possible in the main area of the app.')
            if num_of_parent_columns > 1:
                raise StreamlitAPIException('Columns can only be placed inside other columns up to one level of nesting.')
        if block_type == 'chat_message' and block_type in frozenset(parent_block_types):
            raise StreamlitAPIException('Chat messages cannot nested inside other chat messages.')
        if block_type == 'expandable' and block_type in frozenset(parent_block_types):
            raise StreamlitAPIException('Expanders may not be nested inside other expanders.')
        if dg._root_container is None or dg._cursor is None:
            return dg
        msg = ForwardMsg_pb2.ForwardMsg()
        msg.metadata.delta_path[:] = dg._cursor.delta_path
        msg.delta.add_block.CopyFrom(block_proto)
        block_cursor = cursor.RunningCursor(root_container=dg._root_container, parent_path=dg._cursor.parent_path + (dg._cursor.index,))
        if dg_type is None:
            dg_type = DeltaGenerator
        block_dg = cast(DeltaGenerator, dg_type(root_container=dg._root_container, cursor=block_cursor, parent=dg, block_type=block_type))
        block_dg._form_data = FormData(current_form_id(dg))
        dg._cursor.get_locked_cursor(add_rows_metadata=None)
        _enqueue_message(msg)
        caching.save_block_message(block_proto, invoked_dg_id=self.id, used_dg_id=dg.id, returned_dg_id=block_dg.id)
        return block_dg

    def _arrow_add_rows(self: DG, data: Data=None, **kwargs: DataFrame | npt.NDArray[Any] | Iterable[Any] | dict[Hashable, Any] | None) -> DG | None:
        if False:
            return 10
        "Concatenate a dataframe to the bottom of the current one.\n\n        Parameters\n        ----------\n        data : pandas.DataFrame, pandas.Styler, numpy.ndarray, Iterable, dict, or None\n            Table to concat. Optional.\n\n        **kwargs : pandas.DataFrame, numpy.ndarray, Iterable, dict, or None\n            The named dataset to concat. Optional. You can only pass in 1\n            dataset (including the one in the data parameter).\n\n        Example\n        -------\n        >>> import streamlit as st\n        >>> import pandas as pd\n        >>> import numpy as np\n        >>>\n        >>> df1 = pd.DataFrame(\n        ...    np.random.randn(50, 20),\n        ...    columns=('col %d' % i for i in range(20)))\n        ...\n        >>> my_table = st.table(df1)\n        >>>\n        >>> df2 = pd.DataFrame(\n        ...    np.random.randn(50, 20),\n        ...    columns=('col %d' % i for i in range(20)))\n        ...\n        >>> my_table._arrow_add_rows(df2)\n        >>> # Now the table shown in the Streamlit app contains the data for\n        >>> # df1 followed by the data for df2.\n\n        You can do the same thing with plots. For example, if you want to add\n        more data to a line chart:\n\n        >>> # Assuming df1 and df2 from the example above still exist...\n        >>> my_chart = st.line_chart(df1)\n        >>> my_chart._arrow_add_rows(df2)\n        >>> # Now the chart shown in the Streamlit app contains the data for\n        >>> # df1 followed by the data for df2.\n\n        And for plots whose datasets are named, you can pass the data with a\n        keyword argument where the key is the name:\n\n        >>> my_chart = st._arrow_vega_lite_chart({\n        ...     'mark': 'line',\n        ...     'encoding': {'x': 'a', 'y': 'b'},\n        ...     'datasets': {\n        ...       'some_fancy_name': df1,  # <-- named dataset\n        ...      },\n        ...     'data': {'name': 'some_fancy_name'},\n        ... }),\n        >>> my_chart._arrow_add_rows(some_fancy_name=df2)  # <-- name used as keyword\n\n        "
        if self._root_container is None or self._cursor is None:
            return self
        if not self._cursor.is_locked:
            raise StreamlitAPIException('Only existing elements can `add_rows`.')
        if data is not None and len(kwargs) == 0:
            name = ''
        elif len(kwargs) == 1:
            (name, data) = kwargs.popitem()
        else:
            raise StreamlitAPIException('Wrong number of arguments to add_rows().Command requires exactly one dataset')
        if self._cursor.props['delta_type'] in ARROW_DELTA_TYPES_THAT_MELT_DATAFRAMES and self._cursor.props['add_rows_metadata'].last_index is None:
            st_method_name = self._cursor.props['delta_type'].replace('arrow_', '')
            st_method = getattr(self, st_method_name)
            st_method(data, **kwargs)
            return None
        (new_data, self._cursor.props['add_rows_metadata']) = _prep_data_for_add_rows(data, self._cursor.props['delta_type'], self._cursor.props['add_rows_metadata'])
        msg = ForwardMsg_pb2.ForwardMsg()
        msg.metadata.delta_path[:] = self._cursor.delta_path
        import streamlit.elements.arrow as arrow_proto
        default_uuid = str(hash(self._get_delta_path_str()))
        arrow_proto.marshall(msg.delta.arrow_add_rows.data, new_data, default_uuid)
        if name:
            msg.delta.arrow_add_rows.name = name
            msg.delta.arrow_add_rows.has_name = True
        _enqueue_message(msg)
        return self

def _prep_data_for_add_rows(data: Data, delta_type: str, add_rows_metadata: AddRowsMetadata) -> tuple[Data, AddRowsMetadata]:
    if False:
        print('Hello World!')
    out_data: Data
    if delta_type in ARROW_DELTA_TYPES_THAT_MELT_DATAFRAMES:
        import pandas as pd
        df = cast(pd.DataFrame, type_util.convert_anything_to_df(data))
        if isinstance(df.index, pd.RangeIndex):
            old_step = _get_pandas_index_attr(df, 'step')
            df = df.reset_index(drop=True)
            old_stop = _get_pandas_index_attr(df, 'stop')
            if old_step is None or old_stop is None:
                raise StreamlitAPIException("'RangeIndex' object has no attribute 'step'")
            start = add_rows_metadata.last_index + old_step
            stop = add_rows_metadata.last_index + old_step + old_stop
            df.index = pd.RangeIndex(start=start, stop=stop, step=old_step)
            add_rows_metadata.last_index = stop - 1
        (out_data, *_) = prep_data(df, **add_rows_metadata.columns)
    else:
        out_data = type_util.convert_anything_to_df(data, allow_styler=True)
    return (out_data, add_rows_metadata)

def _get_pandas_index_attr(data: DataFrame | Series, attr: str) -> Any | None:
    if False:
        i = 10
        return i + 15
    return getattr(data.index, attr, None)

@overload
def _value_or_dg(value: None, dg: DG) -> DG:
    if False:
        for i in range(10):
            print('nop')
    ...

@overload
def _value_or_dg(value: Type[NoValue], dg: DG) -> None:
    if False:
        i = 10
        return i + 15
    ...

@overload
def _value_or_dg(value: Value, dg: DG) -> Value:
    if False:
        print('Hello World!')
    ...

def _value_or_dg(value: Type[NoValue] | Value | None, dg: DG) -> DG | Value | None:
    if False:
        return 10
    'Return either value, or None, or dg.\n\n    This is needed because Widgets have meaningful return values. This is\n    unlike other elements, which always return None. Then we internally replace\n    that None with a DeltaGenerator instance.\n\n    However, sometimes a widget may want to return None, and in this case it\n    should not be replaced by a DeltaGenerator. So we have a special NoValue\n    object that gets replaced by None.\n\n    '
    if value is NoValue:
        return None
    if value is None:
        return dg
    return cast(Value, value)

def _enqueue_message(msg: ForwardMsg_pb2.ForwardMsg) -> None:
    if False:
        print('Hello World!')
    'Enqueues a ForwardMsg proto to send to the app.'
    ctx = get_script_run_ctx()
    if ctx is None:
        raise NoSessionContext()
    ctx.enqueue(msg)