from __future__ import annotations
import contextlib
import hashlib
import threading
import types
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterator, Union
from google.protobuf.message import Message
from typing_extensions import Protocol, runtime_checkable
import streamlit as st
from streamlit import runtime, util
from streamlit.elements import NONWIDGET_ELEMENTS, WIDGETS
from streamlit.logger import get_logger
from streamlit.proto.Block_pb2 import Block
from streamlit.runtime.caching.cache_errors import CachedStFunctionWarning, CacheReplayClosureError
from streamlit.runtime.caching.cache_type import CacheType
from streamlit.runtime.caching.hashing import update_hash
from streamlit.runtime.scriptrunner.script_run_context import ScriptRunContext, get_script_run_ctx
from streamlit.runtime.state.common import WidgetMetadata
from streamlit.util import HASHLIB_KWARGS
if TYPE_CHECKING:
    from streamlit.delta_generator import DeltaGenerator
_LOGGER = get_logger(__name__)

@runtime_checkable
class Widget(Protocol):
    id: str

@dataclass(frozen=True)
class WidgetMsgMetadata:
    """Everything needed for replaying a widget and treating it as an implicit
    argument to a cached function, beyond what is stored for all elements.
    """
    widget_id: str
    widget_value: Any
    metadata: WidgetMetadata[Any]

@dataclass(frozen=True)
class MediaMsgData:
    media: bytes | str
    mimetype: str
    media_id: str

@dataclass(frozen=True)
class ElementMsgData:
    """An element's message and related metadata for
    replaying that element's function call.

    widget_metadata is filled in if and only if this element is a widget.
    media_data is filled in iff this is a media element (image, audio, video).
    """
    delta_type: str
    message: Message
    id_of_dg_called_on: str
    returned_dgs_id: str
    widget_metadata: WidgetMsgMetadata | None = None
    media_data: list[MediaMsgData] | None = None

@dataclass(frozen=True)
class BlockMsgData:
    message: Block
    id_of_dg_called_on: str
    returned_dgs_id: str
MsgData = Union[ElementMsgData, BlockMsgData]
'\nNote [Cache result structure]\n\nThe cache for a decorated function\'s results is split into two parts to enable\nhandling widgets invoked by the function.\n\nWidgets act as implicit additional inputs to the cached function, so they should\nbe used when deriving the cache key. However, we don\'t know what widgets are\ninvolved without first calling the function! So, we use the first execution\nof the function with a particular cache key to record what widgets are used,\nand use the current values of those widgets to derive a second cache key to\nlook up the function execution\'s results. The combination of first and second\ncache keys act as one true cache key, just split up because the second part depends\non the first.\n\nWe need to treat widgets as implicit arguments of the cached function, because\nthe behavior of the function, inluding what elements are created and what it\nreturns, can be and usually will be influenced by the values of those widgets.\nFor example:\n> @st.memo\n> def example_fn(x):\n>     y = x + 1\n>     if st.checkbox("hi"):\n>         st.write("you checked the thing")\n>         y = 0\n>     return y\n>\n> example_fn(2)\n\nIf the checkbox is checked, the function call should return 0 and the checkbox and\nmessage should be rendered. If the checkbox isn\'t checked, only the checkbox should\nrender, and the function will return 3.\n\n\nThere is a small catch in this. Since what widgets execute could depend on the values of\nany prior widgets, if we replace the `st.write` call in the example with a slider,\nthe first time it runs, we would miss the slider because it wasn\'t called,\nso when we later execute the function with the checkbox checked, the widget cache key\nwould not include the state of the slider, and would incorrectly get a cache hit\nfor a different slider value.\n\nIn principle the cache could be function+args key -> 1st widget key -> 2nd widget key\n... -> final widget key, with each widget dependent on the exact values of the widgets\nseen prior. This would prevent unnecessary cache misses due to differing values of widgets\nthat wouldn\'t affect the function\'s execution because they aren\'t even created.\nBut this would add even more complexity and both conceptual and runtime overhead, so it is\nunclear if it would be worth doing.\n\nInstead, we can keep the widgets as one cache key, and if we encounter a new widget\nwhile executing the function, we just update the list of widgets to include it.\nThis will cause existing cached results to be invalidated, which is bad, but to\navoid it we would need to keep around the full list of widgets and values for each\nwidget cache key so we could compute the updated key, which is probably too expensive\nto be worth it.\n'

@dataclass
class CachedResult:
    """The full results of calling a cache-decorated function, enough to
    replay the st functions called while executing it.
    """
    value: Any
    messages: list[MsgData]
    main_id: str
    sidebar_id: str

@dataclass
class MultiCacheResults:
    """Widgets called by a cache-decorated function, and a mapping of the
    widget-derived cache key to the final results of executing the function.
    """
    widget_ids: set[str]
    results: dict[str, CachedResult]

    def get_current_widget_key(self, ctx: ScriptRunContext, cache_type: CacheType) -> str:
        if False:
            return 10
        state = ctx.session_state
        widget_values = [(wid, state[wid]) for wid in sorted(self.widget_ids) if wid in state]
        widget_key = _make_widget_key(widget_values, cache_type)
        return widget_key
'\nNote [DeltaGenerator method invocation]\nThere are two top level DG instances defined for all apps:\n`main`, which is for putting elements in the main part of the app\n`sidebar`, for the sidebar\n\nThere are 3 different ways an st function can be invoked:\n1. Implicitly on the main DG instance (plain `st.foo` calls)\n2. Implicitly in an active contextmanager block (`st.foo` within a `with st.container` context)\n3. Explicitly on a DG instance (`st.sidebar.foo`, `my_column_1.foo`)\n\nTo simplify replaying messages from a cached function result, we convert all of these\nto explicit invocations. How they get rewritten depends on if the invocation was\nimplicit vs explicit, and if the target DG has been seen/produced during replay.\n\nImplicit invocation on a known DG -> Explicit invocation on that DG\nImplicit invocation on an unknown DG -> Rewrite as explicit invocation on main\n    with st.container():\n        my_cache_decorated_function()\n\n    This is situation 2 above, and the DG is a block entirely outside our function call,\n    so we interpret it as "put this element in the enclosing contextmanager block"\n    (or main if there isn\'t one), which is achieved by invoking on main.\nExplicit invocation on a known DG -> No change needed\nExplicit invocation on an unknown DG -> Raise an error\n    We have no way to identify the target DG, and it may not even be present in the\n    current script run, so the least surprising thing to do is raise an error.\n\n'

class CachedMessageReplayContext(threading.local):
    """A utility for storing messages generated by `st` commands called inside
    a cached function.

    Data is stored in a thread-local object, so it's safe to use an instance
    of this class across multiple threads.
    """

    def __init__(self, cache_type: CacheType):
        if False:
            i = 10
            return i + 15
        self._cached_func_stack: list[types.FunctionType] = []
        self._suppress_st_function_warning = 0
        self._cached_message_stack: list[list[MsgData]] = []
        self._seen_dg_stack: list[set[str]] = []
        self._most_recent_messages: list[MsgData] = []
        self._registered_metadata: WidgetMetadata[Any] | None = None
        self._media_data: list[MediaMsgData] = []
        self._cache_type = cache_type
        self._allow_widgets: int = 0

    def __repr__(self) -> str:
        if False:
            print('Hello World!')
        return util.repr_(self)

    @contextlib.contextmanager
    def calling_cached_function(self, func: types.FunctionType, allow_widgets: bool) -> Iterator[None]:
        if False:
            for i in range(10):
                print('nop')
        'Context manager that should wrap the invocation of a cached function.\n        It allows us to track any `st.foo` messages that are generated from inside the function\n        for playback during cache retrieval.\n        '
        self._cached_func_stack.append(func)
        self._cached_message_stack.append([])
        self._seen_dg_stack.append(set())
        if allow_widgets:
            self._allow_widgets += 1
        try:
            yield
        finally:
            self._cached_func_stack.pop()
            self._most_recent_messages = self._cached_message_stack.pop()
            self._seen_dg_stack.pop()
            if allow_widgets:
                self._allow_widgets -= 1

    def save_element_message(self, delta_type: str, element_proto: Message, invoked_dg_id: str, used_dg_id: str, returned_dg_id: str) -> None:
        if False:
            i = 10
            return i + 15
        "Record the element protobuf as having been produced during any currently\n        executing cached functions, so they can be replayed any time the function's\n        execution is skipped because they're in the cache.\n        "
        if not runtime.exists():
            return
        if len(self._cached_message_stack) >= 1:
            id_to_save = self.select_dg_to_save(invoked_dg_id, used_dg_id)
            if isinstance(element_proto, Widget) and element_proto.id:
                wid = element_proto.id
                if not self._registered_metadata:
                    _LOGGER.error("Trying to save widget message that wasn't registered. This should not be possible.")
                    raise AttributeError
                widget_meta = WidgetMsgMetadata(wid, None, metadata=self._registered_metadata)
            else:
                widget_meta = None
            media_data = self._media_data
            element_msg_data = ElementMsgData(delta_type, element_proto, id_to_save, returned_dg_id, widget_meta, media_data)
            for msgs in self._cached_message_stack:
                if self._allow_widgets or widget_meta is None:
                    msgs.append(element_msg_data)
        self._media_data = []
        self._registered_metadata = None
        for s in self._seen_dg_stack:
            s.add(returned_dg_id)

    def save_block_message(self, block_proto: Block, invoked_dg_id: str, used_dg_id: str, returned_dg_id: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        id_to_save = self.select_dg_to_save(invoked_dg_id, used_dg_id)
        for msgs in self._cached_message_stack:
            msgs.append(BlockMsgData(block_proto, id_to_save, returned_dg_id))
        for s in self._seen_dg_stack:
            s.add(returned_dg_id)

    def select_dg_to_save(self, invoked_id: str, acting_on_id: str) -> str:
        if False:
            print('Hello World!')
        'Select the id of the DG that this message should be invoked on\n        during message replay.\n\n        See Note [DeltaGenerator method invocation]\n\n        invoked_id is the DG the st function was called on, usually `st._main`.\n        acting_on_id is the DG the st function ultimately runs on, which may be different\n        if the invoked DG delegated to another one because it was in a `with` block.\n        '
        if len(self._seen_dg_stack) > 0 and acting_on_id in self._seen_dg_stack[-1]:
            return acting_on_id
        else:
            return invoked_id

    def save_widget_metadata(self, metadata: WidgetMetadata[Any]) -> None:
        if False:
            return 10
        self._registered_metadata = metadata

    def save_image_data(self, image_data: bytes | str, mimetype: str, image_id: str) -> None:
        if False:
            while True:
                i = 10
        self._media_data.append(MediaMsgData(image_data, mimetype, image_id))

    @contextlib.contextmanager
    def suppress_cached_st_function_warning(self) -> Iterator[None]:
        if False:
            print('Hello World!')
        self._suppress_st_function_warning += 1
        try:
            yield
        finally:
            self._suppress_st_function_warning -= 1
            assert self._suppress_st_function_warning >= 0

    def maybe_show_cached_st_function_warning(self, dg: 'DeltaGenerator', st_func_name: str) -> None:
        if False:
            while True:
                i = 10
        "If appropriate, warn about calling st.foo inside @memo.\n\n        DeltaGenerator's @_with_element and @_widget wrappers use this to warn\n        the user when they're calling st.foo() from within a function that is\n        wrapped in @st.cache.\n\n        Parameters\n        ----------\n        dg : DeltaGenerator\n            The DeltaGenerator to publish the warning to.\n\n        st_func_name : str\n            The name of the Streamlit function that was called.\n\n        "
        if st_func_name in NONWIDGET_ELEMENTS:
            return
        if st_func_name in WIDGETS and self._allow_widgets > 0:
            return
        if len(self._cached_func_stack) > 0 and self._suppress_st_function_warning <= 0:
            cached_func = self._cached_func_stack[-1]
            self._show_cached_st_function_warning(dg, st_func_name, cached_func)

    def _show_cached_st_function_warning(self, dg: 'DeltaGenerator', st_func_name: str, cached_func: types.FunctionType) -> None:
        if False:
            for i in range(10):
                print('nop')
        with self.suppress_cached_st_function_warning():
            e = CachedStFunctionWarning(self._cache_type, st_func_name, cached_func)
            dg.exception(e)

def replay_cached_messages(result: CachedResult, cache_type: CacheType, cached_func: types.FunctionType) -> None:
    if False:
        i = 10
        return i + 15
    'Replay the st element function calls that happened when executing a\n    cache-decorated function.\n\n    When a cache function is executed, we record the element and block messages\n    produced, and use those to reproduce the DeltaGenerator calls, so the elements\n    will appear in the web app even when execution of the function is skipped\n    because the result was cached.\n\n    To make this work, for each st function call we record an identifier for the\n    DG it was effectively called on (see Note [DeltaGenerator method invocation]).\n    We also record the identifier for each DG returned by an st function call, if\n    it returns one. Then, for each recorded message, we get the current DG instance\n    corresponding to the DG the message was originally called on, and enqueue the\n    message using that, recording any new DGs produced in case a later st function\n    call is on one of them.\n    '
    from streamlit.delta_generator import DeltaGenerator
    from streamlit.runtime.state.widgets import register_widget_from_metadata
    returned_dgs: dict[str, DeltaGenerator] = {}
    returned_dgs[result.main_id] = st._main
    returned_dgs[result.sidebar_id] = st.sidebar
    ctx = get_script_run_ctx()
    try:
        for msg in result.messages:
            if isinstance(msg, ElementMsgData):
                if msg.widget_metadata is not None:
                    register_widget_from_metadata(msg.widget_metadata.metadata, ctx, None, msg.delta_type)
                if msg.media_data is not None:
                    for data in msg.media_data:
                        runtime.get_instance().media_file_mgr.add(data.media, data.mimetype, data.media_id)
                dg = returned_dgs[msg.id_of_dg_called_on]
                maybe_dg = dg._enqueue(msg.delta_type, msg.message)
                if isinstance(maybe_dg, DeltaGenerator):
                    returned_dgs[msg.returned_dgs_id] = maybe_dg
            elif isinstance(msg, BlockMsgData):
                dg = returned_dgs[msg.id_of_dg_called_on]
                new_dg = dg._block(msg.message)
                returned_dgs[msg.returned_dgs_id] = new_dg
    except KeyError:
        raise CacheReplayClosureError(cache_type, cached_func)

def _make_widget_key(widgets: list[tuple[str, Any]], cache_type: CacheType) -> str:
    if False:
        while True:
            i = 10
    'Generate a key for the given list of widgets used in a cache-decorated function.\n\n    Keys are generated by hashing the IDs and values of the widgets in the given list.\n    '
    func_hasher = hashlib.new('md5', **HASHLIB_KWARGS)
    for widget_id_val in widgets:
        update_hash(widget_id_val, func_hasher, cache_type)
    return func_hasher.hexdigest()