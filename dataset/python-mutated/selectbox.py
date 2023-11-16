from __future__ import annotations
from dataclasses import dataclass
from textwrap import dedent
from typing import TYPE_CHECKING, Any, Callable, Generic, Sequence, cast
from streamlit.elements.form import current_form_id
from streamlit.elements.utils import check_callback_rules, check_session_state_rules, get_label_visibility_proto_value, maybe_coerce_enum
from streamlit.errors import StreamlitAPIException
from streamlit.proto.Selectbox_pb2 import Selectbox as SelectboxProto
from streamlit.runtime.metrics_util import gather_metrics
from streamlit.runtime.scriptrunner import ScriptRunContext, get_script_run_ctx
from streamlit.runtime.state import WidgetArgs, WidgetCallback, WidgetKwargs, register_widget
from streamlit.runtime.state.common import compute_widget_id
from streamlit.type_util import Key, LabelVisibility, OptionSequence, T, ensure_indexable, maybe_raise_label_warnings, to_key
from streamlit.util import index_
if TYPE_CHECKING:
    from streamlit.delta_generator import DeltaGenerator

@dataclass
class SelectboxSerde(Generic[T]):
    options: Sequence[T]
    index: int | None

    def serialize(self, v: object) -> int | None:
        if False:
            return 10
        if v is None:
            return None
        if len(self.options) == 0:
            return 0
        return index_(self.options, v)

    def deserialize(self, ui_value: int | None, widget_id: str='') -> T | None:
        if False:
            for i in range(10):
                print('nop')
        idx = ui_value if ui_value is not None else self.index
        return self.options[idx] if idx is not None and len(self.options) > 0 else None

class SelectboxMixin:

    @gather_metrics('selectbox')
    def selectbox(self, label: str, options: OptionSequence[T], index: int | None=0, format_func: Callable[[Any], Any]=str, key: Key | None=None, help: str | None=None, on_change: WidgetCallback | None=None, args: WidgetArgs | None=None, kwargs: WidgetKwargs | None=None, *, placeholder: str='Choose an option', disabled: bool=False, label_visibility: LabelVisibility='visible') -> T | None:
        if False:
            while True:
                i = 10
        'Display a select widget.\n\n        Parameters\n        ----------\n        label : str\n            A short label explaining to the user what this select widget is for.\n            The label can optionally contain Markdown and supports the following\n            elements: Bold, Italics, Strikethroughs, Inline Code, Emojis, and Links.\n\n            This also supports:\n\n            * Emoji shortcodes, such as ``:+1:``  and ``:sunglasses:``.\n              For a list of all supported codes,\n              see https://share.streamlit.io/streamlit/emoji-shortcodes.\n\n            * LaTeX expressions, by wrapping them in "$" or "$$" (the "$$"\n              must be on their own lines). Supported LaTeX functions are listed\n              at https://katex.org/docs/supported.html.\n\n            * Colored text, using the syntax ``:color[text to be colored]``,\n              where ``color`` needs to be replaced with any of the following\n              supported colors: blue, green, orange, red, violet, gray/grey, rainbow.\n\n            Unsupported elements are unwrapped so only their children (text contents) render.\n            Display unsupported elements as literal characters by\n            backslash-escaping them. E.g. ``1\\. Not an ordered list``.\n\n            For accessibility reasons, you should never set an empty label (label="")\n            but hide it with label_visibility if needed. In the future, we may disallow\n            empty labels by raising an exception.\n        options : Iterable\n            Labels for the select options in an Iterable. For example, this can\n            be a list, numpy.ndarray, pandas.Series, pandas.DataFrame, or\n            pandas.Index. For pandas.DataFrame, the first column is used.\n            Each label will be cast to str internally by default.\n        index : int\n            The index of the preselected option on first render. If ``None``,\n            will initialize empty and return ``None`` until the user selects an option.\n            Defaults to 0 (the first option).\n        format_func : function\n            Function to modify the display of the labels. It receives the option\n            as an argument and its output will be cast to str.\n        key : str or int\n            An optional string or integer to use as the unique key for the widget.\n            If this is omitted, a key will be generated for the widget\n            based on its content. Multiple widgets of the same type may\n            not share the same key.\n        help : str\n            An optional tooltip that gets displayed next to the selectbox.\n        on_change : callable\n            An optional callback invoked when this selectbox\'s value changes.\n        args : tuple\n            An optional tuple of args to pass to the callback.\n        kwargs : dict\n            An optional dict of kwargs to pass to the callback.\n        placeholder : str\n            A string to display when no options are selected.\n            Defaults to \'Choose an option\'.\n        disabled : bool\n            An optional boolean, which disables the selectbox if set to True.\n            The default is False.\n        label_visibility : "visible", "hidden", or "collapsed"\n            The visibility of the label. If "hidden", the label doesn\'t show but there\n            is still empty space for it above the widget (equivalent to label="").\n            If "collapsed", both the label and the space are removed. Default is\n            "visible".\n\n        Returns\n        -------\n        any\n            The selected option or ``None`` if no option is selected.\n\n        Example\n        -------\n        >>> import streamlit as st\n        >>>\n        >>> option = st.selectbox(\n        ...     \'How would you like to be contacted?\',\n        ...     (\'Email\', \'Home phone\', \'Mobile phone\'))\n        >>>\n        >>> st.write(\'You selected:\', option)\n\n        .. output::\n           https://doc-selectbox.streamlit.app/\n           height: 320px\n\n        To initialize an empty selectbox, use ``None`` as the index value:\n\n        >>> import streamlit as st\n        >>>\n        >>> option = st.selectbox(\n        ...    "How would you like to be contacted?",\n        ...    ("Email", "Home phone", "Mobile phone"),\n        ...    index=None,\n        ...    placeholder="Select contact method...",\n        ... )\n        >>>\n        >>> st.write(\'You selected:\', option)\n\n        .. output::\n           https://doc-selectbox-empty.streamlit.app/\n           height: 320px\n\n        '
        ctx = get_script_run_ctx()
        return self._selectbox(label=label, options=options, index=index, format_func=format_func, key=key, help=help, on_change=on_change, args=args, kwargs=kwargs, placeholder=placeholder, disabled=disabled, label_visibility=label_visibility, ctx=ctx)

    def _selectbox(self, label: str, options: OptionSequence[T], index: int | None=0, format_func: Callable[[Any], Any]=str, key: Key | None=None, help: str | None=None, on_change: WidgetCallback | None=None, args: WidgetArgs | None=None, kwargs: WidgetKwargs | None=None, *, placeholder: str='Choose an option', disabled: bool=False, label_visibility: LabelVisibility='visible', ctx: ScriptRunContext | None=None) -> T | None:
        if False:
            print('Hello World!')
        key = to_key(key)
        check_callback_rules(self.dg, on_change)
        check_session_state_rules(default_value=None if index == 0 else index, key=key)
        maybe_raise_label_warnings(label, label_visibility)
        opt = ensure_indexable(options)
        id = compute_widget_id('selectbox', user_key=key, label=label, options=[str(format_func(option)) for option in opt], index=index, key=key, help=help, placeholder=placeholder, form_id=current_form_id(self.dg), page=ctx.page_script_hash if ctx else None)
        if not isinstance(index, int) and index is not None:
            raise StreamlitAPIException('Selectbox Value has invalid type: %s' % type(index).__name__)
        if index is not None and len(opt) > 0 and (not 0 <= index < len(opt)):
            raise StreamlitAPIException('Selectbox index must be between 0 and length of options')
        selectbox_proto = SelectboxProto()
        selectbox_proto.id = id
        selectbox_proto.label = label
        if index is not None:
            selectbox_proto.default = index
        selectbox_proto.options[:] = [str(format_func(option)) for option in opt]
        selectbox_proto.form_id = current_form_id(self.dg)
        selectbox_proto.placeholder = placeholder
        selectbox_proto.disabled = disabled
        selectbox_proto.label_visibility.value = get_label_visibility_proto_value(label_visibility)
        if help is not None:
            selectbox_proto.help = dedent(help)
        serde = SelectboxSerde(opt, index)
        widget_state = register_widget('selectbox', selectbox_proto, user_key=key, on_change_handler=on_change, args=args, kwargs=kwargs, deserializer=serde.deserialize, serializer=serde.serialize, ctx=ctx)
        widget_state = maybe_coerce_enum(widget_state, options, opt)
        if widget_state.value_changed:
            serialized_value = serde.serialize(widget_state.value)
            if serialized_value is not None:
                selectbox_proto.value = serialized_value
            selectbox_proto.set_value = True
        self.dg._enqueue('selectbox', selectbox_proto)
        return widget_state.value

    @property
    def dg(self) -> 'DeltaGenerator':
        if False:
            while True:
                i = 10
        'Get our DeltaGenerator.'
        return cast('DeltaGenerator', self)