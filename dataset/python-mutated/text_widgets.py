from __future__ import annotations
from dataclasses import dataclass
from textwrap import dedent
from typing import cast, overload
from typing_extensions import Literal
import streamlit
from streamlit.elements.form import current_form_id
from streamlit.elements.utils import check_callback_rules, check_session_state_rules, get_label_visibility_proto_value
from streamlit.errors import StreamlitAPIException
from streamlit.proto.TextArea_pb2 import TextArea as TextAreaProto
from streamlit.proto.TextInput_pb2 import TextInput as TextInputProto
from streamlit.runtime.metrics_util import gather_metrics
from streamlit.runtime.scriptrunner import ScriptRunContext, get_script_run_ctx
from streamlit.runtime.state import WidgetArgs, WidgetCallback, WidgetKwargs, register_widget
from streamlit.runtime.state.common import compute_widget_id
from streamlit.type_util import Key, LabelVisibility, SupportsStr, maybe_raise_label_warnings, to_key

@dataclass
class TextInputSerde:
    value: str | None

    def deserialize(self, ui_value: str | None, widget_id: str='') -> str | None:
        if False:
            while True:
                i = 10
        return ui_value if ui_value is not None else self.value

    def serialize(self, v: str | None) -> str | None:
        if False:
            for i in range(10):
                print('nop')
        return v

@dataclass
class TextAreaSerde:
    value: str | None

    def deserialize(self, ui_value: str | None, widget_id: str='') -> str | None:
        if False:
            for i in range(10):
                print('nop')
        return ui_value if ui_value is not None else self.value

    def serialize(self, v: str | None) -> str | None:
        if False:
            i = 10
            return i + 15
        return v

class TextWidgetsMixin:

    @overload
    def text_input(self, label: str, value: str='', max_chars: int | None=None, key: Key | None=None, type: Literal['default', 'password']='default', help: str | None=None, autocomplete: str | None=None, on_change: WidgetCallback | None=None, args: WidgetArgs | None=None, kwargs: WidgetKwargs | None=None, *, placeholder: str | None=None, disabled: bool=False, label_visibility: LabelVisibility='visible') -> str:
        if False:
            return 10
        pass

    @overload
    def text_input(self, label: str, value: SupportsStr | None=None, max_chars: int | None=None, key: Key | None=None, type: Literal['default', 'password']='default', help: str | None=None, autocomplete: str | None=None, on_change: WidgetCallback | None=None, args: WidgetArgs | None=None, kwargs: WidgetKwargs | None=None, *, placeholder: str | None=None, disabled: bool=False, label_visibility: LabelVisibility='visible') -> str | None:
        if False:
            for i in range(10):
                print('nop')
        pass

    @gather_metrics('text_input')
    def text_input(self, label: str, value: str | SupportsStr | None='', max_chars: int | None=None, key: Key | None=None, type: Literal['default', 'password']='default', help: str | None=None, autocomplete: str | None=None, on_change: WidgetCallback | None=None, args: WidgetArgs | None=None, kwargs: WidgetKwargs | None=None, *, placeholder: str | None=None, disabled: bool=False, label_visibility: LabelVisibility='visible') -> str | None:
        if False:
            print('Hello World!')
        'Display a single-line text input widget.\n\n        Parameters\n        ----------\n        label : str\n            A short label explaining to the user what this input is for.\n            The label can optionally contain Markdown and supports the following\n            elements: Bold, Italics, Strikethroughs, Inline Code, Emojis, and Links.\n\n            This also supports:\n\n            * Emoji shortcodes, such as ``:+1:``  and ``:sunglasses:``.\n              For a list of all supported codes,\n              see https://share.streamlit.io/streamlit/emoji-shortcodes.\n\n            * LaTeX expressions, by wrapping them in "$" or "$$" (the "$$"\n              must be on their own lines). Supported LaTeX functions are listed\n              at https://katex.org/docs/supported.html.\n\n            * Colored text, using the syntax ``:color[text to be colored]``,\n              where ``color`` needs to be replaced with any of the following\n              supported colors: blue, green, orange, red, violet, gray/grey, rainbow.\n\n            Unsupported elements are unwrapped so only their children (text contents) render.\n            Display unsupported elements as literal characters by\n            backslash-escaping them. E.g. ``1\\. Not an ordered list``.\n\n            For accessibility reasons, you should never set an empty label (label="")\n            but hide it with label_visibility if needed. In the future, we may disallow\n            empty labels by raising an exception.\n        value : object or None\n            The text value of this widget when it first renders. This will be\n            cast to str internally. If ``None``, will initialize empty and\n            return ``None`` until the user provides input. Defaults to empty string.\n        max_chars : int or None\n            Max number of characters allowed in text input.\n        key : str or int\n            An optional string or integer to use as the unique key for the widget.\n            If this is omitted, a key will be generated for the widget\n            based on its content. Multiple widgets of the same type may\n            not share the same key.\n        type : "default" or "password"\n            The type of the text input. This can be either "default" (for\n            a regular text input), or "password" (for a text input that\n            masks the user\'s typed value). Defaults to "default".\n        help : str\n            An optional tooltip that gets displayed next to the input.\n        autocomplete : str\n            An optional value that will be passed to the <input> element\'s\n            autocomplete property. If unspecified, this value will be set to\n            "new-password" for "password" inputs, and the empty string for\n            "default" inputs. For more details, see https://developer.mozilla.org/en-US/docs/Web/HTML/Attributes/autocomplete\n        on_change : callable\n            An optional callback invoked when this text input\'s value changes.\n        args : tuple\n            An optional tuple of args to pass to the callback.\n        kwargs : dict\n            An optional dict of kwargs to pass to the callback.\n        placeholder : str or None\n            An optional string displayed when the text input is empty. If None,\n            no text is displayed.\n        disabled : bool\n            An optional boolean, which disables the text input if set to True.\n            The default is False.\n        label_visibility : "visible", "hidden", or "collapsed"\n            The visibility of the label. If "hidden", the label doesn\'t show but there\n            is still empty space for it above the widget (equivalent to label="").\n            If "collapsed", both the label and the space are removed. Default is\n            "visible".\n\n        Returns\n        -------\n        str or None\n            The current value of the text input widget or ``None`` if no value has been\n            provided by the user.\n\n        Example\n        -------\n        >>> import streamlit as st\n        >>>\n        >>> title = st.text_input(\'Movie title\', \'Life of Brian\')\n        >>> st.write(\'The current movie title is\', title)\n\n        .. output::\n           https://doc-text-input.streamlit.app/\n           height: 260px\n\n        '
        ctx = get_script_run_ctx()
        return self._text_input(label=label, value=value, max_chars=max_chars, key=key, type=type, help=help, autocomplete=autocomplete, on_change=on_change, args=args, kwargs=kwargs, placeholder=placeholder, disabled=disabled, label_visibility=label_visibility, ctx=ctx)

    def _text_input(self, label: str, value: SupportsStr | None='', max_chars: int | None=None, key: Key | None=None, type: str='default', help: str | None=None, autocomplete: str | None=None, on_change: WidgetCallback | None=None, args: WidgetArgs | None=None, kwargs: WidgetKwargs | None=None, *, placeholder: str | None=None, disabled: bool=False, label_visibility: LabelVisibility='visible', ctx: ScriptRunContext | None=None) -> str | None:
        if False:
            for i in range(10):
                print('nop')
        key = to_key(key)
        check_callback_rules(self.dg, on_change)
        check_session_state_rules(default_value=None if value == '' else value, key=key)
        maybe_raise_label_warnings(label, label_visibility)
        value = str(value) if value is not None else None
        id = compute_widget_id('text_input', user_key=key, label=label, value=value, max_chars=max_chars, key=key, type=type, help=help, autocomplete=autocomplete, placeholder=str(placeholder), form_id=current_form_id(self.dg), page=ctx.page_script_hash if ctx else None)
        text_input_proto = TextInputProto()
        text_input_proto.id = id
        text_input_proto.label = label
        if value is not None:
            text_input_proto.default = value
        text_input_proto.form_id = current_form_id(self.dg)
        text_input_proto.disabled = disabled
        text_input_proto.label_visibility.value = get_label_visibility_proto_value(label_visibility)
        if help is not None:
            text_input_proto.help = dedent(help)
        if max_chars is not None:
            text_input_proto.max_chars = max_chars
        if placeholder is not None:
            text_input_proto.placeholder = str(placeholder)
        if type == 'default':
            text_input_proto.type = TextInputProto.DEFAULT
        elif type == 'password':
            text_input_proto.type = TextInputProto.PASSWORD
        else:
            raise StreamlitAPIException("'%s' is not a valid text_input type. Valid types are 'default' and 'password'." % type)
        if autocomplete is None:
            autocomplete = 'new-password' if type == 'password' else ''
        text_input_proto.autocomplete = autocomplete
        serde = TextInputSerde(value)
        widget_state = register_widget('text_input', text_input_proto, user_key=key, on_change_handler=on_change, args=args, kwargs=kwargs, deserializer=serde.deserialize, serializer=serde.serialize, ctx=ctx)
        if widget_state.value_changed:
            if widget_state.value is not None:
                text_input_proto.value = widget_state.value
            text_input_proto.set_value = True
        self.dg._enqueue('text_input', text_input_proto)
        return widget_state.value

    @overload
    def text_area(self, label: str, value: str='', height: int | None=None, max_chars: int | None=None, key: Key | None=None, help: str | None=None, on_change: WidgetCallback | None=None, args: WidgetArgs | None=None, kwargs: WidgetKwargs | None=None, *, placeholder: str | None=None, disabled: bool=False, label_visibility: LabelVisibility='visible') -> str:
        if False:
            return 10
        pass

    @overload
    def text_area(self, label: str, value: SupportsStr | None=None, height: int | None=None, max_chars: int | None=None, key: Key | None=None, help: str | None=None, on_change: WidgetCallback | None=None, args: WidgetArgs | None=None, kwargs: WidgetKwargs | None=None, *, placeholder: str | None=None, disabled: bool=False, label_visibility: LabelVisibility='visible') -> str | None:
        if False:
            while True:
                i = 10
        pass

    @gather_metrics('text_area')
    def text_area(self, label: str, value: str | SupportsStr | None='', height: int | None=None, max_chars: int | None=None, key: Key | None=None, help: str | None=None, on_change: WidgetCallback | None=None, args: WidgetArgs | None=None, kwargs: WidgetKwargs | None=None, *, placeholder: str | None=None, disabled: bool=False, label_visibility: LabelVisibility='visible') -> str | None:
        if False:
            i = 10
            return i + 15
        'Display a multi-line text input widget.\n\n        Parameters\n        ----------\n        label : str\n            A short label explaining to the user what this input is for.\n            The label can optionally contain Markdown and supports the following\n            elements: Bold, Italics, Strikethroughs, Inline Code, Emojis, and Links.\n\n            This also supports:\n\n            * Emoji shortcodes, such as ``:+1:``  and ``:sunglasses:``.\n              For a list of all supported codes,\n              see https://share.streamlit.io/streamlit/emoji-shortcodes.\n\n            * LaTeX expressions, by wrapping them in "$" or "$$" (the "$$"\n              must be on their own lines). Supported LaTeX functions are listed\n              at https://katex.org/docs/supported.html.\n\n            * Colored text, using the syntax ``:color[text to be colored]``,\n              where ``color`` needs to be replaced with any of the following\n              supported colors: blue, green, orange, red, violet, gray/grey, rainbow.\n\n            Unsupported elements are unwrapped so only their children (text contents) render.\n            Display unsupported elements as literal characters by\n            backslash-escaping them. E.g. ``1\\. Not an ordered list``.\n\n            For accessibility reasons, you should never set an empty label (label="")\n            but hide it with label_visibility if needed. In the future, we may disallow\n            empty labels by raising an exception.\n        value : object or None\n            The text value of this widget when it first renders. This will be\n            cast to str internally. If ``None``, will initialize empty and\n            return ``None`` until the user provides input. Defaults to empty string.\n        height : int or None\n            Desired height of the UI element expressed in pixels. If None, a\n            default height is used.\n        max_chars : int or None\n            Maximum number of characters allowed in text area.\n        key : str or int\n            An optional string or integer to use as the unique key for the widget.\n            If this is omitted, a key will be generated for the widget\n            based on its content. Multiple widgets of the same type may\n            not share the same key.\n        help : str\n            An optional tooltip that gets displayed next to the textarea.\n        on_change : callable\n            An optional callback invoked when this text_area\'s value changes.\n        args : tuple\n            An optional tuple of args to pass to the callback.\n        kwargs : dict\n            An optional dict of kwargs to pass to the callback.\n        placeholder : str or None\n            An optional string displayed when the text area is empty. If None,\n            no text is displayed.\n        disabled : bool\n            An optional boolean, which disables the text area if set to True.\n            The default is False.\n        label_visibility : "visible", "hidden", or "collapsed"\n            The visibility of the label. If "hidden", the label doesn\'t show but there\n            is still empty space for it above the widget (equivalent to label="").\n            If "collapsed", both the label and the space are removed. Default is\n            "visible".\n\n        Returns\n        -------\n        str or None\n            The current value of the text area widget or ``None`` if no value has been\n            provided by the user.\n\n        Example\n        -------\n        >>> import streamlit as st\n        >>>\n        >>> txt = st.text_area(\n        ...     "Text to analyze",\n        ...     "It was the best of times, it was the worst of times, it was the age of "\n        ...     "wisdom, it was the age of foolishness, it was the epoch of belief, it "\n        ...     "was the epoch of incredulity, it was the season of Light, it was the "\n        ...     "season of Darkness, it was the spring of hope, it was the winter of "\n        ...     "despair, (...)",\n        ...     )\n        >>>\n        >>> st.write(f\'You wrote {len(txt)} characters.\')\n\n        .. output::\n           https://doc-text-area.streamlit.app/\n           height: 300px\n\n        '
        ctx = get_script_run_ctx()
        return self._text_area(label=label, value=value, height=height, max_chars=max_chars, key=key, help=help, on_change=on_change, args=args, kwargs=kwargs, placeholder=placeholder, disabled=disabled, label_visibility=label_visibility, ctx=ctx)

    def _text_area(self, label: str, value: SupportsStr | None='', height: int | None=None, max_chars: int | None=None, key: Key | None=None, help: str | None=None, on_change: WidgetCallback | None=None, args: WidgetArgs | None=None, kwargs: WidgetKwargs | None=None, *, placeholder: str | None=None, disabled: bool=False, label_visibility: LabelVisibility='visible', ctx: ScriptRunContext | None=None) -> str | None:
        if False:
            i = 10
            return i + 15
        key = to_key(key)
        check_callback_rules(self.dg, on_change)
        check_session_state_rules(default_value=None if value == '' else value, key=key)
        maybe_raise_label_warnings(label, label_visibility)
        value = str(value) if value is not None else None
        id = compute_widget_id('text_area', user_key=key, label=label, value=value, height=height, max_chars=max_chars, key=key, help=help, placeholder=str(placeholder), form_id=current_form_id(self.dg), page=ctx.page_script_hash if ctx else None)
        text_area_proto = TextAreaProto()
        text_area_proto.id = id
        text_area_proto.label = label
        if value is not None:
            text_area_proto.default = value
        text_area_proto.form_id = current_form_id(self.dg)
        text_area_proto.disabled = disabled
        text_area_proto.label_visibility.value = get_label_visibility_proto_value(label_visibility)
        if help is not None:
            text_area_proto.help = dedent(help)
        if height is not None:
            text_area_proto.height = height
        if max_chars is not None:
            text_area_proto.max_chars = max_chars
        if placeholder is not None:
            text_area_proto.placeholder = str(placeholder)
        serde = TextAreaSerde(value)
        widget_state = register_widget('text_area', text_area_proto, user_key=key, on_change_handler=on_change, args=args, kwargs=kwargs, deserializer=serde.deserialize, serializer=serde.serialize, ctx=ctx)
        if widget_state.value_changed:
            if widget_state.value is not None:
                text_area_proto.value = widget_state.value
            text_area_proto.set_value = True
        self.dg._enqueue('text_area', text_area_proto)
        return widget_state.value

    @property
    def dg(self) -> 'streamlit.delta_generator.DeltaGenerator':
        if False:
            i = 10
            return i + 15
        'Get our DeltaGenerator.'
        return cast('streamlit.delta_generator.DeltaGenerator', self)