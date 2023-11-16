from __future__ import annotations
import numbers
from dataclasses import dataclass
from textwrap import dedent
from typing import Literal, Union, cast, overload
import streamlit
from streamlit.elements.form import current_form_id
from streamlit.elements.utils import check_callback_rules, check_session_state_rules, get_label_visibility_proto_value
from streamlit.errors import StreamlitAPIException
from streamlit.js_number import JSNumber, JSNumberBoundsException
from streamlit.proto.NumberInput_pb2 import NumberInput as NumberInputProto
from streamlit.runtime.metrics_util import gather_metrics
from streamlit.runtime.scriptrunner import ScriptRunContext, get_script_run_ctx
from streamlit.runtime.state import WidgetArgs, WidgetCallback, WidgetKwargs, register_widget
from streamlit.runtime.state.common import compute_widget_id
from streamlit.type_util import Key, LabelVisibility, maybe_raise_label_warnings, to_key
Number = Union[int, float]

@dataclass
class NumberInputSerde:
    value: Number | None
    data_type: int

    def serialize(self, v: Number | None) -> Number | None:
        if False:
            print('Hello World!')
        return v

    def deserialize(self, ui_value: Number | None, widget_id: str='') -> Number | None:
        if False:
            i = 10
            return i + 15
        val: Number | None = ui_value if ui_value is not None else self.value
        if val is not None and self.data_type == NumberInputProto.INT:
            val = int(val)
        return val

class NumberInputMixin:

    @overload
    def number_input(self, label: str, min_value: Number | None=None, max_value: Number | None=None, value: Number | Literal['min']='min', step: Number | None=None, format: str | None=None, key: Key | None=None, help: str | None=None, on_change: WidgetCallback | None=None, args: WidgetArgs | None=None, kwargs: WidgetKwargs | None=None, *, placeholder: str | None=None, disabled: bool=False, label_visibility: LabelVisibility='visible') -> Number:
        if False:
            for i in range(10):
                print('nop')
        pass

    @overload
    def number_input(self, label: str, min_value: Number | None=None, max_value: Number | None=None, value: None=None, step: Number | None=None, format: str | None=None, key: Key | None=None, help: str | None=None, on_change: WidgetCallback | None=None, args: WidgetArgs | None=None, kwargs: WidgetKwargs | None=None, *, placeholder: str | None=None, disabled: bool=False, label_visibility: LabelVisibility='visible') -> Number | None:
        if False:
            for i in range(10):
                print('nop')
        pass

    @gather_metrics('number_input')
    def number_input(self, label: str, min_value: Number | None=None, max_value: Number | None=None, value: Number | Literal['min'] | None='min', step: Number | None=None, format: str | None=None, key: Key | None=None, help: str | None=None, on_change: WidgetCallback | None=None, args: WidgetArgs | None=None, kwargs: WidgetKwargs | None=None, *, placeholder: str | None=None, disabled: bool=False, label_visibility: LabelVisibility='visible') -> Number | None:
        if False:
            for i in range(10):
                print('nop')
        'Display a numeric input widget.\n\n        .. note::\n            Integer values exceeding +/- ``(1<<53) - 1`` cannot be accurately\n            stored or returned by the widget due to serialization contstraints\n            between the Python server and JavaScript client. You must handle\n            such numbers as floats, leading to a loss in precision.\n\n        Parameters\n        ----------\n        label : str\n            A short label explaining to the user what this input is for.\n            The label can optionally contain Markdown and supports the following\n            elements: Bold, Italics, Strikethroughs, Inline Code, Emojis, and Links.\n\n            This also supports:\n\n            * Emoji shortcodes, such as ``:+1:``  and ``:sunglasses:``.\n              For a list of all supported codes,\n              see https://share.streamlit.io/streamlit/emoji-shortcodes.\n\n            * LaTeX expressions, by wrapping them in "$" or "$$" (the "$$"\n              must be on their own lines). Supported LaTeX functions are listed\n              at https://katex.org/docs/supported.html.\n\n            * Colored text, using the syntax ``:color[text to be colored]``,\n              where ``color`` needs to be replaced with any of the following\n              supported colors: blue, green, orange, red, violet, gray/grey, rainbow.\n\n            Unsupported elements are unwrapped so only their children (text contents) render.\n            Display unsupported elements as literal characters by\n            backslash-escaping them. E.g. ``1\\. Not an ordered list``.\n\n            For accessibility reasons, you should never set an empty label (label="")\n            but hide it with label_visibility if needed. In the future, we may disallow\n            empty labels by raising an exception.\n        min_value : int, float, or None\n            The minimum permitted value.\n            If None, there will be no minimum.\n        max_value : int, float, or None\n            The maximum permitted value.\n            If None, there will be no maximum.\n        value : int, float, "min" or None\n            The value of this widget when it first renders. If ``None``, will initialize\n            empty and return ``None`` until the user provides input.\n            If "min" (default), will initialize with min_value, or 0.0 if\n            min_value is None.\n        step : int, float, or None\n            The stepping interval.\n            Defaults to 1 if the value is an int, 0.01 otherwise.\n            If the value is not specified, the format parameter will be used.\n        format : str or None\n            A printf-style format string controlling how the interface should\n            display numbers. Output must be purely numeric. This does not impact\n            the return value. Valid formatters: %d %e %f %g %i %u\n        key : str or int\n            An optional string or integer to use as the unique key for the widget.\n            If this is omitted, a key will be generated for the widget\n            based on its content. Multiple widgets of the same type may\n            not share the same key.\n        help : str\n            An optional tooltip that gets displayed next to the input.\n        on_change : callable\n            An optional callback invoked when this number_input\'s value changes.\n        args : tuple\n            An optional tuple of args to pass to the callback.\n        kwargs : dict\n            An optional dict of kwargs to pass to the callback.\n        placeholder : str or None\n            An optional string displayed when the number input is empty.\n            If None, no placeholder is displayed.\n        disabled : bool\n            An optional boolean, which disables the number input if set to\n            True. The default is False.\n        label_visibility : "visible", "hidden", or "collapsed"\n            The visibility of the label. If "hidden", the label doesn\'t show but there\n            is still empty space for it above the widget (equivalent to label="").\n            If "collapsed", both the label and the space are removed. Default is\n            "visible".\n\n        Returns\n        -------\n        int or float or None\n            The current value of the numeric input widget or ``None`` if the widget\n            is empty. The return type will match the data type of the value parameter.\n\n        Example\n        -------\n        >>> import streamlit as st\n        >>>\n        >>> number = st.number_input(\'Insert a number\')\n        >>> st.write(\'The current number is \', number)\n\n        .. output::\n           https://doc-number-input.streamlit.app/\n           height: 260px\n\n        To initialize an empty number input, use ``None`` as the value:\n\n        >>> import streamlit as st\n        >>>\n        >>> number = st.number_input("Insert a number", value=None, placeholder="Type a number...")\n        >>> st.write(\'The current number is \', number)\n\n        .. output::\n           https://doc-number-input-empty.streamlit.app/\n           height: 260px\n\n        '
        ctx = get_script_run_ctx()
        return self._number_input(label=label, min_value=min_value, max_value=max_value, value=value, step=step, format=format, key=key, help=help, on_change=on_change, args=args, kwargs=kwargs, placeholder=placeholder, disabled=disabled, label_visibility=label_visibility, ctx=ctx)

    def _number_input(self, label: str, min_value: Number | None=None, max_value: Number | None=None, value: Number | Literal['min'] | None='min', step: Number | None=None, format: str | None=None, key: Key | None=None, help: str | None=None, on_change: WidgetCallback | None=None, args: WidgetArgs | None=None, kwargs: WidgetKwargs | None=None, *, placeholder: str | None=None, disabled: bool=False, label_visibility: LabelVisibility='visible', ctx: ScriptRunContext | None=None) -> Number | None:
        if False:
            i = 10
            return i + 15
        key = to_key(key)
        check_callback_rules(self.dg, on_change)
        check_session_state_rules(default_value=value if value != 'min' else None, key=key)
        maybe_raise_label_warnings(label, label_visibility)
        id = compute_widget_id('number_input', user_key=key, label=label, min_value=min_value, max_value=max_value, value=value, step=step, format=format, key=key, help=help, placeholder=None if placeholder is None else str(placeholder), form_id=current_form_id(self.dg), page=ctx.page_script_hash if ctx else None)
        number_input_args = [min_value, max_value, value, step]
        int_args = all((isinstance(a, (numbers.Integral, type(None), str)) for a in number_input_args))
        float_args = all((isinstance(a, (float, type(None), str)) for a in number_input_args))
        if not int_args and (not float_args):
            raise StreamlitAPIException(f'All numerical arguments must be of the same type.\n`value` has {type(value).__name__} type.\n`min_value` has {type(min_value).__name__} type.\n`max_value` has {type(max_value).__name__} type.\n`step` has {type(step).__name__} type.')
        if value == 'min':
            if min_value is not None:
                value = min_value
            elif int_args and float_args:
                value = 0.0
            elif int_args:
                value = 0
            else:
                value = 0.0
        int_value = isinstance(value, numbers.Integral)
        float_value = isinstance(value, float)
        if value is None:
            if int_args and (not float_args):
                int_value = True
            else:
                float_value = True
        if format is None:
            format = '%d' if int_value else '%0.2f'
        if format in ['%d', '%u', '%i'] and float_value:
            import streamlit as st
            st.warning(f'Warning: NumberInput value below has type float, but format {format} displays as integer.')
        elif format[-1] == 'f' and int_value:
            import streamlit as st
            st.warning(f'Warning: NumberInput value below has type int so is displayed as int despite format string {format}.')
        if step is None:
            step = 1 if int_value else 0.01
        try:
            float(format % 2)
        except (TypeError, ValueError):
            raise StreamlitAPIException('Format string for st.number_input contains invalid characters: %s' % format)
        all_ints = int_value and int_args
        if min_value is not None and value is not None and (min_value > value):
            raise StreamlitAPIException(f'The default `value` {value} must be greater than or equal to the `min_value` {min_value}')
        if max_value is not None and value is not None and (max_value < value):
            raise StreamlitAPIException(f'The default `value` {value} must be less than or equal to the `max_value` {max_value}')
        try:
            if all_ints:
                if min_value is not None:
                    JSNumber.validate_int_bounds(min_value, '`min_value`')
                if max_value is not None:
                    JSNumber.validate_int_bounds(max_value, '`max_value`')
                if step is not None:
                    JSNumber.validate_int_bounds(step, '`step`')
                if value is not None:
                    JSNumber.validate_int_bounds(value, '`value`')
            else:
                if min_value is not None:
                    JSNumber.validate_float_bounds(min_value, '`min_value`')
                if max_value is not None:
                    JSNumber.validate_float_bounds(max_value, '`max_value`')
                if step is not None:
                    JSNumber.validate_float_bounds(step, '`step`')
                if value is not None:
                    JSNumber.validate_float_bounds(value, '`value`')
        except JSNumberBoundsException as e:
            raise StreamlitAPIException(str(e))
        data_type = NumberInputProto.INT if all_ints else NumberInputProto.FLOAT
        number_input_proto = NumberInputProto()
        number_input_proto.id = id
        number_input_proto.data_type = data_type
        number_input_proto.label = label
        if value is not None:
            number_input_proto.default = value
        if placeholder is not None:
            number_input_proto.placeholder = str(placeholder)
        number_input_proto.form_id = current_form_id(self.dg)
        number_input_proto.disabled = disabled
        number_input_proto.label_visibility.value = get_label_visibility_proto_value(label_visibility)
        if help is not None:
            number_input_proto.help = dedent(help)
        if min_value is not None:
            number_input_proto.min = min_value
            number_input_proto.has_min = True
        if max_value is not None:
            number_input_proto.max = max_value
            number_input_proto.has_max = True
        if step is not None:
            number_input_proto.step = step
        if format is not None:
            number_input_proto.format = format
        serde = NumberInputSerde(value, data_type)
        widget_state = register_widget('number_input', number_input_proto, user_key=key, on_change_handler=on_change, args=args, kwargs=kwargs, deserializer=serde.deserialize, serializer=serde.serialize, ctx=ctx)
        if widget_state.value_changed:
            if widget_state.value is not None:
                number_input_proto.value = widget_state.value
            number_input_proto.set_value = True
        self.dg._enqueue('number_input', number_input_proto)
        return widget_state.value

    @property
    def dg(self) -> 'streamlit.delta_generator.DeltaGenerator':
        if False:
            return 10
        'Get our DeltaGenerator.'
        return cast('streamlit.delta_generator.DeltaGenerator', self)