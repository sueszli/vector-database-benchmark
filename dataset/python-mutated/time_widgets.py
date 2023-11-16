from __future__ import annotations
import re
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from textwrap import dedent
from typing import TYPE_CHECKING, Any, List, Literal, Sequence, Tuple, Union, cast, overload
from dateutil import relativedelta
from typing_extensions import TypeAlias
from streamlit.elements.form import current_form_id
from streamlit.elements.utils import check_callback_rules, check_session_state_rules, get_label_visibility_proto_value
from streamlit.errors import StreamlitAPIException
from streamlit.proto.DateInput_pb2 import DateInput as DateInputProto
from streamlit.proto.TimeInput_pb2 import TimeInput as TimeInputProto
from streamlit.runtime.metrics_util import gather_metrics
from streamlit.runtime.scriptrunner import ScriptRunContext, get_script_run_ctx
from streamlit.runtime.state import WidgetArgs, WidgetCallback, WidgetKwargs, register_widget
from streamlit.runtime.state.common import compute_widget_id
from streamlit.type_util import Key, LabelVisibility, maybe_raise_label_warnings, to_key
if TYPE_CHECKING:
    from streamlit.delta_generator import DeltaGenerator
SingleDateValue: TypeAlias = Union[date, datetime, None]
DateValue: TypeAlias = Union[SingleDateValue, Sequence[SingleDateValue]]
DateWidgetReturn: TypeAlias = Union[date, Tuple[()], Tuple[date], Tuple[date, date], None]
DEFAULT_STEP_MINUTES = 15
ALLOWED_DATE_FORMATS = re.compile('^(YYYY[/.\\-]MM[/.\\-]DD|DD[/.\\-]MM[/.\\-]YYYY|MM[/.\\-]DD[/.\\-]YYYY)$')

def _parse_date_value(value: DateValue | Literal['today']) -> Tuple[List[date] | None, bool]:
    if False:
        print('Hello World!')
    parsed_dates: List[date]
    range_value: bool = False
    if value is None:
        return (None, range_value)
    if value == 'today':
        parsed_dates = [datetime.now().date()]
    elif isinstance(value, datetime):
        parsed_dates = [value.date()]
    elif isinstance(value, date):
        parsed_dates = [value]
    elif isinstance(value, (list, tuple)):
        if not len(value) in (0, 1, 2):
            raise StreamlitAPIException('DateInput value should either be an date/datetime or a list/tuple of 0 - 2 date/datetime values')
        parsed_dates = [v.date() if isinstance(v, datetime) else v for v in value]
        range_value = True
    else:
        raise StreamlitAPIException('DateInput value should either be an date/datetime or a list/tuple of 0 - 2 date/datetime values')
    return (parsed_dates, range_value)

def _parse_min_date(min_value: SingleDateValue, parsed_dates: Sequence[date] | None) -> date:
    if False:
        print('Hello World!')
    parsed_min_date: date
    if isinstance(min_value, datetime):
        parsed_min_date = min_value.date()
    elif isinstance(min_value, date):
        parsed_min_date = min_value
    elif min_value is None:
        if parsed_dates:
            parsed_min_date = parsed_dates[0] - relativedelta.relativedelta(years=10)
        else:
            parsed_min_date = date.today() - relativedelta.relativedelta(years=10)
    else:
        raise StreamlitAPIException('DateInput min should either be a date/datetime or None')
    return parsed_min_date

def _parse_max_date(max_value: SingleDateValue, parsed_dates: Sequence[date] | None) -> date:
    if False:
        return 10
    parsed_max_date: date
    if isinstance(max_value, datetime):
        parsed_max_date = max_value.date()
    elif isinstance(max_value, date):
        parsed_max_date = max_value
    elif max_value is None:
        if parsed_dates:
            parsed_max_date = parsed_dates[-1] + relativedelta.relativedelta(years=10)
        else:
            parsed_max_date = date.today() + relativedelta.relativedelta(years=10)
    else:
        raise StreamlitAPIException('DateInput max should either be a date/datetime or None')
    return parsed_max_date

@dataclass(frozen=True)
class _DateInputValues:
    value: Sequence[date] | None
    is_range: bool
    max: date
    min: date

    @classmethod
    def from_raw_values(cls, value: DateValue | Literal['today'], min_value: SingleDateValue, max_value: SingleDateValue) -> '_DateInputValues':
        if False:
            return 10
        (parsed_value, is_range) = _parse_date_value(value=value)
        return cls(value=parsed_value, is_range=is_range, min=_parse_min_date(min_value=min_value, parsed_dates=parsed_value), max=_parse_max_date(max_value=max_value, parsed_dates=parsed_value))

    def __post_init__(self) -> None:
        if False:
            print('Hello World!')
        if self.min > self.max:
            raise StreamlitAPIException(f"The `min_value`, set to {self.min}, shouldn't be larger than the `max_value`, set to {self.max}.")
        if self.value:
            start_value = self.value[0]
            end_value = self.value[-1]
            if start_value < self.min or end_value > self.max:
                raise StreamlitAPIException(f'The default `value` of {self.value} must lie between the `min_value` of {self.min} and the `max_value` of {self.max}, inclusively.')

@dataclass
class TimeInputSerde:
    value: time | None

    def deserialize(self, ui_value: str | None, widget_id: Any='') -> time | None:
        if False:
            while True:
                i = 10
        return datetime.strptime(ui_value, '%H:%M').time() if ui_value is not None else self.value

    def serialize(self, v: datetime | time | None) -> str | None:
        if False:
            print('Hello World!')
        if v is None:
            return None
        if isinstance(v, datetime):
            v = v.time()
        return time.strftime(v, '%H:%M')

@dataclass
class DateInputSerde:
    value: _DateInputValues

    def deserialize(self, ui_value: Any, widget_id: str='') -> DateWidgetReturn:
        if False:
            i = 10
            return i + 15
        return_value: Sequence[date] | None
        if ui_value is not None:
            return_value = tuple((datetime.strptime(v, '%Y/%m/%d').date() for v in ui_value))
        else:
            return_value = self.value.value
        if return_value is None or len(return_value) == 0:
            return () if self.value.is_range else None
        if not self.value.is_range:
            return return_value[0]
        return cast(DateWidgetReturn, tuple(return_value))

    def serialize(self, v: DateWidgetReturn) -> List[str]:
        if False:
            for i in range(10):
                print('nop')
        if v is None:
            return []
        to_serialize = list(v) if isinstance(v, (list, tuple)) else [v]
        return [date.strftime(v, '%Y/%m/%d') for v in to_serialize]

class TimeWidgetsMixin:

    @overload
    def time_input(self, label: str, value: time | datetime | Literal['now']='now', key: Key | None=None, help: str | None=None, on_change: WidgetCallback | None=None, args: WidgetArgs | None=None, kwargs: WidgetKwargs | None=None, *, disabled: bool=False, label_visibility: LabelVisibility='visible', step: Union[int, timedelta]=timedelta(minutes=DEFAULT_STEP_MINUTES)) -> time:
        if False:
            i = 10
            return i + 15
        pass

    @overload
    def time_input(self, label: str, value: None=None, key: Key | None=None, help: str | None=None, on_change: WidgetCallback | None=None, args: WidgetArgs | None=None, kwargs: WidgetKwargs | None=None, *, disabled: bool=False, label_visibility: LabelVisibility='visible', step: Union[int, timedelta]=timedelta(minutes=DEFAULT_STEP_MINUTES)) -> time | None:
        if False:
            i = 10
            return i + 15
        pass

    @gather_metrics('time_input')
    def time_input(self, label: str, value: time | datetime | Literal['now'] | None='now', key: Key | None=None, help: str | None=None, on_change: WidgetCallback | None=None, args: WidgetArgs | None=None, kwargs: WidgetKwargs | None=None, *, disabled: bool=False, label_visibility: LabelVisibility='visible', step: Union[int, timedelta]=timedelta(minutes=DEFAULT_STEP_MINUTES)) -> time | None:
        if False:
            print('Hello World!')
        'Display a time input widget.\n\n        Parameters\n        ----------\n        label : str\n            A short label explaining to the user what this time input is for.\n            The label can optionally contain Markdown and supports the following\n            elements: Bold, Italics, Strikethroughs, Inline Code, Emojis, and Links.\n\n            This also supports:\n\n            * Emoji shortcodes, such as ``:+1:``  and ``:sunglasses:``.\n              For a list of all supported codes,\n              see https://share.streamlit.io/streamlit/emoji-shortcodes.\n\n            * LaTeX expressions, by wrapping them in "$" or "$$" (the "$$"\n              must be on their own lines). Supported LaTeX functions are listed\n              at https://katex.org/docs/supported.html.\n\n            * Colored text, using the syntax ``:color[text to be colored]``,\n              where ``color`` needs to be replaced with any of the following\n              supported colors: blue, green, orange, red, violet, gray/grey, rainbow.\n\n            Unsupported elements are unwrapped so only their children (text contents) render.\n            Display unsupported elements as literal characters by\n            backslash-escaping them. E.g. ``1\\. Not an ordered list``.\n\n            For accessibility reasons, you should never set an empty label (label="")\n            but hide it with label_visibility if needed. In the future, we may disallow\n            empty labels by raising an exception.\n        value : datetime.time/datetime.datetime, "now" or None\n            The value of this widget when it first renders. This will be\n            cast to str internally. If ``None``, will initialize empty and\n            return ``None`` until the user selects a time. If "now" (default),\n            will initialize with the current time.\n        key : str or int\n            An optional string or integer to use as the unique key for the widget.\n            If this is omitted, a key will be generated for the widget\n            based on its content. Multiple widgets of the same type may\n            not share the same key.\n        help : str\n            An optional tooltip that gets displayed next to the input.\n        on_change : callable\n            An optional callback invoked when this time_input\'s value changes.\n        args : tuple\n            An optional tuple of args to pass to the callback.\n        kwargs : dict\n            An optional dict of kwargs to pass to the callback.\n        disabled : bool\n            An optional boolean, which disables the time input if set to True.\n            The default is False.\n        label_visibility : "visible", "hidden", or "collapsed"\n            The visibility of the label. If "hidden", the label doesn\'t show but there\n            is still empty space for it above the widget (equivalent to label="").\n            If "collapsed", both the label and the space are removed. Default is\n            "visible".\n        step : int or timedelta\n            The stepping interval in seconds. Defaults to 900, i.e. 15 minutes.\n            You can also pass a datetime.timedelta object.\n\n        Returns\n        -------\n        datetime.time or None\n            The current value of the time input widget or ``None`` if no time has been\n            selected.\n\n        Example\n        -------\n        >>> import datetime\n        >>> import streamlit as st\n        >>>\n        >>> t = st.time_input(\'Set an alarm for\', datetime.time(8, 45))\n        >>> st.write(\'Alarm is set for\', t)\n\n        .. output::\n           https://doc-time-input.streamlit.app/\n           height: 260px\n\n        To initialize an empty time input, use ``None`` as the value:\n\n        >>> import datetime\n        >>> import streamlit as st\n        >>>\n        >>> t = st.time_input(\'Set an alarm for\', value=None)\n        >>> st.write(\'Alarm is set for\', t)\n\n        .. output::\n           https://doc-time-input-empty.streamlit.app/\n           height: 260px\n\n        '
        ctx = get_script_run_ctx()
        return self._time_input(label=label, value=value, key=key, help=help, on_change=on_change, args=args, kwargs=kwargs, disabled=disabled, label_visibility=label_visibility, step=step, ctx=ctx)

    def _time_input(self, label: str, value: time | datetime | Literal['now'] | None='now', key: Key | None=None, help: str | None=None, on_change: WidgetCallback | None=None, args: WidgetArgs | None=None, kwargs: WidgetKwargs | None=None, *, disabled: bool=False, label_visibility: LabelVisibility='visible', step: Union[int, timedelta]=timedelta(minutes=DEFAULT_STEP_MINUTES), ctx: ScriptRunContext | None=None) -> time | None:
        if False:
            return 10
        key = to_key(key)
        check_callback_rules(self.dg, on_change)
        check_session_state_rules(default_value=value if value != 'now' else None, key=key)
        maybe_raise_label_warnings(label, label_visibility)
        parsed_time: time | None
        if value is None:
            parsed_time = None
        elif value == 'now':
            parsed_time = datetime.now().time().replace(second=0, microsecond=0)
        elif isinstance(value, datetime):
            parsed_time = value.time().replace(second=0, microsecond=0)
        elif isinstance(value, time):
            parsed_time = value
        else:
            raise StreamlitAPIException('The type of value should be one of datetime, time or None')
        id = compute_widget_id('time_input', user_key=key, label=label, value=parsed_time if isinstance(value, (datetime, time)) else value, key=key, help=help, step=step, form_id=current_form_id(self.dg), page=ctx.page_script_hash if ctx else None)
        del value
        time_input_proto = TimeInputProto()
        time_input_proto.id = id
        time_input_proto.label = label
        if parsed_time is not None:
            time_input_proto.default = time.strftime(parsed_time, '%H:%M')
        time_input_proto.form_id = current_form_id(self.dg)
        if not isinstance(step, (int, timedelta)):
            raise StreamlitAPIException(f'`step` can only be `int` or `timedelta` but {type(step)} is provided.')
        if isinstance(step, timedelta):
            step = step.seconds
        if step < 60 or step > timedelta(hours=23).seconds:
            raise StreamlitAPIException(f'`step` must be between 60 seconds and 23 hours but is currently set to {step} seconds.')
        time_input_proto.step = step
        time_input_proto.disabled = disabled
        time_input_proto.label_visibility.value = get_label_visibility_proto_value(label_visibility)
        if help is not None:
            time_input_proto.help = dedent(help)
        serde = TimeInputSerde(parsed_time)
        widget_state = register_widget('time_input', time_input_proto, user_key=key, on_change_handler=on_change, args=args, kwargs=kwargs, deserializer=serde.deserialize, serializer=serde.serialize, ctx=ctx)
        if widget_state.value_changed:
            if (serialized_value := serde.serialize(widget_state.value)) is not None:
                time_input_proto.value = serialized_value
            time_input_proto.set_value = True
        self.dg._enqueue('time_input', time_input_proto)
        return widget_state.value

    @gather_metrics('date_input')
    def date_input(self, label: str, value: DateValue | Literal['today']='today', min_value: SingleDateValue=None, max_value: SingleDateValue=None, key: Key | None=None, help: str | None=None, on_change: WidgetCallback | None=None, args: WidgetArgs | None=None, kwargs: WidgetKwargs | None=None, *, format: str='YYYY/MM/DD', disabled: bool=False, label_visibility: LabelVisibility='visible') -> DateWidgetReturn:
        if False:
            for i in range(10):
                print('nop')
        'Display a date input widget.\n\n        Parameters\n        ----------\n        label : str\n            A short label explaining to the user what this date input is for.\n            The label can optionally contain Markdown and supports the following\n            elements: Bold, Italics, Strikethroughs, Inline Code, Emojis, and Links.\n\n            This also supports:\n\n            * Emoji shortcodes, such as ``:+1:``  and ``:sunglasses:``.\n              For a list of all supported codes,\n              see https://share.streamlit.io/streamlit/emoji-shortcodes.\n\n            * LaTeX expressions, by wrapping them in "$" or "$$" (the "$$"\n              must be on their own lines). Supported LaTeX functions are listed\n              at https://katex.org/docs/supported.html.\n\n            * Colored text, using the syntax ``:color[text to be colored]``,\n              where ``color`` needs to be replaced with any of the following\n              supported colors: blue, green, orange, red, violet, gray/grey, rainbow.\n\n            Unsupported elements are unwrapped so only their children (text contents) render.\n            Display unsupported elements as literal characters by\n            backslash-escaping them. E.g. ``1\\. Not an ordered list``.\n\n            For accessibility reasons, you should never set an empty label (label="")\n            but hide it with label_visibility if needed. In the future, we may disallow\n            empty labels by raising an exception.\n        value : datetime.date or datetime.datetime or list/tuple of datetime.date or datetime.datetime, "today", or None\n            The value of this widget when it first renders. If a list/tuple with\n            0 to 2 date/datetime values is provided, the datepicker will allow\n            users to provide a range. If ``None``, will initialize empty and\n            return ``None`` until the user provides input. If "today" (default),\n            will initialize with today as a single-date picker.\n        min_value : datetime.date or datetime.datetime\n            The minimum selectable date. If value is a date, defaults to value - 10 years.\n            If value is the interval [start, end], defaults to start - 10 years.\n        max_value : datetime.date or datetime.datetime\n            The maximum selectable date. If value is a date, defaults to value + 10 years.\n            If value is the interval [start, end], defaults to end + 10 years.\n        key : str or int\n            An optional string or integer to use as the unique key for the widget.\n            If this is omitted, a key will be generated for the widget\n            based on its content. Multiple widgets of the same type may\n            not share the same key.\n        help : str\n            An optional tooltip that gets displayed next to the input.\n        on_change : callable\n            An optional callback invoked when this date_input\'s value changes.\n        args : tuple\n            An optional tuple of args to pass to the callback.\n        kwargs : dict\n            An optional dict of kwargs to pass to the callback.\n        format : str\n            A format string controlling how the interface should display dates.\n            Supports “YYYY/MM/DD” (default), “DD/MM/YYYY”, or “MM/DD/YYYY”.\n            You may also use a period (.) or hyphen (-) as separators.\n        disabled : bool\n            An optional boolean, which disables the date input if set to True.\n            The default is False.\n        label_visibility : "visible", "hidden", or "collapsed"\n            The visibility of the label. If "hidden", the label doesn\'t show but there\n            is still empty space for it above the widget (equivalent to label="").\n            If "collapsed", both the label and the space are removed. Default is\n            "visible".\n\n\n        Returns\n        -------\n        datetime.date or a tuple with 0-2 dates or None\n            The current value of the date input widget or ``None`` if no date has been\n            selected.\n\n        Examples\n        --------\n        >>> import datetime\n        >>> import streamlit as st\n        >>>\n        >>> d = st.date_input("When\'s your birthday", datetime.date(2019, 7, 6))\n        >>> st.write(\'Your birthday is:\', d)\n\n        .. output::\n           https://doc-date-input.streamlit.app/\n           height: 380px\n\n        >>> import datetime\n        >>> import streamlit as st\n        >>>\n        >>> today = datetime.datetime.now()\n        >>> next_year = today.year + 1\n        >>> jan_1 = datetime.date(next_year, 1, 1)\n        >>> dec_31 = datetime.date(next_year, 12, 31)\n        >>>\n        >>> d = st.date_input(\n        ...     "Select your vacation for next year",\n        ...     (jan_1, datetime.date(next_year, 1, 7)),\n        ...     jan_1,\n        ...     dec_31,\n        ...     format="MM.DD.YYYY",\n        ... )\n        >>> d\n\n        .. output::\n           https://doc-date-input1.streamlit.app/\n           height: 380px\n\n        To initialize an empty date input, use ``None`` as the value:\n\n        >>> import datetime\n        >>> import streamlit as st\n        >>>\n        >>> d = st.date_input("When\'s your birthday", value=None)\n        >>> st.write(\'Your birthday is:\', d)\n\n        .. output::\n           https://doc-date-input-empty.streamlit.app/\n           height: 380px\n\n        '
        ctx = get_script_run_ctx()
        return self._date_input(label=label, value=value, min_value=min_value, max_value=max_value, key=key, help=help, on_change=on_change, args=args, kwargs=kwargs, disabled=disabled, label_visibility=label_visibility, format=format, ctx=ctx)

    def _date_input(self, label: str, value: DateValue | Literal['today']='today', min_value: SingleDateValue=None, max_value: SingleDateValue=None, key: Key | None=None, help: str | None=None, on_change: WidgetCallback | None=None, args: WidgetArgs | None=None, kwargs: WidgetKwargs | None=None, *, format: str='YYYY/MM/DD', disabled: bool=False, label_visibility: LabelVisibility='visible', ctx: ScriptRunContext | None=None) -> DateWidgetReturn:
        if False:
            return 10
        key = to_key(key)
        check_callback_rules(self.dg, on_change)
        check_session_state_rules(default_value=value if value != 'today' else None, key=key)
        maybe_raise_label_warnings(label, label_visibility)

        def parse_date_deterministic(v: SingleDateValue | Literal['today']) -> str | None:
            if False:
                for i in range(10):
                    print('nop')
            if isinstance(v, datetime):
                return date.strftime(v.date(), '%Y/%m/%d')
            elif isinstance(v, date):
                return date.strftime(v, '%Y/%m/%d')
            return None
        parsed_min_date = parse_date_deterministic(min_value)
        parsed_max_date = parse_date_deterministic(max_value)
        parsed: str | None | List[str | None]
        if value == 'today' or value is None:
            parsed = None
        elif isinstance(value, (datetime, date)):
            parsed = parse_date_deterministic(value)
        else:
            parsed = [parse_date_deterministic(cast(SingleDateValue, v)) for v in value]
        id = compute_widget_id('date_input', user_key=key, label=label, value=parsed, min_value=parsed_min_date, max_value=parsed_max_date, key=key, help=help, format=format, form_id=current_form_id(self.dg), page=ctx.page_script_hash if ctx else None)
        if not bool(ALLOWED_DATE_FORMATS.match(format)):
            raise StreamlitAPIException(f'The provided format (`{format}`) is not valid. DateInput format should be one of `YYYY/MM/DD`, `DD/MM/YYYY`, or `MM/DD/YYYY` and can also use a period (.) or hyphen (-) as separators.')
        parsed_values = _DateInputValues.from_raw_values(value=value, min_value=min_value, max_value=max_value)
        del value, min_value, max_value
        date_input_proto = DateInputProto()
        date_input_proto.id = id
        date_input_proto.is_range = parsed_values.is_range
        date_input_proto.disabled = disabled
        date_input_proto.label_visibility.value = get_label_visibility_proto_value(label_visibility)
        date_input_proto.format = format
        date_input_proto.label = label
        if parsed_values.value is None:
            date_input_proto.default[:] = []
        else:
            date_input_proto.default[:] = [date.strftime(v, '%Y/%m/%d') for v in parsed_values.value]
        date_input_proto.min = date.strftime(parsed_values.min, '%Y/%m/%d')
        date_input_proto.max = date.strftime(parsed_values.max, '%Y/%m/%d')
        date_input_proto.form_id = current_form_id(self.dg)
        if help is not None:
            date_input_proto.help = dedent(help)
        serde = DateInputSerde(parsed_values)
        widget_state = register_widget('date_input', date_input_proto, user_key=key, on_change_handler=on_change, args=args, kwargs=kwargs, deserializer=serde.deserialize, serializer=serde.serialize, ctx=ctx)
        if widget_state.value_changed:
            date_input_proto.value[:] = serde.serialize(widget_state.value)
            date_input_proto.set_value = True
        self.dg._enqueue('date_input', date_input_proto)
        return widget_state.value

    @property
    def dg(self) -> 'DeltaGenerator':
        if False:
            return 10
        'Get our DeltaGenerator.'
        return cast('DeltaGenerator', self)