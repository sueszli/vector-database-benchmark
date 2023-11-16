"""
Tools for formatting logging events.
"""
from datetime import datetime as DateTime
from typing import Any, Callable, Iterator, Mapping, Optional, Union, cast
from constantly import NamedConstant
from twisted.python._tzhelper import FixedOffsetTimeZone
from twisted.python.failure import Failure
from twisted.python.reflect import safe_repr
from ._flatten import aFormatter, flatFormat
from ._interfaces import LogEvent
timeFormatRFC3339 = '%Y-%m-%dT%H:%M:%S%z'

def formatEvent(event: LogEvent) -> str:
    if False:
        return 10
    '\n    Formats an event as text, using the format in C{event["log_format"]}.\n\n    This implementation should never raise an exception; if the formatting\n    cannot be done, the returned string will describe the event generically so\n    that a useful message is emitted regardless.\n\n    @param event: A logging event.\n\n    @return: A formatted string.\n    '
    return eventAsText(event, includeTraceback=False, includeTimestamp=False, includeSystem=False)

def formatUnformattableEvent(event: LogEvent, error: BaseException) -> str:
    if False:
        return 10
    '\n    Formats an event as text that describes the event generically and a\n    formatting error.\n\n    @param event: A logging event.\n    @param error: The formatting error.\n\n    @return: A formatted string.\n    '
    try:
        return 'Unable to format event {event!r}: {error}'.format(event=event, error=error)
    except BaseException:
        failure = Failure()
        text = ', '.join((' = '.join((safe_repr(key), safe_repr(value))) for (key, value) in event.items()))
        return 'MESSAGE LOST: unformattable object logged: {error}\nRecoverable data: {text}\nException during formatting:\n{failure}'.format(error=safe_repr(error), failure=failure, text=text)

def formatTime(when: Optional[float], timeFormat: Optional[str]=timeFormatRFC3339, default: str='-') -> str:
    if False:
        return 10
    '\n    Format a timestamp as text.\n\n    Example::\n\n        >>> from time import time\n        >>> from twisted.logger import formatTime\n        >>>\n        >>> t = time()\n        >>> formatTime(t)\n        u\'2013-10-22T14:19:11-0700\'\n        >>> formatTime(t, timeFormat="%Y/%W")  # Year and week number\n        u\'2013/42\'\n        >>>\n\n    @param when: A timestamp.\n    @param timeFormat: A time format.\n    @param default: Text to return if C{when} or C{timeFormat} is L{None}.\n\n    @return: A formatted time.\n    '
    if timeFormat is None or when is None:
        return default
    else:
        tz = FixedOffsetTimeZone.fromLocalTimeStamp(when)
        datetime = DateTime.fromtimestamp(when, tz)
        return str(datetime.strftime(timeFormat))

def formatEventAsClassicLogText(event: LogEvent, formatTime: Callable[[Optional[float]], str]=formatTime) -> Optional[str]:
    if False:
        for i in range(10):
            print('nop')
    '\n    Format an event as a line of human-readable text for, e.g. traditional log\n    file output.\n\n    The output format is C{"{timeStamp} [{system}] {event}\\n"}, where:\n\n        - C{timeStamp} is computed by calling the given C{formatTime} callable\n          on the event\'s C{"log_time"} value\n\n        - C{system} is the event\'s C{"log_system"} value, if set, otherwise,\n          the C{"log_namespace"} and C{"log_level"}, joined by a C{"#"}.  Each\n          defaults to C{"-"} is not set.\n\n        - C{event} is the event, as formatted by L{formatEvent}.\n\n    Example::\n\n        >>> from time import time\n        >>> from twisted.logger import formatEventAsClassicLogText\n        >>> from twisted.logger import LogLevel\n        >>>\n        >>> formatEventAsClassicLogText(dict())  # No format, returns None\n        >>> formatEventAsClassicLogText(dict(log_format="Hello!"))\n        u\'- [-#-] Hello!\\n\'\n        >>> formatEventAsClassicLogText(dict(\n        ...     log_format="Hello!",\n        ...     log_time=time(),\n        ...     log_namespace="my_namespace",\n        ...     log_level=LogLevel.info,\n        ... ))\n        u\'2013-10-22T17:30:02-0700 [my_namespace#info] Hello!\\n\'\n        >>> formatEventAsClassicLogText(dict(\n        ...     log_format="Hello!",\n        ...     log_time=time(),\n        ...     log_system="my_system",\n        ... ))\n        u\'2013-11-11T17:22:06-0800 [my_system] Hello!\\n\'\n        >>>\n\n    @param event: an event.\n    @param formatTime: A time formatter\n\n    @return: A formatted event, or L{None} if no output is appropriate.\n    '
    eventText = eventAsText(event, formatTime=formatTime)
    if not eventText:
        return None
    eventText = eventText.replace('\n', '\n\t')
    return eventText + '\n'

class CallMapping(Mapping[str, Any]):
    """
    Read-only mapping that turns a C{()}-suffix in key names into an invocation
    of the key rather than a lookup of the key.

    Implementation support for L{formatWithCall}.
    """

    def __init__(self, submapping: Mapping[str, Any]) -> None:
        if False:
            return 10
        '\n        @param submapping: Another read-only mapping which will be used to look\n            up items.\n        '
        self._submapping = submapping

    def __iter__(self) -> Iterator[Any]:
        if False:
            for i in range(10):
                print('nop')
        return iter(self._submapping)

    def __len__(self) -> int:
        if False:
            while True:
                i = 10
        return len(self._submapping)

    def __getitem__(self, key: str) -> Any:
        if False:
            return 10
        '\n        Look up an item in the submapping for this L{CallMapping}, calling it\n        if C{key} ends with C{"()"}.\n        '
        callit = key.endswith('()')
        realKey = key[:-2] if callit else key
        value = self._submapping[realKey]
        if callit:
            value = value()
        return value

def formatWithCall(formatString: str, mapping: Mapping[str, Any]) -> str:
    if False:
        return 10
    '\n    Format a string like L{str.format}, but:\n\n        - taking only a name mapping; no positional arguments\n\n        - with the additional syntax that an empty set of parentheses\n          correspond to a formatting item that should be called, and its result\n          C{str}\'d, rather than calling C{str} on the element directly as\n          normal.\n\n    For example::\n\n        >>> formatWithCall("{string}, {function()}.",\n        ...                dict(string="just a string",\n        ...                     function=lambda: "a function"))\n        \'just a string, a function.\'\n\n    @param formatString: A PEP-3101 format string.\n    @param mapping: A L{dict}-like object to format.\n\n    @return: The string with formatted values interpolated.\n    '
    return str(aFormatter.vformat(formatString, (), CallMapping(mapping)))

def _formatEvent(event: LogEvent) -> str:
    if False:
        print('Hello World!')
    '\n    Formats an event as a string, using the format in C{event["log_format"]}.\n\n    This implementation should never raise an exception; if the formatting\n    cannot be done, the returned string will describe the event generically so\n    that a useful message is emitted regardless.\n\n    @param event: A logging event.\n\n    @return: A formatted string.\n    '
    try:
        if 'log_flattened' in event:
            return flatFormat(event)
        format = cast(Optional[Union[str, bytes]], event.get('log_format', None))
        if format is None:
            return ''
        if isinstance(format, str):
            pass
        elif isinstance(format, bytes):
            format = format.decode('utf-8')
        else:
            raise TypeError(f'Log format must be str, not {format!r}')
        return formatWithCall(format, event)
    except BaseException as e:
        return formatUnformattableEvent(event, e)

def _formatTraceback(failure: Failure) -> str:
    if False:
        return 10
    '\n    Format a failure traceback, assuming UTF-8 and using a replacement\n    strategy for errors.  Every effort is made to provide a usable\n    traceback, but should not that not be possible, a message and the\n    captured exception are logged.\n\n    @param failure: The failure to retrieve a traceback from.\n\n    @return: The formatted traceback.\n    '
    try:
        traceback = failure.getTraceback()
    except BaseException as e:
        traceback = '(UNABLE TO OBTAIN TRACEBACK FROM EVENT):' + str(e)
    return traceback

def _formatSystem(event: LogEvent) -> str:
    if False:
        print('Hello World!')
    '\n    Format the system specified in the event in the "log_system" key if set,\n    otherwise the C{"log_namespace"} and C{"log_level"}, joined by a C{"#"}.\n    Each defaults to C{"-"} is not set.  If formatting fails completely,\n    "UNFORMATTABLE" is returned.\n\n    @param event: The event containing the system specification.\n\n    @return: A formatted string representing the "log_system" key.\n    '
    system = cast(Optional[str], event.get('log_system', None))
    if system is None:
        level = cast(Optional[NamedConstant], event.get('log_level', None))
        if level is None:
            levelName = '-'
        else:
            levelName = level.name
        system = '{namespace}#{level}'.format(namespace=cast(str, event.get('log_namespace', '-')), level=levelName)
    else:
        try:
            system = str(system)
        except Exception:
            system = 'UNFORMATTABLE'
    return system

def eventAsText(event: LogEvent, includeTraceback: bool=True, includeTimestamp: bool=True, includeSystem: bool=True, formatTime: Callable[[float], str]=formatTime) -> str:
    if False:
        return 10
    '\n    Format an event as text.  Optionally, attach timestamp, traceback, and\n    system information.\n\n    The full output format is:\n    C{"{timeStamp} [{system}] {event}\\n{traceback}\\n"} where:\n\n        - C{timeStamp} is the event\'s C{"log_time"} value formatted with\n          the provided C{formatTime} callable.\n\n        - C{system} is the event\'s C{"log_system"} value, if set, otherwise,\n          the C{"log_namespace"} and C{"log_level"}, joined by a C{"#"}.  Each\n          defaults to C{"-"} is not set.\n\n        - C{event} is the event, as formatted by L{formatEvent}.\n\n        - C{traceback} is the traceback if the event contains a\n          C{"log_failure"} key.  In the event the original traceback cannot\n          be formatted, a message indicating the failure will be substituted.\n\n    If the event cannot be formatted, and no traceback exists, an empty string\n    is returned, even if includeSystem or includeTimestamp are true.\n\n    @param event: A logging event.\n    @param includeTraceback: If true and a C{"log_failure"} key exists, append\n        a traceback.\n    @param includeTimestamp: If true include a formatted timestamp before the\n        event.\n    @param includeSystem:  If true, include the event\'s C{"log_system"} value.\n    @param formatTime: A time formatter\n\n    @return: A formatted string with specified options.\n\n    @since: Twisted 18.9.0\n    '
    eventText = _formatEvent(event)
    if includeTraceback and 'log_failure' in event:
        f = event['log_failure']
        traceback = _formatTraceback(f)
        eventText = '\n'.join((eventText, traceback))
    if not eventText:
        return eventText
    timeStamp = ''
    if includeTimestamp:
        timeStamp = ''.join([formatTime(cast(float, event.get('log_time', None))), ' '])
    system = ''
    if includeSystem:
        system = ''.join(['[', _formatSystem(event), ']', ' '])
    return '{timeStamp}{system}{eventText}'.format(timeStamp=timeStamp, system=system, eventText=eventText)