"""
Contains all mappers (formatters) for CloudWatch logs
"""
import json
import logging
from json import JSONDecodeError
from typing import Any
from samcli.lib.observability.cw_logs.cw_log_event import CWLogEvent
from samcli.lib.observability.observability_info_puller import ObservabilityEventMapper
from samcli.lib.utils.colors import Colored
from samcli.lib.utils.time import timestamp_to_iso
LOG = logging.getLogger(__name__)

class CWKeywordHighlighterFormatter(ObservabilityEventMapper[CWLogEvent]):
    """
    Mapper implementation which will highlight given keywords in CloudWatch logs
    """

    def __init__(self, colored: Colored, keyword=None):
        if False:
            i = 10
            return i + 15
        '\n        Parameters\n        ----------\n        colored : Colored\n            Colored class that will be used to highlight the keywords in log event\n        keyword : str\n            Keyword that will be highlighted\n        '
        self._keyword = keyword
        self._colored = colored

    def map(self, event: CWLogEvent) -> CWLogEvent:
        if False:
            while True:
                i = 10
        if self._keyword:
            highlight = self._colored.underline(self._keyword)
            event.message = event.message.replace(self._keyword, highlight)
        return event

class CWColorizeErrorsFormatter(ObservabilityEventMapper[CWLogEvent]):
    """
    Mapper implementation which will colorize some pre-defined error messages
    """
    NODEJS_CRASH_MESSAGE = 'Process exited before completing request'
    TIMEOUT_MSG = 'Task timed out'

    def __init__(self, colored: Colored):
        if False:
            for i in range(10):
                print('nop')
        self._colored = colored

    def map(self, event: CWLogEvent) -> CWLogEvent:
        if False:
            return 10
        if CWColorizeErrorsFormatter.NODEJS_CRASH_MESSAGE in event.message or CWColorizeErrorsFormatter.TIMEOUT_MSG in event.message:
            event.message = self._colored.red(event.message)
        return event

class CWJsonFormatter(ObservabilityEventMapper[CWLogEvent]):
    """
    Mapper implementation which will auto indent the input if the input is a JSON object
    """

    def map(self, event: CWLogEvent) -> CWLogEvent:
        if False:
            i = 10
            return i + 15
        try:
            if event.message.startswith('{'):
                msg_dict = json.loads(event.message)
                event.message = json.dumps(msg_dict, indent=2)
        except JSONDecodeError as err:
            LOG.debug("Can't decode string (%s) as JSON. Error (%s)", event.message, err)
        return event

class CWPrettyPrintFormatter(ObservabilityEventMapper[CWLogEvent]):
    """
    Mapper implementation which will format given CloudWatch log event into string with coloring
    log stream name and timestamp
    """

    def __init__(self, colored: Colored):
        if False:
            print('Hello World!')
        self._colored = colored

    def map(self, event: CWLogEvent) -> CWLogEvent:
        if False:
            for i in range(10):
                print('nop')
        timestamp = self._colored.yellow(timestamp_to_iso(int(event.timestamp)))
        log_stream_name = self._colored.cyan(event.log_stream_name)
        event.message = f'{log_stream_name} {timestamp} {event.message}'
        return event

class CWAddNewLineIfItDoesntExist(ObservabilityEventMapper):
    """
    Mapper implementation which will add new lines at the end of events if it is not already there
    """

    def map(self, event: Any) -> Any:
        if False:
            i = 10
            return i + 15
        if isinstance(event, CWLogEvent) and (not event.message.endswith('\n')):
            event.message = f'{event.message}\n'
            return event
        if isinstance(event, str) and (not event.endswith('\n')):
            return f'{event}\n'
        return event

class CWLogEventJSONMapper(ObservabilityEventMapper[CWLogEvent]):
    """
    Converts given CWLogEvent into JSON string
    """

    def map(self, event: CWLogEvent) -> CWLogEvent:
        if False:
            i = 10
            return i + 15
        event.message = json.dumps(event.event)
        return event