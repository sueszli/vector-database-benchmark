import abc
import re
from re import Pattern
from typing import Optional
from ..language_server import daemon_connection
from .daemon_query import DaemonQueryFailure

class AbstractDaemonQueryFailer(abc.ABC):

    @abc.abstractmethod
    def query_failure(self, path: str) -> Optional[DaemonQueryFailure]:
        if False:
            while True:
                i = 10
        'A result of None indicates that failure should not take place'
        raise NotImplementedError()

    def query_connection_failure(self, path: str) -> Optional[daemon_connection.DaemonConnectionFailure]:
        if False:
            print('Hello World!')
        'A result of None indicates that failure should not take place'
        raise NotImplementedError()

class DaemonQueryNoOpFailer(AbstractDaemonQueryFailer):

    def query_failure(self, path: str) -> Optional[DaemonQueryFailure]:
        if False:
            while True:
                i = 10
        return None

    def query_connection_failure(self, path: str) -> Optional[daemon_connection.DaemonConnectionFailure]:
        if False:
            while True:
                i = 10
        return None

class RegexDaemonQueryFailer(AbstractDaemonQueryFailer):
    """Fails daemon queries matching a specified regex pattern"""

    def __init__(self, reject_regex: str) -> None:
        if False:
            return 10
        self.reject_regex = reject_regex
        self.compiled_reject_regex: Pattern[str] = re.compile(reject_regex)

    def _matches_regex(self, path: str) -> Optional[str]:
        if False:
            return 10
        if self.compiled_reject_regex.match(path):
            return f'Not querying daemon for path: {path} as matches regex: {self.reject_regex}'

    def query_failure(self, path: str) -> Optional[DaemonQueryFailure]:
        if False:
            print('Hello World!')
        if (fail_message := self._matches_regex(path)) is not None:
            return DaemonQueryFailure(fail_message)

    def query_connection_failure(self, path: str) -> Optional[daemon_connection.DaemonConnectionFailure]:
        if False:
            i = 10
            return i + 15
        if (fail_message := self._matches_regex(path)) is not None:
            return daemon_connection.DaemonConnectionFailure(fail_message)