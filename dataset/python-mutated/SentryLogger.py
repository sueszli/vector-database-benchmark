from UM.Logger import LogOutput
from typing import Set
from cura.CrashHandler import CrashHandler
try:
    from sentry_sdk import add_breadcrumb
except ImportError:
    pass
from typing import Optional

class SentryLogger(LogOutput):
    _levels = {'w': 'warning', 'i': 'info', 'c': 'fatal', 'e': 'error', 'd': 'debug'}

    def __init__(self) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__()
        self._show_once = set()

    def log(self, log_type: str, message: str) -> None:
        if False:
            return 10
        'Log the message to the sentry hub as a breadcrumb\n\n        :param log_type: "e" (error), "i"(info), "d"(debug), "w"(warning) or "c"(critical) (can postfix with "_once")\n        :param message: String containing message to be logged\n        '
        level = self._translateLogType(log_type)
        message = CrashHandler.pruneSensitiveData(message)
        if level is None:
            if message not in self._show_once:
                level = self._translateLogType(log_type[0])
                if level is not None:
                    self._show_once.add(message)
                    add_breadcrumb(level=level, message=message)
        else:
            add_breadcrumb(level=level, message=message)

    @staticmethod
    def _translateLogType(log_type: str) -> Optional[str]:
        if False:
            print('Hello World!')
        return SentryLogger._levels.get(log_type)