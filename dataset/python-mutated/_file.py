"""
File log observer.
"""
from typing import IO, Any, Callable, Optional
from zope.interface import implementer
from twisted.python.compat import ioType
from ._format import formatEventAsClassicLogText, formatTime, timeFormatRFC3339
from ._interfaces import ILogObserver, LogEvent

@implementer(ILogObserver)
class FileLogObserver:
    """
    Log observer that writes to a file-like object.
    """

    def __init__(self, outFile: IO[Any], formatEvent: Callable[[LogEvent], Optional[str]]) -> None:
        if False:
            i = 10
            return i + 15
        '\n        @param outFile: A file-like object.  Ideally one should be passed which\n            accepts text data.  Otherwise, UTF-8 L{bytes} will be used.\n        @param formatEvent: A callable that formats an event.\n        '
        if ioType(outFile) is not str:
            self._encoding: Optional[str] = 'utf-8'
        else:
            self._encoding = None
        self._outFile = outFile
        self.formatEvent = formatEvent

    def __call__(self, event: LogEvent) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Write event to file.\n\n        @param event: An event.\n        '
        text = self.formatEvent(event)
        if text:
            if self._encoding is None:
                self._outFile.write(text)
            else:
                self._outFile.write(text.encode(self._encoding))
            self._outFile.flush()

def textFileLogObserver(outFile: IO[Any], timeFormat: Optional[str]=timeFormatRFC3339) -> FileLogObserver:
    if False:
        return 10
    '\n    Create a L{FileLogObserver} that emits text to a specified (writable)\n    file-like object.\n\n    @param outFile: A file-like object.  Ideally one should be passed which\n        accepts text data.  Otherwise, UTF-8 L{bytes} will be used.\n    @param timeFormat: The format to use when adding timestamp prefixes to\n        logged events.  If L{None}, or for events with no C{"log_timestamp"}\n        key, the default timestamp prefix of C{"-"} is used.\n\n    @return: A file log observer.\n    '

    def formatEvent(event: LogEvent) -> Optional[str]:
        if False:
            print('Hello World!')
        return formatEventAsClassicLogText(event, formatTime=lambda e: formatTime(e, timeFormat))
    return FileLogObserver(outFile, formatEvent)