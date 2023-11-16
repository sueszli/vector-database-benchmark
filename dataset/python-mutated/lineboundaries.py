import re
from twisted.internet import defer
from twisted.logger import Logger
from buildbot.warnings import warn_deprecated
log = Logger()

class LineBoundaryFinder:
    __slots__ = ['partialLine', 'callback', 'warned']
    MAX_LINELENGTH = 4096
    newline_re = re.compile('(\\r\\n|\\r(?=.)|\\033\\[u|\\033\\[[0-9]+;[0-9]+[Hf]|\\033\\[2J|\\x08+)')

    def __init__(self, callback=None):
        if False:
            while True:
                i = 10
        if callback is not None:
            warn_deprecated('3.6.0', f'{self.__class__.__name__} does not accept callback anymore')
        self.partialLine = None
        self.callback = callback
        self.warned = False

    def adjust_line(self, text):
        if False:
            i = 10
            return i + 15
        if self.partialLine:
            if len(self.partialLine) > self.MAX_LINELENGTH:
                if not self.warned:
                    log.warn('Splitting long line: {line_start} {length} (not warning anymore for this log)', line_start=self.partialLine[:30], length=len(self.partialLine))
                    self.warned = True
                (self.partialLine, text) = (text, self.partialLine)
                ret = []
                while len(text) > self.MAX_LINELENGTH:
                    ret.append(text[:self.MAX_LINELENGTH])
                    text = text[self.MAX_LINELENGTH:]
                ret.append(text)
                result = '\n'.join(ret) + '\n'
                return result
            text = self.partialLine + text
            self.partialLine = None
        text = self.newline_re.sub('\n', text)
        if text:
            if text[-1] != '\n':
                i = text.rfind('\n')
                if i >= 0:
                    i = i + 1
                    (text, self.partialLine) = (text[:i], text[i:])
                else:
                    self.partialLine = text
                    return None
            return text
        return None

    def append(self, text):
        if False:
            while True:
                i = 10
        lines = self.adjust_line(text)
        if self.callback is None:
            return lines
        if lines is None:
            return defer.succeed(None)
        return self.callback(lines)

    def flush(self):
        if False:
            i = 10
            return i + 15
        if self.partialLine is not None:
            return self.append('\n')
        if self.callback is not None:
            return defer.succeed(None)
        return None