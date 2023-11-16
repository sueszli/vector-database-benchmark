import logging
import re
import sys
from lib.core.settings import IS_WIN
if IS_WIN:
    import ctypes
    import ctypes.wintypes
    ctypes.windll.kernel32.SetConsoleTextAttribute.argtypes = [ctypes.wintypes.HANDLE, ctypes.wintypes.WORD]
    ctypes.windll.kernel32.SetConsoleTextAttribute.restype = ctypes.wintypes.BOOL

def stdoutEncode(data):
    if False:
        print('Hello World!')
    return data

class ColorizingStreamHandler(logging.StreamHandler):
    color_map = {'black': 0, 'red': 1, 'green': 2, 'yellow': 3, 'blue': 4, 'magenta': 5, 'cyan': 6, 'white': 7}
    level_map = {logging.DEBUG: (None, 'blue', False), logging.INFO: (None, 'green', False), logging.WARNING: (None, 'yellow', False), logging.ERROR: (None, 'red', False), logging.CRITICAL: ('red', 'white', False)}
    csi = '\x1b['
    reset = '\x1b[0m'
    bold = '\x1b[1m'
    disable_coloring = False

    @property
    def is_tty(self):
        if False:
            while True:
                i = 10
        isatty = getattr(self.stream, 'isatty', None)
        return isatty and isatty() and (not self.disable_coloring)

    def emit(self, record):
        if False:
            print('Hello World!')
        try:
            message = stdoutEncode(self.format(record))
            stream = self.stream
            if not self.is_tty:
                if message and message[0] == '\r':
                    message = message[1:]
                stream.write(message)
            else:
                self.output_colorized(message)
            stream.write(getattr(self, 'terminator', '\n'))
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except IOError:
            pass
        except:
            self.handleError(record)
    if not IS_WIN:

        def output_colorized(self, message):
            if False:
                print('Hello World!')
            self.stream.write(message)
    else:
        ansi_esc = re.compile('\\x1b\\[((?:\\d+)(?:;(?:\\d+))*)m')
        nt_color_map = {0: 0, 1: 4, 2: 2, 3: 6, 4: 1, 5: 5, 6: 3, 7: 7}

        def output_colorized(self, message):
            if False:
                return 10
            parts = self.ansi_esc.split(message)
            h = None
            fd = getattr(self.stream, 'fileno', None)
            if fd is not None:
                fd = fd()
                if fd in (1, 2):
                    h = ctypes.windll.kernel32.GetStdHandle(-10 - fd)
            while parts:
                text = parts.pop(0)
                if text:
                    self.stream.write(text)
                    self.stream.flush()
                if parts:
                    params = parts.pop(0)
                    if h is not None:
                        params = [int(p) for p in params.split(';')]
                        color = 0
                        for p in params:
                            if 40 <= p <= 47:
                                color |= self.nt_color_map[p - 40] << 4
                            elif 30 <= p <= 37:
                                color |= self.nt_color_map[p - 30]
                            elif p == 1:
                                color |= 8
                            elif p == 0:
                                color = 7
                            else:
                                pass
                        ctypes.windll.kernel32.SetConsoleTextAttribute(h, color)

    def _reset(self, message):
        if False:
            print('Hello World!')
        if not message.endswith(self.reset):
            reset = self.reset
        elif self.bold in message:
            reset = self.reset + self.bold
        else:
            reset = self.reset
        return reset

    def colorize(self, message, levelno):
        if False:
            print('Hello World!')
        if levelno in self.level_map and self.is_tty:
            (bg, fg, bold) = self.level_map[levelno]
            params = []
            if bg in self.color_map:
                params.append(str(self.color_map[bg] + 40))
            if fg in self.color_map:
                params.append(str(self.color_map[fg] + 30))
            if bold:
                params.append('1')
            if params and message:
                if message.lstrip() != message:
                    prefix = re.search('\\s+', message).group(0)
                    message = message[len(prefix):]
                else:
                    prefix = ''
                message = '%s%s' % (prefix, ''.join((self.csi, ';'.join(params), 'm', message, self.reset)))
        return message

    def format(self, record):
        if False:
            while True:
                i = 10
        message = logging.StreamHandler.format(self, record)
        return self.colorize(message, record.levelno)