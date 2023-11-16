"""Logging functions for Mininet."""
import logging
from logging import Logger
import types
OUTPUT = 25
LEVELS = {'debug': logging.DEBUG, 'info': logging.INFO, 'output': OUTPUT, 'warning': logging.WARNING, 'warn': logging.WARNING, 'error': logging.ERROR, 'critical': logging.CRITICAL}
LOGLEVELDEFAULT = OUTPUT
LOGMSGFORMAT = '%(message)s'

class StreamHandlerNoNewline(logging.StreamHandler):
    """StreamHandler that doesn't print newlines by default.
       Since StreamHandler automatically adds newlines, define a mod to more
       easily support interactive mode when we want it, or errors-only logging
       for running unit tests."""

    def emit(self, record):
        if False:
            i = 10
            return i + 15
        'Emit a record.\n           If a formatter is specified, it is used to format the record.\n           The record is then written to the stream with a trailing newline\n           [ N.B. this may be removed depending on feedback ]. If exception\n           information is present, it is formatted using\n           traceback.printException and appended to the stream.'
        try:
            msg = self.format(record)
            fs = '%s'
            if not hasattr(types, 'UnicodeType'):
                self.stream.write(fs % msg)
            else:
                try:
                    self.stream.write(fs % msg)
                except UnicodeError:
                    self.stream.write(fs % msg.encode('UTF-8'))
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)

class Singleton(type):
    """Singleton pattern from Wikipedia
       See http://en.wikipedia.org/wiki/Singleton_Pattern

       Intended to be used as a __metaclass_ param, as shown for the class
       below."""

    def __init__(cls, name, bases, dict_):
        if False:
            i = 10
            return i + 15
        super(Singleton, cls).__init__(name, bases, dict_)
        cls.instance = None

    def __call__(cls, *args, **kw):
        if False:
            for i in range(10):
                print('nop')
        if cls.instance is None:
            cls.instance = super(Singleton, cls).__call__(*args, **kw)
        return cls.instance

class MininetLogger(Logger, object):
    """Mininet-specific logger
       Enable each mininet .py file to with one import:

       from mininet.log import [lg, info, error]

       ...get a default logger that doesn't require one newline per logging
       call.

       Inherit from object to ensure that we have at least one new-style base
       class, and can then use the __metaclass__ directive, to prevent this
       error:

       TypeError: Error when calling the metaclass bases
       a new-style class can't have only classic bases

       If Python2.5/logging/__init__.py defined Filterer as a new-style class,
       via Filterer( object ): rather than Filterer, we wouldn't need this.

       Use singleton pattern to ensure only one logger is ever created."""
    __metaclass__ = Singleton

    def __init__(self, name='mininet'):
        if False:
            while True:
                i = 10
        Logger.__init__(self, name)
        ch = StreamHandlerNoNewline()
        formatter = logging.Formatter(LOGMSGFORMAT)
        ch.setFormatter(formatter)
        self.addHandler(ch)
        self.ch = ch
        self.setLogLevel()

    def setLogLevel(self, levelname=None):
        if False:
            for i in range(10):
                print('nop')
        'Setup loglevel.\n           Convenience function to support lowercase names.\n           levelName: level name from LEVELS'
        if levelname and levelname not in LEVELS:
            print(LEVELS)
            raise Exception('setLogLevel: unknown levelname %s' % levelname)
        level = LEVELS.get(levelname, LOGLEVELDEFAULT)
        self.setLevel(level)
        self.ch.setLevel(level)

    def output(self, msg, *args, **kwargs):
        if False:
            return 10
        'Log \'msg % args\' with severity \'OUTPUT\'.\n\n           To pass exception information, use the keyword argument exc_info\n           with a true value, e.g.\n\n           logger.warning("Houston, we have a %s", "cli output", exc_info=1)\n        '
        if getattr(self.manager, 'disabled', 0) >= OUTPUT:
            return
        if self.isEnabledFor(OUTPUT):
            self._log(OUTPUT, msg, args, kwargs)

def makeListCompatible(fn):
    if False:
        i = 10
        return i + 15
    "Return a new function allowing fn( 'a 1 b' ) to be called as\n       newfn( 'a', 1, 'b' )"

    def newfn(*args):
        if False:
            i = 10
            return i + 15
        'Generated function. Closure-ish.'
        if len(args) == 1:
            return fn(*args)
        args = ' '.join((str(arg) for arg in args))
        return fn(args)
    setattr(newfn, '__name__', fn.__name__)
    setattr(newfn, '__doc__', fn.__doc__)
    return newfn
logging.setLoggerClass(MininetLogger)
lg = logging.getLogger('mininet')
_loggers = (lg.info, lg.output, lg.warning, lg.error, lg.debug)
_loggers = tuple((makeListCompatible(logger) for logger in _loggers))
(lg.info, lg.output, lg.warning, lg.error, lg.debug) = _loggers
(info, output, warning, error, debug) = _loggers
warn = warning
setLogLevel = lg.setLogLevel