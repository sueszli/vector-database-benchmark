from __future__ import absolute_import, print_function
from logging.handlers import BufferingHandler
import logging
import functools
import re

class RecordFilter(object):
    """Implement logging record filtering as per the configuration
    --logging-filter option.
    """

    def __init__(self, names):
        if False:
            i = 10
            return i + 15
        self.include = set()
        self.exclude = set()
        for name in names.split(','):
            if name[0] == '-':
                self.exclude.add(name[1:])
            else:
                self.include.add(name)

    def filter(self, record):
        if False:
            while True:
                i = 10
        if self.exclude:
            return record.name not in self.exclude
        return record.name in self.include

class LoggingCapture(BufferingHandler):
    """Capture logging events in a memory buffer for later display or query.

    Captured logging events are stored on the attribute
    :attr:`~LoggingCapture.buffer`:

    .. attribute:: buffer

       This is a list of captured logging events as `logging.LogRecords`_.

    .. _`logging.LogRecords`:
       http://docs.python.org/library/logging.html#logrecord-objects

    By default the format of the messages will be::

        '%(levelname)s:%(name)s:%(message)s'

    This may be overridden using standard logging formatter names in the
    configuration variable ``logging_format``.

    The level of logging captured is set to ``logging.NOTSET`` by default. You
    may override this using the configuration setting ``logging_level`` (which
    is set to a level name.)

    Finally there may be `filtering of logging events`__ specified by the
    configuration variable ``logging_filter``.

    .. __: behave.html#command-line-arguments
    """

    def __init__(self, config, level=None):
        if False:
            while True:
                i = 10
        BufferingHandler.__init__(self, 1000)
        self.config = config
        self.old_handlers = []
        self.old_level = None
        log_format = datefmt = None
        if config.logging_format:
            log_format = config.logging_format
        else:
            log_format = '%(levelname)s:%(name)s:%(message)s'
        if config.logging_datefmt:
            datefmt = config.logging_datefmt
        formatter = logging.Formatter(log_format, datefmt)
        self.setFormatter(formatter)
        if level is not None:
            self.level = level
        elif config.logging_level:
            self.level = config.logging_level
        else:
            self.level = logging.NOTSET
        if config.logging_filter:
            self.addFilter(RecordFilter(config.logging_filter))

    def __bool__(self):
        if False:
            for i in range(10):
                print('nop')
        return bool(self.buffer)

    def flush(self):
        if False:
            return 10
        pass

    def truncate(self):
        if False:
            for i in range(10):
                print('nop')
        self.buffer = []

    def getvalue(self):
        if False:
            return 10
        return '\n'.join((self.formatter.format(r) for r in self.buffer))

    def find_event(self, pattern):
        if False:
            for i in range(10):
                print('nop')
        'Search through the buffer for a message that matches the given\n        regular expression.\n\n        Returns boolean indicating whether a match was found.\n        '
        pattern = re.compile(pattern)
        for record in self.buffer:
            if pattern.search(record.getMessage()) is not None:
                return True
        return False

    def any_errors(self):
        if False:
            print('Hello World!')
        'Search through the buffer for any ERROR or CRITICAL events.\n\n        Returns boolean indicating whether a match was found.\n        '
        return any((record for record in self.buffer if record.levelname in ('ERROR', 'CRITICAL')))

    def inveigle(self):
        if False:
            return 10
        'Turn on logging capture by replacing all existing handlers\n        configured in the logging module.\n\n        If the config var logging_clear_handlers is set then we also remove\n        all existing handlers.\n\n        We also set the level of the root logger.\n\n        The opposite of this is :meth:`~LoggingCapture.abandon`.\n        '
        root_logger = logging.getLogger()
        if self.config.logging_clear_handlers:
            for logger in logging.Logger.manager.loggerDict.values():
                if hasattr(logger, 'handlers'):
                    for handler in logger.handlers:
                        self.old_handlers.append((logger, handler))
                        logger.removeHandler(handler)
        for handler in root_logger.handlers[:]:
            if isinstance(handler, LoggingCapture):
                root_logger.handlers.remove(handler)
            elif self.config.logging_clear_handlers:
                self.old_handlers.append((root_logger, handler))
                root_logger.removeHandler(handler)
        root_logger.addHandler(self)
        self.old_level = root_logger.level
        root_logger.setLevel(self.level)

    def abandon(self):
        if False:
            for i in range(10):
                print('nop')
        'Turn off logging capture.\n\n        If other handlers were removed by :meth:`~LoggingCapture.inveigle` then\n        they are reinstated.\n        '
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            if handler is self:
                root_logger.handlers.remove(handler)
        if self.config.logging_clear_handlers:
            for (logger, handler) in self.old_handlers:
                logger.addHandler(handler)
        if self.old_level is not None:
            root_logger.setLevel(self.old_level)
            self.old_level = None
MemoryHandler = LoggingCapture

def capture(*args, **kw):
    if False:
        while True:
            i = 10
    'Decorator to wrap an *environment file function* in log file capture.\n\n    It configures the logging capture using the *behave* context,\n    the first argument to the function being decorated\n    (so don\'t use this to decorate something that\n    doesn\'t have *context* as the first argument).\n\n    The basic usage is:\n\n    .. code-block: python\n\n        @capture\n        def after_scenario(context, scenario):\n            ...\n\n    The function prints any captured logging\n    (at the level determined by the ``log_level`` configuration setting)\n    directly to stdout, regardless of error conditions.\n\n    It is mostly useful for debugging in situations where you are seeing a\n    message like::\n\n        No handlers could be found for logger "name"\n\n    The decorator takes an optional "level" keyword argument which limits the\n    level of logging captured, overriding the level in the run\'s configuration:\n\n    .. code-block: python\n\n        @capture(level=logging.ERROR)\n        def after_scenario(context, scenario):\n            ...\n\n    This would limit the logging captured to just ERROR and above,\n    and thus only display logged events if they are interesting.\n    '

    def create_decorator(func, level=None):
        if False:
            print('Hello World!')

        def f(context, *args):
            if False:
                while True:
                    i = 10
            h = LoggingCapture(context.config, level=level)
            h.inveigle()
            try:
                func(context, *args)
            finally:
                h.abandon()
            v = h.getvalue()
            if v:
                print('Captured Logging:')
                print(v)
        return f
    if not args:
        return functools.partial(create_decorator, level=kw.get('level'))
    else:
        return create_decorator(args[0])