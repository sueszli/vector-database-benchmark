import logging
try:
    import logger
    _found_logger = logger.Logger.sectionStart is not None
except (ImportError, AttributeError) as exc:
    _found_logger = False

def set_level(level):
    if False:
        i = 10
        return i + 15
    'Set a logging level for the pywinauto logger.'
    ActionLogger.set_level(level)

def reset_level():
    if False:
        return 10
    'Reset a logging level to a default'
    ActionLogger.reset_level()

def disable():
    if False:
        while True:
            i = 10
    'Disable pywinauto logging actions'
    ActionLogger.disable()

def enable():
    if False:
        for i in range(10):
            print('nop')
    'Enable pywinauto logging actions'
    reset_level()

class _CustomLogger(object):
    """
    Custom logger to use for pywinatuo logging actions.

    The usage of the class is optional and only if the standard
    logging facilities are not enough
    """

    def __init__(self, logFilePath=None):
        if False:
            while True:
                i = 10
        'Init the custom logger'
        self.logger = logger.Logger(logFilePath)

    @staticmethod
    def set_level(level):
        if False:
            while True:
                i = 10
        'Set a logging level'
        pass

    @staticmethod
    def reset_level():
        if False:
            while True:
                i = 10
        'Reset a logging level to a default'
        pass

    @staticmethod
    def disable():
        if False:
            while True:
                i = 10
        'Set a logging level to one above INFO to disable logs emitting'
        pass

    def log(self, *args):
        if False:
            i = 10
            return i + 15
        'Process a log message'
        for msg in args:
            self.logger.message(msg)

    def logSectionStart(self, msg):
        if False:
            while True:
                i = 10
        self.logger.sectionStart(msg)

    def logSectionEnd(self):
        if False:
            while True:
                i = 10
        self.logger.sectionEnd()

def _setup_standard_logger():
    if False:
        return 10
    'A helper to init the standard logger'
    logger = logging.getLogger(__package__)
    if logger.handlers:
        return logger
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
    ch.setFormatter(formatter)
    ch.setLevel(logging.DEBUG)
    logger.addHandler(ch)
    return logger

class _StandardLogger(object):
    """
    Wrapper around the standard python logger
    """
    logger = _setup_standard_logger()

    @staticmethod
    def set_level(level):
        if False:
            i = 10
            return i + 15
        'Set a logging level'
        _StandardLogger.logger.setLevel(level)

    @staticmethod
    def reset_level():
        if False:
            print('Hello World!')
        "Reset a logging level to a default one\n\n        We use logging.INFO because 'logger.info' is called in 'log' method.\n        Notice that setting up the level with logging.NOTSET results in delegating the filtering\n        to other active loggers so that if another logger had set a higher level than we need,\n        the messages for pywinauto logger will be dropped even if it was 'enabled'.\n        "
        _StandardLogger.logger.setLevel(logging.INFO)

    @staticmethod
    def disable():
        if False:
            return 10
        'Set a logging level to one above INFO to disable logs emitting'
        set_level(logging.WARNING)

    def __init__(self, logFilePath=None):
        if False:
            while True:
                i = 10
        'Init the wrapper'
        self.logFilePath = logFilePath
        self.logger = _StandardLogger.logger

    def log(self, *args):
        if False:
            print('Hello World!')
        'Process a log message'
        self.logger.info(*args)
        for handler in self.logger.handlers:
            if hasattr(handler, 'flush'):
                handler.flush()

    def logSectionStart(self, msg):
        if False:
            for i in range(10):
                print('nop')
        'Empty for now, just to conform with _CustomLogger'
        pass

    def logSectionEnd(self):
        if False:
            print('Hello World!')
        'Empty for now, just to conform with _CustomLogger'
        pass
if _found_logger:
    ActionLogger = _CustomLogger
else:
    ActionLogger = _StandardLogger
disable()