import traceback
import logging
from coalib.output.printers.LOG_LEVEL import LOG_LEVEL
from coalib.processes.communication.LogMessage import LogMessage

class LogPrinterMixin:
    """
    Provides access to the logging interfaces (e.g. err, warn, info) by routing
    them to the log_message method, which should be implemented by descendants
    of this class.
    """

    def debug(self, *messages, delimiter=' ', timestamp=None, **kwargs):
        if False:
            print('Hello World!')
        self.log_message(LogMessage(LOG_LEVEL.DEBUG, *messages, delimiter=delimiter, timestamp=timestamp), **kwargs)

    def info(self, *messages, delimiter=' ', timestamp=None, **kwargs):
        if False:
            while True:
                i = 10
        self.log_message(LogMessage(LOG_LEVEL.INFO, *messages, delimiter=delimiter, timestamp=timestamp), **kwargs)

    def warn(self, *messages, delimiter=' ', timestamp=None, **kwargs):
        if False:
            return 10
        self.log_message(LogMessage(LOG_LEVEL.WARNING, *messages, delimiter=delimiter, timestamp=timestamp), **kwargs)

    def err(self, *messages, delimiter=' ', timestamp=None, **kwargs):
        if False:
            while True:
                i = 10
        self.log_message(LogMessage(LOG_LEVEL.ERROR, *messages, delimiter=delimiter, timestamp=timestamp), **kwargs)

    def log(self, log_level, message, timestamp=None, **kwargs):
        if False:
            i = 10
            return i + 15
        self.log_message(LogMessage(log_level, message, timestamp=timestamp), **kwargs)

    def log_exception(self, message, exception, log_level=LOG_LEVEL.ERROR, timestamp=None, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        If the log_level of the printer is greater than DEBUG, it prints\n        only the message. If it is DEBUG or lower, it shows the message\n        along with the traceback of the exception.\n\n        :param message:   The message to print.\n        :param exception: The exception to print.\n        :param log_level: The log_level of this message (not used when\n                          logging the traceback. Tracebacks always have\n                          a level of DEBUG).\n        :param timestamp: The time at which this log occurred. Defaults to\n                          the current time.\n        :param kwargs:    Keyword arguments to be passed when logging the\n                          message (not used when logging the traceback).\n        '
        if not isinstance(exception, BaseException):
            raise TypeError('log_exception can only log derivatives of BaseException.')
        traceback_str = '\n'.join(traceback.format_exception(type(exception), exception, exception.__traceback__))
        self.log(log_level, message, timestamp=timestamp, **kwargs)
        self.log_message(LogMessage(LOG_LEVEL.INFO, 'Exception was:' + '\n' + traceback_str, timestamp=timestamp), **kwargs)

    def log_message(self, log_message, **kwargs):
        if False:
            while True:
                i = 10
        "\n        It is your responsibility to implement this method, if you're using this\n        mixin.\n        "
        raise NotImplementedError

class LogPrinter(LogPrinterMixin):
    """
    This class is deprecated and will be soon removed. To get logger use
    logging.getLogger(__name__). Make sure that you're getting it when the
    logging configuration is loaded.

    The LogPrinter class allows to print log messages to an underlying Printer.

    This class is an adapter, means you can create a LogPrinter from every
    existing Printer instance.
    """

    def __init__(self, printer=None, log_level=LOG_LEVEL.DEBUG, timestamp_format='%X'):
        if False:
            i = 10
            return i + 15
        '\n        Creates a new log printer from an existing Printer.\n\n        :param printer:          The underlying Printer where log messages\n                                 shall be written to. If you inherit from\n                                 LogPrinter, set it to self.\n        :param log_level:        The minimum log level, everything below will\n                                 not be logged.\n        :param timestamp_format: The format string for the\n                                 datetime.today().strftime(format) method.\n        '
        self.logger = logging.getLogger()
        self._printer = printer
        self.log_level = log_level
        self.timestamp_format = timestamp_format

    @property
    def log_level(self):
        if False:
            while True:
                i = 10
        '\n        Returns current log_level used in logger.\n        '
        return self.logger.getEffectiveLevel()

    @log_level.setter
    def log_level(self, log_level):
        if False:
            i = 10
            return i + 15
        '\n        Sets log_level for logger.\n        '
        self.logger.setLevel(log_level)

    @property
    def printer(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns the underlying printer where logs are printed to.\n        '
        return self._printer

    def log_message(self, log_message, **kwargs):
        if False:
            while True:
                i = 10
        if not isinstance(log_message, LogMessage):
            raise TypeError('log_message should be of type LogMessage.')
        self.logger.log(log_message.log_level, log_message.message)

    def __getstate__(self):
        if False:
            i = 10
            return i + 15
        oldict = self.__dict__.copy()
        del oldict['logger']
        return oldict

    def __setstate__(self, newdict):
        if False:
            i = 10
            return i + 15
        self.__dict__.update(newdict)
        self.logger = logging.getLogger()