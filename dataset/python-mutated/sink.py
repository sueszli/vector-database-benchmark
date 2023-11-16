import collections
import logging

class Sink(object):

    def info(self, message, exc=None):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError

    def warning(self, message, exc=None):
        if False:
            return 10
        raise NotImplementedError

    def error(self, message, exc=None):
        if False:
            print('Hello World!')
        raise NotImplementedError

class NullSink(Sink):

    def info(self, message, exc=None):
        if False:
            for i in range(10):
                print('nop')
        pass

    def warning(self, message, exc=None):
        if False:
            print('Hello World!')
        pass

    def error(self, message, exc=None):
        if False:
            print('Hello World!')
        pass

class LoggingSink(Sink):
    _DEFAULT_LOGGER = logging.getLogger(__name__)

    def __init__(self, name=None, logger=None):
        if False:
            print('Hello World!')
        self.logger = logger
        if self.logger is None and name is not None:
            self.logger = logging.getLogger(name)
        if self.logger is None:
            self.logger = self._DEFAULT_LOGGER

    def info(self, message, exc=None):
        if False:
            for i in range(10):
                print('nop')
        self.logger.info(message)

    def warning(self, message, exc=None):
        if False:
            for i in range(10):
                print('nop')
        self.logger.warning(message)

    def error(self, message, exc=None):
        if False:
            for i in range(10):
                print('nop')
        self.logger.error(message)
Message = collections.namedtuple('Message', ['type', 'content', 'exc'])

class CollectingSink(Sink):

    def __init__(self):
        if False:
            return 10
        self.messages = list()

    def info(self, message, exc=None):
        if False:
            i = 10
            return i + 15
        self.messages.append(Message('info', message, exc))

    def warning(self, message, exc=None):
        if False:
            while True:
                i = 10
        self.messages.append(Message('warning', message, exc))

    def error(self, message, exc=None):
        if False:
            while True:
                i = 10
        self.messages.append(Message('error', message, exc))

    @property
    def infos(self):
        if False:
            return 10
        return [m for m in self.messages if m.type == 'info']

    @property
    def warnings(self):
        if False:
            print('Hello World!')
        return [m for m in self.messages if m.type == 'warning']

    @property
    def errors(self):
        if False:
            for i in range(10):
                print('nop')
        return [m for m in self.messages if m.type == 'error']