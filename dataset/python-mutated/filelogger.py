from robot.utils import file_writer
from .loggerhelper import AbstractLogger
from .loggerapi import LoggerApi

class FileLogger(AbstractLogger, LoggerApi):

    def __init__(self, path, level):
        if False:
            i = 10
            return i + 15
        super().__init__(level)
        self._writer = self._get_writer(path)

    def _get_writer(self, path):
        if False:
            while True:
                i = 10
        return file_writer(path, usage='syslog')

    def message(self, msg):
        if False:
            while True:
                i = 10
        if self._is_logged(msg.level) and (not self._writer.closed):
            entry = '%s | %s | %s\n' % (msg.timestamp, msg.level.ljust(5), msg.message)
            self._writer.write(entry)

    def start_suite(self, data, result):
        if False:
            for i in range(10):
                print('nop')
        self.info("Started suite '%s'." % result.name)

    def end_suite(self, data, result):
        if False:
            for i in range(10):
                print('nop')
        self.info("Ended suite '%s'." % result.name)

    def start_test(self, data, result):
        if False:
            for i in range(10):
                print('nop')
        self.info("Started test '%s'." % result.name)

    def end_test(self, data, result):
        if False:
            while True:
                i = 10
        self.info("Ended test '%s'." % result.name)

    def start_body_item(self, data, result):
        if False:
            for i in range(10):
                print('nop')
        self.debug(lambda : "Started keyword '%s'." % result.name if result.type in result.KEYWORD_TYPES else result._log_name)

    def end_body_item(self, data, result):
        if False:
            for i in range(10):
                print('nop')
        self.debug(lambda : "Ended keyword '%s'." % result.name if result.type in result.KEYWORD_TYPES else result._log_name)

    def output_file(self, name, path):
        if False:
            return 10
        self.info('%s: %s' % (name, path))

    def close(self):
        if False:
            print('Hello World!')
        self._writer.close()