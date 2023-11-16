import sys

class Listener:
    ROBOT_LISTENER_API_VERSION = 2

    def __init__(self, name='X'):
        if False:
            i = 10
            return i + 15
        self.name = name

    def start_suite(self, name, attrs):
        if False:
            for i in range(10):
                print('nop')
        self._log('from listener {0}'.format(self.name))

    def close(self):
        if False:
            return 10
        self._log('listener close')

    def report_file(self, path):
        if False:
            i = 10
            return i + 15
        self._log('report {0}'.format(path))

    def log_file(self, path):
        if False:
            for i in range(10):
                print('nop')
        self._log('log {0}'.format(path))

    def output_file(self, path):
        if False:
            while True:
                i = 10
        self._log('output {0}'.format(path))

    def _log(self, message):
        if False:
            for i in range(10):
                print('nop')
        sys.__stdout__.write('[{0}]\n'.format(message))