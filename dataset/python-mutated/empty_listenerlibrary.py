from robot.api.deco import library
import sys

class listener:
    ROBOT_LISTENER_API_VERSION = 2

    def start_test(self, name, attrs):
        if False:
            print('Hello World!')
        self._stderr('START TEST')

    def end_test(self, name, attrs):
        if False:
            while True:
                i = 10
        self._stderr('END TEST')

    def log_message(self, msg):
        if False:
            return 10
        self._stderr('MESSAGE %s' % msg['message'])

    def close(self):
        if False:
            i = 10
            return i + 15
        self._stderr('CLOSE')

    def _stderr(self, msg):
        if False:
            i = 10
            return i + 15
        sys.__stderr__.write('%s\n' % msg)

@library(scope='TEST CASE', listener=listener())
class empty_listenerlibrary:
    pass