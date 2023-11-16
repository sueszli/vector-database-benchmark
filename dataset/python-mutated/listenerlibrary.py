import sys

class listenerlibrary:
    ROBOT_LISTENER_API_VERSION = 2
    ROBOT_LIBRARY_SCOPE = 'TEST CASE'

    def __init__(self):
        if False:
            print('Hello World!')
        self.ROBOT_LIBRARY_LISTENER = self
        self.events = []
        self.level = 'suite'

    def get_events(self):
        if False:
            for i in range(10):
                print('nop')
        return self.events[:]

    def _start_suite(self, name, attrs):
        if False:
            print('Hello World!')
        self.events.append('Start suite: %s' % name)

    def endSuite(self, name, attrs):
        if False:
            while True:
                i = 10
        self.events.append('End suite: %s' % name)

    def _start_test(self, name, attrs):
        if False:
            while True:
                i = 10
        self.events.append('Start test: %s' % name)
        self.level = 'test'

    def end_test(self, name, attrs):
        if False:
            i = 10
            return i + 15
        self.events.append('End test: %s' % name)

    def _startKeyword(self, name, attrs):
        if False:
            return 10
        self.events.append('Start kw: %s' % name)

    def _end_keyword(self, name, attrs):
        if False:
            print('Hello World!')
        self.events.append('End kw: %s' % name)

    def _close(self):
        if False:
            print('Hello World!')
        if self.ROBOT_LIBRARY_SCOPE == 'TEST CASE':
            level = ' (%s)' % self.level
        else:
            level = ''
        sys.__stderr__.write('CLOSING %s%s\n' % (self.ROBOT_LIBRARY_SCOPE, level))

    def events_should_be(self, *expected):
        if False:
            for i in range(10):
                print('nop')
        self._assert(self.events == list(expected), 'Expected events:\n%s\n\nActual events:\n%s' % (self._format(expected), self._format(self.events)))

    def events_should_be_empty(self):
        if False:
            for i in range(10):
                print('nop')
        self._assert(not self.events, 'Expected no events, got:\n%s' % self._format(self.events))

    def _assert(self, condition, message):
        if False:
            while True:
                i = 10
        assert condition, message

    def _format(self, events):
        if False:
            for i in range(10):
                print('nop')
        return '\n'.join(events)