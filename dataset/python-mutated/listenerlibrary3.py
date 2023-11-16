import sys

class listenerlibrary3:
    ROBOT_LISTENER_API_VERSION = 3
    ROBOT_LIBRARY_SCOPE = 'TEST CASE'

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.ROBOT_LIBRARY_LISTENER = self

    def start_suite(self, data, result):
        if False:
            while True:
                i = 10
        result.doc = (result.doc + ' [start suite]').strip()
        result.metadata['suite'] = '[start]'
        result.metadata['tests'] = ''
        assert len(data.tests) == 2
        assert len(result.tests) == 0
        data.tests.create(name='New')

    def end_suite(self, data, result):
        if False:
            print('Hello World!')
        assert len(data.tests) == 3
        assert len(result.tests) == 3
        assert result.doc.endswith('[start suite]')
        assert result.metadata['suite'] == '[start]'
        result.name += ' [end suite]'
        result.doc += ' [end suite]'
        result.metadata['suite'] += ' [end]'

    def start_test(self, data, result):
        if False:
            return 10
        result.doc = (result.doc + ' [start test]').strip()
        result.tags.add('[start]')
        result.message = 'Message: [start]'
        result.parent.metadata['tests'] += 'x'
        data.body.create_keyword('No Operation')

    def end_test(self, data, result):
        if False:
            while True:
                i = 10
        result.doc += ' [end test]'
        result.tags.add('[end]')
        result.passed = not result.passed
        result.message += ' [end]'

    def log_message(self, msg):
        if False:
            return 10
        msg.message += ' [log_message]'
        msg.timestamp = '2015-12-16 15:51:20.141'

    def foo(self):
        if False:
            while True:
                i = 10
        print('*WARN* Foo')

    def message(self, msg):
        if False:
            while True:
                i = 10
        msg.message += ' [message]'
        msg.timestamp = '2015-12-16 15:51:20.141'

    def close(self):
        if False:
            print('Hello World!')
        sys.__stderr__.write('CLOSING Listener library 3\n')