import unittest
from robot.reporting.logreportwriters import LogWriter
from robot.utils.asserts import assert_true, assert_equal

class LogWriterWithMockedWriting(LogWriter):

    def __init__(self, model):
        if False:
            for i in range(10):
                print('nop')
        LogWriter.__init__(self, model)
        self.split_write_calls = []
        self.write_called = False

    def _write_split_log(self, index, keywords, strings, path):
        if False:
            print('Hello World!')
        self.split_write_calls.append((index, keywords, strings, path))

    def _write_file(self, output, config, template):
        if False:
            i = 10
            return i + 15
        self.write_called = True

class TestLogWriter(unittest.TestCase):

    def test_splitting_log(self):
        if False:
            for i in range(10):
                print('nop')

        class model:
            split_results = [((0, 1, 2, -1), ('*', '*1', '*2')), ((0, 1, 0, 42), ('*', '*x')), (((1, 2), (3, 4, ())), ('*',))]
        writer = LogWriterWithMockedWriting(model)
        writer.write('mylog.html', None)
        assert_true(writer.write_called)
        assert_equal([(1, (0, 1, 2, -1), ('*', '*1', '*2'), 'mylog-1.js'), (2, (0, 1, 0, 42), ('*', '*x'), 'mylog-2.js'), (3, ((1, 2), (3, 4, ())), ('*',), 'mylog-3.js')], writer.split_write_calls)
if __name__ == '__main__':
    unittest.main()