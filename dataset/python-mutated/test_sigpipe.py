import os
from helpers import unittest
from luigi.format import InputPipeProcessWrapper
BASH_SCRIPT = '\n#!/bin/bash\n\ntrap "touch /tmp/luigi_sigpipe.marker; exit 141" SIGPIPE\n\n\nfor i in {1..3}\ndo\n    sleep 0.1\n    echo "Welcome $i times"\ndone\n'
FAIL_SCRIPT = BASH_SCRIPT + '\nexit 1\n'

class TestSigpipe(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        with open('/tmp/luigi_test_sigpipe.sh', 'w') as fp:
            fp.write(BASH_SCRIPT)

    def tearDown(self):
        if False:
            return 10
        os.remove('/tmp/luigi_test_sigpipe.sh')
        if os.path.exists('/tmp/luigi_sigpipe.marker'):
            os.remove('/tmp/luigi_sigpipe.marker')

    def test_partial_read(self):
        if False:
            while True:
                i = 10
        p1 = InputPipeProcessWrapper(['bash', '/tmp/luigi_test_sigpipe.sh'])
        self.assertEqual(p1.readline().decode('utf8'), 'Welcome 1 times\n')
        p1.close()
        self.assertTrue(os.path.exists('/tmp/luigi_sigpipe.marker'))

    def test_full_read(self):
        if False:
            i = 10
            return i + 15
        p1 = InputPipeProcessWrapper(['bash', '/tmp/luigi_test_sigpipe.sh'])
        counter = 1
        for line in p1:
            self.assertEqual(line.decode('utf8'), 'Welcome %i times\n' % counter)
            counter += 1
        p1.close()
        self.assertFalse(os.path.exists('/tmp/luigi_sigpipe.marker'))

class TestSubprocessException(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        with open('/tmp/luigi_test_sigpipe.sh', 'w') as fp:
            fp.write(FAIL_SCRIPT)

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        os.remove('/tmp/luigi_test_sigpipe.sh')
        if os.path.exists('/tmp/luigi_sigpipe.marker'):
            os.remove('/tmp/luigi_sigpipe.marker')

    def test_partial_read(self):
        if False:
            for i in range(10):
                print('nop')
        p1 = InputPipeProcessWrapper(['bash', '/tmp/luigi_test_sigpipe.sh'])
        self.assertEqual(p1.readline().decode('utf8'), 'Welcome 1 times\n')
        p1.close()
        self.assertTrue(os.path.exists('/tmp/luigi_sigpipe.marker'))

    def test_full_read(self):
        if False:
            print('Hello World!')

        def run():
            if False:
                for i in range(10):
                    print('nop')
            p1 = InputPipeProcessWrapper(['bash', '/tmp/luigi_test_sigpipe.sh'])
            counter = 1
            for line in p1:
                self.assertEqual(line.decode('utf8'), 'Welcome %i times\n' % counter)
                counter += 1
            p1.close()
        self.assertRaises(RuntimeError, run)