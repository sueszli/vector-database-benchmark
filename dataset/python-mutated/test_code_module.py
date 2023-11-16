"""Test InteractiveConsole and InteractiveInterpreter from code module"""
import sys
import unittest
from textwrap import dedent
from contextlib import ExitStack
from unittest import mock
from test.support import import_helper
code = import_helper.import_module('code')

class TestInteractiveConsole(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.console = code.InteractiveConsole()
        self.mock_sys()

    def mock_sys(self):
        if False:
            for i in range(10):
                print('nop')
        'Mock system environment for InteractiveConsole'
        stack = ExitStack()
        self.addCleanup(stack.close)
        self.infunc = stack.enter_context(mock.patch('code.input', create=True))
        self.stdout = stack.enter_context(mock.patch('code.sys.stdout'))
        self.stderr = stack.enter_context(mock.patch('code.sys.stderr'))
        prepatch = mock.patch('code.sys', wraps=code.sys, spec=code.sys)
        self.sysmod = stack.enter_context(prepatch)
        if sys.excepthook is sys.__excepthook__:
            self.sysmod.excepthook = self.sysmod.__excepthook__
        del self.sysmod.ps1
        del self.sysmod.ps2

    def test_ps1(self):
        if False:
            while True:
                i = 10
        self.infunc.side_effect = EOFError('Finished')
        self.console.interact()
        self.assertEqual(self.sysmod.ps1, '>>> ')
        self.sysmod.ps1 = 'custom1> '
        self.console.interact()
        self.assertEqual(self.sysmod.ps1, 'custom1> ')

    def test_ps2(self):
        if False:
            while True:
                i = 10
        self.infunc.side_effect = EOFError('Finished')
        self.console.interact()
        self.assertEqual(self.sysmod.ps2, '... ')
        self.sysmod.ps1 = 'custom2> '
        self.console.interact()
        self.assertEqual(self.sysmod.ps1, 'custom2> ')

    def test_console_stderr(self):
        if False:
            while True:
                i = 10
        self.infunc.side_effect = ["'antioch'", '', EOFError('Finished')]
        self.console.interact()
        for call in list(self.stdout.method_calls):
            if 'antioch' in ''.join(call[1]):
                break
        else:
            raise AssertionError('no console stdout')

    def test_syntax_error(self):
        if False:
            for i in range(10):
                print('nop')
        self.infunc.side_effect = ['undefined', EOFError('Finished')]
        self.console.interact()
        for call in self.stderr.method_calls:
            if 'NameError' in ''.join(call[1]):
                break
        else:
            raise AssertionError('No syntax error from console')

    def test_sysexcepthook(self):
        if False:
            print('Hello World!')
        self.infunc.side_effect = ["raise ValueError('')", EOFError('Finished')]
        hook = mock.Mock()
        self.sysmod.excepthook = hook
        self.console.interact()
        self.assertTrue(hook.called)

    def test_banner(self):
        if False:
            return 10
        self.infunc.side_effect = EOFError('Finished')
        self.console.interact(banner='Foo')
        self.assertEqual(len(self.stderr.method_calls), 3)
        banner_call = self.stderr.method_calls[0]
        self.assertEqual(banner_call, ['write', ('Foo\n',), {}])
        self.stderr.reset_mock()
        self.infunc.side_effect = EOFError('Finished')
        self.console.interact(banner='')
        self.assertEqual(len(self.stderr.method_calls), 2)

    def test_exit_msg(self):
        if False:
            print('Hello World!')
        self.infunc.side_effect = EOFError('Finished')
        self.console.interact(banner='')
        self.assertEqual(len(self.stderr.method_calls), 2)
        err_msg = self.stderr.method_calls[1]
        expected = 'now exiting InteractiveConsole...\n'
        self.assertEqual(err_msg, ['write', (expected,), {}])
        self.stderr.reset_mock()
        self.infunc.side_effect = EOFError('Finished')
        self.console.interact(banner='', exitmsg='')
        self.assertEqual(len(self.stderr.method_calls), 1)
        self.stderr.reset_mock()
        message = 'bye! ζж'
        self.infunc.side_effect = EOFError('Finished')
        self.console.interact(banner='', exitmsg=message)
        self.assertEqual(len(self.stderr.method_calls), 2)
        err_msg = self.stderr.method_calls[1]
        expected = message + '\n'
        self.assertEqual(err_msg, ['write', (expected,), {}])

    def test_cause_tb(self):
        if False:
            print('Hello World!')
        self.infunc.side_effect = ["raise ValueError('') from AttributeError", EOFError('Finished')]
        self.console.interact()
        output = ''.join((''.join(call[1]) for call in self.stderr.method_calls))
        expected = dedent('\n        AttributeError\n\n        The above exception was the direct cause of the following exception:\n\n        Traceback (most recent call last):\n          File "<console>", line 1, in <module>\n        ValueError\n        ')
        self.assertIn(expected, output)

    def test_context_tb(self):
        if False:
            while True:
                i = 10
        self.infunc.side_effect = ['try: ham\nexcept: eggs\n', EOFError('Finished')]
        self.console.interact()
        output = ''.join((''.join(call[1]) for call in self.stderr.method_calls))
        expected = dedent('\n        Traceback (most recent call last):\n          File "<console>", line 1, in <module>\n        NameError: name \'ham\' is not defined\n\n        During handling of the above exception, another exception occurred:\n\n        Traceback (most recent call last):\n          File "<console>", line 2, in <module>\n        NameError: name \'eggs\' is not defined\n        ')
        self.assertIn(expected, output)
if __name__ == '__main__':
    unittest.main()