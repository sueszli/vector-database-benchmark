from contextlib import ExitStack
import os
import sys
from tempfile import NamedTemporaryFile
import unittest
from coalib.misc.Shell import run_interactive_shell_command, run_shell_command

class RunShellCommandTest(unittest.TestCase):

    @staticmethod
    def construct_testscript_command(scriptname):
        if False:
            i = 10
            return i + 15
        return (sys.executable, os.path.join(os.path.dirname(os.path.realpath(__file__)), 'run_shell_command_testfiles', scriptname))

    def test_run_interactive_shell_command(self):
        if False:
            while True:
                i = 10
        command = RunShellCommandTest.construct_testscript_command('test_interactive_program.py')
        with run_interactive_shell_command(command) as p:
            self.assertEqual(p.stdout.readline(), 'test_program X\n')
            self.assertEqual(p.stdout.readline(), 'Type in a number:\n')
            p.stdin.write('33\n')
            p.stdin.flush()
            self.assertEqual(p.stdout.readline(), '33\n')
            self.assertEqual(p.stdout.readline(), 'Exiting program.\n')
            self.assertEqual(p.stdout.read(), '')
            self.assertEqual(p.stderr.read(), '')

    def test_run_interactive_shell_command_custom_streams(self):
        if False:
            for i in range(10):
                print('nop')
        command = RunShellCommandTest.construct_testscript_command('test_interactive_program.py')
        with ExitStack() as stack:
            streams = {s: stack.enter_context(NamedTemporaryFile(mode='w+')) for s in ['stdout', 'stderr', 'stdin']}
            with run_interactive_shell_command(command, **streams) as p:
                streams['stdin'].write('712\n')
                streams['stdin'].flush()
                streams['stdin'].seek(0)
            self.assertFalse(streams['stdout'].closed)
            self.assertFalse(streams['stderr'].closed)
            self.assertFalse(streams['stdin'].closed)
            streams['stdout'].seek(0)
            self.assertEqual(streams['stdout'].read(), 'test_program X\nType in a number:\n712\nExiting program.\n')
            streams['stderr'].seek(0)
            self.assertEqual(streams['stderr'].read(), '')

    def test_run_interactive_shell_command_kwargs_delegation(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaises(TypeError):
            with run_interactive_shell_command('some_command', weird_parameter=30):
                pass

    def test_run_shell_command_without_stdin(self):
        if False:
            return 10
        command = RunShellCommandTest.construct_testscript_command('test_program.py')
        (stdout, stderr) = run_shell_command(command)
        expected = 'test_program Z\nnon-interactive mode.\nExiting...\n'
        self.assertEqual(stdout, expected)
        self.assertEqual(stderr, '')

    def test_run_shell_command_with_stdin(self):
        if False:
            i = 10
            return i + 15
        command = RunShellCommandTest.construct_testscript_command('test_input_program.py')
        (stdout, stderr) = run_shell_command(command, '1  4  10  22')
        self.assertEqual(stdout, '37\n')
        self.assertEqual(stderr, '')
        (stdout, stderr) = run_shell_command(command, '1 p 5')
        self.assertEqual(stdout, '')
        self.assertEqual(stderr, 'INVALID INPUT\n')

    def test_run_shell_command_kwargs_delegation(self):
        if False:
            return 10
        with self.assertRaises(TypeError):
            run_shell_command('super-cool-command', weird_parameter2='abc')