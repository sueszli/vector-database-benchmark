import dis
import os.path
import re
import subprocess
import sys
import types
import unittest
from test.support import findfile

def abspath(filename):
    if False:
        while True:
            i = 10
    return os.path.abspath(findfile(filename, subdir='dtracedata'))

def normalize_trace_output(output):
    if False:
        for i in range(10):
            print('nop')
    'Normalize DTrace output for comparison.\n\n    DTrace keeps a per-CPU buffer, and when showing the fired probes, buffers\n    are concatenated. So if the operating system moves our thread around, the\n    straight result can be "non-causal". So we add timestamps to the probe\n    firing, sort by that field, then strip it from the output'
    output = re.sub('\\[[0-9]+ refs\\]', '', output)
    try:
        result = [row.split('\t') for row in output.splitlines() if row and (not row.startswith('#'))]
        result.sort(key=lambda row: int(row[0]))
        result = [row[1] for row in result]
        return '\n'.join(result)
    except (IndexError, ValueError):
        raise AssertionError('tracer produced unparsable output:\n{}'.format(output))

class TraceBackend:
    EXTENSION = None
    COMMAND = None
    COMMAND_ARGS = []

    def run_case(self, name, optimize_python=None):
        if False:
            i = 10
            return i + 15
        actual_output = normalize_trace_output(self.trace_python(script_file=abspath(name + self.EXTENSION), python_file=abspath(name + '.py'), optimize_python=optimize_python))
        with open(abspath(name + self.EXTENSION + '.expected')) as f:
            expected_output = f.read().rstrip()
        return (expected_output, actual_output)

    def generate_trace_command(self, script_file, subcommand=None):
        if False:
            while True:
                i = 10
        command = self.COMMAND + [script_file]
        if subcommand:
            command += ['-c', subcommand]
        return command

    def trace(self, script_file, subcommand=None):
        if False:
            print('Hello World!')
        command = self.generate_trace_command(script_file, subcommand)
        (stdout, _) = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True).communicate()
        return stdout

    def trace_python(self, script_file, python_file, optimize_python=None):
        if False:
            i = 10
            return i + 15
        python_flags = []
        if optimize_python:
            python_flags.extend(['-O'] * optimize_python)
        subcommand = ' '.join([sys.executable] + python_flags + [python_file])
        return self.trace(script_file, subcommand)

    def assert_usable(self):
        if False:
            return 10
        try:
            output = self.trace(abspath('assert_usable' + self.EXTENSION))
            output = output.strip()
        except (FileNotFoundError, NotADirectoryError, PermissionError) as fnfe:
            output = str(fnfe)
        if output != 'probe: success':
            raise unittest.SkipTest('{}(1) failed: {}'.format(self.COMMAND[0], output))

class DTraceBackend(TraceBackend):
    EXTENSION = '.d'
    COMMAND = ['dtrace', '-q', '-s']

class SystemTapBackend(TraceBackend):
    EXTENSION = '.stp'
    COMMAND = ['stap', '-g']

class TraceTests:
    maxDiff = None
    backend = None
    optimize_python = 0

    @classmethod
    def setUpClass(self):
        if False:
            for i in range(10):
                print('nop')
        self.backend.assert_usable()

    def run_case(self, name):
        if False:
            for i in range(10):
                print('nop')
        (actual_output, expected_output) = self.backend.run_case(name, optimize_python=self.optimize_python)
        self.assertEqual(actual_output, expected_output)

    def test_function_entry_return(self):
        if False:
            for i in range(10):
                print('nop')
        self.run_case('call_stack')

    def test_verify_call_opcodes(self):
        if False:
            while True:
                i = 10
        'Ensure our call stack test hits all function call opcodes'
        opcodes = set(['CALL_FUNCTION', 'CALL_FUNCTION_EX', 'CALL_FUNCTION_KW'])
        with open(abspath('call_stack.py')) as f:
            code_string = f.read()

        def get_function_instructions(funcname):
            if False:
                print('Hello World!')
            code = compile(source=code_string, filename='<string>', mode='exec', optimize=self.optimize_python)
            for c in code.co_consts:
                if isinstance(c, types.CodeType) and c.co_name == funcname:
                    return dis.get_instructions(c)
            return []
        for instruction in get_function_instructions('start'):
            opcodes.discard(instruction.opname)
        self.assertEqual(set(), opcodes)

    def test_gc(self):
        if False:
            while True:
                i = 10
        self.run_case('gc')

    def test_line(self):
        if False:
            for i in range(10):
                print('nop')
        self.run_case('line')

class DTraceNormalTests(TraceTests, unittest.TestCase):
    backend = DTraceBackend()
    optimize_python = 0

class DTraceOptimizedTests(TraceTests, unittest.TestCase):
    backend = DTraceBackend()
    optimize_python = 2

class SystemTapNormalTests(TraceTests, unittest.TestCase):
    backend = SystemTapBackend()
    optimize_python = 0

class SystemTapOptimizedTests(TraceTests, unittest.TestCase):
    backend = SystemTapBackend()
    optimize_python = 2
if __name__ == '__main__':
    unittest.main()