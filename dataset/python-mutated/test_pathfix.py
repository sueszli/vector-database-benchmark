import os
import subprocess
import sys
import unittest
from test import support
from test.support import os_helper
from test.test_tools import scriptsdir, skip_if_missing
skip_if_missing()

class TestPathfixFunctional(unittest.TestCase):
    script = os.path.join(scriptsdir, 'pathfix.py')

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.addCleanup(os_helper.unlink, os_helper.TESTFN)

    def pathfix(self, shebang, pathfix_flags, exitcode=0, stdout='', stderr='', directory=''):
        if False:
            i = 10
            return i + 15
        if directory:
            filename = os.path.join(directory, 'script-A_1.py')
            pathfix_arg = directory
        else:
            filename = os_helper.TESTFN
            pathfix_arg = filename
        with open(filename, 'w', encoding='utf8') as f:
            f.write(f'{shebang}\n' + 'print("Hello world")\n')
        encoding = sys.getfilesystemencoding()
        proc = subprocess.run([sys.executable, self.script, *pathfix_flags, '-n', pathfix_arg], env={**os.environ, 'PYTHONIOENCODING': encoding}, capture_output=True)
        if stdout == '' and proc.returncode == 0:
            stdout = f'{filename}: updating\n'
        self.assertEqual(proc.returncode, exitcode, proc)
        self.assertEqual(proc.stdout.decode(encoding), stdout.replace('\n', os.linesep), proc)
        self.assertEqual(proc.stderr.decode(encoding), stderr.replace('\n', os.linesep), proc)
        with open(filename, 'r', encoding='utf8') as f:
            output = f.read()
        lines = output.split('\n')
        self.assertEqual(lines[1:], ['print("Hello world")', ''])
        new_shebang = lines[0]
        if proc.returncode != 0:
            self.assertEqual(shebang, new_shebang)
        return new_shebang

    def test_recursive(self):
        if False:
            return 10
        tmpdir = os_helper.TESTFN + '.d'
        self.addCleanup(os_helper.rmtree, tmpdir)
        os.mkdir(tmpdir)
        expected_stderr = f"recursedown('{os.path.basename(tmpdir)}')\n"
        self.assertEqual(self.pathfix('#! /usr/bin/env python', ['-i', '/usr/bin/python3'], directory=tmpdir, stderr=expected_stderr), '#! /usr/bin/python3')

    def test_pathfix(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.pathfix('#! /usr/bin/env python', ['-i', '/usr/bin/python3']), '#! /usr/bin/python3')
        self.assertEqual(self.pathfix('#! /usr/bin/env python -R', ['-i', '/usr/bin/python3']), '#! /usr/bin/python3')

    def test_pathfix_keeping_flags(self):
        if False:
            return 10
        self.assertEqual(self.pathfix('#! /usr/bin/env python -R', ['-i', '/usr/bin/python3', '-k']), '#! /usr/bin/python3 -R')
        self.assertEqual(self.pathfix('#! /usr/bin/env python', ['-i', '/usr/bin/python3', '-k']), '#! /usr/bin/python3')

    def test_pathfix_adding_flag(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.pathfix('#! /usr/bin/env python', ['-i', '/usr/bin/python3', '-a', 's']), '#! /usr/bin/python3 -s')
        self.assertEqual(self.pathfix('#! /usr/bin/env python -S', ['-i', '/usr/bin/python3', '-a', 's']), '#! /usr/bin/python3 -s')
        self.assertEqual(self.pathfix('#! /usr/bin/env python -V', ['-i', '/usr/bin/python3', '-a', 'v', '-k']), '#! /usr/bin/python3 -vV')
        self.assertEqual(self.pathfix('#! /usr/bin/env python', ['-i', '/usr/bin/python3', '-a', 'Rs']), '#! /usr/bin/python3 -Rs')
        self.assertEqual(self.pathfix('#! /usr/bin/env python -W default', ['-i', '/usr/bin/python3', '-a', 's', '-k']), '#! /usr/bin/python3 -sW default')

    def test_pathfix_adding_errors(self):
        if False:
            while True:
                i = 10
        self.pathfix('#! /usr/bin/env python -E', ['-i', '/usr/bin/python3', '-a', 'W default', '-k'], exitcode=2, stderr="-a option doesn't support whitespaces")
if __name__ == '__main__':
    unittest.main()