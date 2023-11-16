"""
Tests for running ``gevent.monkey`` as a module to launch a
patched script.

Uses files in the ``monkey_package/`` directory.
"""
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import os
import os.path
import sys
from gevent import testing as greentest
from gevent.testing.util import absolute_pythonpath
from gevent.testing.util import run

class TestRun(greentest.TestCase):
    maxDiff = None

    def setUp(self):
        if False:
            return 10
        self.abs_pythonpath = absolute_pythonpath()
        self.cwd = os.getcwd()
        os.chdir(os.path.dirname(__file__))

    def tearDown(self):
        if False:
            print('Hello World!')
        os.chdir(self.cwd)

    def _run(self, script, module=False):
        if False:
            while True:
                i = 10
        env = os.environ.copy()
        env['PYTHONWARNINGS'] = 'ignore'
        if self.abs_pythonpath:
            env['PYTHONPATH'] = self.abs_pythonpath
        run_kwargs = dict(buffer_output=True, quiet=True, nested=True, env=env, timeout=10)
        args = [sys.executable, '-m', 'gevent.monkey']
        if module:
            args.append('--module')
        args += [script, 'patched']
        monkey_result = run(args, **run_kwargs)
        self.assertTrue(monkey_result)
        if module:
            args = [sys.executable, '-m', script, 'stdlib']
        else:
            args = [sys.executable, script, 'stdlib']
        std_result = run(args, **run_kwargs)
        self.assertTrue(std_result)
        monkey_out_lines = monkey_result.output_lines
        std_out_lines = std_result.output_lines
        self.assertEqual(monkey_out_lines, std_out_lines)
        self.assertEqual(monkey_result.error, std_result.error)
        return monkey_out_lines

    def test_run_simple(self):
        if False:
            while True:
                i = 10
        self._run(os.path.join('monkey_package', 'script.py'))

    def _run_package(self, module):
        if False:
            i = 10
            return i + 15
        lines = self._run('monkey_package', module=module)
        self.assertTrue(lines[0].endswith(u'__main__.py'), lines[0])
        self.assertEqual(lines[1].strip(), u'__main__')

    def test_run_package(self):
        if False:
            while True:
                i = 10
        self._run_package(module=False)

    def test_run_module(self):
        if False:
            while True:
                i = 10
        self._run_package(module=True)

    def test_issue_302(self):
        if False:
            i = 10
            return i + 15
        monkey_lines = self._run(os.path.join('monkey_package', 'issue302monkey.py'))
        self.assertEqual(monkey_lines[0].strip(), u'True')
        monkey_lines[1] = monkey_lines[1].replace(u'\\', u'/')
        self.assertTrue(monkey_lines[1].strip().endswith(u'monkey_package/issue302monkey.py'))
        self.assertEqual(monkey_lines[2].strip(), u'True', monkey_lines)

    def test_threadpool_in_patched_after_patch(self):
        if False:
            while True:
                i = 10
        out = self._run(os.path.join('monkey_package', 'threadpool_monkey_patches.py'))
        self.assertEqual(out, ['False', '2'])

    def test_threadpool_in_patched_after_patch_module(self):
        if False:
            while True:
                i = 10
        out = self._run('monkey_package.threadpool_monkey_patches', module=True)
        self.assertEqual(out, ['False', '2'])

    def test_threadpool_not_patched_after_patch_module(self):
        if False:
            return 10
        out = self._run('monkey_package.threadpool_no_monkey', module=True)
        self.assertEqual(out, ['False', 'False', '2'])
if __name__ == '__main__':
    greentest.main()