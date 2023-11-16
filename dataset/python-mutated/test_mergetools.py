import os
import sys
import tempfile
from bzrlib import mergetools, tests

class TestFilenameSubstitution(tests.TestCaseInTempDir):

    def test_simple_filename(self):
        if False:
            print('Hello World!')
        cmd_list = ['kdiff3', '{base}', '{this}', '{other}', '-o', '{result}']
        (args, tmpfile) = mergetools._subst_filename(cmd_list, 'test.txt')
        self.assertEqual(['kdiff3', 'test.txt.BASE', 'test.txt.THIS', 'test.txt.OTHER', '-o', 'test.txt'], args)

    def test_spaces(self):
        if False:
            for i in range(10):
                print('nop')
        cmd_list = ['kdiff3', '{base}', '{this}', '{other}', '-o', '{result}']
        (args, tmpfile) = mergetools._subst_filename(cmd_list, 'file with space.txt')
        self.assertEqual(['kdiff3', 'file with space.txt.BASE', 'file with space.txt.THIS', 'file with space.txt.OTHER', '-o', 'file with space.txt'], args)

    def test_spaces_and_quotes(self):
        if False:
            print('Hello World!')
        cmd_list = ['kdiff3', '{base}', '{this}', '{other}', '-o', '{result}']
        (args, tmpfile) = mergetools._subst_filename(cmd_list, 'file with "space and quotes".txt')
        self.assertEqual(['kdiff3', 'file with "space and quotes".txt.BASE', 'file with "space and quotes".txt.THIS', 'file with "space and quotes".txt.OTHER', '-o', 'file with "space and quotes".txt'], args)

    def test_tempfile(self):
        if False:
            for i in range(10):
                print('nop')
        self.build_tree(('test.txt', 'test.txt.BASE', 'test.txt.THIS', 'test.txt.OTHER'))
        cmd_list = ['some_tool', '{this_temp}']
        (args, tmpfile) = mergetools._subst_filename(cmd_list, 'test.txt')
        self.assertPathExists(tmpfile)
        os.remove(tmpfile)

class TestCheckAvailability(tests.TestCaseInTempDir):

    def test_full_path(self):
        if False:
            while True:
                i = 10
        self.assertTrue(mergetools.check_availability(sys.executable))

    def test_exe_on_path(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertTrue(mergetools.check_availability('python'))

    def test_nonexistent(self):
        if False:
            while True:
                i = 10
        self.assertFalse(mergetools.check_availability('DOES NOT EXIST'))

    def test_non_executable(self):
        if False:
            i = 10
            return i + 15
        (f, name) = tempfile.mkstemp()
        try:
            self.log('temp filename: %s', name)
            self.assertFalse(mergetools.check_availability(name))
        finally:
            os.close(f)
            os.unlink(name)

class TestInvoke(tests.TestCaseInTempDir):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super(tests.TestCaseInTempDir, self).setUp()
        self._exe = None
        self._args = None
        self.build_tree_contents((('test.txt', 'stuff'), ('test.txt.BASE', 'base stuff'), ('test.txt.THIS', 'this stuff'), ('test.txt.OTHER', 'other stuff')))

    def test_invoke_expands_exe_path(self):
        if False:
            for i in range(10):
                print('nop')
        self.overrideEnv('PATH', os.path.dirname(sys.executable))

        def dummy_invoker(exe, args, cleanup):
            if False:
                i = 10
                return i + 15
            self._exe = exe
            self._args = args
            cleanup(0)
            return 0
        command = '%s {result}' % os.path.basename(sys.executable)
        retcode = mergetools.invoke(command, 'test.txt', dummy_invoker)
        self.assertEqual(0, retcode)
        self.assertEqual(sys.executable, self._exe)
        self.assertEqual(['test.txt'], self._args)

    def test_success(self):
        if False:
            print('Hello World!')

        def dummy_invoker(exe, args, cleanup):
            if False:
                return 10
            self._exe = exe
            self._args = args
            cleanup(0)
            return 0
        retcode = mergetools.invoke('tool {result}', 'test.txt', dummy_invoker)
        self.assertEqual(0, retcode)
        self.assertEqual('tool', self._exe)
        self.assertEqual(['test.txt'], self._args)

    def test_failure(self):
        if False:
            for i in range(10):
                print('nop')

        def dummy_invoker(exe, args, cleanup):
            if False:
                return 10
            self._exe = exe
            self._args = args
            cleanup(1)
            return 1
        retcode = mergetools.invoke('tool {result}', 'test.txt', dummy_invoker)
        self.assertEqual(1, retcode)
        self.assertEqual('tool', self._exe)
        self.assertEqual(['test.txt'], self._args)

    def test_success_tempfile(self):
        if False:
            i = 10
            return i + 15

        def dummy_invoker(exe, args, cleanup):
            if False:
                for i in range(10):
                    print('nop')
            self._exe = exe
            self._args = args
            self.assertPathExists(args[0])
            f = open(args[0], 'wt')
            f.write('temp stuff')
            f.close()
            cleanup(0)
            return 0
        retcode = mergetools.invoke('tool {this_temp}', 'test.txt', dummy_invoker)
        self.assertEqual(0, retcode)
        self.assertEqual('tool', self._exe)
        self.assertPathDoesNotExist(self._args[0])
        self.assertFileEqual('temp stuff', 'test.txt')

    def test_failure_tempfile(self):
        if False:
            i = 10
            return i + 15

        def dummy_invoker(exe, args, cleanup):
            if False:
                print('Hello World!')
            self._exe = exe
            self._args = args
            self.assertPathExists(args[0])
            self.log(repr(args))
            f = open(args[0], 'wt')
            self.log(repr(f))
            f.write('temp stuff')
            f.close()
            cleanup(1)
            return 1
        retcode = mergetools.invoke('tool {this_temp}', 'test.txt', dummy_invoker)
        self.assertEqual(1, retcode)
        self.assertEqual('tool', self._exe)
        self.assertFileEqual('stuff', 'test.txt')