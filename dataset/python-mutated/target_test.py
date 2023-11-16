from helpers import unittest, skipOnTravisAndGithubActions
from mock import Mock
import re
import random
import luigi.target
import luigi.format

class TestException(Exception):
    pass

class TargetTest(unittest.TestCase):

    def test_cannot_instantiate(self):
        if False:
            for i in range(10):
                print('nop')

        def instantiate_target():
            if False:
                for i in range(10):
                    print('nop')
            luigi.target.Target()
        self.assertRaises(TypeError, instantiate_target)

    def test_abstract_subclass(self):
        if False:
            i = 10
            return i + 15

        class ExistsLessTarget(luigi.target.Target):
            pass

        def instantiate_target():
            if False:
                print('Hello World!')
            ExistsLessTarget()
        self.assertRaises(TypeError, instantiate_target)

    def test_instantiate_subclass(self):
        if False:
            for i in range(10):
                print('nop')

        class GoodTarget(luigi.target.Target):

            def exists(self):
                if False:
                    print('Hello World!')
                return True

            def open(self, mode):
                if False:
                    return 10
                return None
        GoodTarget()

class FileSystemTargetTestMixin:
    """All Target that take bytes (python2: str) should pass those
    tests. In addition, a test to verify the method `exists`should be added
    """

    def create_target(self, format=None):
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError()

    def assertCleanUp(self, tmp_path=''):
        if False:
            for i in range(10):
                print('nop')
        pass

    def test_atomicity(self):
        if False:
            i = 10
            return i + 15
        target = self.create_target()
        fobj = target.open('w')
        self.assertFalse(target.exists())
        fobj.close()
        self.assertTrue(target.exists())

    def test_readback(self):
        if False:
            print('Hello World!')
        target = self.create_target()
        origdata = 'lol\n'
        fobj = target.open('w')
        fobj.write(origdata)
        fobj.close()
        fobj = target.open('r')
        data = fobj.read()
        self.assertEqual(origdata, data)

    def test_unicode_obj(self):
        if False:
            i = 10
            return i + 15
        target = self.create_target()
        origdata = u'lol\n'
        fobj = target.open('w')
        fobj.write(origdata)
        fobj.close()
        fobj = target.open('r')
        data = fobj.read()
        self.assertEqual(origdata, data)

    def test_with_close(self):
        if False:
            while True:
                i = 10
        target = self.create_target()
        with target.open('w') as fobj:
            tp = getattr(fobj, 'tmp_path', '')
            fobj.write('hej\n')
        self.assertCleanUp(tp)
        self.assertTrue(target.exists())

    def test_with_exception(self):
        if False:
            for i in range(10):
                print('nop')
        target = self.create_target()
        a = {}

        def foo():
            if False:
                i = 10
                return i + 15
            with target.open('w') as fobj:
                fobj.write('hej\n')
                a['tp'] = getattr(fobj, 'tmp_path', '')
                raise TestException('Test triggered exception')
        self.assertRaises(TestException, foo)
        self.assertCleanUp(a['tp'])
        self.assertFalse(target.exists())

    def test_del(self):
        if False:
            return 10
        t = self.create_target()
        p = t.open('w')
        print('test', file=p)
        tp = getattr(p, 'tmp_path', '')
        del p
        self.assertCleanUp(tp)
        self.assertFalse(t.exists())

    def test_write_cleanup_no_close(self):
        if False:
            while True:
                i = 10
        t = self.create_target()

        def context():
            if False:
                i = 10
                return i + 15
            f = t.open('w')
            f.write('stuff')
            return getattr(f, 'tmp_path', '')
        tp = context()
        import gc
        gc.collect()
        self.assertCleanUp(tp)
        self.assertFalse(t.exists())

    def test_text(self):
        if False:
            i = 10
            return i + 15
        t = self.create_target(luigi.format.UTF8)
        a = u'我éçф'
        with t.open('w') as f:
            f.write(a)
        with t.open('r') as f:
            b = f.read()
        self.assertEqual(a, b)

    def test_del_with_Text(self):
        if False:
            print('Hello World!')
        t = self.create_target(luigi.format.UTF8)
        p = t.open('w')
        print(u'test', file=p)
        tp = getattr(p, 'tmp_path', '')
        del p
        self.assertCleanUp(tp)
        self.assertFalse(t.exists())

    def test_format_injection(self):
        if False:
            return 10

        class CustomFormat(luigi.format.Format):

            def pipe_reader(self, input_pipe):
                if False:
                    while True:
                        i = 10
                input_pipe.foo = 'custom read property'
                return input_pipe

            def pipe_writer(self, output_pipe):
                if False:
                    i = 10
                    return i + 15
                output_pipe.foo = 'custom write property'
                return output_pipe
        t = self.create_target(CustomFormat())
        with t.open('w') as f:
            self.assertEqual(f.foo, 'custom write property')
        with t.open('r') as f:
            self.assertEqual(f.foo, 'custom read property')

    @skipOnTravisAndGithubActions('https://travis-ci.org/spotify/luigi/jobs/73693470')
    def test_binary_write(self):
        if False:
            for i in range(10):
                print('nop')
        t = self.create_target(luigi.format.Nop)
        with t.open('w') as f:
            f.write(b'a\xf2\xf3\r\nfd')
        with t.open('r') as f:
            c = f.read()
        self.assertEqual(c, b'a\xf2\xf3\r\nfd')

    def test_writelines(self):
        if False:
            while True:
                i = 10
        t = self.create_target()
        with t.open('w') as f:
            f.writelines(['a\n', 'b\n', 'c\n'])
        with t.open('r') as f:
            c = f.read()
        self.assertEqual(c, 'a\nb\nc\n')

    def test_read_iterator(self):
        if False:
            print('Hello World!')
        t = self.create_target()
        with t.open('w') as f:
            f.write('a\nb\nc\n')
        c = []
        with t.open('r') as f:
            for x in f:
                c.append(x)
        self.assertEqual(c, ['a\n', 'b\n', 'c\n'])

    def test_gzip(self):
        if False:
            print('Hello World!')
        t = self.create_target(luigi.format.Gzip)
        p = t.open('w')
        test_data = b'test'
        p.write(test_data)
        tp = getattr(p, 'tmp_path', '')
        self.assertFalse(t.exists())
        p.close()
        self.assertCleanUp(tp)
        self.assertTrue(t.exists())

    def test_gzip_works_and_cleans_up(self):
        if False:
            return 10
        t = self.create_target(luigi.format.Gzip)
        test_data = b'123testing'
        with t.open('w') as f:
            tp = getattr(f, 'tmp_path', '')
            f.write(test_data)
        self.assertCleanUp(tp)
        with t.open() as f:
            result = f.read()
        self.assertEqual(test_data, result)

    def test_move_on_fs(self):
        if False:
            i = 10
            return i + 15
        t = self.create_target()
        other_path = t.path + '-' + str(random.randint(0, 999999999))
        t._touchz()
        fs = t.fs
        self.assertTrue(t.exists())
        fs.move(t.path, other_path)
        self.assertFalse(t.exists())

    def test_rename_dont_move_on_fs(self):
        if False:
            for i in range(10):
                print('nop')
        t = self.create_target()
        other_path = t.path + '-' + str(random.randint(0, 999999999))
        t._touchz()
        fs = t.fs
        self.assertTrue(t.exists())
        fs.rename_dont_move(t.path, other_path)
        self.assertFalse(t.exists())
        self.assertRaises(luigi.target.FileAlreadyExists, lambda : fs.rename_dont_move(t.path, other_path))

class TemporaryPathTest(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        super(TemporaryPathTest, self).setUp()
        self.fs = Mock()

        class MyFileSystemTarget(luigi.target.FileSystemTarget):
            open = None
            fs = self.fs
        self.target_cls = MyFileSystemTarget

    def test_temporary_path_files(self):
        if False:
            while True:
                i = 10
        target_outer = self.target_cls('/tmp/notreal.xls')
        target_inner = self.target_cls('/tmp/blah.txt')

        class MyException(Exception):
            pass
        orig_ex = MyException()
        try:
            with target_outer.temporary_path() as tmp_path_outer:
                self.assertIn('notreal', tmp_path_outer)
                with target_inner.temporary_path() as tmp_path_inner:
                    self.assertIn('blah', tmp_path_inner)
                    with target_inner.temporary_path() as tmp_path_inner_2:
                        self.assertNotEqual(tmp_path_inner, tmp_path_inner_2)
                    self.fs.rename_dont_move.assert_called_once_with(tmp_path_inner_2, target_inner.path)
                self.fs.rename_dont_move.assert_called_with(tmp_path_inner, target_inner.path)
                self.assertEqual(self.fs.rename_dont_move.call_count, 2)
                raise orig_ex
        except MyException as ex:
            self.assertIs(ex, orig_ex)
        else:
            assert False
        self.assertEqual(self.fs.rename_dont_move.call_count, 2)

    def test_temporary_path_directory(self):
        if False:
            i = 10
            return i + 15
        target_slash = self.target_cls('/tmp/dir/')
        target_noslash = self.target_cls('/tmp/dir')
        with target_slash.temporary_path() as tmp_path:
            assert re.match('/tmp/dir-luigi-tmp-\\d{10}/', tmp_path)
        self.fs.rename_dont_move.assert_called_once_with(tmp_path, target_slash.path)
        with target_noslash.temporary_path() as tmp_path:
            assert re.match('/tmp/dir-luigi-tmp-\\d{10}', tmp_path)
        self.fs.rename_dont_move.assert_called_with(tmp_path, target_noslash.path)

    def test_windowsish_dir(self):
        if False:
            print('Hello World!')
        target = self.target_cls('C:\\my\\folder' + '\\')
        pattern = 'C:\\\\my\\\\folder-luigi-tmp-\\d{10}' + '\\\\'
        with target.temporary_path() as tmp_path:
            assert re.match(pattern, tmp_path)
        self.fs.rename_dont_move.assert_called_once_with(tmp_path, target.path)

    def test_hadoopish_dir(self):
        if False:
            for i in range(10):
                print('nop')
        target = self.target_cls('hdfs:///user/arash/myfile.uids')
        with target.temporary_path() as tmp_path:
            assert re.match('hdfs:///user/arash/myfile.uids-luigi-tmp-\\d{10}', tmp_path)
        self.fs.rename_dont_move.assert_called_once_with(tmp_path, target.path)

    def test_creates_dir_for_file(self):
        if False:
            print('Hello World!')
        target = self.target_cls('/my/file/is/awesome.txt')
        with target.temporary_path():
            self.fs.mkdir.assert_called_once_with('/my/file/is', parents=True, raise_if_exists=False)

    def test_creates_dir_for_dir(self):
        if False:
            i = 10
            return i + 15
        target = self.target_cls('/my/dir/is/awesome/')
        with target.temporary_path():
            self.fs.mkdir.assert_called_once_with('/my/dir/is', parents=True, raise_if_exists=False)

    def test_file_in_current_dir(self):
        if False:
            for i in range(10):
                print('nop')
        target = self.target_cls('foo.txt')
        with target.temporary_path() as tmp_path:
            self.fs.mkdir.assert_not_called()
        self.fs.rename_dont_move.assert_called_once_with(tmp_path, target.path)