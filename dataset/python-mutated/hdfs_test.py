import re
import random
import pickle
import luigi
import luigi.format
from luigi.contrib import hdfs
import luigi.contrib.hdfs.clients
from target_test import FileSystemTargetTestMixin

class ComplexOldFormat(luigi.format.Format):
    """Should take unicode but output bytes
    """

    def hdfs_writer(self, output_pipe):
        if False:
            return 10
        return self.pipe_writer(luigi.contrib.hdfs.Plain.hdfs_writer(output_pipe))

    def pipe_writer(self, output_pipe):
        if False:
            print('Hello World!')
        return luigi.format.UTF8.pipe_writer(output_pipe)

    def pipe_reader(self, output_pipe):
        if False:
            return 10
        return output_pipe

class TestException(Exception):
    pass

class HdfsTargetTestMixin(FileSystemTargetTestMixin):

    def create_target(self, format=None):
        if False:
            i = 10
            return i + 15
        target = hdfs.HdfsTarget(self._test_file(), format=format)
        if target.exists():
            target.remove(skip_trash=True)
        return target

    def test_slow_exists(self):
        if False:
            while True:
                i = 10
        target = hdfs.HdfsTarget(self._test_file())
        try:
            target.remove(skip_trash=True)
        except BaseException:
            pass
        self.assertFalse(self.fs.exists(target.path))
        target.open('w').close()
        self.assertTrue(self.fs.exists(target.path))

        def should_raise():
            if False:
                while True:
                    i = 10
            self.fs.exists('hdfs://doesnotexist/foo')
        self.assertRaises(hdfs.HDFSCliError, should_raise)

        def should_raise_2():
            if False:
                while True:
                    i = 10
            self.fs.exists('hdfs://_doesnotexist_/foo')
        self.assertRaises(hdfs.HDFSCliError, should_raise_2)

    def test_create_ancestors(self):
        if False:
            print('Hello World!')
        parent = self._test_dir()
        target = hdfs.HdfsTarget('%s/foo/bar/baz' % parent)
        if self.fs.exists(parent):
            self.fs.remove(parent, skip_trash=True)
        self.assertFalse(self.fs.exists(parent))
        fobj = target.open('w')
        fobj.write('lol\n')
        fobj.close()
        self.assertTrue(self.fs.exists(parent))
        self.assertTrue(target.exists())

    def test_tmp_cleanup(self):
        if False:
            i = 10
            return i + 15
        path = self._test_file()
        target = hdfs.HdfsTarget(path, is_tmp=True)
        if target.exists():
            target.remove(skip_trash=True)
        with target.open('w') as fobj:
            fobj.write('lol\n')
        self.assertTrue(target.exists())
        del target
        import gc
        gc.collect()
        self.assertFalse(self.fs.exists(path))

    def test_luigi_tmp(self):
        if False:
            for i in range(10):
                print('nop')
        target = hdfs.HdfsTarget(is_tmp=True)
        self.assertFalse(target.exists())
        with target.open('w'):
            pass
        self.assertTrue(target.exists())

    def test_tmp_move(self):
        if False:
            for i in range(10):
                print('nop')
        target = hdfs.HdfsTarget(is_tmp=True)
        target2 = hdfs.HdfsTarget(self._test_file())
        if target2.exists():
            target2.remove(skip_trash=True)
        with target.open('w'):
            pass
        self.assertTrue(target.exists())
        target.move(target2.path)
        self.assertFalse(target.exists())
        self.assertTrue(target2.exists())

    def test_rename_no_parent(self):
        if False:
            return 10
        parent = self._test_dir() + '/foo'
        if self.fs.exists(parent):
            self.fs.remove(parent, skip_trash=True)
        target1 = hdfs.HdfsTarget(is_tmp=True)
        target2 = hdfs.HdfsTarget(parent + '/bar')
        with target1.open('w'):
            pass
        self.assertTrue(target1.exists())
        target1.move(target2.path)
        self.assertFalse(target1.exists())
        self.assertTrue(target2.exists())

    def test_rename_no_grandparent(self):
        if False:
            print('Hello World!')
        grandparent = self._test_dir() + '/foo'
        if self.fs.exists(grandparent):
            self.fs.remove(grandparent, skip_trash=True)
        target1 = hdfs.HdfsTarget(is_tmp=True)
        target2 = hdfs.HdfsTarget(grandparent + '/bar/baz')
        with target1.open('w'):
            pass
        self.assertTrue(target1.exists())
        target1.move(target2.path)
        self.assertFalse(target1.exists())
        self.assertTrue(target2.exists())

    def test_glob_exists(self):
        if False:
            print('Hello World!')
        target_dir = hdfs.HdfsTarget(self._test_dir())
        if target_dir.exists():
            target_dir.remove(skip_trash=True)
        self.fs.mkdir(target_dir.path)
        t1 = hdfs.HdfsTarget(target_dir.path + '/part-00001')
        t2 = hdfs.HdfsTarget(target_dir.path + '/part-00002')
        t3 = hdfs.HdfsTarget(target_dir.path + '/another')
        with t1.open('w') as f:
            f.write('foo\n')
        with t2.open('w') as f:
            f.write('bar\n')
        with t3.open('w') as f:
            f.write('biz\n')
        files = hdfs.HdfsTarget('%s/part-0000*' % target_dir.path)
        self.assertTrue(files.glob_exists(2))
        self.assertFalse(files.glob_exists(3))
        self.assertFalse(files.glob_exists(1))

    def assertRegexpMatches(self, text, expected_regexp, msg=None):
        if False:
            while True:
                i = 10
        'Python 2.7 backport.'
        if isinstance(expected_regexp, str):
            expected_regexp = re.compile(expected_regexp)
        if not expected_regexp.search(text):
            msg = msg or "Regexp didn't match"
            msg = '%s: %r not found in %r' % (msg, expected_regexp.pattern, text)
            raise self.failureException(msg)

    def test_tmppath_not_configured(self):
        if False:
            i = 10
            return i + 15
        path1 = '/dir1/dir2/file'
        path2 = 'hdfs:///dir1/dir2/file'
        path3 = 'hdfs://somehost/dir1/dir2/file'
        path4 = 'file:///dir1/dir2/file'
        path5 = '/tmp/dir/file'
        path6 = 'file:///tmp/dir/file'
        path7 = 'hdfs://somehost/tmp/dir/file'
        path8 = None
        path9 = '/tmpdir/file'
        res1 = hdfs.tmppath(path1, include_unix_username=False)
        res2 = hdfs.tmppath(path2, include_unix_username=False)
        res3 = hdfs.tmppath(path3, include_unix_username=False)
        res4 = hdfs.tmppath(path4, include_unix_username=False)
        res5 = hdfs.tmppath(path5, include_unix_username=False)
        res6 = hdfs.tmppath(path6, include_unix_username=False)
        res7 = hdfs.tmppath(path7, include_unix_username=False)
        res8 = hdfs.tmppath(path8, include_unix_username=False)
        res9 = hdfs.tmppath(path9, include_unix_username=False)
        self.assertRegexpMatches(res1, '^/tmp/dir1/dir2/file-luigitemp-\\d+')
        self.assertRegexpMatches(res2, '^hdfs:/tmp/dir1/dir2/file-luigitemp-\\d+')
        self.assertRegexpMatches(res3, '^hdfs://somehost/tmp/dir1/dir2/file-luigitemp-\\d+')
        self.assertRegexpMatches(res4, '^file:///tmp/dir1/dir2/file-luigitemp-\\d+')
        self.assertRegexpMatches(res5, '^/tmp/dir/file-luigitemp-\\d+')
        self.assertRegexpMatches(res6, '^file:///tmp/tmp/dir/file-luigitemp-\\d+')
        self.assertRegexpMatches(res7, '^hdfs://somehost/tmp/tmp/dir/file-luigitemp-\\d+')
        self.assertRegexpMatches(res8, '^/tmp/luigitemp-\\d+')
        self.assertRegexpMatches(res9, '/tmp/tmpdir/file')

    def test_tmppath_username(self):
        if False:
            print('Hello World!')
        self.assertRegexpMatches(hdfs.tmppath('/path/to/stuff', include_unix_username=True), '^/tmp/[a-z0-9_]+/path/to/stuff-luigitemp-\\d+')

    def test_pickle(self):
        if False:
            for i in range(10):
                print('nop')
        t = hdfs.HdfsTarget('/tmp/dir')
        pickle.dumps(t)

    def test_flag_target(self):
        if False:
            print('Hello World!')
        target = hdfs.HdfsFlagTarget('/some/dir/', format=format)
        if target.exists():
            target.remove(skip_trash=True)
        self.assertFalse(target.exists())
        t1 = hdfs.HdfsTarget(target.path + 'part-00000', format=format)
        with t1.open('w'):
            pass
        t2 = hdfs.HdfsTarget(target.path + '_SUCCESS', format=format)
        with t2.open('w'):
            pass
        self.assertTrue(target.exists())

    def test_flag_target_fails_if_not_directory(self):
        if False:
            print('Hello World!')
        with self.assertRaises(ValueError):
            hdfs.HdfsFlagTarget('/home/file.txt')

class _MiscOperationsMixin:

    def get_target(self):
        if False:
            while True:
                i = 10
        fn = '/tmp/foo-%09d' % random.randint(0, 999999999)
        t = luigi.contrib.hdfs.HdfsTarget(fn)
        with t.open('w') as f:
            f.write('test')
        return t

    def test_count(self):
        if False:
            print('Hello World!')
        t = self.get_target()
        res = self.get_client().count(t.path)
        for key in ['content_size', 'dir_count', 'file_count']:
            self.assertTrue(key in res)

    def test_chmod(self):
        if False:
            print('Hello World!')
        t = self.get_target()
        self.get_client().chmod(t.path, '777')

    def test_chown(self):
        if False:
            print('Hello World!')
        t = self.get_target()
        self.get_client().chown(t.path, 'root', 'root')