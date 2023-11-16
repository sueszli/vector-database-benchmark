import luigi
import mock
import random
import unittest
from luigi.contrib.opener import OpenerTarget, NoOpenerError
from luigi.mock import MockTarget
from luigi.local_target import LocalTarget
import pytest

@pytest.mark.contrib
class TestOpenerTarget(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        MockTarget.fs.clear()
        self.local_file = '/tmp/{}/xyz/test.txt'.format(random.randint(0, 999999999))
        if LocalTarget.fs.exists(self.local_file):
            LocalTarget.fs.remove(self.local_file)

    def tearDown(self):
        if False:
            while True:
                i = 10
        if LocalTarget.fs.exists(self.local_file):
            LocalTarget.fs.remove(self.local_file)

    def test_invalid_target(self):
        if False:
            for i in range(10):
                print('nop')
        'Verify invalid types raises NoOpenerError\n\n        '
        self.assertRaises(NoOpenerError, OpenerTarget, 'foo://bar.txt')

    def test_mock_target(self):
        if False:
            for i in range(10):
                print('nop')
        'Verify mock target url\n\n        '
        target = OpenerTarget('mock://foo/bar.txt')
        self.assertEqual(type(target), MockTarget)
        target.open('w').close()
        self.assertTrue(MockTarget.fs.exists('foo/bar.txt'))

    def test_mock_target_root(self):
        if False:
            i = 10
            return i + 15
        'Verify mock target url\n\n        '
        target = OpenerTarget('mock:///foo/bar.txt')
        self.assertEqual(type(target), MockTarget)
        target.open('w').close()
        self.assertTrue(MockTarget.fs.exists('/foo/bar.txt'))

    def test_default_target(self):
        if False:
            return 10
        'Verify default local target url\n\n        '
        target = OpenerTarget(self.local_file)
        self.assertEqual(type(target), LocalTarget)
        target.open('w').close()
        self.assertTrue(LocalTarget.fs.exists(self.local_file))

    def test_local_target(self):
        if False:
            i = 10
            return i + 15
        'Verify basic local target url\n\n        '
        local_file = 'file://{}'.format(self.local_file)
        target = OpenerTarget(local_file)
        self.assertEqual(type(target), LocalTarget)
        target.open('w').close()
        self.assertTrue(LocalTarget.fs.exists(self.local_file))

    @mock.patch('luigi.local_target.LocalTarget.__init__')
    @mock.patch('luigi.local_target.LocalTarget.__del__')
    def test_local_tmp_target(self, lt_del_patch, lt_init_patch):
        if False:
            return 10
        'Verify local target url with query string\n\n        '
        lt_init_patch.return_value = None
        lt_del_patch.return_value = None
        local_file = 'file://{}?is_tmp'.format(self.local_file)
        OpenerTarget(local_file)
        lt_init_patch.assert_called_with(self.local_file, is_tmp=True)

    @mock.patch('luigi.contrib.s3.S3Target.__init__')
    def test_s3_parse(self, s3_init_patch):
        if False:
            print('Hello World!')
        'Verify basic s3 target url\n\n        '
        s3_init_patch.return_value = None
        local_file = 's3://zefr/foo/bar.txt'
        OpenerTarget(local_file)
        s3_init_patch.assert_called_with('s3://zefr/foo/bar.txt')

    @mock.patch('luigi.contrib.s3.S3Target.__init__')
    def test_s3_parse_param(self, s3_init_patch):
        if False:
            print('Hello World!')
        'Verify s3 target url with params\n\n        '
        s3_init_patch.return_value = None
        local_file = 's3://zefr/foo/bar.txt?foo=hello&bar=true'
        OpenerTarget(local_file)
        s3_init_patch.assert_called_with('s3://zefr/foo/bar.txt', foo='hello', bar='true')

    def test_binary_support(self):
        if False:
            print('Hello World!')
        '\n        Make sure keyword arguments are preserved through the OpenerTarget\n        '
        fp = OpenerTarget('mock://file.txt').open('w')
        self.assertRaises(TypeError, fp.write, b'\x07\x08\x07')
        fp = OpenerTarget('mock://file.txt', format=luigi.format.MixedUnicodeBytes).open('w')
        fp.write(b'\x07\x08\x07')
        fp.close()