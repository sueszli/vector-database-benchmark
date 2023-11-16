import datetime
import ftplib
import os
import shutil
import sys
from helpers import unittest
from io import StringIO
from luigi.contrib.ftp import RemoteFileSystem, RemoteTarget
FILE1 = 'this is file1'
FILE2 = 'this is file2'
FILE3 = 'this is file3'
HOST = 'localhost'
USER = 'luigi'
PWD = 'some_password'

class TestFTPFilesystem(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        ' Creates structure\n\n        /test\n        /test/file1\n        /test/hola/\n        /test/hola/file2\n        /test/hola/singlefile\n        /test/hola/file3\n        '
        ftp = ftplib.FTP(HOST, USER, PWD)
        ftp.cwd('/')
        ftp.mkd('test')
        ftp.cwd('test')
        ftp.mkd('hola')
        ftp.cwd('hola')
        f2 = StringIO(FILE2)
        ftp.storbinary('STOR file2', f2)
        f3 = StringIO(FILE3)
        ftp.storbinary('STOR file3', f3)
        ftp.cwd('..')
        f1 = StringIO(FILE1)
        ftp.storbinary('STOR file1', f1)
        ftp.close()

    def test_file_remove(self):
        if False:
            i = 10
            return i + 15
        ' Delete with recursive deactivated '
        rfs = RemoteFileSystem(HOST, USER, PWD)
        rfs.remove('/test/hola/file3', recursive=False)
        rfs.remove('/test/hola/file2', recursive=False)
        rfs.remove('/test/hola', recursive=False)
        rfs.remove('/test/file1', recursive=False)
        rfs.remove('/test', recursive=False)
        ftp = ftplib.FTP(HOST, USER, PWD)
        list_dir = ftp.nlst()
        self.assertFalse('test' in list_dir)

    def test_recursive_remove(self):
        if False:
            return 10
        ' Test FTP filesystem removing files recursive '
        rfs = RemoteFileSystem(HOST, USER, PWD)
        rfs.remove('/test')
        ftp = ftplib.FTP(HOST, USER, PWD)
        list_dir = ftp.nlst()
        self.assertFalse('test' in list_dir)

class TestFTPFilesystemUpload(unittest.TestCase):

    def test_single(self):
        if False:
            i = 10
            return i + 15
        ' Test upload file with creation of intermediate folders '
        ftp_path = '/test/nest/luigi-test'
        local_filepath = '/tmp/luigi-test-ftp'
        with open(local_filepath, 'w') as outfile:
            outfile.write('something to fill')
        rfs = RemoteFileSystem(HOST, USER, PWD)
        rfs.put(local_filepath, ftp_path)
        ftp = ftplib.FTP(HOST, USER, PWD)
        ftp.cwd('/test/nest')
        list_dir = ftp.nlst()
        self.assertTrue('luigi-test' in list_dir)
        ftp.delete('luigi-test')
        ftp.cwd('/')
        ftp.rmd('/test/nest')
        ftp.rmd('test')
        os.remove(local_filepath)
        ftp.close()

class TestRemoteTarget(unittest.TestCase):

    def test_put(self):
        if False:
            i = 10
            return i + 15
        ' Test RemoteTarget put method with uploading to an FTP '
        local_filepath = '/tmp/luigi-remotetarget-write-test'
        remote_file = '/test/example.put.file'
        with open(local_filepath, 'w') as outfile:
            outfile.write('something to fill')
        remotetarget = RemoteTarget(remote_file, HOST, username=USER, password=PWD)
        remotetarget.put(local_filepath)
        ftp = ftplib.FTP(HOST, USER, PWD)
        ftp.cwd('/test')
        list_dir = ftp.nlst()
        self.assertTrue(remote_file.split('/')[-1] in list_dir)
        os.remove(local_filepath)
        ftp.delete(remote_file)
        ftp.cwd('/')
        ftp.rmd('test')
        ftp.close()

    def test_get(self):
        if False:
            i = 10
            return i + 15
        ' Test Remote target get method downloading a file from ftp '
        local_filepath = '/tmp/luigi-remotetarget-read-test'
        tmp_filepath = '/tmp/tmp-luigi-remotetarget-read-test'
        remote_file = '/test/example.get.file'
        with open(tmp_filepath, 'w') as outfile:
            outfile.write('something to fill')
        ftp = ftplib.FTP(HOST, USER, PWD)
        ftp.mkd('test')
        ftp.storbinary('STOR %s' % remote_file, open(tmp_filepath, 'rb'))
        ftp.close()
        remotetarget = RemoteTarget(remote_file, HOST, username=USER, password=PWD)
        remotetarget.get(local_filepath)
        with remotetarget.open('r') as fin:
            self.assertEqual(fin.read(), 'something to fill')
        if sys.version_info >= (3, 2):
            temppath = remotetarget._RemoteTarget__tmp_path
            self.assertTrue(os.path.exists(temppath))
            remotetarget = None
            self.assertFalse(os.path.exists(temppath))
        self.assertTrue(os.path.exists(local_filepath))
        ts = datetime.datetime.now() - datetime.timedelta(days=2)
        delayed_remotetarget = RemoteTarget(remote_file, HOST, username=USER, password=PWD, mtime=ts)
        self.assertTrue(delayed_remotetarget.exists())
        ts = datetime.datetime.now() + datetime.timedelta(days=2)
        delayed_remotetarget = RemoteTarget(remote_file, HOST, username=USER, password=PWD, mtime=ts)
        self.assertFalse(delayed_remotetarget.exists())
        os.remove(local_filepath)
        os.remove(tmp_filepath)
        ftp = ftplib.FTP(HOST, USER, PWD)
        ftp.delete(remote_file)
        ftp.cwd('/')
        ftp.rmd('test')
        ftp.close()

def _run_ftp_server():
    if False:
        return 10
    from pyftpdlib.authorizers import DummyAuthorizer
    from pyftpdlib.handlers import FTPHandler
    from pyftpdlib.servers import FTPServer
    authorizer = DummyAuthorizer()
    tmp_folder = '/tmp/luigi-test-ftp-server/'
    if os.path.exists(tmp_folder):
        shutil.rmtree(tmp_folder)
    os.mkdir(tmp_folder)
    authorizer.add_user(USER, PWD, tmp_folder, perm='elradfmwM')
    handler = FTPHandler
    handler.authorizer = authorizer
    address = ('localhost', 21)
    server = FTPServer(address, handler)
    server.serve_forever()
if __name__ == '__main__':
    _run_ftp_server()