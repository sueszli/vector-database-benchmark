"""
Compress module tests
"""
import os
import tarfile
import tempfile
import unittest
from zipfile import ZipFile, ZIP_DEFLATED
from txtai.archive import ArchiveFactory, Compress
from utils import Utils

class TestArchive(unittest.TestCase):
    """
    Archive tests.
    """

    def testInvalidTar(self):
        if False:
            print('Hello World!')
        '\n        Test invalid tar file\n        '
        path = os.path.join(tempfile.gettempdir(), 'badtar')
        with tarfile.open(path, 'w') as tar:
            tar.add(Utils.PATH, arcname='..')
        archive = ArchiveFactory.create(path)
        with self.assertRaises(IOError):
            archive.load(path, 'tar')

    def testInvalidZip(self):
        if False:
            while True:
                i = 10
        '\n        Test invalid zip file\n        '
        path = os.path.join(tempfile.gettempdir(), 'badzip')
        with ZipFile(path, 'w', ZIP_DEFLATED) as zfile:
            zfile.write(Utils.PATH + '/article.pdf', arcname='../article.pdf')
        archive = ArchiveFactory.create(path)
        with self.assertRaises(IOError):
            archive.load(path, 'zip')

    def testNotImplemented(self):
        if False:
            return 10
        '\n        Test exceptions for non-implemented methods\n        '
        compress = Compress()
        self.assertRaises(NotImplementedError, compress.pack, None, None)
        self.assertRaises(NotImplementedError, compress.unpack, None, None)