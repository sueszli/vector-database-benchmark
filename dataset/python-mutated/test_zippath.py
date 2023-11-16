"""
Test cases covering L{twisted.python.zippath}.
"""
from __future__ import annotations
import os
import zipfile
from typing import Union
from twisted.python.filepath import _coerceToFilesystemEncoding
from twisted.python.zippath import ZipArchive, ZipPath
from twisted.test.test_paths import AbstractFilePathTests

def zipit(dirname: str | bytes, zfname: str | bytes) -> None:
    if False:
        while True:
            i = 10
    "\n    Create a zipfile on zfname, containing the contents of dirname'\n    "
    coercedDirname = _coerceToFilesystemEncoding('', dirname)
    coercedZfname = _coerceToFilesystemEncoding('', zfname)
    with zipfile.ZipFile(coercedZfname, 'w') as zf:
        for (root, ignored, files) in os.walk(coercedDirname):
            for fname in files:
                fspath = os.path.join(root, fname)
                arcpath = os.path.join(root, fname)[len(dirname) + 1:]
                zf.write(fspath, arcpath)

class ZipFilePathTests(AbstractFilePathTests):
    """
    Test various L{ZipPath} path manipulations as well as reprs for L{ZipPath}
    and L{ZipArchive}.
    """
    path: ZipArchive[bytes]
    root: ZipArchive[bytes]

    def setUp(self) -> None:
        if False:
            i = 10
            return i + 15
        AbstractFilePathTests.setUp(self)
        zipit(self.cmn, self.cmn + b'.zip')
        self.nativecmn = _coerceToFilesystemEncoding('', self.cmn)
        self.path = ZipArchive(self.cmn + b'.zip')
        self.root = self.path
        self.all = [x.replace(self.cmn, self.cmn + b'.zip') for x in self.all]

    def test_sibling(self) -> None:
        if False:
            return 10
        '\n        L{ZipPath.sibling} returns a path at the same level.\n        '
        self.assertEqual(self.path.child('one').sibling('two'), self.path.child('two'))

    def test_zipPathRepr(self) -> None:
        if False:
            return 10
        "\n        Make sure that invoking ZipPath's repr prints the correct class name\n        and an absolute path to the zip file.\n        "
        child: Union[ZipPath[str, bytes], ZipPath[str, str]] = self.path.child('foo')
        pathRepr = 'ZipPath({!r})'.format(os.path.abspath(self.nativecmn + '.zip' + os.sep + 'foo'))
        self.assertEqual(repr(child), pathRepr)
        relativeCommon = self.nativecmn.replace(os.getcwd() + os.sep, '', 1) + '.zip'
        relpath = ZipArchive(relativeCommon)
        child = relpath.child('foo')
        self.assertEqual(repr(child), pathRepr)

    def test_zipPathReprParentDirSegment(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        The repr of a ZipPath with C{".."} in the internal part of its path\n        includes the C{".."} rather than applying the usual parent directory\n        meaning.\n        '
        child = self.path.child('foo').child('..').child('bar')
        pathRepr = 'ZipPath(%r)' % (self.nativecmn + '.zip' + os.sep.join(['', 'foo', '..', 'bar']))
        self.assertEqual(repr(child), pathRepr)

    def test_zipArchiveRepr(self) -> None:
        if False:
            i = 10
            return i + 15
        "\n        Make sure that invoking ZipArchive's repr prints the correct class\n        name and an absolute path to the zip file.\n        "
        path = ZipArchive(self.nativecmn + '.zip')
        pathRepr = 'ZipArchive({!r})'.format(os.path.abspath(self.nativecmn + '.zip'))
        self.assertEqual(repr(path), pathRepr)
        relativeCommon = self.nativecmn.replace(os.getcwd() + os.sep, '', 1) + '.zip'
        relpath = ZipArchive(relativeCommon)
        self.assertEqual(repr(relpath), pathRepr)