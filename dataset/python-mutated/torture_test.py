import sys
import os
import unittest
from io import StringIO
from test.test_email import TestEmailBase
from test.support import run_unittest
import email
from email import __file__ as testfile
from email.iterators import _structure

def openfile(filename):
    if False:
        while True:
            i = 10
    from os.path import join, dirname, abspath
    path = abspath(join(dirname(testfile), os.pardir, 'moredata', filename))
    return open(path, 'r')
try:
    openfile('crispin-torture.txt')
except OSError:
    raise unittest.SkipTest

class TortureBase(TestEmailBase):

    def _msgobj(self, filename):
        if False:
            while True:
                i = 10
        fp = openfile(filename)
        try:
            msg = email.message_from_file(fp)
        finally:
            fp.close()
        return msg

class TestCrispinTorture(TortureBase):

    def test_mondo_message(self):
        if False:
            while True:
                i = 10
        eq = self.assertEqual
        neq = self.ndiffAssertEqual
        msg = self._msgobj('crispin-torture.txt')
        payload = msg.get_payload()
        eq(type(payload), list)
        eq(len(payload), 12)
        eq(msg.preamble, None)
        eq(msg.epilogue, '\n')
        fp = StringIO()
        _structure(msg, fp=fp)
        neq(fp.getvalue(), 'multipart/mixed\n    text/plain\n    message/rfc822\n        multipart/alternative\n            text/plain\n            multipart/mixed\n                text/richtext\n            application/andrew-inset\n    message/rfc822\n        audio/basic\n    audio/basic\n    image/pbm\n    message/rfc822\n        multipart/mixed\n            multipart/mixed\n                text/plain\n                audio/x-sun\n            multipart/mixed\n                image/gif\n                image/gif\n                application/x-be2\n                application/atomicmail\n            audio/x-sun\n    message/rfc822\n        multipart/mixed\n            text/plain\n            image/pgm\n            text/plain\n    message/rfc822\n        multipart/mixed\n            text/plain\n            image/pbm\n    message/rfc822\n        application/postscript\n    image/gif\n    message/rfc822\n        multipart/mixed\n            audio/basic\n            audio/basic\n    message/rfc822\n        multipart/mixed\n            application/postscript\n            text/plain\n            message/rfc822\n                multipart/mixed\n                    text/plain\n                    multipart/parallel\n                        image/gif\n                        audio/basic\n                    application/atomicmail\n                    message/rfc822\n                        audio/x-sun\n')

def _testclasses():
    if False:
        for i in range(10):
            print('nop')
    mod = sys.modules[__name__]
    return [getattr(mod, name) for name in dir(mod) if name.startswith('Test')]

def suite():
    if False:
        for i in range(10):
            print('nop')
    suite = unittest.TestSuite()
    for testclass in _testclasses():
        suite.addTest(unittest.makeSuite(testclass))
    return suite

def test_main():
    if False:
        while True:
            i = 10
    for testclass in _testclasses():
        run_unittest(testclass)
if __name__ == '__main__':
    unittest.main(defaultTest='suite')