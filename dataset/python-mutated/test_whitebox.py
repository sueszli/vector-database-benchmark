import os
import bzrlib
from bzrlib import errors, osutils, tests
from bzrlib.osutils import relpath, pathjoin, abspath, realpath

class MoreTests(tests.TestCaseWithTransport):

    def test_relpath(self):
        if False:
            i = 10
            return i + 15
        'test for branch path lookups\n\n        bzrlib.osutils._relpath do a simple but subtle\n        job: given a path (either relative to cwd or absolute), work out\n        if it is inside a branch and return the path relative to the base.\n        '
        dtmp = osutils.mkdtemp()
        self.addCleanup(osutils.rmtree, dtmp)
        dtmp = realpath(dtmp)

        def rp(p):
            if False:
                return 10
            return relpath(dtmp, p)
        self.assertEqual('foo', rp(pathjoin(dtmp, 'foo')))
        self.assertEqual('', rp(dtmp))
        self.assertRaises(errors.PathNotChild, rp, '/etc')
        self.assertRaises(errors.PathNotChild, rp, dtmp.rstrip('\\/') + '2')
        self.assertRaises(errors.PathNotChild, rp, dtmp.rstrip('\\/') + '2/foo')
        os.chdir(dtmp)
        self.assertEqual('foo/bar/quux', rp('foo/bar/quux'))
        self.assertEqual('foo', rp('foo'))
        self.assertEqual('foo', rp('./foo'))
        self.assertEqual('foo', rp(abspath('foo')))
        self.assertRaises(errors.PathNotChild, rp, '../foo')