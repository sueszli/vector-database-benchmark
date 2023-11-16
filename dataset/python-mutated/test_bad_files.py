"""Tests being able to ignore bad filetypes."""
from cStringIO import StringIO
import os
from bzrlib import errors
from bzrlib.status import show_tree_status
from bzrlib.tests import TestCaseWithTransport
from bzrlib.tests.features import OsFifoFeature

def verify_status(tester, tree, value):
    if False:
        i = 10
        return i + 15
    'Verify the output of show_tree_status'
    tof = StringIO()
    show_tree_status(tree, to_file=tof)
    tof.seek(0)
    tester.assertEqual(value, tof.readlines())

class TestBadFiles(TestCaseWithTransport):

    def test_bad_files(self):
        if False:
            i = 10
            return i + 15
        "Test that bzr will ignore files it doesn't like"
        self.requireFeature(OsFifoFeature)
        wt = self.make_branch_and_tree('.')
        b = wt.branch
        files = ['one', 'two', 'three']
        file_ids = ['one-id', 'two-id', 'three-id']
        self.build_tree(files)
        wt.add(files, file_ids)
        wt.commit('Commit one', rev_id='a@u-0-0')
        verify_status(self, wt, [])
        os.mkfifo('a-fifo')
        self.build_tree(['six'])
        verify_status(self, wt, ['unknown:\n', '  a-fifo\n', '  six\n'])
        self.assertRaises(errors.BadFileKindError, wt.smart_add, ['a-fifo'])
        verify_status(self, wt, ['unknown:\n', '  a-fifo\n', '  six\n'])
        wt.smart_add([])
        verify_status(self, wt, ['added:\n', '  six\n', 'unknown:\n', '  a-fifo\n'])
        wt.commit('Commit four', rev_id='a@u-0-3')
        verify_status(self, wt, ['unknown:\n', '  a-fifo\n'])