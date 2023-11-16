import os
from bzrlib import tests
from bzrlib.tests import features

class TestTreeShape(tests.TestCaseWithTransport):

    def test_build_tree(self):
        if False:
            return 10
        'Test tree-building test helper'
        self.build_tree_contents([('foo', 'new contents'), ('.bzr/',), ('.bzr/README', 'hello')])
        self.assertPathExists('foo')
        self.assertPathExists('.bzr/README')
        self.assertFileEqual('hello', '.bzr/README')

    def test_build_tree_symlink(self):
        if False:
            return 10
        self.requireFeature(features.SymlinkFeature)
        self.build_tree_contents([('link@', 'target')])
        self.assertEqual('target', os.readlink('link'))