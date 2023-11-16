"""Test read_bundle works properly across various transports."""
import cStringIO
import bzrlib.bundle
from bzrlib.bundle.serializer import write_bundle
import bzrlib.bzrdir
import bzrlib.errors as errors
from bzrlib import tests
from bzrlib.tests.test_transport import TestTransportImplementation
from bzrlib.tests.per_transport import transport_test_permutations
import bzrlib.transport
import bzrlib.urlutils
from bzrlib.tests.scenarios import load_tests_apply_scenarios
load_tests = load_tests_apply_scenarios

def create_bundle_file(test_case):
    if False:
        while True:
            i = 10
    test_case.build_tree(['tree/', 'tree/a', 'tree/subdir/'])
    format = bzrlib.bzrdir.BzrDirFormat.get_default_format()
    bzrdir = format.initialize('tree')
    repo = bzrdir.create_repository()
    branch = repo.bzrdir.create_branch()
    wt = branch.bzrdir.create_workingtree()
    wt.add(['a', 'subdir/'])
    wt.commit('new project', rev_id='commit-1')
    out = cStringIO.StringIO()
    rev_ids = write_bundle(wt.branch.repository, wt.get_parent_ids()[0], 'null:', out)
    out.seek(0)
    return (out, wt)

class TestReadMergeableBundleFromURL(TestTransportImplementation):
    """Test that read_bundle works properly across multiple transports"""
    scenarios = transport_test_permutations()

    def setUp(self):
        if False:
            return 10
        super(TestReadMergeableBundleFromURL, self).setUp()
        self.bundle_name = 'test_bundle'
        self.possible_transports = [self.get_transport(self.bundle_name)]
        self.overrideEnv('BZR_NO_SMART_VFS', None)
        wt = self.create_test_bundle()

    def read_mergeable_from_url(self, url):
        if False:
            for i in range(10):
                print('nop')
        return bzrlib.bundle.read_mergeable_from_url(url, possible_transports=self.possible_transports)

    def get_url(self, relpath=''):
        if False:
            return 10
        return bzrlib.urlutils.join(self._server.get_url(), relpath)

    def create_test_bundle(self):
        if False:
            print('Hello World!')
        (out, wt) = create_bundle_file(self)
        if self.get_transport().is_readonly():
            self.build_tree_contents([(self.bundle_name, out.getvalue())])
        else:
            self.get_transport().put_file(self.bundle_name, out)
            self.log('Put to: %s', self.get_url(self.bundle_name))
        return wt

    def test_read_mergeable_from_url(self):
        if False:
            i = 10
            return i + 15
        info = self.read_mergeable_from_url(unicode(self.get_url(self.bundle_name)))
        revision = info.real_revisions[-1]
        self.assertEqual('commit-1', revision.revision_id)

    def test_read_fail(self):
        if False:
            while True:
                i = 10
        self.assertRaises(errors.NotABundle, self.read_mergeable_from_url, self.get_url('tree'))
        self.assertRaises(errors.NotABundle, self.read_mergeable_from_url, self.get_url('tree/a'))

    def test_read_mergeable_respects_possible_transports(self):
        if False:
            i = 10
            return i + 15
        if not isinstance(self.get_transport(self.bundle_name), bzrlib.transport.ConnectedTransport):
            raise tests.TestSkipped('Need a ConnectedTransport to test transport reuse')
        url = unicode(self.get_url(self.bundle_name))
        info = self.read_mergeable_from_url(url)
        self.assertEqual(1, len(self.possible_transports))