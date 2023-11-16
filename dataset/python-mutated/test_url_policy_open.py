"""Tests for the branch open with specific URL policy code."""
from bzrlib import urlutils
from bzrlib.branch import Branch, BranchReferenceFormat
from bzrlib.bzrdir import BzrProber
from bzrlib.controldir import ControlDir, ControlDirFormat
from bzrlib.errors import NotBranchError
from bzrlib.url_policy_open import BadUrl, _BlacklistPolicy, BranchLoopError, BranchReferenceForbidden, open_only_scheme, BranchOpener, WhitelistPolicy
from bzrlib.tests import TestCase, TestCaseWithTransport
from bzrlib.transport import chroot

class TestBranchOpenerCheckAndFollowBranchReference(TestCase):
    """Unit tests for `BranchOpener.check_and_follow_branch_reference`."""

    def setUp(self):
        if False:
            while True:
                i = 10
        super(TestBranchOpenerCheckAndFollowBranchReference, self).setUp()
        BranchOpener.install_hook()

    class StubbedBranchOpener(BranchOpener):
        """BranchOpener that provides canned answers.

        We implement the methods we need to to be able to control all the
        inputs to the `follow_reference` method, which is what is
        being tested in this class.
        """

        def __init__(self, references, policy):
            if False:
                for i in range(10):
                    print('nop')
            parent_cls = TestBranchOpenerCheckAndFollowBranchReference
            super(parent_cls.StubbedBranchOpener, self).__init__(policy)
            self._reference_values = {}
            for i in range(len(references) - 1):
                self._reference_values[references[i]] = references[i + 1]
            self.follow_reference_calls = []

        def follow_reference(self, url):
            if False:
                i = 10
                return i + 15
            self.follow_reference_calls.append(url)
            return self._reference_values[url]

    def make_branch_opener(self, should_follow_references, references, unsafe_urls=None):
        if False:
            print('Hello World!')
        policy = _BlacklistPolicy(should_follow_references, unsafe_urls)
        opener = self.StubbedBranchOpener(references, policy)
        return opener

    def test_check_initial_url(self):
        if False:
            for i in range(10):
                print('nop')
        opener = self.make_branch_opener(None, [], set(['a']))
        self.assertRaises(BadUrl, opener.check_and_follow_branch_reference, 'a')

    def test_not_reference(self):
        if False:
            i = 10
            return i + 15
        opener = self.make_branch_opener(False, ['a', None])
        self.assertEqual('a', opener.check_and_follow_branch_reference('a'))
        self.assertEqual(['a'], opener.follow_reference_calls)

    def test_branch_reference_forbidden(self):
        if False:
            return 10
        opener = self.make_branch_opener(False, ['a', 'b'])
        self.assertRaises(BranchReferenceForbidden, opener.check_and_follow_branch_reference, 'a')
        self.assertEqual(['a'], opener.follow_reference_calls)

    def test_allowed_reference(self):
        if False:
            print('Hello World!')
        opener = self.make_branch_opener(True, ['a', 'b', None])
        self.assertEqual('b', opener.check_and_follow_branch_reference('a'))
        self.assertEqual(['a', 'b'], opener.follow_reference_calls)

    def test_check_referenced_urls(self):
        if False:
            print('Hello World!')
        opener = self.make_branch_opener(True, ['a', 'b', None], unsafe_urls=set('b'))
        self.assertRaises(BadUrl, opener.check_and_follow_branch_reference, 'a')
        self.assertEqual(['a'], opener.follow_reference_calls)

    def test_self_referencing_branch(self):
        if False:
            for i in range(10):
                print('nop')
        opener = self.make_branch_opener(True, ['a', 'a'])
        self.assertRaises(BranchLoopError, opener.check_and_follow_branch_reference, 'a')
        self.assertEqual(['a'], opener.follow_reference_calls)

    def test_branch_reference_loop(self):
        if False:
            i = 10
            return i + 15
        references = ['a', 'b', 'a']
        opener = self.make_branch_opener(True, references)
        self.assertRaises(BranchLoopError, opener.check_and_follow_branch_reference, 'a')
        self.assertEqual(['a', 'b'], opener.follow_reference_calls)

class TrackingProber(BzrProber):
    """Subclass of BzrProber which tracks URLs it has been asked to open."""
    seen_urls = []

    @classmethod
    def probe_transport(klass, transport):
        if False:
            print('Hello World!')
        klass.seen_urls.append(transport.base)
        return BzrProber.probe_transport(transport)

class TestBranchOpenerStacking(TestCaseWithTransport):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        super(TestBranchOpenerStacking, self).setUp()
        BranchOpener.install_hook()

    def make_branch_opener(self, allowed_urls, probers=None):
        if False:
            return 10
        policy = WhitelistPolicy(True, allowed_urls, True)
        return BranchOpener(policy, probers)

    def test_probers(self):
        if False:
            print('Hello World!')
        b = self.make_branch('branch')
        opener = self.make_branch_opener([b.base], probers=[])
        self.assertRaises(NotBranchError, opener.open, b.base)
        opener = self.make_branch_opener([b.base], probers=[BzrProber])
        self.assertEqual(b.base, opener.open(b.base).base)

    def test_default_probers(self):
        if False:
            return 10
        self.addCleanup(ControlDirFormat.unregister_prober, TrackingProber)
        ControlDirFormat.register_prober(TrackingProber)
        TrackingProber.seen_urls = []
        opener = self.make_branch_opener(['.'], probers=[TrackingProber])
        self.assertRaises(NotBranchError, opener.open, '.')
        self.assertEqual(1, len(TrackingProber.seen_urls))
        TrackingProber.seen_urls = []
        self.assertRaises(NotBranchError, ControlDir.open, '.')
        self.assertEqual(1, len(TrackingProber.seen_urls))

    def test_allowed_url(self):
        if False:
            print('Hello World!')
        stacked_on_branch = self.make_branch('base-branch', format='1.6')
        stacked_branch = self.make_branch('stacked-branch', format='1.6')
        stacked_branch.set_stacked_on_url(stacked_on_branch.base)
        opener = self.make_branch_opener([stacked_branch.base, stacked_on_branch.base])
        opener.open(stacked_branch.base)

    def test_nstackable_repository(self):
        if False:
            i = 10
            return i + 15
        branch = self.make_branch('unstacked', format='knit')
        opener = self.make_branch_opener([branch.base])
        opener.open(branch.base)

    def test_allowed_relative_url(self):
        if False:
            i = 10
            return i + 15
        stacked_on_branch = self.make_branch('base-branch', format='1.6')
        stacked_branch = self.make_branch('stacked-branch', format='1.6')
        stacked_branch.set_stacked_on_url('../base-branch')
        opener = self.make_branch_opener([stacked_branch.base, stacked_on_branch.base])
        self.assertNotEqual('../base-branch', stacked_on_branch.base)
        opener.open(stacked_branch.base)

    def test_allowed_relative_nested(self):
        if False:
            return 10
        self.get_transport().mkdir('subdir')
        a = self.make_branch('subdir/a', format='1.6')
        b = self.make_branch('b', format='1.6')
        b.set_stacked_on_url('../subdir/a')
        c = self.make_branch('subdir/c', format='1.6')
        c.set_stacked_on_url('../../b')
        opener = self.make_branch_opener([c.base, b.base, a.base])
        opener.open(c.base)

    def test_forbidden_url(self):
        if False:
            return 10
        stacked_on_branch = self.make_branch('base-branch', format='1.6')
        stacked_branch = self.make_branch('stacked-branch', format='1.6')
        stacked_branch.set_stacked_on_url(stacked_on_branch.base)
        opener = self.make_branch_opener([stacked_branch.base])
        self.assertRaises(BadUrl, opener.open, stacked_branch.base)

    def test_forbidden_url_nested(self):
        if False:
            return 10
        a = self.make_branch('a', format='1.6')
        b = self.make_branch('b', format='1.6')
        b.set_stacked_on_url(a.base)
        c = self.make_branch('c', format='1.6')
        c.set_stacked_on_url(b.base)
        opener = self.make_branch_opener([c.base, b.base])
        self.assertRaises(BadUrl, opener.open, c.base)

    def test_self_stacked_branch(self):
        if False:
            print('Hello World!')
        a = self.make_branch('a', format='1.6')
        a.get_config().set_user_option('stacked_on_location', a.base)
        opener = self.make_branch_opener([a.base])
        self.assertRaises(BranchLoopError, opener.open, a.base)

    def test_loop_stacked_branch(self):
        if False:
            for i in range(10):
                print('nop')
        a = self.make_branch('a', format='1.6')
        b = self.make_branch('b', format='1.6')
        a.set_stacked_on_url(b.base)
        b.set_stacked_on_url(a.base)
        opener = self.make_branch_opener([a.base, b.base])
        self.assertRaises(BranchLoopError, opener.open, a.base)
        self.assertRaises(BranchLoopError, opener.open, b.base)

    def test_custom_opener(self):
        if False:
            for i in range(10):
                print('nop')
        a = self.make_branch('a', format='2a')
        b = self.make_branch('b', format='2a')
        b.set_stacked_on_url(a.base)
        TrackingProber.seen_urls = []
        opener = self.make_branch_opener([a.base, b.base], probers=[TrackingProber])
        opener.open(b.base)
        self.assertEqual(set(TrackingProber.seen_urls), set([b.base, a.base]))

    def test_custom_opener_with_branch_reference(self):
        if False:
            return 10
        a = self.make_branch('a', format='2a')
        b_dir = self.make_bzrdir('b')
        b = BranchReferenceFormat().initialize(b_dir, target_branch=a)
        TrackingProber.seen_urls = []
        opener = self.make_branch_opener([a.base, b.base], probers=[TrackingProber])
        opener.open(b.base)
        self.assertEqual(set(TrackingProber.seen_urls), set([b.base, a.base]))

class TestOpenOnlyScheme(TestCaseWithTransport):
    """Tests for `open_only_scheme`."""

    def setUp(self):
        if False:
            print('Hello World!')
        super(TestOpenOnlyScheme, self).setUp()
        BranchOpener.install_hook()

    def test_hook_does_not_interfere(self):
        if False:
            return 10
        self.make_branch('stacked')
        self.make_branch('stacked-on')
        Branch.open('stacked').set_stacked_on_url('../stacked-on')
        Branch.open('stacked')

    def get_chrooted_scheme(self, relpath):
        if False:
            while True:
                i = 10
        'Create a server that is chrooted to `relpath`.\n\n        :return: ``(scheme, get_url)`` where ``scheme`` is the scheme of the\n            chroot server and ``get_url`` returns URLs on said server.\n        '
        transport = self.get_transport(relpath)
        chroot_server = chroot.ChrootServer(transport)
        chroot_server.start_server()
        self.addCleanup(chroot_server.stop_server)

        def get_url(relpath):
            if False:
                i = 10
                return i + 15
            return chroot_server.get_url() + relpath
        return (urlutils.URL.from_string(chroot_server.get_url()).scheme, get_url)

    def test_stacked_within_scheme(self):
        if False:
            return 10
        self.get_transport().mkdir('inside')
        self.make_branch('inside/stacked')
        self.make_branch('inside/stacked-on')
        (scheme, get_chrooted_url) = self.get_chrooted_scheme('inside')
        Branch.open(get_chrooted_url('stacked')).set_stacked_on_url(get_chrooted_url('stacked-on'))
        open_only_scheme(scheme, get_chrooted_url('stacked'))

    def test_stacked_outside_scheme(self):
        if False:
            i = 10
            return i + 15
        self.get_transport().mkdir('inside')
        self.get_transport().mkdir('outside')
        self.make_branch('inside/stacked')
        self.make_branch('outside/stacked-on')
        (scheme, get_chrooted_url) = self.get_chrooted_scheme('inside')
        Branch.open(get_chrooted_url('stacked')).set_stacked_on_url(self.get_url('outside/stacked-on'))
        self.assertRaises(BadUrl, open_only_scheme, scheme, get_chrooted_url('stacked'))