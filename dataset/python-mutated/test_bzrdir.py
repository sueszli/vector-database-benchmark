"""Tests for the BzrDir facility and any format specific tests.

For interface contract tests, see tests/per_bzr_dir.
"""
import os
import subprocess
import sys
from bzrlib import branch, bzrdir, config, controldir, errors, help_topics, lock, repository, revision as _mod_revision, osutils, remote, transport as _mod_transport, urlutils, win32utils, workingtree_3, workingtree_4
import bzrlib.branch
from bzrlib.branchfmt.fullhistory import BzrBranchFormat5
from bzrlib.errors import NotBranchError, NoColocatedBranchSupport, UnknownFormatError, UnsupportedFormatError
from bzrlib.tests import TestCase, TestCaseWithMemoryTransport, TestCaseWithTransport, TestSkipped
from bzrlib.tests import http_server, http_utils
from bzrlib.tests.test_http import TestWithTransport_pycurl
from bzrlib.transport import memory, pathfilter
from bzrlib.transport.http._urllib import HttpTransport_urllib
from bzrlib.transport.nosmart import NoSmartTransportDecorator
from bzrlib.transport.readonly import ReadonlyTransportDecorator
from bzrlib.repofmt import knitrepo, knitpack_repo

class TestDefaultFormat(TestCase):

    def test_get_set_default_format(self):
        if False:
            return 10
        old_format = bzrdir.BzrDirFormat.get_default_format()
        self.assertIsInstance(old_format, bzrdir.BzrDirMetaFormat1)
        controldir.ControlDirFormat._set_default_format(SampleBzrDirFormat())
        try:
            result = bzrdir.BzrDir.create('memory:///')
            self.assertIsInstance(result, SampleBzrDir)
        finally:
            controldir.ControlDirFormat._set_default_format(old_format)
        self.assertEqual(old_format, bzrdir.BzrDirFormat.get_default_format())

class DeprecatedBzrDirFormat(bzrdir.BzrDirFormat):
    """A deprecated bzr dir format."""

class TestFormatRegistry(TestCase):

    def make_format_registry(self):
        if False:
            for i in range(10):
                print('nop')
        my_format_registry = controldir.ControlDirFormatRegistry()
        my_format_registry.register('deprecated', DeprecatedBzrDirFormat, 'Some format.  Slower and unawesome and deprecated.', deprecated=True)
        my_format_registry.register_lazy('lazy', 'bzrlib.tests.test_bzrdir', 'DeprecatedBzrDirFormat', 'Format registered lazily', deprecated=True)
        bzrdir.register_metadir(my_format_registry, 'knit', 'bzrlib.repofmt.knitrepo.RepositoryFormatKnit1', 'Format using knits')
        my_format_registry.set_default('knit')
        bzrdir.register_metadir(my_format_registry, 'branch6', 'bzrlib.repofmt.knitrepo.RepositoryFormatKnit3', 'Experimental successor to knit.  Use at your own risk.', branch_format='bzrlib.branch.BzrBranchFormat6', experimental=True)
        bzrdir.register_metadir(my_format_registry, 'hidden format', 'bzrlib.repofmt.knitrepo.RepositoryFormatKnit3', 'Experimental successor to knit.  Use at your own risk.', branch_format='bzrlib.branch.BzrBranchFormat6', hidden=True)
        my_format_registry.register('hiddendeprecated', DeprecatedBzrDirFormat, 'Old format.  Slower and does not support things. ', hidden=True)
        my_format_registry.register_lazy('hiddenlazy', 'bzrlib.tests.test_bzrdir', 'DeprecatedBzrDirFormat', 'Format registered lazily', deprecated=True, hidden=True)
        return my_format_registry

    def test_format_registry(self):
        if False:
            return 10
        my_format_registry = self.make_format_registry()
        my_bzrdir = my_format_registry.make_bzrdir('lazy')
        self.assertIsInstance(my_bzrdir, DeprecatedBzrDirFormat)
        my_bzrdir = my_format_registry.make_bzrdir('deprecated')
        self.assertIsInstance(my_bzrdir, DeprecatedBzrDirFormat)
        my_bzrdir = my_format_registry.make_bzrdir('default')
        self.assertIsInstance(my_bzrdir.repository_format, knitrepo.RepositoryFormatKnit1)
        my_bzrdir = my_format_registry.make_bzrdir('knit')
        self.assertIsInstance(my_bzrdir.repository_format, knitrepo.RepositoryFormatKnit1)
        my_bzrdir = my_format_registry.make_bzrdir('branch6')
        self.assertIsInstance(my_bzrdir.get_branch_format(), bzrlib.branch.BzrBranchFormat6)

    def test_get_help(self):
        if False:
            for i in range(10):
                print('nop')
        my_format_registry = self.make_format_registry()
        self.assertEqual('Format registered lazily', my_format_registry.get_help('lazy'))
        self.assertEqual('Format using knits', my_format_registry.get_help('knit'))
        self.assertEqual('Format using knits', my_format_registry.get_help('default'))
        self.assertEqual('Some format.  Slower and unawesome and deprecated.', my_format_registry.get_help('deprecated'))

    def test_help_topic(self):
        if False:
            return 10
        topics = help_topics.HelpTopicRegistry()
        registry = self.make_format_registry()
        topics.register('current-formats', registry.help_topic, 'Current formats')
        topics.register('other-formats', registry.help_topic, 'Other formats')
        new = topics.get_detail('current-formats')
        rest = topics.get_detail('other-formats')
        (experimental, deprecated) = rest.split('Deprecated formats')
        self.assertContainsRe(new, 'formats-help')
        self.assertContainsRe(new, ':knit:\n    \\(native\\) \\(default\\) Format using knits\n')
        self.assertContainsRe(experimental, ':branch6:\n    \\(native\\) Experimental successor to knit')
        self.assertContainsRe(deprecated, ':lazy:\n    \\(native\\) Format registered lazily\n')
        self.assertNotContainsRe(new, 'hidden')

    def test_set_default_repository(self):
        if False:
            while True:
                i = 10
        default_factory = controldir.format_registry.get('default')
        old_default = [k for (k, v) in controldir.format_registry.iteritems() if v == default_factory and k != 'default'][0]
        controldir.format_registry.set_default_repository('dirstate-with-subtree')
        try:
            self.assertIs(controldir.format_registry.get('dirstate-with-subtree'), controldir.format_registry.get('default'))
            self.assertIs(repository.format_registry.get_default().__class__, knitrepo.RepositoryFormatKnit3)
        finally:
            controldir.format_registry.set_default_repository(old_default)

    def test_aliases(self):
        if False:
            return 10
        a_registry = controldir.ControlDirFormatRegistry()
        a_registry.register('deprecated', DeprecatedBzrDirFormat, 'Old format.  Slower and does not support stuff', deprecated=True)
        a_registry.register('deprecatedalias', DeprecatedBzrDirFormat, 'Old format.  Slower and does not support stuff', deprecated=True, alias=True)
        self.assertEqual(frozenset(['deprecatedalias']), a_registry.aliases())

class SampleBranch(bzrlib.branch.Branch):
    """A dummy branch for guess what, dummy use."""

    def __init__(self, dir):
        if False:
            while True:
                i = 10
        self.bzrdir = dir

class SampleRepository(bzrlib.repository.Repository):
    """A dummy repo."""

    def __init__(self, dir):
        if False:
            return 10
        self.bzrdir = dir

class SampleBzrDir(bzrdir.BzrDir):
    """A sample BzrDir implementation to allow testing static methods."""

    def create_repository(self, shared=False):
        if False:
            print('Hello World!')
        'See ControlDir.create_repository.'
        return 'A repository'

    def open_repository(self):
        if False:
            i = 10
            return i + 15
        'See ControlDir.open_repository.'
        return SampleRepository(self)

    def create_branch(self, name=None):
        if False:
            for i in range(10):
                print('nop')
        'See ControlDir.create_branch.'
        if name is not None:
            raise NoColocatedBranchSupport(self)
        return SampleBranch(self)

    def create_workingtree(self):
        if False:
            while True:
                i = 10
        'See ControlDir.create_workingtree.'
        return 'A tree'

class SampleBzrDirFormat(bzrdir.BzrDirFormat):
    """A sample format

    this format is initializable, unsupported to aid in testing the
    open and open_downlevel routines.
    """

    def get_format_string(self):
        if False:
            i = 10
            return i + 15
        'See BzrDirFormat.get_format_string().'
        return 'Sample .bzr dir format.'

    def initialize_on_transport(self, t):
        if False:
            while True:
                i = 10
        'Create a bzr dir.'
        t.mkdir('.bzr')
        t.put_bytes('.bzr/branch-format', self.get_format_string())
        return SampleBzrDir(t, self)

    def is_supported(self):
        if False:
            i = 10
            return i + 15
        return False

    def open(self, transport, _found=None):
        if False:
            i = 10
            return i + 15
        return 'opened branch.'

    @classmethod
    def from_string(cls, format_string):
        if False:
            return 10
        return cls()

class BzrDirFormatTest1(bzrdir.BzrDirMetaFormat1):

    @staticmethod
    def get_format_string():
        if False:
            return 10
        return 'Test format 1'

class BzrDirFormatTest2(bzrdir.BzrDirMetaFormat1):

    @staticmethod
    def get_format_string():
        if False:
            print('Hello World!')
        return 'Test format 2'

class TestBzrDirFormat(TestCaseWithTransport):
    """Tests for the BzrDirFormat facility."""

    def test_find_format(self):
        if False:
            while True:
                i = 10
        bzrdir.BzrProber.formats.register(BzrDirFormatTest1.get_format_string(), BzrDirFormatTest1())
        self.addCleanup(bzrdir.BzrProber.formats.remove, BzrDirFormatTest1.get_format_string())
        bzrdir.BzrProber.formats.register(BzrDirFormatTest2.get_format_string(), BzrDirFormatTest2())
        self.addCleanup(bzrdir.BzrProber.formats.remove, BzrDirFormatTest2.get_format_string())
        t = self.get_transport()
        self.build_tree(['foo/', 'bar/'], transport=t)

        def check_format(format, url):
            if False:
                return 10
            format.initialize(url)
            t = _mod_transport.get_transport_from_path(url)
            found_format = bzrdir.BzrDirFormat.find_format(t)
            self.assertIsInstance(found_format, format.__class__)
        check_format(BzrDirFormatTest1(), 'foo')
        check_format(BzrDirFormatTest2(), 'bar')

    def test_find_format_nothing_there(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertRaises(NotBranchError, bzrdir.BzrDirFormat.find_format, _mod_transport.get_transport_from_path('.'))

    def test_find_format_unknown_format(self):
        if False:
            while True:
                i = 10
        t = self.get_transport()
        t.mkdir('.bzr')
        t.put_bytes('.bzr/branch-format', '')
        self.assertRaises(UnknownFormatError, bzrdir.BzrDirFormat.find_format, _mod_transport.get_transport_from_path('.'))

    def test_register_unregister_format(self):
        if False:
            while True:
                i = 10
        format = SampleBzrDirFormat()
        url = self.get_url()
        format.initialize(url)
        bzrdir.BzrProber.formats.register(format.get_format_string(), format)
        self.assertRaises(UnsupportedFormatError, bzrdir.BzrDir.open, url)
        self.assertRaises(UnsupportedFormatError, bzrdir.BzrDir.open_containing, url)
        t = _mod_transport.get_transport_from_url(url)
        self.assertEqual(format.open(t), bzrdir.BzrDir.open_unsupported(url))
        bzrdir.BzrProber.formats.remove(format.get_format_string())
        self.assertRaises(UnknownFormatError, bzrdir.BzrDir.open_unsupported, url)

    def test_create_branch_and_repo_uses_default(self):
        if False:
            return 10
        format = SampleBzrDirFormat()
        branch = bzrdir.BzrDir.create_branch_and_repo(self.get_url(), format=format)
        self.assertTrue(isinstance(branch, SampleBranch))

    def test_create_branch_and_repo_under_shared(self):
        if False:
            i = 10
            return i + 15
        format = controldir.format_registry.make_bzrdir('knit')
        self.make_repository('.', shared=True, format=format)
        branch = bzrdir.BzrDir.create_branch_and_repo(self.get_url('child'), format=format)
        self.assertRaises(errors.NoRepositoryPresent, branch.bzrdir.open_repository)

    def test_create_branch_and_repo_under_shared_force_new(self):
        if False:
            return 10
        format = controldir.format_registry.make_bzrdir('knit')
        self.make_repository('.', shared=True, format=format)
        branch = bzrdir.BzrDir.create_branch_and_repo(self.get_url('child'), force_new_repo=True, format=format)
        branch.bzrdir.open_repository()

    def test_create_standalone_working_tree(self):
        if False:
            for i in range(10):
                print('nop')
        format = SampleBzrDirFormat()
        self.assertRaises(errors.NotLocalUrl, bzrdir.BzrDir.create_standalone_workingtree, self.get_readonly_url(), format=format)
        tree = bzrdir.BzrDir.create_standalone_workingtree('.', format=format)
        self.assertEqual('A tree', tree)

    def test_create_standalone_working_tree_under_shared_repo(self):
        if False:
            for i in range(10):
                print('nop')
        format = controldir.format_registry.make_bzrdir('knit')
        self.make_repository('.', shared=True, format=format)
        self.assertRaises(errors.NotLocalUrl, bzrdir.BzrDir.create_standalone_workingtree, self.get_readonly_url('child'), format=format)
        tree = bzrdir.BzrDir.create_standalone_workingtree('child', format=format)
        tree.bzrdir.open_repository()

    def test_create_branch_convenience(self):
        if False:
            return 10
        format = controldir.format_registry.make_bzrdir('knit')
        branch = bzrdir.BzrDir.create_branch_convenience('.', format=format)
        branch.bzrdir.open_workingtree()
        branch.bzrdir.open_repository()

    def test_create_branch_convenience_possible_transports(self):
        if False:
            while True:
                i = 10
        "Check that the optional 'possible_transports' is recognized"
        format = controldir.format_registry.make_bzrdir('knit')
        t = self.get_transport()
        branch = bzrdir.BzrDir.create_branch_convenience('.', format=format, possible_transports=[t])
        branch.bzrdir.open_workingtree()
        branch.bzrdir.open_repository()

    def test_create_branch_convenience_root(self):
        if False:
            i = 10
            return i + 15
        'Creating a branch at the root of a fs should work.'
        self.vfs_transport_factory = memory.MemoryServer
        format = controldir.format_registry.make_bzrdir('knit')
        branch = bzrdir.BzrDir.create_branch_convenience(self.get_url(), format=format)
        self.assertRaises(errors.NoWorkingTree, branch.bzrdir.open_workingtree)
        branch.bzrdir.open_repository()

    def test_create_branch_convenience_under_shared_repo(self):
        if False:
            return 10
        format = controldir.format_registry.make_bzrdir('knit')
        self.make_repository('.', shared=True, format=format)
        branch = bzrdir.BzrDir.create_branch_convenience('child', format=format)
        branch.bzrdir.open_workingtree()
        self.assertRaises(errors.NoRepositoryPresent, branch.bzrdir.open_repository)

    def test_create_branch_convenience_under_shared_repo_force_no_tree(self):
        if False:
            return 10
        format = controldir.format_registry.make_bzrdir('knit')
        self.make_repository('.', shared=True, format=format)
        branch = bzrdir.BzrDir.create_branch_convenience('child', force_new_tree=False, format=format)
        self.assertRaises(errors.NoWorkingTree, branch.bzrdir.open_workingtree)
        self.assertRaises(errors.NoRepositoryPresent, branch.bzrdir.open_repository)

    def test_create_branch_convenience_under_shared_repo_no_tree_policy(self):
        if False:
            return 10
        format = controldir.format_registry.make_bzrdir('knit')
        repo = self.make_repository('.', shared=True, format=format)
        repo.set_make_working_trees(False)
        branch = bzrdir.BzrDir.create_branch_convenience('child', format=format)
        self.assertRaises(errors.NoWorkingTree, branch.bzrdir.open_workingtree)
        self.assertRaises(errors.NoRepositoryPresent, branch.bzrdir.open_repository)

    def test_create_branch_convenience_under_shared_repo_no_tree_policy_force_tree(self):
        if False:
            i = 10
            return i + 15
        format = controldir.format_registry.make_bzrdir('knit')
        repo = self.make_repository('.', shared=True, format=format)
        repo.set_make_working_trees(False)
        branch = bzrdir.BzrDir.create_branch_convenience('child', force_new_tree=True, format=format)
        branch.bzrdir.open_workingtree()
        self.assertRaises(errors.NoRepositoryPresent, branch.bzrdir.open_repository)

    def test_create_branch_convenience_under_shared_repo_force_new_repo(self):
        if False:
            return 10
        format = controldir.format_registry.make_bzrdir('knit')
        self.make_repository('.', shared=True, format=format)
        branch = bzrdir.BzrDir.create_branch_convenience('child', force_new_repo=True, format=format)
        branch.bzrdir.open_repository()
        branch.bzrdir.open_workingtree()

class TestRepositoryAcquisitionPolicy(TestCaseWithTransport):

    def test_acquire_repository_standalone(self):
        if False:
            return 10
        'The default acquisition policy should create a standalone branch.'
        my_bzrdir = self.make_bzrdir('.')
        repo_policy = my_bzrdir.determine_repository_policy()
        (repo, is_new) = repo_policy.acquire_repository()
        self.assertEqual(repo.bzrdir.root_transport.base, my_bzrdir.root_transport.base)
        self.assertFalse(repo.is_shared())

    def test_determine_stacking_policy(self):
        if False:
            return 10
        parent_bzrdir = self.make_bzrdir('.')
        child_bzrdir = self.make_bzrdir('child')
        parent_bzrdir.get_config().set_default_stack_on('http://example.org')
        repo_policy = child_bzrdir.determine_repository_policy()
        self.assertEqual('http://example.org', repo_policy._stack_on)

    def test_determine_stacking_policy_relative(self):
        if False:
            while True:
                i = 10
        parent_bzrdir = self.make_bzrdir('.')
        child_bzrdir = self.make_bzrdir('child')
        parent_bzrdir.get_config().set_default_stack_on('child2')
        repo_policy = child_bzrdir.determine_repository_policy()
        self.assertEqual('child2', repo_policy._stack_on)
        self.assertEqual(parent_bzrdir.root_transport.base, repo_policy._stack_on_pwd)

    def prepare_default_stacking(self, child_format='1.6'):
        if False:
            print('Hello World!')
        parent_bzrdir = self.make_bzrdir('.')
        child_branch = self.make_branch('child', format=child_format)
        parent_bzrdir.get_config().set_default_stack_on(child_branch.base)
        new_child_transport = parent_bzrdir.transport.clone('child2')
        return (child_branch, new_child_transport)

    def test_clone_on_transport_obeys_stacking_policy(self):
        if False:
            for i in range(10):
                print('nop')
        (child_branch, new_child_transport) = self.prepare_default_stacking()
        new_child = child_branch.bzrdir.clone_on_transport(new_child_transport)
        self.assertEqual(child_branch.base, new_child.open_branch().get_stacked_on_url())

    def test_default_stacking_with_stackable_branch_unstackable_repo(self):
        if False:
            while True:
                i = 10
        source_bzrdir = self.make_bzrdir('source')
        knitpack_repo.RepositoryFormatKnitPack1().initialize(source_bzrdir)
        source_branch = bzrlib.branch.BzrBranchFormat7().initialize(source_bzrdir)
        parent_bzrdir = self.make_bzrdir('parent')
        stacked_on = self.make_branch('parent/stacked-on', format='pack-0.92')
        parent_bzrdir.get_config().set_default_stack_on(stacked_on.base)
        target = source_bzrdir.clone(self.get_url('parent/target'))

    def test_format_initialize_on_transport_ex_stacked_on(self):
        if False:
            while True:
                i = 10
        trunk = self.make_branch('trunk', format='1.9')
        t = self.get_transport('stacked')
        old_fmt = controldir.format_registry.make_bzrdir('pack-0.92')
        repo_name = old_fmt.repository_format.network_name()
        (repo, control, require_stacking, repo_policy) = old_fmt.initialize_on_transport_ex(t, repo_format_name=repo_name, stacked_on='../trunk', stack_on_pwd=t.base)
        if repo is not None:
            self.assertTrue(repo.is_write_locked())
            self.addCleanup(repo.unlock)
        else:
            repo = control.open_repository()
        self.assertIsInstance(control, bzrdir.BzrDir)
        opened = bzrdir.BzrDir.open(t.base)
        if not isinstance(old_fmt, remote.RemoteBzrDirFormat):
            self.assertEqual(control._format.network_name(), old_fmt.network_name())
            self.assertEqual(control._format.network_name(), opened._format.network_name())
        self.assertEqual(control.__class__, opened.__class__)
        self.assertLength(1, repo._fallback_repositories)

    def test_sprout_obeys_stacking_policy(self):
        if False:
            i = 10
            return i + 15
        (child_branch, new_child_transport) = self.prepare_default_stacking()
        new_child = child_branch.bzrdir.sprout(new_child_transport.base)
        self.assertEqual(child_branch.base, new_child.open_branch().get_stacked_on_url())

    def test_clone_ignores_policy_for_unsupported_formats(self):
        if False:
            print('Hello World!')
        (child_branch, new_child_transport) = self.prepare_default_stacking(child_format='pack-0.92')
        new_child = child_branch.bzrdir.clone_on_transport(new_child_transport)
        self.assertRaises(errors.UnstackableBranchFormat, new_child.open_branch().get_stacked_on_url)

    def test_sprout_ignores_policy_for_unsupported_formats(self):
        if False:
            print('Hello World!')
        (child_branch, new_child_transport) = self.prepare_default_stacking(child_format='pack-0.92')
        new_child = child_branch.bzrdir.sprout(new_child_transport.base)
        self.assertRaises(errors.UnstackableBranchFormat, new_child.open_branch().get_stacked_on_url)

    def test_sprout_upgrades_format_if_stacked_specified(self):
        if False:
            for i in range(10):
                print('nop')
        (child_branch, new_child_transport) = self.prepare_default_stacking(child_format='pack-0.92')
        new_child = child_branch.bzrdir.sprout(new_child_transport.base, stacked=True)
        self.assertEqual(child_branch.bzrdir.root_transport.base, new_child.open_branch().get_stacked_on_url())
        repo = new_child.open_repository()
        self.assertTrue(repo._format.supports_external_lookups)
        self.assertFalse(repo.supports_rich_root())

    def test_clone_on_transport_upgrades_format_if_stacked_on_specified(self):
        if False:
            print('Hello World!')
        (child_branch, new_child_transport) = self.prepare_default_stacking(child_format='pack-0.92')
        new_child = child_branch.bzrdir.clone_on_transport(new_child_transport, stacked_on=child_branch.bzrdir.root_transport.base)
        self.assertEqual(child_branch.bzrdir.root_transport.base, new_child.open_branch().get_stacked_on_url())
        repo = new_child.open_repository()
        self.assertTrue(repo._format.supports_external_lookups)
        self.assertFalse(repo.supports_rich_root())

    def test_sprout_upgrades_to_rich_root_format_if_needed(self):
        if False:
            return 10
        (child_branch, new_child_transport) = self.prepare_default_stacking(child_format='rich-root-pack')
        new_child = child_branch.bzrdir.sprout(new_child_transport.base, stacked=True)
        repo = new_child.open_repository()
        self.assertTrue(repo._format.supports_external_lookups)
        self.assertTrue(repo.supports_rich_root())

    def test_add_fallback_repo_handles_absolute_urls(self):
        if False:
            for i in range(10):
                print('nop')
        stack_on = self.make_branch('stack_on', format='1.6')
        repo = self.make_repository('repo', format='1.6')
        policy = bzrdir.UseExistingRepository(repo, stack_on.base)
        policy._add_fallback(repo)

    def test_add_fallback_repo_handles_relative_urls(self):
        if False:
            return 10
        stack_on = self.make_branch('stack_on', format='1.6')
        repo = self.make_repository('repo', format='1.6')
        policy = bzrdir.UseExistingRepository(repo, '.', stack_on.base)
        policy._add_fallback(repo)

    def test_configure_relative_branch_stacking_url(self):
        if False:
            return 10
        stack_on = self.make_branch('stack_on', format='1.6')
        stacked = self.make_branch('stack_on/stacked', format='1.6')
        policy = bzrdir.UseExistingRepository(stacked.repository, '.', stack_on.base)
        policy.configure_branch(stacked)
        self.assertEqual('..', stacked.get_stacked_on_url())

    def test_relative_branch_stacking_to_absolute(self):
        if False:
            while True:
                i = 10
        stack_on = self.make_branch('stack_on', format='1.6')
        stacked = self.make_branch('stack_on/stacked', format='1.6')
        policy = bzrdir.UseExistingRepository(stacked.repository, '.', self.get_readonly_url('stack_on'))
        policy.configure_branch(stacked)
        self.assertEqual(self.get_readonly_url('stack_on'), stacked.get_stacked_on_url())

class ChrootedTests(TestCaseWithTransport):
    """A support class that provides readonly urls outside the local namespace.

    This is done by checking if self.transport_server is a MemoryServer. if it
    is then we are chrooted already, if it is not then an HttpServer is used
    for readonly urls.
    """

    def setUp(self):
        if False:
            print('Hello World!')
        super(ChrootedTests, self).setUp()
        if not self.vfs_transport_factory == memory.MemoryServer:
            self.transport_readonly_server = http_server.HttpServer

    def local_branch_path(self, branch):
        if False:
            i = 10
            return i + 15
        return os.path.realpath(urlutils.local_path_from_url(branch.base))

    def test_open_containing(self):
        if False:
            while True:
                i = 10
        self.assertRaises(NotBranchError, bzrdir.BzrDir.open_containing, self.get_readonly_url(''))
        self.assertRaises(NotBranchError, bzrdir.BzrDir.open_containing, self.get_readonly_url('g/p/q'))
        control = bzrdir.BzrDir.create(self.get_url())
        (branch, relpath) = bzrdir.BzrDir.open_containing(self.get_readonly_url(''))
        self.assertEqual('', relpath)
        (branch, relpath) = bzrdir.BzrDir.open_containing(self.get_readonly_url('g/p/q'))
        self.assertEqual('g/p/q', relpath)

    def test_open_containing_tree_branch_or_repository_empty(self):
        if False:
            while True:
                i = 10
        self.assertRaises(errors.NotBranchError, bzrdir.BzrDir.open_containing_tree_branch_or_repository, self.get_readonly_url(''))

    def test_open_containing_tree_branch_or_repository_all(self):
        if False:
            print('Hello World!')
        self.make_branch_and_tree('topdir')
        (tree, branch, repo, relpath) = bzrdir.BzrDir.open_containing_tree_branch_or_repository('topdir/foo')
        self.assertEqual(os.path.realpath('topdir'), os.path.realpath(tree.basedir))
        self.assertEqual(os.path.realpath('topdir'), self.local_branch_path(branch))
        self.assertEqual(osutils.realpath(os.path.join('topdir', '.bzr', 'repository')), repo.bzrdir.transport.local_abspath('repository'))
        self.assertEqual(relpath, 'foo')

    def test_open_containing_tree_branch_or_repository_no_tree(self):
        if False:
            while True:
                i = 10
        self.make_branch('branch')
        (tree, branch, repo, relpath) = bzrdir.BzrDir.open_containing_tree_branch_or_repository('branch/foo')
        self.assertEqual(tree, None)
        self.assertEqual(os.path.realpath('branch'), self.local_branch_path(branch))
        self.assertEqual(osutils.realpath(os.path.join('branch', '.bzr', 'repository')), repo.bzrdir.transport.local_abspath('repository'))
        self.assertEqual(relpath, 'foo')

    def test_open_containing_tree_branch_or_repository_repo(self):
        if False:
            i = 10
            return i + 15
        self.make_repository('repo')
        (tree, branch, repo, relpath) = bzrdir.BzrDir.open_containing_tree_branch_or_repository('repo')
        self.assertEqual(tree, None)
        self.assertEqual(branch, None)
        self.assertEqual(osutils.realpath(os.path.join('repo', '.bzr', 'repository')), repo.bzrdir.transport.local_abspath('repository'))
        self.assertEqual(relpath, '')

    def test_open_containing_tree_branch_or_repository_shared_repo(self):
        if False:
            i = 10
            return i + 15
        self.make_repository('shared', shared=True)
        bzrdir.BzrDir.create_branch_convenience('shared/branch', force_new_tree=False)
        (tree, branch, repo, relpath) = bzrdir.BzrDir.open_containing_tree_branch_or_repository('shared/branch')
        self.assertEqual(tree, None)
        self.assertEqual(os.path.realpath('shared/branch'), self.local_branch_path(branch))
        self.assertEqual(osutils.realpath(os.path.join('shared', '.bzr', 'repository')), repo.bzrdir.transport.local_abspath('repository'))
        self.assertEqual(relpath, '')

    def test_open_containing_tree_branch_or_repository_branch_subdir(self):
        if False:
            return 10
        self.make_branch_and_tree('foo')
        self.build_tree(['foo/bar/'])
        (tree, branch, repo, relpath) = bzrdir.BzrDir.open_containing_tree_branch_or_repository('foo/bar')
        self.assertEqual(os.path.realpath('foo'), os.path.realpath(tree.basedir))
        self.assertEqual(os.path.realpath('foo'), self.local_branch_path(branch))
        self.assertEqual(osutils.realpath(os.path.join('foo', '.bzr', 'repository')), repo.bzrdir.transport.local_abspath('repository'))
        self.assertEqual(relpath, 'bar')

    def test_open_containing_tree_branch_or_repository_repo_subdir(self):
        if False:
            i = 10
            return i + 15
        self.make_repository('bar')
        self.build_tree(['bar/baz/'])
        (tree, branch, repo, relpath) = bzrdir.BzrDir.open_containing_tree_branch_or_repository('bar/baz')
        self.assertEqual(tree, None)
        self.assertEqual(branch, None)
        self.assertEqual(osutils.realpath(os.path.join('bar', '.bzr', 'repository')), repo.bzrdir.transport.local_abspath('repository'))
        self.assertEqual(relpath, 'baz')

    def test_open_containing_from_transport(self):
        if False:
            return 10
        self.assertRaises(NotBranchError, bzrdir.BzrDir.open_containing_from_transport, _mod_transport.get_transport_from_url(self.get_readonly_url('')))
        self.assertRaises(NotBranchError, bzrdir.BzrDir.open_containing_from_transport, _mod_transport.get_transport_from_url(self.get_readonly_url('g/p/q')))
        control = bzrdir.BzrDir.create(self.get_url())
        (branch, relpath) = bzrdir.BzrDir.open_containing_from_transport(_mod_transport.get_transport_from_url(self.get_readonly_url('')))
        self.assertEqual('', relpath)
        (branch, relpath) = bzrdir.BzrDir.open_containing_from_transport(_mod_transport.get_transport_from_url(self.get_readonly_url('g/p/q')))
        self.assertEqual('g/p/q', relpath)

    def test_open_containing_tree_or_branch(self):
        if False:
            for i in range(10):
                print('nop')
        self.make_branch_and_tree('topdir')
        (tree, branch, relpath) = bzrdir.BzrDir.open_containing_tree_or_branch('topdir/foo')
        self.assertEqual(os.path.realpath('topdir'), os.path.realpath(tree.basedir))
        self.assertEqual(os.path.realpath('topdir'), self.local_branch_path(branch))
        self.assertIs(tree.bzrdir, branch.bzrdir)
        self.assertEqual('foo', relpath)
        (tree, branch, relpath) = bzrdir.BzrDir.open_containing_tree_or_branch(self.get_readonly_url('topdir/foo'))
        self.assertEqual(None, tree)
        self.assertEqual('foo', relpath)
        self.make_branch('topdir/foo')
        (tree, branch, relpath) = bzrdir.BzrDir.open_containing_tree_or_branch('topdir/foo')
        self.assertIs(tree, None)
        self.assertEqual(os.path.realpath('topdir/foo'), self.local_branch_path(branch))
        self.assertEqual('', relpath)

    def test_open_tree_or_branch(self):
        if False:
            i = 10
            return i + 15
        self.make_branch_and_tree('topdir')
        (tree, branch) = bzrdir.BzrDir.open_tree_or_branch('topdir')
        self.assertEqual(os.path.realpath('topdir'), os.path.realpath(tree.basedir))
        self.assertEqual(os.path.realpath('topdir'), self.local_branch_path(branch))
        self.assertIs(tree.bzrdir, branch.bzrdir)
        (tree, branch) = bzrdir.BzrDir.open_tree_or_branch(self.get_readonly_url('topdir'))
        self.assertEqual(None, tree)
        self.make_branch('topdir/foo')
        (tree, branch) = bzrdir.BzrDir.open_tree_or_branch('topdir/foo')
        self.assertIs(tree, None)
        self.assertEqual(os.path.realpath('topdir/foo'), self.local_branch_path(branch))

    def test_open_from_transport(self):
        if False:
            return 10
        control = bzrdir.BzrDir.create(self.get_url())
        t = self.get_transport()
        opened_bzrdir = bzrdir.BzrDir.open_from_transport(t)
        self.assertEqual(t.base, opened_bzrdir.root_transport.base)
        self.assertIsInstance(opened_bzrdir, bzrdir.BzrDir)

    def test_open_from_transport_no_bzrdir(self):
        if False:
            print('Hello World!')
        t = self.get_transport()
        self.assertRaises(NotBranchError, bzrdir.BzrDir.open_from_transport, t)

    def test_open_from_transport_bzrdir_in_parent(self):
        if False:
            i = 10
            return i + 15
        control = bzrdir.BzrDir.create(self.get_url())
        t = self.get_transport()
        t.mkdir('subdir')
        t = t.clone('subdir')
        self.assertRaises(NotBranchError, bzrdir.BzrDir.open_from_transport, t)

    def test_sprout_recursive(self):
        if False:
            print('Hello World!')
        tree = self.make_branch_and_tree('tree1', format='development-subtree')
        sub_tree = self.make_branch_and_tree('tree1/subtree', format='development-subtree')
        sub_tree.set_root_id('subtree-root')
        tree.add_reference(sub_tree)
        self.build_tree(['tree1/subtree/file'])
        sub_tree.add('file')
        tree.commit('Initial commit')
        tree2 = tree.bzrdir.sprout('tree2').open_workingtree()
        tree2.lock_read()
        self.addCleanup(tree2.unlock)
        self.assertPathExists('tree2/subtree/file')
        self.assertEqual('tree-reference', tree2.kind('subtree-root'))

    def test_cloning_metadir(self):
        if False:
            for i in range(10):
                print('nop')
        'Ensure that cloning metadir is suitable'
        bzrdir = self.make_bzrdir('bzrdir')
        bzrdir.cloning_metadir()
        branch = self.make_branch('branch', format='knit')
        format = branch.bzrdir.cloning_metadir()
        self.assertIsInstance(format.workingtree_format, workingtree_4.WorkingTreeFormat6)

    def test_sprout_recursive_treeless(self):
        if False:
            print('Hello World!')
        tree = self.make_branch_and_tree('tree1', format='development-subtree')
        sub_tree = self.make_branch_and_tree('tree1/subtree', format='development-subtree')
        tree.add_reference(sub_tree)
        self.build_tree(['tree1/subtree/file'])
        sub_tree.add('file')
        tree.commit('Initial commit')
        tree.branch.get_config_stack().set('bzr.transform.orphan_policy', 'move')
        tree.bzrdir.destroy_workingtree()
        repo = self.make_repository('repo', shared=True, format='development-subtree')
        repo.set_make_working_trees(False)
        self.assertRaises(errors.NotBranchError, tree.bzrdir.sprout, 'repo/tree2')

    def make_foo_bar_baz(self):
        if False:
            print('Hello World!')
        foo = bzrdir.BzrDir.create_branch_convenience('foo').bzrdir
        bar = self.make_branch('foo/bar').bzrdir
        baz = self.make_branch('baz').bzrdir
        return (foo, bar, baz)

    def test_find_bzrdirs(self):
        if False:
            i = 10
            return i + 15
        (foo, bar, baz) = self.make_foo_bar_baz()
        t = self.get_transport()
        self.assertEqualBzrdirs([baz, foo, bar], bzrdir.BzrDir.find_bzrdirs(t))

    def make_fake_permission_denied_transport(self, transport, paths):
        if False:
            print('Hello World!')
        'Create a transport that raises PermissionDenied for some paths.'

        def filter(path):
            if False:
                i = 10
                return i + 15
            if path in paths:
                raise errors.PermissionDenied(path)
            return path
        path_filter_server = pathfilter.PathFilteringServer(transport, filter)
        path_filter_server.start_server()
        self.addCleanup(path_filter_server.stop_server)
        path_filter_transport = pathfilter.PathFilteringTransport(path_filter_server, '.')
        return (path_filter_server, path_filter_transport)

    def assertBranchUrlsEndWith(self, expect_url_suffix, actual_bzrdirs):
        if False:
            print('Hello World!')
        'Check that each branch url ends with the given suffix.'
        for actual_bzrdir in actual_bzrdirs:
            self.assertEndsWith(actual_bzrdir.user_url, expect_url_suffix)

    def test_find_bzrdirs_permission_denied(self):
        if False:
            for i in range(10):
                print('nop')
        (foo, bar, baz) = self.make_foo_bar_baz()
        t = self.get_transport()
        (path_filter_server, path_filter_transport) = self.make_fake_permission_denied_transport(t, ['foo'])
        self.assertBranchUrlsEndWith('/baz/', bzrdir.BzrDir.find_bzrdirs(path_filter_transport))
        smart_transport = self.make_smart_server('.', backing_server=path_filter_server)
        self.assertBranchUrlsEndWith('/baz/', bzrdir.BzrDir.find_bzrdirs(smart_transport))

    def test_find_bzrdirs_list_current(self):
        if False:
            while True:
                i = 10

        def list_current(transport):
            if False:
                i = 10
                return i + 15
            return [s for s in transport.list_dir('') if s != 'baz']
        (foo, bar, baz) = self.make_foo_bar_baz()
        t = self.get_transport()
        self.assertEqualBzrdirs([foo, bar], bzrdir.BzrDir.find_bzrdirs(t, list_current=list_current))

    def test_find_bzrdirs_evaluate(self):
        if False:
            while True:
                i = 10

        def evaluate(bzrdir):
            if False:
                for i in range(10):
                    print('nop')
            try:
                repo = bzrdir.open_repository()
            except errors.NoRepositoryPresent:
                return (True, bzrdir.root_transport.base)
            else:
                return (False, bzrdir.root_transport.base)
        (foo, bar, baz) = self.make_foo_bar_baz()
        t = self.get_transport()
        self.assertEqual([baz.root_transport.base, foo.root_transport.base], list(bzrdir.BzrDir.find_bzrdirs(t, evaluate=evaluate)))

    def assertEqualBzrdirs(self, first, second):
        if False:
            i = 10
            return i + 15
        first = list(first)
        second = list(second)
        self.assertEqual(len(first), len(second))
        for (x, y) in zip(first, second):
            self.assertEqual(x.root_transport.base, y.root_transport.base)

    def test_find_branches(self):
        if False:
            while True:
                i = 10
        root = self.make_repository('', shared=True)
        (foo, bar, baz) = self.make_foo_bar_baz()
        qux = self.make_bzrdir('foo/qux')
        t = self.get_transport()
        branches = bzrdir.BzrDir.find_branches(t)
        self.assertEqual(baz.root_transport.base, branches[0].base)
        self.assertEqual(foo.root_transport.base, branches[1].base)
        self.assertEqual(bar.root_transport.base, branches[2].base)
        branches = bzrdir.BzrDir.find_branches(t.clone('foo'))
        self.assertEqual(foo.root_transport.base, branches[0].base)
        self.assertEqual(bar.root_transport.base, branches[1].base)

class TestMissingRepoBranchesSkipped(TestCaseWithMemoryTransport):

    def test_find_bzrdirs_missing_repo(self):
        if False:
            return 10
        t = self.get_transport()
        arepo = self.make_repository('arepo', shared=True)
        abranch_url = arepo.user_url + '/abranch'
        abranch = bzrdir.BzrDir.create(abranch_url).create_branch()
        t.delete_tree('arepo/.bzr')
        self.assertRaises(errors.NoRepositoryPresent, branch.Branch.open, abranch_url)
        self.make_branch('baz')
        for actual_bzrdir in bzrdir.BzrDir.find_branches(t):
            self.assertEndsWith(actual_bzrdir.user_url, '/baz/')

class TestMeta1DirFormat(TestCaseWithTransport):
    """Tests specific to the meta1 dir format."""

    def test_right_base_dirs(self):
        if False:
            print('Hello World!')
        dir = bzrdir.BzrDirMetaFormat1().initialize(self.get_url())
        t = dir.transport
        branch_base = t.clone('branch').base
        self.assertEqual(branch_base, dir.get_branch_transport(None).base)
        self.assertEqual(branch_base, dir.get_branch_transport(BzrBranchFormat5()).base)
        repository_base = t.clone('repository').base
        self.assertEqual(repository_base, dir.get_repository_transport(None).base)
        repository_format = repository.format_registry.get_default()
        self.assertEqual(repository_base, dir.get_repository_transport(repository_format).base)
        checkout_base = t.clone('checkout').base
        self.assertEqual(checkout_base, dir.get_workingtree_transport(None).base)
        self.assertEqual(checkout_base, dir.get_workingtree_transport(workingtree_3.WorkingTreeFormat3()).base)

    def test_meta1dir_uses_lockdir(self):
        if False:
            for i in range(10):
                print('nop')
        'Meta1 format uses a LockDir to guard the whole directory, not a file.'
        dir = bzrdir.BzrDirMetaFormat1().initialize(self.get_url())
        t = dir.transport
        self.assertIsDirectory('branch-lock', t)

    def test_comparison(self):
        if False:
            return 10
        'Equality and inequality behave properly.\n\n        Metadirs should compare equal iff they have the same repo, branch and\n        tree formats.\n        '
        mydir = controldir.format_registry.make_bzrdir('knit')
        self.assertEqual(mydir, mydir)
        self.assertFalse(mydir != mydir)
        otherdir = controldir.format_registry.make_bzrdir('knit')
        self.assertEqual(otherdir, mydir)
        self.assertFalse(otherdir != mydir)
        otherdir2 = controldir.format_registry.make_bzrdir('development-subtree')
        self.assertNotEqual(otherdir2, mydir)
        self.assertFalse(otherdir2 == mydir)

    def test_with_features(self):
        if False:
            print('Hello World!')
        tree = self.make_branch_and_tree('tree', format='2a')
        tree.bzrdir.update_feature_flags({'bar': 'required'})
        self.assertRaises(errors.MissingFeature, bzrdir.BzrDir.open, 'tree')
        bzrdir.BzrDirMetaFormat1.register_feature('bar')
        self.addCleanup(bzrdir.BzrDirMetaFormat1.unregister_feature, 'bar')
        dir = bzrdir.BzrDir.open('tree')
        self.assertEqual('required', dir._format.features.get('bar'))
        tree.bzrdir.update_feature_flags({'bar': None, 'nonexistant': None})
        dir = bzrdir.BzrDir.open('tree')
        self.assertEqual({}, dir._format.features)

    def test_needs_conversion_different_working_tree(self):
        if False:
            i = 10
            return i + 15
        new_format = controldir.format_registry.make_bzrdir('dirstate')
        tree = self.make_branch_and_tree('tree', format='knit')
        self.assertTrue(tree.bzrdir.needs_format_conversion(new_format))

    def test_initialize_on_format_uses_smart_transport(self):
        if False:
            while True:
                i = 10
        self.setup_smart_server_with_call_log()
        new_format = controldir.format_registry.make_bzrdir('dirstate')
        transport = self.get_transport('target')
        transport.ensure_base()
        self.reset_smart_call_log()
        instance = new_format.initialize_on_transport(transport)
        self.assertIsInstance(instance, remote.RemoteBzrDir)
        rpc_count = len(self.hpss_calls)
        self.assertEqual(2, rpc_count)

class NonLocalTests(TestCaseWithTransport):
    """Tests for bzrdir static behaviour on non local paths."""

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super(NonLocalTests, self).setUp()
        self.vfs_transport_factory = memory.MemoryServer

    def test_create_branch_convenience(self):
        if False:
            i = 10
            return i + 15
        format = controldir.format_registry.make_bzrdir('knit')
        branch = bzrdir.BzrDir.create_branch_convenience(self.get_url('foo'), format=format)
        self.assertRaises(errors.NoWorkingTree, branch.bzrdir.open_workingtree)
        branch.bzrdir.open_repository()

    def test_create_branch_convenience_force_tree_not_local_fails(self):
        if False:
            while True:
                i = 10
        format = controldir.format_registry.make_bzrdir('knit')
        self.assertRaises(errors.NotLocalUrl, bzrdir.BzrDir.create_branch_convenience, self.get_url('foo'), force_new_tree=True, format=format)
        t = self.get_transport()
        self.assertFalse(t.has('foo'))

    def test_clone(self):
        if False:
            i = 10
            return i + 15
        format = controldir.format_registry.make_bzrdir('knit')
        branch = bzrdir.BzrDir.create_branch_convenience('local', format=format)
        branch.bzrdir.open_workingtree()
        result = branch.bzrdir.clone(self.get_url('remote'))
        self.assertRaises(errors.NoWorkingTree, result.open_workingtree)
        result.open_branch()
        result.open_repository()

    def test_checkout_metadir(self):
        if False:
            i = 10
            return i + 15
        self.make_branch('branch-knit2', format='dirstate-with-subtree')
        my_bzrdir = bzrdir.BzrDir.open(self.get_url('branch-knit2'))
        checkout_format = my_bzrdir.checkout_metadir()
        self.assertIsInstance(checkout_format.workingtree_format, workingtree_4.WorkingTreeFormat4)

class TestHTTPRedirections(object):
    """Test redirection between two http servers.

    This MUST be used by daughter classes that also inherit from
    TestCaseWithTwoWebservers.

    We can't inherit directly from TestCaseWithTwoWebservers or the
    test framework will try to create an instance which cannot
    run, its implementation being incomplete.
    """

    def create_transport_readonly_server(self):
        if False:
            return 10
        return http_utils.HTTPServerRedirecting()

    def create_transport_secondary_server(self):
        if False:
            while True:
                i = 10
        return http_utils.HTTPServerRedirecting()

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super(TestHTTPRedirections, self).setUp()
        self.new_server = self.get_readonly_server()
        self.old_server = self.get_secondary_server()
        self.old_server.redirect_to(self.new_server.host, self.new_server.port)

    def test_loop(self):
        if False:
            print('Hello World!')
        self.new_server.redirect_to(self.old_server.host, self.old_server.port)
        old_url = self._qualified_url(self.old_server.host, self.old_server.port)
        oldt = self._transport(old_url)
        self.assertRaises(errors.NotBranchError, bzrdir.BzrDir.open_from_transport, oldt)
        new_url = self._qualified_url(self.new_server.host, self.new_server.port)
        newt = self._transport(new_url)
        self.assertRaises(errors.NotBranchError, bzrdir.BzrDir.open_from_transport, newt)

    def test_qualifier_preserved(self):
        if False:
            return 10
        wt = self.make_branch_and_tree('branch')
        old_url = self._qualified_url(self.old_server.host, self.old_server.port)
        start = self._transport(old_url).clone('branch')
        bdir = bzrdir.BzrDir.open_from_transport(start)
        self.assertIsInstance(bdir.root_transport, type(start))

class TestHTTPRedirections_urllib(TestHTTPRedirections, http_utils.TestCaseWithTwoWebservers):
    """Tests redirections for urllib implementation"""
    _transport = HttpTransport_urllib

    def _qualified_url(self, host, port):
        if False:
            while True:
                i = 10
        result = 'http+urllib://%s:%s' % (host, port)
        self.permit_url(result)
        return result

class TestHTTPRedirections_pycurl(TestWithTransport_pycurl, TestHTTPRedirections, http_utils.TestCaseWithTwoWebservers):
    """Tests redirections for pycurl implementation"""

    def _qualified_url(self, host, port):
        if False:
            print('Hello World!')
        result = 'http+pycurl://%s:%s' % (host, port)
        self.permit_url(result)
        return result

class TestHTTPRedirections_nosmart(TestHTTPRedirections, http_utils.TestCaseWithTwoWebservers):
    """Tests redirections for the nosmart decorator"""
    _transport = NoSmartTransportDecorator

    def _qualified_url(self, host, port):
        if False:
            return 10
        result = 'nosmart+http://%s:%s' % (host, port)
        self.permit_url(result)
        return result

class TestHTTPRedirections_readonly(TestHTTPRedirections, http_utils.TestCaseWithTwoWebservers):
    """Tests redirections for readonly decoratror"""
    _transport = ReadonlyTransportDecorator

    def _qualified_url(self, host, port):
        if False:
            for i in range(10):
                print('nop')
        result = 'readonly+http://%s:%s' % (host, port)
        self.permit_url(result)
        return result

class TestDotBzrHidden(TestCaseWithTransport):
    ls = ['ls']
    if sys.platform == 'win32':
        ls = [os.environ['COMSPEC'], '/C', 'dir', '/B']

    def get_ls(self):
        if False:
            i = 10
            return i + 15
        f = subprocess.Popen(self.ls, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        (out, err) = f.communicate()
        self.assertEqual(0, f.returncode, 'Calling %s failed: %s' % (self.ls, err))
        return out.splitlines()

    def test_dot_bzr_hidden(self):
        if False:
            print('Hello World!')
        if sys.platform == 'win32' and (not win32utils.has_win32file):
            raise TestSkipped('unable to make file hidden without pywin32 library')
        b = bzrdir.BzrDir.create('.')
        self.build_tree(['a'])
        self.assertEqual(['a'], self.get_ls())

    def test_dot_bzr_hidden_with_url(self):
        if False:
            return 10
        if sys.platform == 'win32' and (not win32utils.has_win32file):
            raise TestSkipped('unable to make file hidden without pywin32 library')
        b = bzrdir.BzrDir.create(urlutils.local_path_to_url('.'))
        self.build_tree(['a'])
        self.assertEqual(['a'], self.get_ls())

class _TestBzrDirFormat(bzrdir.BzrDirMetaFormat1):
    """Test BzrDirFormat implementation for TestBzrDirSprout."""

    def _open(self, transport):
        if False:
            print('Hello World!')
        return _TestBzrDir(transport, self)

class _TestBzrDir(bzrdir.BzrDirMeta1):
    """Test BzrDir implementation for TestBzrDirSprout.

    When created a _TestBzrDir already has repository and a branch.  The branch
    is a test double as well.
    """

    def __init__(self, *args, **kwargs):
        if False:
            print('Hello World!')
        super(_TestBzrDir, self).__init__(*args, **kwargs)
        self.test_branch = _TestBranch(self.transport)
        self.test_branch.repository = self.create_repository()

    def open_branch(self, unsupported=False, possible_transports=None):
        if False:
            for i in range(10):
                print('nop')
        return self.test_branch

    def cloning_metadir(self, require_stacking=False):
        if False:
            while True:
                i = 10
        return _TestBzrDirFormat()

class _TestBranchFormat(bzrlib.branch.BranchFormat):
    """Test Branch format for TestBzrDirSprout."""

class _TestBranch(bzrlib.branch.Branch):
    """Test Branch implementation for TestBzrDirSprout."""

    def __init__(self, transport, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        self._format = _TestBranchFormat()
        self._transport = transport
        self.base = transport.base
        super(_TestBranch, self).__init__(*args, **kwargs)
        self.calls = []
        self._parent = None

    def sprout(self, *args, **kwargs):
        if False:
            return 10
        self.calls.append('sprout')
        return _TestBranch(self._transport)

    def copy_content_into(self, destination, revision_id=None):
        if False:
            i = 10
            return i + 15
        self.calls.append('copy_content_into')

    def last_revision(self):
        if False:
            for i in range(10):
                print('nop')
        return _mod_revision.NULL_REVISION

    def get_parent(self):
        if False:
            return 10
        return self._parent

    def _get_config(self):
        if False:
            while True:
                i = 10
        return config.TransportConfig(self._transport, 'branch.conf')

    def _get_config_store(self):
        if False:
            print('Hello World!')
        return config.BranchStore(self)

    def set_parent(self, parent):
        if False:
            while True:
                i = 10
        self._parent = parent

    def lock_read(self):
        if False:
            print('Hello World!')
        return lock.LogicalLockResult(self.unlock)

    def unlock(self):
        if False:
            i = 10
            return i + 15
        return

class TestBzrDirSprout(TestCaseWithMemoryTransport):

    def test_sprout_uses_branch_sprout(self):
        if False:
            while True:
                i = 10
        "BzrDir.sprout calls Branch.sprout.\n\n        Usually, BzrDir.sprout should delegate to the branch's sprout method\n        for part of the work.  This allows the source branch to control the\n        choice of format for the new branch.\n\n        There are exceptions, but this tests avoids them:\n          - if there's no branch in the source bzrdir,\n          - or if the stacking has been requested and the format needs to be\n            overridden to satisfy that.\n        "
        t = self.get_transport('source')
        t.ensure_base()
        source_bzrdir = _TestBzrDirFormat().initialize_on_transport(t)
        self.assertEqual([], source_bzrdir.test_branch.calls)
        target_url = self.get_url('target')
        result = source_bzrdir.sprout(target_url, recurse='no')
        self.assertSubset(['sprout'], source_bzrdir.test_branch.calls)

    def test_sprout_parent(self):
        if False:
            return 10
        grandparent_tree = self.make_branch('grandparent')
        parent = grandparent_tree.bzrdir.sprout('parent').open_branch()
        branch_tree = parent.bzrdir.sprout('branch').open_branch()
        self.assertContainsRe(branch_tree.get_parent(), '/parent/$')

class TestBzrDirHooks(TestCaseWithMemoryTransport):

    def test_pre_open_called(self):
        if False:
            print('Hello World!')
        calls = []
        bzrdir.BzrDir.hooks.install_named_hook('pre_open', calls.append, None)
        transport = self.get_transport('foo')
        url = transport.base
        self.assertRaises(errors.NotBranchError, bzrdir.BzrDir.open, url)
        self.assertEqual([transport.base], [t.base for t in calls])

    def test_pre_open_actual_exceptions_raised(self):
        if False:
            for i in range(10):
                print('nop')
        count = [0]

        def fail_once(transport):
            if False:
                while True:
                    i = 10
            count[0] += 1
            if count[0] == 1:
                raise errors.BzrError('fail')
        bzrdir.BzrDir.hooks.install_named_hook('pre_open', fail_once, None)
        transport = self.get_transport('foo')
        url = transport.base
        err = self.assertRaises(errors.BzrError, bzrdir.BzrDir.open, url)
        self.assertEqual('fail', err._preformatted_string)

    def test_post_repo_init(self):
        if False:
            print('Hello World!')
        from bzrlib.controldir import RepoInitHookParams
        calls = []
        bzrdir.BzrDir.hooks.install_named_hook('post_repo_init', calls.append, None)
        self.make_repository('foo')
        self.assertLength(1, calls)
        params = calls[0]
        self.assertIsInstance(params, RepoInitHookParams)
        self.assertTrue(hasattr(params, 'bzrdir'))
        self.assertTrue(hasattr(params, 'repository'))

    def test_post_repo_init_hook_repr(self):
        if False:
            i = 10
            return i + 15
        param_reprs = []
        bzrdir.BzrDir.hooks.install_named_hook('post_repo_init', lambda params: param_reprs.append(repr(params)), None)
        self.make_repository('foo')
        self.assertLength(1, param_reprs)
        param_repr = param_reprs[0]
        self.assertStartsWith(param_repr, '<RepoInitHookParams for ')

class TestGenerateBackupName(TestCaseWithMemoryTransport):

    def setUp(self):
        if False:
            return 10
        super(TestGenerateBackupName, self).setUp()
        self._transport = self.get_transport()
        bzrdir.BzrDir.create(self.get_url(), possible_transports=[self._transport])
        self._bzrdir = bzrdir.BzrDir.open_from_transport(self._transport)

    def test_new(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual('a.~1~', self._bzrdir._available_backup_name('a'))

    def test_exiting(self):
        if False:
            print('Hello World!')
        self._transport.put_bytes('a.~1~', 'some content')
        self.assertEqual('a.~2~', self._bzrdir._available_backup_name('a'))

class TestMeta1DirColoFormat(TestCaseWithTransport):
    """Tests specific to the meta1 dir with colocated branches format."""

    def test_supports_colo(self):
        if False:
            while True:
                i = 10
        format = bzrdir.BzrDirMetaFormat1Colo()
        self.assertTrue(format.colocated_branches)

    def test_upgrade_from_2a(self):
        if False:
            while True:
                i = 10
        tree = self.make_branch_and_tree('.', format='2a')
        format = bzrdir.BzrDirMetaFormat1Colo()
        self.assertTrue(tree.bzrdir.needs_format_conversion(format))
        converter = tree.bzrdir._format.get_converter(format)
        result = converter.convert(tree.bzrdir, None)
        self.assertIsInstance(result._format, bzrdir.BzrDirMetaFormat1Colo)
        self.assertFalse(result.needs_format_conversion(format))

    def test_downgrade_to_2a(self):
        if False:
            i = 10
            return i + 15
        tree = self.make_branch_and_tree('.', format='development-colo')
        format = bzrdir.BzrDirMetaFormat1()
        self.assertTrue(tree.bzrdir.needs_format_conversion(format))
        converter = tree.bzrdir._format.get_converter(format)
        result = converter.convert(tree.bzrdir, None)
        self.assertIsInstance(result._format, bzrdir.BzrDirMetaFormat1)
        self.assertFalse(result.needs_format_conversion(format))

    def test_downgrade_to_2a_too_many_branches(self):
        if False:
            while True:
                i = 10
        tree = self.make_branch_and_tree('.', format='development-colo')
        tree.bzrdir.create_branch(name='another-colocated-branch')
        converter = tree.bzrdir._format.get_converter(bzrdir.BzrDirMetaFormat1())
        result = converter.convert(tree.bzrdir, bzrdir.BzrDirMetaFormat1())
        self.assertIsInstance(result._format, bzrdir.BzrDirMetaFormat1)

    def test_nested(self):
        if False:
            i = 10
            return i + 15
        tree = self.make_branch_and_tree('.', format='development-colo')
        tree.bzrdir.create_branch(name='foo')
        tree.bzrdir.create_branch(name='fool/bla')
        self.assertRaises(errors.ParentBranchExists, tree.bzrdir.create_branch, name='foo/bar')

    def test_parent(self):
        if False:
            print('Hello World!')
        tree = self.make_branch_and_tree('.', format='development-colo')
        tree.bzrdir.create_branch(name='fool/bla')
        tree.bzrdir.create_branch(name='foo/bar')
        self.assertRaises(errors.AlreadyBranchError, tree.bzrdir.create_branch, name='foo')

class SampleBzrFormat(bzrdir.BzrFormat):

    @classmethod
    def get_format_string(cls):
        if False:
            for i in range(10):
                print('nop')
        return 'First line\n'

class TestBzrFormat(TestCase):
    """Tests for BzrFormat."""

    def test_as_string(self):
        if False:
            return 10
        format = SampleBzrFormat()
        format.features = {'foo': 'required'}
        self.assertEqual(format.as_string(), 'First line\nrequired foo\n')
        format.features['another'] = 'optional'
        self.assertEqual(format.as_string(), 'First line\nrequired foo\noptional another\n')

    def test_network_name(self):
        if False:
            for i in range(10):
                print('nop')
        format = SampleBzrFormat()
        format.features = {'foo': 'required'}
        self.assertEqual('First line\nrequired foo\n', format.network_name())

    def test_from_string_no_features(self):
        if False:
            return 10
        format = SampleBzrFormat.from_string('First line\n')
        self.assertEqual({}, format.features)

    def test_from_string_with_feature(self):
        if False:
            i = 10
            return i + 15
        format = SampleBzrFormat.from_string('First line\nrequired foo\n')
        self.assertEqual('required', format.features.get('foo'))

    def test_from_string_format_string_mismatch(self):
        if False:
            while True:
                i = 10
        self.assertRaises(AssertionError, SampleBzrFormat.from_string, 'Second line\nrequired foo\n')

    def test_from_string_missing_space(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertRaises(errors.ParseFormatError, SampleBzrFormat.from_string, 'First line\nfoo\n')

    def test_from_string_with_spaces(self):
        if False:
            return 10
        format = SampleBzrFormat.from_string('First line\nrequired foo with spaces\n')
        self.assertEqual('required', format.features.get('foo with spaces'))

    def test_eq(self):
        if False:
            i = 10
            return i + 15
        format1 = SampleBzrFormat()
        format1.features = {'nested-trees': 'optional'}
        format2 = SampleBzrFormat()
        format2.features = {'nested-trees': 'optional'}
        self.assertEqual(format1, format1)
        self.assertEqual(format1, format2)
        format3 = SampleBzrFormat()
        self.assertNotEqual(format1, format3)

    def test_check_support_status_optional(self):
        if False:
            i = 10
            return i + 15
        format = SampleBzrFormat()
        format.features = {'nested-trees': 'optional'}
        format.check_support_status(True)
        self.addCleanup(SampleBzrFormat.unregister_feature, 'nested-trees')
        SampleBzrFormat.register_feature('nested-trees')
        format.check_support_status(True)

    def test_check_support_status_required(self):
        if False:
            return 10
        format = SampleBzrFormat()
        format.features = {'nested-trees': 'required'}
        self.assertRaises(errors.MissingFeature, format.check_support_status, True)
        self.addCleanup(SampleBzrFormat.unregister_feature, 'nested-trees')
        SampleBzrFormat.register_feature('nested-trees')
        format.check_support_status(True)

    def test_check_support_status_unknown(self):
        if False:
            return 10
        format = SampleBzrFormat()
        format.features = {'nested-trees': 'unknown'}
        self.assertRaises(errors.MissingFeature, format.check_support_status, True)
        self.addCleanup(SampleBzrFormat.unregister_feature, 'nested-trees')
        SampleBzrFormat.register_feature('nested-trees')
        format.check_support_status(True)

    def test_feature_already_registered(self):
        if False:
            i = 10
            return i + 15
        self.addCleanup(SampleBzrFormat.unregister_feature, 'nested-trees')
        SampleBzrFormat.register_feature('nested-trees')
        self.assertRaises(errors.FeatureAlreadyRegistered, SampleBzrFormat.register_feature, 'nested-trees')

    def test_feature_with_space(self):
        if False:
            while True:
                i = 10
        self.assertRaises(ValueError, SampleBzrFormat.register_feature, 'nested trees')