"""Tests for the test framework."""
from cStringIO import StringIO
import gc
import doctest
import os
import signal
import sys
import threading
import time
import unittest
import warnings
from testtools import ExtendedToOriginalDecorator, MultiTestResult
from testtools.content import Content
from testtools.content_type import ContentType
from testtools.matchers import DocTestMatches, Equals
import testtools.testresult.doubles
import bzrlib
from bzrlib import branchbuilder, bzrdir, controldir, errors, hooks, lockdir, memorytree, osutils, remote, repository, symbol_versioning, tests, transport, workingtree, workingtree_3, workingtree_4
from bzrlib.repofmt import groupcompress_repo
from bzrlib.symbol_versioning import deprecated_function, deprecated_in, deprecated_method
from bzrlib.tests import features, test_lsprof, test_server, TestUtil
from bzrlib.trace import note, mutter
from bzrlib.transport import memory

def _test_ids(test_suite):
    if False:
        while True:
            i = 10
    'Get the ids for the tests in a test suite.'
    return [t.id() for t in tests.iter_suite_tests(test_suite)]

class MetaTestLog(tests.TestCase):

    def test_logging(self):
        if False:
            i = 10
            return i + 15
        'Test logs are captured when a test fails.'
        self.log('a test message')
        details = self.getDetails()
        log = details['log']
        self.assertThat(log.content_type, Equals(ContentType('text', 'plain', {'charset': 'utf8'})))
        self.assertThat(u''.join(log.iter_text()), Equals(self.get_log()))
        self.assertThat(self.get_log(), DocTestMatches(u'...a test message\n', doctest.ELLIPSIS))

class TestTreeShape(tests.TestCaseInTempDir):

    def test_unicode_paths(self):
        if False:
            while True:
                i = 10
        self.requireFeature(features.UnicodeFilenameFeature)
        filename = u'hellØ'
        self.build_tree_contents([(filename, 'contents of hello')])
        self.assertPathExists(filename)

class TestClassesAvailable(tests.TestCase):
    """As a convenience we expose Test* classes from bzrlib.tests"""

    def test_test_case(self):
        if False:
            print('Hello World!')
        from bzrlib.tests import TestCase

    def test_test_loader(self):
        if False:
            print('Hello World!')
        from bzrlib.tests import TestLoader

    def test_test_suite(self):
        if False:
            while True:
                i = 10
        from bzrlib.tests import TestSuite

class TestTransportScenarios(tests.TestCase):
    """A group of tests that test the transport implementation adaption core.

    This is a meta test that the tests are applied to all available
    transports.

    This will be generalised in the future which is why it is in this
    test file even though it is specific to transport tests at the moment.
    """

    def test_get_transport_permutations(self):
        if False:
            return 10

        class MockModule(object):

            def get_test_permutations(self):
                if False:
                    print('Hello World!')
                return sample_permutation
        sample_permutation = [(1, 2), (3, 4)]
        from bzrlib.tests.per_transport import get_transport_test_permutations
        self.assertEqual(sample_permutation, get_transport_test_permutations(MockModule()))

    def test_scenarios_include_all_modules(self):
        if False:
            print('Hello World!')
        from bzrlib.tests.per_transport import transport_test_permutations
        from bzrlib.transport import _get_transport_modules
        modules = _get_transport_modules()
        permutation_count = 0
        for module in modules:
            try:
                permutation_count += len(reduce(getattr, (module + '.get_test_permutations').split('.')[1:], __import__(module))())
            except errors.DependencyNotPresent:
                pass
        scenarios = transport_test_permutations()
        self.assertEqual(permutation_count, len(scenarios))

    def test_scenarios_include_transport_class(self):
        if False:
            for i in range(10):
                print('nop')
        from bzrlib.tests.per_transport import transport_test_permutations
        scenarios = transport_test_permutations()
        self.assertTrue(len(scenarios) > 6)
        one_scenario = scenarios[0]
        self.assertIsInstance(one_scenario[0], str)
        self.assertTrue(issubclass(one_scenario[1]['transport_class'], bzrlib.transport.Transport))
        self.assertTrue(issubclass(one_scenario[1]['transport_server'], bzrlib.transport.Server))

class TestBranchScenarios(tests.TestCase):

    def test_scenarios(self):
        if False:
            i = 10
            return i + 15
        from bzrlib.tests.per_branch import make_scenarios
        server1 = 'a'
        server2 = 'b'
        formats = [('c', 'C'), ('d', 'D')]
        scenarios = make_scenarios(server1, server2, formats)
        self.assertEqual(2, len(scenarios))
        self.assertEqual([('str', {'branch_format': 'c', 'bzrdir_format': 'C', 'transport_readonly_server': 'b', 'transport_server': 'a'}), ('str', {'branch_format': 'd', 'bzrdir_format': 'D', 'transport_readonly_server': 'b', 'transport_server': 'a'})], scenarios)

class TestBzrDirScenarios(tests.TestCase):

    def test_scenarios(self):
        if False:
            print('Hello World!')
        from bzrlib.tests.per_controldir import make_scenarios
        vfs_factory = 'v'
        server1 = 'a'
        server2 = 'b'
        formats = ['c', 'd']
        scenarios = make_scenarios(vfs_factory, server1, server2, formats)
        self.assertEqual([('str', {'bzrdir_format': 'c', 'transport_readonly_server': 'b', 'transport_server': 'a', 'vfs_transport_factory': 'v'}), ('str', {'bzrdir_format': 'd', 'transport_readonly_server': 'b', 'transport_server': 'a', 'vfs_transport_factory': 'v'})], scenarios)

class TestRepositoryScenarios(tests.TestCase):

    def test_formats_to_scenarios(self):
        if False:
            print('Hello World!')
        from bzrlib.tests.per_repository import formats_to_scenarios
        formats = [('(c)', remote.RemoteRepositoryFormat()), ('(d)', repository.format_registry.get('Bazaar repository format 2a (needs bzr 1.16 or later)\n'))]
        no_vfs_scenarios = formats_to_scenarios(formats, 'server', 'readonly', None)
        vfs_scenarios = formats_to_scenarios(formats, 'server', 'readonly', vfs_transport_factory='vfs')
        expected = [('RemoteRepositoryFormat(c)', {'bzrdir_format': remote.RemoteBzrDirFormat(), 'repository_format': remote.RemoteRepositoryFormat(), 'transport_readonly_server': 'readonly', 'transport_server': 'server'}), ('RepositoryFormat2a(d)', {'bzrdir_format': bzrdir.BzrDirMetaFormat1(), 'repository_format': groupcompress_repo.RepositoryFormat2a(), 'transport_readonly_server': 'readonly', 'transport_server': 'server'})]
        self.assertEqual(expected, no_vfs_scenarios)
        self.assertEqual([('RemoteRepositoryFormat(c)', {'bzrdir_format': remote.RemoteBzrDirFormat(), 'repository_format': remote.RemoteRepositoryFormat(), 'transport_readonly_server': 'readonly', 'transport_server': 'server', 'vfs_transport_factory': 'vfs'}), ('RepositoryFormat2a(d)', {'bzrdir_format': bzrdir.BzrDirMetaFormat1(), 'repository_format': groupcompress_repo.RepositoryFormat2a(), 'transport_readonly_server': 'readonly', 'transport_server': 'server', 'vfs_transport_factory': 'vfs'})], vfs_scenarios)

class TestTestScenarioApplication(tests.TestCase):
    """Tests for the test adaption facilities."""

    def test_apply_scenario(self):
        if False:
            print('Hello World!')
        from bzrlib.tests import apply_scenario
        input_test = TestTestScenarioApplication('test_apply_scenario')
        adapted_test1 = apply_scenario(input_test, ('new id', {'bzrdir_format': 'bzr_format', 'repository_format': 'repo_fmt', 'transport_server': 'transport_server', 'transport_readonly_server': 'readonly-server'}))
        adapted_test2 = apply_scenario(input_test, ('new id 2', {'bzrdir_format': None}))
        self.assertRaises(AttributeError, getattr, input_test, 'bzrdir_format')
        self.assertEqual('bzr_format', adapted_test1.bzrdir_format)
        self.assertEqual('repo_fmt', adapted_test1.repository_format)
        self.assertEqual('transport_server', adapted_test1.transport_server)
        self.assertEqual('readonly-server', adapted_test1.transport_readonly_server)
        self.assertEqual('bzrlib.tests.test_selftest.TestTestScenarioApplication.test_apply_scenario(new id)', adapted_test1.id())
        self.assertEqual(None, adapted_test2.bzrdir_format)
        self.assertEqual('bzrlib.tests.test_selftest.TestTestScenarioApplication.test_apply_scenario(new id 2)', adapted_test2.id())

class TestInterRepositoryScenarios(tests.TestCase):

    def test_scenarios(self):
        if False:
            while True:
                i = 10
        from bzrlib.tests.per_interrepository import make_scenarios
        server1 = 'a'
        server2 = 'b'
        formats = [('C0', 'C1', 'C2', 'C3'), ('D0', 'D1', 'D2', 'D3')]
        scenarios = make_scenarios(server1, server2, formats)
        self.assertEqual([('C0,str,str', {'repository_format': 'C1', 'repository_format_to': 'C2', 'transport_readonly_server': 'b', 'transport_server': 'a', 'extra_setup': 'C3'}), ('D0,str,str', {'repository_format': 'D1', 'repository_format_to': 'D2', 'transport_readonly_server': 'b', 'transport_server': 'a', 'extra_setup': 'D3'})], scenarios)

class TestWorkingTreeScenarios(tests.TestCase):

    def test_scenarios(self):
        if False:
            while True:
                i = 10
        from bzrlib.tests.per_workingtree import make_scenarios
        server1 = 'a'
        server2 = 'b'
        formats = [workingtree_4.WorkingTreeFormat4(), workingtree_3.WorkingTreeFormat3(), workingtree_4.WorkingTreeFormat6()]
        scenarios = make_scenarios(server1, server2, formats, remote_server='c', remote_readonly_server='d', remote_backing_server='e')
        self.assertEqual([('WorkingTreeFormat4', {'bzrdir_format': formats[0]._matchingbzrdir, 'transport_readonly_server': 'b', 'transport_server': 'a', 'workingtree_format': formats[0]}), ('WorkingTreeFormat3', {'bzrdir_format': formats[1]._matchingbzrdir, 'transport_readonly_server': 'b', 'transport_server': 'a', 'workingtree_format': formats[1]}), ('WorkingTreeFormat6', {'bzrdir_format': formats[2]._matchingbzrdir, 'transport_readonly_server': 'b', 'transport_server': 'a', 'workingtree_format': formats[2]}), ('WorkingTreeFormat6,remote', {'bzrdir_format': formats[2]._matchingbzrdir, 'repo_is_remote': True, 'transport_readonly_server': 'd', 'transport_server': 'c', 'vfs_transport_factory': 'e', 'workingtree_format': formats[2]})], scenarios)

class TestTreeScenarios(tests.TestCase):

    def test_scenarios(self):
        if False:
            for i in range(10):
                print('nop')
        from bzrlib.tests.per_tree import _dirstate_tree_from_workingtree, make_scenarios, preview_tree_pre, preview_tree_post, return_parameter, revision_tree_from_workingtree
        server1 = 'a'
        server2 = 'b'
        smart_server = test_server.SmartTCPServer_for_testing
        smart_readonly_server = test_server.ReadonlySmartTCPServer_for_testing
        mem_server = memory.MemoryServer
        formats = [workingtree_4.WorkingTreeFormat4(), workingtree_3.WorkingTreeFormat3()]
        scenarios = make_scenarios(server1, server2, formats)
        self.assertEqual(8, len(scenarios))
        default_wt_format = workingtree.format_registry.get_default()
        wt4_format = workingtree_4.WorkingTreeFormat4()
        wt5_format = workingtree_4.WorkingTreeFormat5()
        wt6_format = workingtree_4.WorkingTreeFormat6()
        expected_scenarios = [('WorkingTreeFormat4', {'bzrdir_format': formats[0]._matchingbzrdir, 'transport_readonly_server': 'b', 'transport_server': 'a', 'workingtree_format': formats[0], '_workingtree_to_test_tree': return_parameter}), ('WorkingTreeFormat3', {'bzrdir_format': formats[1]._matchingbzrdir, 'transport_readonly_server': 'b', 'transport_server': 'a', 'workingtree_format': formats[1], '_workingtree_to_test_tree': return_parameter}), ('WorkingTreeFormat6,remote', {'bzrdir_format': wt6_format._matchingbzrdir, 'repo_is_remote': True, 'transport_readonly_server': smart_readonly_server, 'transport_server': smart_server, 'vfs_transport_factory': mem_server, 'workingtree_format': wt6_format, '_workingtree_to_test_tree': return_parameter}), ('RevisionTree', {'_workingtree_to_test_tree': revision_tree_from_workingtree, 'bzrdir_format': default_wt_format._matchingbzrdir, 'transport_readonly_server': 'b', 'transport_server': 'a', 'workingtree_format': default_wt_format}), ('DirStateRevisionTree,WT4', {'_workingtree_to_test_tree': _dirstate_tree_from_workingtree, 'bzrdir_format': wt4_format._matchingbzrdir, 'transport_readonly_server': 'b', 'transport_server': 'a', 'workingtree_format': wt4_format}), ('DirStateRevisionTree,WT5', {'_workingtree_to_test_tree': _dirstate_tree_from_workingtree, 'bzrdir_format': wt5_format._matchingbzrdir, 'transport_readonly_server': 'b', 'transport_server': 'a', 'workingtree_format': wt5_format}), ('PreviewTree', {'_workingtree_to_test_tree': preview_tree_pre, 'bzrdir_format': default_wt_format._matchingbzrdir, 'transport_readonly_server': 'b', 'transport_server': 'a', 'workingtree_format': default_wt_format}), ('PreviewTreePost', {'_workingtree_to_test_tree': preview_tree_post, 'bzrdir_format': default_wt_format._matchingbzrdir, 'transport_readonly_server': 'b', 'transport_server': 'a', 'workingtree_format': default_wt_format})]
        self.assertEqual(expected_scenarios, scenarios)

class TestInterTreeScenarios(tests.TestCase):
    """A group of tests that test the InterTreeTestAdapter."""

    def test_scenarios(self):
        if False:
            print('Hello World!')
        from bzrlib.tests.per_tree import return_parameter
        from bzrlib.tests.per_intertree import make_scenarios
        from bzrlib.workingtree_3 import WorkingTreeFormat3
        from bzrlib.workingtree_4 import WorkingTreeFormat4
        input_test = TestInterTreeScenarios('test_scenarios')
        server1 = 'a'
        server2 = 'b'
        format1 = WorkingTreeFormat4()
        format2 = WorkingTreeFormat3()
        formats = [('1', str, format1, format2, 'converter1'), ('2', int, format2, format1, 'converter2')]
        scenarios = make_scenarios(server1, server2, formats)
        self.assertEqual(2, len(scenarios))
        expected_scenarios = [('1', {'bzrdir_format': format1._matchingbzrdir, 'intertree_class': formats[0][1], 'workingtree_format': formats[0][2], 'workingtree_format_to': formats[0][3], 'mutable_trees_to_test_trees': formats[0][4], '_workingtree_to_test_tree': return_parameter, 'transport_server': server1, 'transport_readonly_server': server2}), ('2', {'bzrdir_format': format2._matchingbzrdir, 'intertree_class': formats[1][1], 'workingtree_format': formats[1][2], 'workingtree_format_to': formats[1][3], 'mutable_trees_to_test_trees': formats[1][4], '_workingtree_to_test_tree': return_parameter, 'transport_server': server1, 'transport_readonly_server': server2})]
        self.assertEqual(scenarios, expected_scenarios)

class TestTestCaseInTempDir(tests.TestCaseInTempDir):

    def test_home_is_not_working(self):
        if False:
            i = 10
            return i + 15
        self.assertNotEqual(self.test_dir, self.test_home_dir)
        cwd = osutils.getcwd()
        self.assertIsSameRealPath(self.test_dir, cwd)
        self.assertIsSameRealPath(self.test_home_dir, os.environ['HOME'])

    def test_assertEqualStat_equal(self):
        if False:
            while True:
                i = 10
        from bzrlib.tests.test_dirstate import _FakeStat
        self.build_tree(['foo'])
        real = os.lstat('foo')
        fake = _FakeStat(real.st_size, real.st_mtime, real.st_ctime, real.st_dev, real.st_ino, real.st_mode)
        self.assertEqualStat(real, fake)

    def test_assertEqualStat_notequal(self):
        if False:
            while True:
                i = 10
        self.build_tree(['foo', 'longname'])
        self.assertRaises(AssertionError, self.assertEqualStat, os.lstat('foo'), os.lstat('longname'))

    def test_failUnlessExists(self):
        if False:
            return 10
        'Deprecated failUnlessExists and failIfExists'
        self.applyDeprecated(deprecated_in((2, 4)), self.failUnlessExists, '.')
        self.build_tree(['foo/', 'foo/bar'])
        self.applyDeprecated(deprecated_in((2, 4)), self.failUnlessExists, 'foo/bar')
        self.applyDeprecated(deprecated_in((2, 4)), self.failIfExists, 'foo/foo')

    def test_assertPathExists(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertPathExists('.')
        self.build_tree(['foo/', 'foo/bar'])
        self.assertPathExists('foo/bar')
        self.assertPathDoesNotExist('foo/foo')

class TestTestCaseWithMemoryTransport(tests.TestCaseWithMemoryTransport):

    def test_home_is_non_existant_dir_under_root(self):
        if False:
            i = 10
            return i + 15
        'The test_home_dir for TestCaseWithMemoryTransport is missing.\n\n        This is because TestCaseWithMemoryTransport is for tests that do not\n        need any disk resources: they should be hooked into bzrlib in such a\n        way that no global settings are being changed by the test (only a\n        few tests should need to do that), and having a missing dir as home is\n        an effective way to ensure that this is the case.\n        '
        self.assertIsSameRealPath(self.TEST_ROOT + '/MemoryTransportMissingHomeDir', self.test_home_dir)
        self.assertIsSameRealPath(self.test_home_dir, os.environ['HOME'])

    def test_cwd_is_TEST_ROOT(self):
        if False:
            return 10
        self.assertIsSameRealPath(self.test_dir, self.TEST_ROOT)
        cwd = osutils.getcwd()
        self.assertIsSameRealPath(self.test_dir, cwd)

    def test_BZR_HOME_and_HOME_are_bytestrings(self):
        if False:
            while True:
                i = 10
        'The $BZR_HOME and $HOME environment variables should not be unicode.\n\n        See https://bugs.launchpad.net/bzr/+bug/464174\n        '
        self.assertIsInstance(os.environ['BZR_HOME'], str)
        self.assertIsInstance(os.environ['HOME'], str)

    def test_make_branch_and_memory_tree(self):
        if False:
            return 10
        'In TestCaseWithMemoryTransport we should not make the branch on disk.\n\n        This is hard to comprehensively robustly test, so we settle for making\n        a branch and checking no directory was created at its relpath.\n        '
        tree = self.make_branch_and_memory_tree('dir')
        self.assertFalse(osutils.lexists('dir'))
        self.assertIsInstance(tree, memorytree.MemoryTree)

    def test_make_branch_and_memory_tree_with_format(self):
        if False:
            while True:
                i = 10
        'make_branch_and_memory_tree should accept a format option.'
        format = bzrdir.BzrDirMetaFormat1()
        format.repository_format = repository.format_registry.get_default()
        tree = self.make_branch_and_memory_tree('dir', format=format)
        self.assertFalse(osutils.lexists('dir'))
        self.assertIsInstance(tree, memorytree.MemoryTree)
        self.assertEqual(format.repository_format.__class__, tree.branch.repository._format.__class__)

    def test_make_branch_builder(self):
        if False:
            while True:
                i = 10
        builder = self.make_branch_builder('dir')
        self.assertIsInstance(builder, branchbuilder.BranchBuilder)
        self.assertFalse(osutils.lexists('dir'))

    def test_make_branch_builder_with_format(self):
        if False:
            print('Hello World!')
        format = bzrdir.BzrDirMetaFormat1()
        repo_format = repository.format_registry.get_default()
        format.repository_format = repo_format
        builder = self.make_branch_builder('dir', format=format)
        the_branch = builder.get_branch()
        self.assertFalse(osutils.lexists('dir'))
        self.assertEqual(format.repository_format.__class__, the_branch.repository._format.__class__)
        self.assertEqual(repo_format.get_format_string(), self.get_transport().get_bytes('dir/.bzr/repository/format'))

    def test_make_branch_builder_with_format_name(self):
        if False:
            while True:
                i = 10
        builder = self.make_branch_builder('dir', format='knit')
        the_branch = builder.get_branch()
        self.assertFalse(osutils.lexists('dir'))
        dir_format = controldir.format_registry.make_bzrdir('knit')
        self.assertEqual(dir_format.repository_format.__class__, the_branch.repository._format.__class__)
        self.assertEqual('Bazaar-NG Knit Repository Format 1', self.get_transport().get_bytes('dir/.bzr/repository/format'))

    def test_dangling_locks_cause_failures(self):
        if False:
            print('Hello World!')

        class TestDanglingLock(tests.TestCaseWithMemoryTransport):

            def test_function(self):
                if False:
                    return 10
                t = self.get_transport_from_path('.')
                l = lockdir.LockDir(t, 'lock')
                l.create()
                l.attempt_lock()
        test = TestDanglingLock('test_function')
        result = test.run()
        total_failures = result.errors + result.failures
        if self._lock_check_thorough:
            self.assertEqual(1, len(total_failures))
        else:
            self.assertEqual(0, len(total_failures))

class TestTestCaseWithTransport(tests.TestCaseWithTransport):
    """Tests for the convenience functions TestCaseWithTransport introduces."""

    def test_get_readonly_url_none(self):
        if False:
            while True:
                i = 10
        from bzrlib.transport.readonly import ReadonlyTransportDecorator
        self.vfs_transport_factory = memory.MemoryServer
        self.transport_readonly_server = None
        url = self.get_readonly_url()
        url2 = self.get_readonly_url('foo/bar')
        t = transport.get_transport_from_url(url)
        t2 = transport.get_transport_from_url(url2)
        self.assertIsInstance(t, ReadonlyTransportDecorator)
        self.assertIsInstance(t2, ReadonlyTransportDecorator)
        self.assertEqual(t2.base[:-1], t.abspath('foo/bar'))

    def test_get_readonly_url_http(self):
        if False:
            return 10
        from bzrlib.tests.http_server import HttpServer
        from bzrlib.transport.http import HttpTransportBase
        self.transport_server = test_server.LocalURLServer
        self.transport_readonly_server = HttpServer
        url = self.get_readonly_url()
        url2 = self.get_readonly_url('foo/bar')
        t = transport.get_transport_from_url(url)
        t2 = transport.get_transport_from_url(url2)
        self.assertIsInstance(t, HttpTransportBase)
        self.assertIsInstance(t2, HttpTransportBase)
        self.assertEqual(t2.base[:-1], t.abspath('foo/bar'))

    def test_is_directory(self):
        if False:
            print('Hello World!')
        'Test assertIsDirectory assertion'
        t = self.get_transport()
        self.build_tree(['a_dir/', 'a_file'], transport=t)
        self.assertIsDirectory('a_dir', t)
        self.assertRaises(AssertionError, self.assertIsDirectory, 'a_file', t)
        self.assertRaises(AssertionError, self.assertIsDirectory, 'not_here', t)

    def test_make_branch_builder(self):
        if False:
            return 10
        builder = self.make_branch_builder('dir')
        rev_id = builder.build_commit()
        self.assertPathExists('dir')
        a_dir = controldir.ControlDir.open('dir')
        self.assertRaises(errors.NoWorkingTree, a_dir.open_workingtree)
        a_branch = a_dir.open_branch()
        builder_branch = builder.get_branch()
        self.assertEqual(a_branch.base, builder_branch.base)
        self.assertEqual((1, rev_id), builder_branch.last_revision_info())
        self.assertEqual((1, rev_id), a_branch.last_revision_info())

class TestTestCaseTransports(tests.TestCaseWithTransport):

    def setUp(self):
        if False:
            print('Hello World!')
        super(TestTestCaseTransports, self).setUp()
        self.vfs_transport_factory = memory.MemoryServer

    def test_make_bzrdir_preserves_transport(self):
        if False:
            for i in range(10):
                print('nop')
        t = self.get_transport()
        result_bzrdir = self.make_bzrdir('subdir')
        self.assertIsInstance(result_bzrdir.transport, memory.MemoryTransport)
        self.assertPathDoesNotExist('subdir')

class TestChrootedTest(tests.ChrootedTestCase):

    def test_root_is_root(self):
        if False:
            return 10
        t = transport.get_transport_from_url(self.get_readonly_url())
        url = t.base
        self.assertEqual(url, t.clone('..').base)

class TestProfileResult(tests.TestCase):

    def test_profiles_tests(self):
        if False:
            return 10
        self.requireFeature(features.lsprof_feature)
        terminal = testtools.testresult.doubles.ExtendedTestResult()
        result = tests.ProfileResult(terminal)

        class Sample(tests.TestCase):

            def a(self):
                if False:
                    return 10
                self.sample_function()

            def sample_function(self):
                if False:
                    return 10
                pass
        test = Sample('a')
        test.run(result)
        case = terminal._events[0][1]
        self.assertLength(1, case._benchcalls)
        ((_, _, _), stats) = case._benchcalls[0]
        self.assertTrue(callable(stats.pprint))

class TestTestResult(tests.TestCase):

    def check_timing(self, test_case, expected_re):
        if False:
            for i in range(10):
                print('nop')
        result = bzrlib.tests.TextTestResult(self._log_file, descriptions=0, verbosity=1)
        capture = testtools.testresult.doubles.ExtendedTestResult()
        test_case.run(MultiTestResult(result, capture))
        run_case = capture._events[0][1]
        timed_string = result._testTimeString(run_case)
        self.assertContainsRe(timed_string, expected_re)

    def test_test_reporting(self):
        if False:
            for i in range(10):
                print('nop')

        class ShortDelayTestCase(tests.TestCase):

            def test_short_delay(self):
                if False:
                    return 10
                time.sleep(0.003)

            def test_short_benchmark(self):
                if False:
                    return 10
                self.time(time.sleep, 0.003)
        self.check_timing(ShortDelayTestCase('test_short_delay'), '^ +[0-9]+ms$')
        self.check_timing(ShortDelayTestCase('test_short_benchmark'), '^ +[0-9]+ms\\*$')

    def test_unittest_reporting_unittest_class(self):
        if False:
            while True:
                i = 10

        class ShortDelayTestCase(unittest.TestCase):

            def test_short_delay(self):
                if False:
                    return 10
                time.sleep(0.003)
        self.check_timing(ShortDelayTestCase('test_short_delay'), '^ +[0-9]+ms$')

    def _time_hello_world_encoding(self):
        if False:
            return 10
        'Profile two sleep calls\n\n        This is used to exercise the test framework.\n        '
        self.time(unicode, 'hello', errors='replace')
        self.time(unicode, 'world', errors='replace')

    def test_lsprofiling(self):
        if False:
            return 10
        'Verbose test result prints lsprof statistics from test cases.'
        self.requireFeature(features.lsprof_feature)
        result_stream = StringIO()
        result = bzrlib.tests.VerboseTestResult(result_stream, descriptions=0, verbosity=2)
        example_test_case = TestTestResult('_time_hello_world_encoding')
        example_test_case._gather_lsprof_in_benchmarks = True
        example_test_case.run(result)
        output = result_stream.getvalue()
        self.assertContainsRe(output, "LSProf output for <type 'unicode'>\\(\\('hello',\\), {'errors': 'replace'}\\)")
        self.assertContainsRe(output, ' *CallCount *Recursive *Total\\(ms\\) *Inline\\(ms\\) *module:lineno\\(function\\)\\n')
        self.assertContainsRe(output, "( +1 +0 +0\\.\\d+ +0\\.\\d+ +<method 'disable' of '_lsprof\\.Profiler' objects>\\n)?")
        self.assertContainsRe(output, "LSProf output for <type 'unicode'>\\(\\('world',\\), {'errors': 'replace'}\\)\\n")

    def test_uses_time_from_testtools(self):
        if False:
            print('Hello World!')
        'Test case timings in verbose results should use testtools times'
        import datetime

        class TimeAddedVerboseTestResult(tests.VerboseTestResult):

            def startTest(self, test):
                if False:
                    i = 10
                    return i + 15
                self.time(datetime.datetime.utcfromtimestamp(1.145))
                super(TimeAddedVerboseTestResult, self).startTest(test)

            def addSuccess(self, test):
                if False:
                    for i in range(10):
                        print('nop')
                self.time(datetime.datetime.utcfromtimestamp(51.147))
                super(TimeAddedVerboseTestResult, self).addSuccess(test)

            def report_tests_starting(self):
                if False:
                    while True:
                        i = 10
                pass
        sio = StringIO()
        self.get_passing_test().run(TimeAddedVerboseTestResult(sio, 0, 2))
        self.assertEndsWith(sio.getvalue(), 'OK    50002ms\n')

    def test_known_failure(self):
        if False:
            i = 10
            return i + 15
        'Using knownFailure should trigger several result actions.'

        class InstrumentedTestResult(tests.ExtendedTestResult):

            def stopTestRun(self):
                if False:
                    print('Hello World!')
                pass

            def report_tests_starting(self):
                if False:
                    for i in range(10):
                        print('nop')
                pass

            def report_known_failure(self, test, err=None, details=None):
                if False:
                    print('Hello World!')
                self._call = (test, 'known failure')
        result = InstrumentedTestResult(None, None, None, None)

        class Test(tests.TestCase):

            def test_function(self):
                if False:
                    while True:
                        i = 10
                self.knownFailure('failed!')
        test = Test('test_function')
        test.run(result)
        self.assertEqual(2, len(result._call))
        self.assertEqual(test.id(), result._call[0].id())
        self.assertEqual('known failure', result._call[1])
        self.assertEqual(1, result.known_failure_count)
        self.assertTrue(result.wasSuccessful())

    def test_verbose_report_known_failure(self):
        if False:
            i = 10
            return i + 15
        result_stream = StringIO()
        result = bzrlib.tests.VerboseTestResult(result_stream, descriptions=0, verbosity=2)
        _get_test('test_xfail').run(result)
        self.assertContainsRe(result_stream.getvalue(), '\n\\S+\\.test_xfail\\s+XFAIL\\s+\\d+ms\n\\s*(?:Text attachment: )?reason(?:\n-+\n|: {{{)this_fails(?:\n-+\n|}}}\n)')

    def get_passing_test(self):
        if False:
            i = 10
            return i + 15
        "Return a test object that can't be run usefully."

        def passing_test():
            if False:
                print('Hello World!')
            pass
        return unittest.FunctionTestCase(passing_test)

    def test_add_not_supported(self):
        if False:
            return 10
        'Test the behaviour of invoking addNotSupported.'

        class InstrumentedTestResult(tests.ExtendedTestResult):

            def stopTestRun(self):
                if False:
                    i = 10
                    return i + 15
                pass

            def report_tests_starting(self):
                if False:
                    print('Hello World!')
                pass

            def report_unsupported(self, test, feature):
                if False:
                    while True:
                        i = 10
                self._call = (test, feature)
        result = InstrumentedTestResult(None, None, None, None)
        test = SampleTestCase('_test_pass')
        feature = features.Feature()
        result.startTest(test)
        result.addNotSupported(test, feature)
        self.assertEqual(2, len(result._call))
        self.assertEqual(test, result._call[0])
        self.assertEqual(feature, result._call[1])
        self.assertTrue(result.wasSuccessful())
        self.assertEqual(1, result.unsupported['Feature'])
        result.addNotSupported(test, feature)
        self.assertEqual(2, result.unsupported['Feature'])

    def test_verbose_report_unsupported(self):
        if False:
            i = 10
            return i + 15
        result_stream = StringIO()
        result = bzrlib.tests.VerboseTestResult(result_stream, descriptions=0, verbosity=2)
        test = self.get_passing_test()
        feature = features.Feature()
        result.startTest(test)
        prefix = len(result_stream.getvalue())
        result.report_unsupported(test, feature)
        output = result_stream.getvalue()[prefix:]
        lines = output.splitlines()
        self.assertStartsWith(lines[0], 'NODEP')
        self.assertEqual(lines[1], "    The feature 'Feature' is not available.")

    def test_unavailable_exception(self):
        if False:
            return 10
        'An UnavailableFeature being raised should invoke addNotSupported.'

        class InstrumentedTestResult(tests.ExtendedTestResult):

            def stopTestRun(self):
                if False:
                    print('Hello World!')
                pass

            def report_tests_starting(self):
                if False:
                    while True:
                        i = 10
                pass

            def addNotSupported(self, test, feature):
                if False:
                    return 10
                self._call = (test, feature)
        result = InstrumentedTestResult(None, None, None, None)
        feature = features.Feature()

        class Test(tests.TestCase):

            def test_function(self):
                if False:
                    print('Hello World!')
                raise tests.UnavailableFeature(feature)
        test = Test('test_function')
        test.run(result)
        self.assertEqual(2, len(result._call))
        self.assertEqual(test.id(), result._call[0].id())
        self.assertEqual(feature, result._call[1])
        self.assertEqual(0, result.error_count)

    def test_strict_with_unsupported_feature(self):
        if False:
            i = 10
            return i + 15
        result = bzrlib.tests.TextTestResult(self._log_file, descriptions=0, verbosity=1)
        test = self.get_passing_test()
        feature = 'Unsupported Feature'
        result.addNotSupported(test, feature)
        self.assertFalse(result.wasStrictlySuccessful())
        self.assertEqual(None, result._extractBenchmarkTime(test))

    def test_strict_with_known_failure(self):
        if False:
            print('Hello World!')
        result = bzrlib.tests.TextTestResult(self._log_file, descriptions=0, verbosity=1)
        test = _get_test('test_xfail')
        test.run(result)
        self.assertFalse(result.wasStrictlySuccessful())
        self.assertEqual(None, result._extractBenchmarkTime(test))

    def test_strict_with_success(self):
        if False:
            while True:
                i = 10
        result = bzrlib.tests.TextTestResult(self._log_file, descriptions=0, verbosity=1)
        test = self.get_passing_test()
        result.addSuccess(test)
        self.assertTrue(result.wasStrictlySuccessful())
        self.assertEqual(None, result._extractBenchmarkTime(test))

    def test_startTests(self):
        if False:
            print('Hello World!')
        'Starting the first test should trigger startTests.'

        class InstrumentedTestResult(tests.ExtendedTestResult):
            calls = 0

            def startTests(self):
                if False:
                    while True:
                        i = 10
                self.calls += 1
        result = InstrumentedTestResult(None, None, None, None)

        def test_function():
            if False:
                i = 10
                return i + 15
            pass
        test = unittest.FunctionTestCase(test_function)
        test.run(result)
        self.assertEqual(1, result.calls)

    def test_startTests_only_once(self):
        if False:
            for i in range(10):
                print('nop')
        'With multiple tests startTests should still only be called once'

        class InstrumentedTestResult(tests.ExtendedTestResult):
            calls = 0

            def startTests(self):
                if False:
                    return 10
                self.calls += 1
        result = InstrumentedTestResult(None, None, None, None)
        suite = unittest.TestSuite([unittest.FunctionTestCase(lambda : None), unittest.FunctionTestCase(lambda : None)])
        suite.run(result)
        self.assertEqual(1, result.calls)
        self.assertEqual(2, result.count)

class TestRunner(tests.TestCase):

    def dummy_test(self):
        if False:
            print('Hello World!')
        pass

    def run_test_runner(self, testrunner, test):
        if False:
            i = 10
            return i + 15
        'Run suite in testrunner, saving global state and restoring it.\n\n        This current saves and restores:\n        TestCaseInTempDir.TEST_ROOT\n\n        There should be no tests in this file that use\n        bzrlib.tests.TextTestRunner without using this convenience method,\n        because of our use of global state.\n        '
        old_root = tests.TestCaseInTempDir.TEST_ROOT
        try:
            tests.TestCaseInTempDir.TEST_ROOT = None
            return testrunner.run(test)
        finally:
            tests.TestCaseInTempDir.TEST_ROOT = old_root

    def test_known_failure_failed_run(self):
        if False:
            i = 10
            return i + 15

        class Test(tests.TestCase):

            def known_failure_test(self):
                if False:
                    return 10
                self.expectFailure('failed', self.assertTrue, False)
        test = unittest.TestSuite()
        test.addTest(Test('known_failure_test'))

        def failing_test():
            if False:
                for i in range(10):
                    print('nop')
            raise AssertionError('foo')
        test.addTest(unittest.FunctionTestCase(failing_test))
        stream = StringIO()
        runner = tests.TextTestRunner(stream=stream)
        result = self.run_test_runner(runner, test)
        lines = stream.getvalue().splitlines()
        self.assertContainsRe(stream.getvalue(), "(?sm)^bzr selftest.*$.*^======================================================================\n^FAIL: failing_test\n^----------------------------------------------------------------------\nTraceback \\(most recent call last\\):\n  .*    raise AssertionError\\('foo'\\)\n.*^----------------------------------------------------------------------\n.*FAILED \\(failures=1, known_failure_count=1\\)")

    def test_known_failure_ok_run(self):
        if False:
            return 10

        class Test(tests.TestCase):

            def known_failure_test(self):
                if False:
                    print('Hello World!')
                self.knownFailure('Never works...')
        test = Test('known_failure_test')
        stream = StringIO()
        runner = tests.TextTestRunner(stream=stream)
        result = self.run_test_runner(runner, test)
        self.assertContainsRe(stream.getvalue(), '\n-*\nRan 1 test in .*\n\nOK \\(known_failures=1\\)\n')

    def test_unexpected_success_bad(self):
        if False:
            print('Hello World!')

        class Test(tests.TestCase):

            def test_truth(self):
                if False:
                    while True:
                        i = 10
                self.expectFailure('No absolute truth', self.assertTrue, True)
        runner = tests.TextTestRunner(stream=StringIO())
        result = self.run_test_runner(runner, Test('test_truth'))
        self.assertContainsRe(runner.stream.getvalue(), '=+\nFAIL: \\S+\\.test_truth\n-+\n(?:.*\n)*\\s*(?:Text attachment: )?reason(?:\n-+\n|: {{{)No absolute truth(?:\n-+\n|}}}\n)(?:.*\n)*-+\nRan 1 test in .*\n\nFAILED \\(failures=1\\)\n\\Z')

    def test_result_decorator(self):
        if False:
            print('Hello World!')
        calls = []

        class LoggingDecorator(ExtendedToOriginalDecorator):

            def startTest(self, test):
                if False:
                    for i in range(10):
                        print('nop')
                ExtendedToOriginalDecorator.startTest(self, test)
                calls.append('start')
        test = unittest.FunctionTestCase(lambda : None)
        stream = StringIO()
        runner = tests.TextTestRunner(stream=stream, result_decorators=[LoggingDecorator])
        result = self.run_test_runner(runner, test)
        self.assertLength(1, calls)

    def test_skipped_test(self):
        if False:
            while True:
                i = 10

        class SkippingTest(tests.TestCase):

            def skipping_test(self):
                if False:
                    i = 10
                    return i + 15
                raise tests.TestSkipped('test intentionally skipped')
        runner = tests.TextTestRunner(stream=self._log_file)
        test = SkippingTest('skipping_test')
        result = self.run_test_runner(runner, test)
        self.assertTrue(result.wasSuccessful())

    def test_skipped_from_setup(self):
        if False:
            print('Hello World!')
        calls = []

        class SkippedSetupTest(tests.TestCase):

            def setUp(self):
                if False:
                    print('Hello World!')
                calls.append('setUp')
                self.addCleanup(self.cleanup)
                raise tests.TestSkipped('skipped setup')

            def test_skip(self):
                if False:
                    i = 10
                    return i + 15
                self.fail('test reached')

            def cleanup(self):
                if False:
                    return 10
                calls.append('cleanup')
        runner = tests.TextTestRunner(stream=self._log_file)
        test = SkippedSetupTest('test_skip')
        result = self.run_test_runner(runner, test)
        self.assertTrue(result.wasSuccessful())
        self.assertEqual(['setUp', 'cleanup'], calls)

    def test_skipped_from_test(self):
        if False:
            for i in range(10):
                print('nop')
        calls = []

        class SkippedTest(tests.TestCase):

            def setUp(self):
                if False:
                    i = 10
                    return i + 15
                super(SkippedTest, self).setUp()
                calls.append('setUp')
                self.addCleanup(self.cleanup)

            def test_skip(self):
                if False:
                    print('Hello World!')
                raise tests.TestSkipped('skipped test')

            def cleanup(self):
                if False:
                    return 10
                calls.append('cleanup')
        runner = tests.TextTestRunner(stream=self._log_file)
        test = SkippedTest('test_skip')
        result = self.run_test_runner(runner, test)
        self.assertTrue(result.wasSuccessful())
        self.assertEqual(['setUp', 'cleanup'], calls)

    def test_not_applicable(self):
        if False:
            i = 10
            return i + 15

        class Test(tests.TestCase):

            def not_applicable_test(self):
                if False:
                    while True:
                        i = 10
                raise tests.TestNotApplicable('this test never runs')
        out = StringIO()
        runner = tests.TextTestRunner(stream=out, verbosity=2)
        test = Test('not_applicable_test')
        result = self.run_test_runner(runner, test)
        self._log_file.write(out.getvalue())
        self.assertTrue(result.wasSuccessful())
        self.assertTrue(result.wasStrictlySuccessful())
        self.assertContainsRe(out.getvalue(), '(?m)not_applicable_test   * N/A')
        self.assertContainsRe(out.getvalue(), '(?m)^    this test never runs')

    def test_unsupported_features_listed(self):
        if False:
            while True:
                i = 10
        'When unsupported features are encountered they are detailed.'

        class Feature1(features.Feature):

            def _probe(self):
                if False:
                    return 10
                return False

        class Feature2(features.Feature):

            def _probe(self):
                if False:
                    print('Hello World!')
                return False
        test1 = SampleTestCase('_test_pass')
        test1._test_needs_features = [Feature1()]
        test2 = SampleTestCase('_test_pass')
        test2._test_needs_features = [Feature2()]
        test = unittest.TestSuite()
        test.addTest(test1)
        test.addTest(test2)
        stream = StringIO()
        runner = tests.TextTestRunner(stream=stream)
        result = self.run_test_runner(runner, test)
        lines = stream.getvalue().splitlines()
        self.assertEqual(['OK', "Missing feature 'Feature1' skipped 1 tests.", "Missing feature 'Feature2' skipped 1 tests."], lines[-3:])

    def test_verbose_test_count(self):
        if False:
            return 10
        'A verbose test run reports the right test count at the start'
        suite = TestUtil.TestSuite([unittest.FunctionTestCase(lambda : None), unittest.FunctionTestCase(lambda : None)])
        self.assertEqual(suite.countTestCases(), 2)
        stream = StringIO()
        runner = tests.TextTestRunner(stream=stream, verbosity=2)
        result = self.run_test_runner(runner, tests.CountingDecorator(suite))
        self.assertStartsWith(stream.getvalue(), 'running 2 tests')

    def test_startTestRun(self):
        if False:
            while True:
                i = 10
        'run should call result.startTestRun()'
        calls = []

        class LoggingDecorator(ExtendedToOriginalDecorator):

            def startTestRun(self):
                if False:
                    return 10
                ExtendedToOriginalDecorator.startTestRun(self)
                calls.append('startTestRun')
        test = unittest.FunctionTestCase(lambda : None)
        stream = StringIO()
        runner = tests.TextTestRunner(stream=stream, result_decorators=[LoggingDecorator])
        result = self.run_test_runner(runner, test)
        self.assertLength(1, calls)

    def test_stopTestRun(self):
        if False:
            print('Hello World!')
        'run should call result.stopTestRun()'
        calls = []

        class LoggingDecorator(ExtendedToOriginalDecorator):

            def stopTestRun(self):
                if False:
                    print('Hello World!')
                ExtendedToOriginalDecorator.stopTestRun(self)
                calls.append('stopTestRun')
        test = unittest.FunctionTestCase(lambda : None)
        stream = StringIO()
        runner = tests.TextTestRunner(stream=stream, result_decorators=[LoggingDecorator])
        result = self.run_test_runner(runner, test)
        self.assertLength(1, calls)

    def test_unicode_test_output_on_ascii_stream(self):
        if False:
            return 10
        'Showing results should always succeed even on an ascii console'

        class FailureWithUnicode(tests.TestCase):

            def test_log_unicode(self):
                if False:
                    i = 10
                    return i + 15
                self.log(u'☆')
                self.fail('Now print that log!')
        out = StringIO()
        self.overrideAttr(osutils, 'get_terminal_encoding', lambda trace=False: 'ascii')
        result = self.run_test_runner(tests.TextTestRunner(stream=out), FailureWithUnicode('test_log_unicode'))
        self.assertContainsRe(out.getvalue(), '(?:Text attachment: )?log(?:\n-+\n|: {{{)\\d+\\.\\d+  \\\\u2606(?:\n-+\n|}}}\n)')

class SampleTestCase(tests.TestCase):

    def _test_pass(self):
        if False:
            while True:
                i = 10
        pass

class _TestException(Exception):
    pass

class TestTestCase(tests.TestCase):
    """Tests that test the core bzrlib TestCase."""

    def test_assertLength_matches_empty(self):
        if False:
            i = 10
            return i + 15
        a_list = []
        self.assertLength(0, a_list)

    def test_assertLength_matches_nonempty(self):
        if False:
            for i in range(10):
                print('nop')
        a_list = [1, 2, 3]
        self.assertLength(3, a_list)

    def test_assertLength_fails_different(self):
        if False:
            i = 10
            return i + 15
        a_list = []
        self.assertRaises(AssertionError, self.assertLength, 1, a_list)

    def test_assertLength_shows_sequence_in_failure(self):
        if False:
            print('Hello World!')
        a_list = [1, 2, 3]
        exception = self.assertRaises(AssertionError, self.assertLength, 2, a_list)
        self.assertEqual('Incorrect length: wanted 2, got 3 for [1, 2, 3]', exception.args[0])

    def test_base_setUp_not_called_causes_failure(self):
        if False:
            while True:
                i = 10

        class TestCaseWithBrokenSetUp(tests.TestCase):

            def setUp(self):
                if False:
                    print('Hello World!')
                pass

            def test_foo(self):
                if False:
                    i = 10
                    return i + 15
                pass
        test = TestCaseWithBrokenSetUp('test_foo')
        result = unittest.TestResult()
        test.run(result)
        self.assertFalse(result.wasSuccessful())
        self.assertEqual(1, result.testsRun)

    def test_base_tearDown_not_called_causes_failure(self):
        if False:
            print('Hello World!')

        class TestCaseWithBrokenTearDown(tests.TestCase):

            def tearDown(self):
                if False:
                    i = 10
                    return i + 15
                pass

            def test_foo(self):
                if False:
                    for i in range(10):
                        print('nop')
                pass
        test = TestCaseWithBrokenTearDown('test_foo')
        result = unittest.TestResult()
        test.run(result)
        self.assertFalse(result.wasSuccessful())
        self.assertEqual(1, result.testsRun)

    def test_debug_flags_sanitised(self):
        if False:
            return 10
        'The bzrlib debug flags should be sanitised by setUp.'
        if 'allow_debug' in tests.selftest_debug_flags:
            raise tests.TestNotApplicable('-Eallow_debug option prevents debug flag sanitisation')
        flags = set()
        if self._lock_check_thorough:
            flags.add('strict_locks')
        self.assertEqual(flags, bzrlib.debug.debug_flags)

    def change_selftest_debug_flags(self, new_flags):
        if False:
            for i in range(10):
                print('nop')
        self.overrideAttr(tests, 'selftest_debug_flags', set(new_flags))

    def test_allow_debug_flag(self):
        if False:
            i = 10
            return i + 15
        'The -Eallow_debug flag prevents bzrlib.debug.debug_flags from being\n        sanitised (i.e. cleared) before running a test.\n        '
        self.change_selftest_debug_flags(set(['allow_debug']))
        bzrlib.debug.debug_flags = set(['a-flag'])

        class TestThatRecordsFlags(tests.TestCase):

            def test_foo(nested_self):
                if False:
                    print('Hello World!')
                self.flags = set(bzrlib.debug.debug_flags)
        test = TestThatRecordsFlags('test_foo')
        test.run(self.make_test_result())
        flags = set(['a-flag'])
        if 'disable_lock_checks' not in tests.selftest_debug_flags:
            flags.add('strict_locks')
        self.assertEqual(flags, self.flags)

    def test_disable_lock_checks(self):
        if False:
            for i in range(10):
                print('nop')
        'The -Edisable_lock_checks flag disables thorough checks.'

        class TestThatRecordsFlags(tests.TestCase):

            def test_foo(nested_self):
                if False:
                    while True:
                        i = 10
                self.flags = set(bzrlib.debug.debug_flags)
                self.test_lock_check_thorough = nested_self._lock_check_thorough
        self.change_selftest_debug_flags(set())
        test = TestThatRecordsFlags('test_foo')
        test.run(self.make_test_result())
        self.assertTrue(self.test_lock_check_thorough)
        self.assertEqual(set(['strict_locks']), self.flags)
        self.change_selftest_debug_flags(set(['disable_lock_checks']))
        test = TestThatRecordsFlags('test_foo')
        test.run(self.make_test_result())
        self.assertFalse(self.test_lock_check_thorough)
        self.assertEqual(set(), self.flags)

    def test_this_fails_strict_lock_check(self):
        if False:
            while True:
                i = 10

        class TestThatRecordsFlags(tests.TestCase):

            def test_foo(nested_self):
                if False:
                    print('Hello World!')
                self.flags1 = set(bzrlib.debug.debug_flags)
                self.thisFailsStrictLockCheck()
                self.flags2 = set(bzrlib.debug.debug_flags)
        self.change_selftest_debug_flags(set())
        test = TestThatRecordsFlags('test_foo')
        test.run(self.make_test_result())
        self.assertEqual(set(['strict_locks']), self.flags1)
        self.assertEqual(set(), self.flags2)

    def test_debug_flags_restored(self):
        if False:
            return 10
        'The bzrlib debug flags should be restored to their original state\n        after the test was run, even if allow_debug is set.\n        '
        self.change_selftest_debug_flags(set(['allow_debug']))
        bzrlib.debug.debug_flags = set(['original-state'])

        class TestThatModifiesFlags(tests.TestCase):

            def test_foo(self):
                if False:
                    i = 10
                    return i + 15
                bzrlib.debug.debug_flags = set(['modified'])
        test = TestThatModifiesFlags('test_foo')
        test.run(self.make_test_result())
        self.assertEqual(set(['original-state']), bzrlib.debug.debug_flags)

    def make_test_result(self):
        if False:
            i = 10
            return i + 15
        'Get a test result that writes to the test log file.'
        return tests.TextTestResult(self._log_file, descriptions=0, verbosity=1)

    def inner_test(self):
        if False:
            for i in range(10):
                print('nop')
        note('inner_test')

    def outer_child(self):
        if False:
            print('Hello World!')
        note('outer_start')
        self.inner_test = TestTestCase('inner_child')
        result = self.make_test_result()
        self.inner_test.run(result)
        note('outer finish')
        self.addCleanup(osutils.delete_any, self._log_file_name)

    def test_trace_nesting(self):
        if False:
            print('Hello World!')
        original_trace = bzrlib.trace._trace_file
        outer_test = TestTestCase('outer_child')
        result = self.make_test_result()
        outer_test.run(result)
        self.assertEqual(original_trace, bzrlib.trace._trace_file)

    def method_that_times_a_bit_twice(self):
        if False:
            while True:
                i = 10
        self.time(time.sleep, 0.007)
        self.time(time.sleep, 0.007)

    def test_time_creates_benchmark_in_result(self):
        if False:
            for i in range(10):
                print('nop')
        'Test that the TestCase.time() method accumulates a benchmark time.'
        sample_test = TestTestCase('method_that_times_a_bit_twice')
        output_stream = StringIO()
        result = bzrlib.tests.VerboseTestResult(output_stream, descriptions=0, verbosity=2)
        sample_test.run(result)
        self.assertContainsRe(output_stream.getvalue(), '\\d+ms\\*\\n$')

    def test_hooks_sanitised(self):
        if False:
            while True:
                i = 10
        'The bzrlib hooks should be sanitised by setUp.'
        self.assertEqual(bzrlib.branch.BranchHooks(), bzrlib.branch.Branch.hooks)
        self.assertEqual(bzrlib.smart.server.SmartServerHooks(), bzrlib.smart.server.SmartTCPServer.hooks)
        self.assertEqual(bzrlib.commands.CommandHooks(), bzrlib.commands.Command.hooks)

    def test__gather_lsprof_in_benchmarks(self):
        if False:
            print('Hello World!')
        'When _gather_lsprof_in_benchmarks is on, accumulate profile data.\n\n        Each self.time() call is individually and separately profiled.\n        '
        self.requireFeature(features.lsprof_feature)
        self._gather_lsprof_in_benchmarks = True
        self.time(time.sleep, 0.0)
        self.time(time.sleep, 0.003)
        self.assertEqual(2, len(self._benchcalls))
        self.assertEqual((time.sleep, (0.0,), {}), self._benchcalls[0][0])
        self.assertEqual((time.sleep, (0.003,), {}), self._benchcalls[1][0])
        self.assertIsInstance(self._benchcalls[0][1], bzrlib.lsprof.Stats)
        self.assertIsInstance(self._benchcalls[1][1], bzrlib.lsprof.Stats)
        del self._benchcalls[:]

    def test_knownFailure(self):
        if False:
            print('Hello World!')
        'Self.knownFailure() should raise a KnownFailure exception.'
        self.assertRaises(tests.KnownFailure, self.knownFailure, 'A Failure')

    def test_open_bzrdir_safe_roots(self):
        if False:
            while True:
                i = 10
        transport_server = memory.MemoryServer()
        transport_server.start_server()
        self.addCleanup(transport_server.stop_server)
        t = transport.get_transport_from_url(transport_server.get_url())
        controldir.ControlDir.create(t.base)
        self.assertRaises(errors.BzrError, controldir.ControlDir.open_from_transport, t)
        self.permit_url(t.base)
        self._bzr_selftest_roots.append(t.base)
        controldir.ControlDir.open_from_transport(t)

    def test_requireFeature_available(self):
        if False:
            i = 10
            return i + 15
        'self.requireFeature(available) is a no-op.'

        class Available(features.Feature):

            def _probe(self):
                if False:
                    return 10
                return True
        feature = Available()
        self.requireFeature(feature)

    def test_requireFeature_unavailable(self):
        if False:
            return 10
        'self.requireFeature(unavailable) raises UnavailableFeature.'

        class Unavailable(features.Feature):

            def _probe(self):
                if False:
                    for i in range(10):
                        print('nop')
                return False
        feature = Unavailable()
        self.assertRaises(tests.UnavailableFeature, self.requireFeature, feature)

    def test_run_no_parameters(self):
        if False:
            while True:
                i = 10
        test = SampleTestCase('_test_pass')
        test.run()

    def test_run_enabled_unittest_result(self):
        if False:
            i = 10
            return i + 15
        'Test we revert to regular behaviour when the test is enabled.'
        test = SampleTestCase('_test_pass')

        class EnabledFeature(object):

            def available(self):
                if False:
                    print('Hello World!')
                return True
        test._test_needs_features = [EnabledFeature()]
        result = unittest.TestResult()
        test.run(result)
        self.assertEqual(1, result.testsRun)
        self.assertEqual([], result.errors)
        self.assertEqual([], result.failures)

    def test_run_disabled_unittest_result(self):
        if False:
            return 10
        'Test our compatability for disabled tests with unittest results.'
        test = SampleTestCase('_test_pass')

        class DisabledFeature(object):

            def available(self):
                if False:
                    for i in range(10):
                        print('nop')
                return False
        test._test_needs_features = [DisabledFeature()]
        result = unittest.TestResult()
        test.run(result)
        self.assertEqual(1, result.testsRun)
        self.assertEqual([], result.errors)
        self.assertEqual([], result.failures)

    def test_run_disabled_supporting_result(self):
        if False:
            return 10
        'Test disabled tests behaviour with support aware results.'
        test = SampleTestCase('_test_pass')

        class DisabledFeature(object):

            def __eq__(self, other):
                if False:
                    while True:
                        i = 10
                return isinstance(other, DisabledFeature)

            def available(self):
                if False:
                    for i in range(10):
                        print('nop')
                return False
        the_feature = DisabledFeature()
        test._test_needs_features = [the_feature]

        class InstrumentedTestResult(unittest.TestResult):

            def __init__(self):
                if False:
                    return 10
                unittest.TestResult.__init__(self)
                self.calls = []

            def startTest(self, test):
                if False:
                    for i in range(10):
                        print('nop')
                self.calls.append(('startTest', test))

            def stopTest(self, test):
                if False:
                    i = 10
                    return i + 15
                self.calls.append(('stopTest', test))

            def addNotSupported(self, test, feature):
                if False:
                    i = 10
                    return i + 15
                self.calls.append(('addNotSupported', test, feature))
        result = InstrumentedTestResult()
        test.run(result)
        case = result.calls[0][1]
        self.assertEqual([('startTest', case), ('addNotSupported', case, the_feature), ('stopTest', case)], result.calls)

    def test_start_server_registers_url(self):
        if False:
            while True:
                i = 10
        transport_server = memory.MemoryServer()
        self.assertEqual([], self._bzr_selftest_roots)
        self.start_server(transport_server)
        self.assertSubset([transport_server.get_url()], self._bzr_selftest_roots)

    def test_assert_list_raises_on_generator(self):
        if False:
            return 10

        def generator_which_will_raise():
            if False:
                while True:
                    i = 10
            yield 1
            raise _TestException()
        e = self.assertListRaises(_TestException, generator_which_will_raise)
        self.assertIsInstance(e, _TestException)
        e = self.assertListRaises(Exception, generator_which_will_raise)
        self.assertIsInstance(e, _TestException)

    def test_assert_list_raises_on_plain(self):
        if False:
            i = 10
            return i + 15

        def plain_exception():
            if False:
                i = 10
                return i + 15
            raise _TestException()
            return []
        e = self.assertListRaises(_TestException, plain_exception)
        self.assertIsInstance(e, _TestException)
        e = self.assertListRaises(Exception, plain_exception)
        self.assertIsInstance(e, _TestException)

    def test_assert_list_raises_assert_wrong_exception(self):
        if False:
            while True:
                i = 10

        class _NotTestException(Exception):
            pass

        def wrong_exception():
            if False:
                for i in range(10):
                    print('nop')
            raise _NotTestException()

        def wrong_exception_generator():
            if False:
                while True:
                    i = 10
            yield 1
            yield 2
            raise _NotTestException()
        self.assertRaises(_NotTestException, self.assertListRaises, _TestException, wrong_exception)
        self.assertRaises(_NotTestException, self.assertListRaises, _TestException, wrong_exception_generator)

    def test_assert_list_raises_no_exception(self):
        if False:
            while True:
                i = 10

        def success():
            if False:
                for i in range(10):
                    print('nop')
            return []

        def success_generator():
            if False:
                while True:
                    i = 10
            yield 1
            yield 2
        self.assertRaises(AssertionError, self.assertListRaises, _TestException, success)
        self.assertRaises(AssertionError, self.assertListRaises, _TestException, success_generator)

    def _run_successful_test(self, test):
        if False:
            for i in range(10):
                print('nop')
        result = testtools.TestResult()
        test.run(result)
        self.assertTrue(result.wasSuccessful())
        return result

    def test_overrideAttr_without_value(self):
        if False:
            i = 10
            return i + 15
        self.test_attr = 'original'
        obj = self

        class Test(tests.TestCase):

            def setUp(self):
                if False:
                    print('Hello World!')
                super(Test, self).setUp()
                self.orig = self.overrideAttr(obj, 'test_attr')

            def test_value(self):
                if False:
                    while True:
                        i = 10
                self.assertEqual('original', self.orig)
                self.assertEqual('original', obj.test_attr)
                obj.test_attr = 'modified'
                self.assertEqual('modified', obj.test_attr)
        self._run_successful_test(Test('test_value'))
        self.assertEqual('original', obj.test_attr)

    def test_overrideAttr_with_value(self):
        if False:
            for i in range(10):
                print('nop')
        self.test_attr = 'original'
        obj = self

        class Test(tests.TestCase):

            def setUp(self):
                if False:
                    for i in range(10):
                        print('nop')
                super(Test, self).setUp()
                self.orig = self.overrideAttr(obj, 'test_attr', new='modified')

            def test_value(self):
                if False:
                    i = 10
                    return i + 15
                self.assertEqual('original', self.orig)
                self.assertEqual('modified', obj.test_attr)
        self._run_successful_test(Test('test_value'))
        self.assertEqual('original', obj.test_attr)

    def test_overrideAttr_with_no_existing_value_and_value(self):
        if False:
            i = 10
            return i + 15
        obj = self

        class Test(tests.TestCase):

            def setUp(self):
                if False:
                    i = 10
                    return i + 15
                tests.TestCase.setUp(self)
                self.orig = self.overrideAttr(obj, 'test_attr', new='modified')

            def test_value(self):
                if False:
                    i = 10
                    return i + 15
                self.assertEqual(tests._unitialized_attr, self.orig)
                self.assertEqual('modified', obj.test_attr)
        self._run_successful_test(Test('test_value'))
        self.assertRaises(AttributeError, getattr, obj, 'test_attr')

    def test_overrideAttr_with_no_existing_value_and_no_value(self):
        if False:
            while True:
                i = 10
        obj = self

        class Test(tests.TestCase):

            def setUp(self):
                if False:
                    print('Hello World!')
                tests.TestCase.setUp(self)
                self.orig = self.overrideAttr(obj, 'test_attr')

            def test_value(self):
                if False:
                    while True:
                        i = 10
                self.assertEqual(tests._unitialized_attr, self.orig)
                self.assertRaises(AttributeError, getattr, obj, 'test_attr')
        self._run_successful_test(Test('test_value'))
        self.assertRaises(AttributeError, getattr, obj, 'test_attr')

    def test_recordCalls(self):
        if False:
            for i in range(10):
                print('nop')
        from bzrlib.tests import test_selftest
        calls = self.recordCalls(test_selftest, '_add_numbers')
        self.assertEqual(test_selftest._add_numbers(2, 10), 12)
        self.assertEqual(calls, [((2, 10), {})])

def _add_numbers(a, b):
    if False:
        while True:
            i = 10
    return a + b

class _MissingFeature(features.Feature):

    def _probe(self):
        if False:
            return 10
        return False
missing_feature = _MissingFeature()

def _get_test(name):
    if False:
        for i in range(10):
            print('nop')
    "Get an instance of a specific example test.\n\n    We protect this in a function so that they don't auto-run in the test\n    suite.\n    "

    class ExampleTests(tests.TestCase):

        def test_fail(self):
            if False:
                while True:
                    i = 10
            mutter('this was a failing test')
            self.fail('this test will fail')

        def test_error(self):
            if False:
                return 10
            mutter('this test errored')
            raise RuntimeError('gotcha')

        def test_missing_feature(self):
            if False:
                for i in range(10):
                    print('nop')
            mutter('missing the feature')
            self.requireFeature(missing_feature)

        def test_skip(self):
            if False:
                for i in range(10):
                    print('nop')
            mutter('this test will be skipped')
            raise tests.TestSkipped('reason')

        def test_success(self):
            if False:
                for i in range(10):
                    print('nop')
            mutter('this test succeeds')

        def test_xfail(self):
            if False:
                while True:
                    i = 10
            mutter('test with expected failure')
            self.knownFailure('this_fails')

        def test_unexpected_success(self):
            if False:
                i = 10
                return i + 15
            mutter('test with unexpected success')
            self.expectFailure('should_fail', lambda : None)
    return ExampleTests(name)

class TestTestCaseLogDetails(tests.TestCase):

    def _run_test(self, test_name):
        if False:
            return 10
        test = _get_test(test_name)
        result = testtools.TestResult()
        test.run(result)
        return result

    def test_fail_has_log(self):
        if False:
            print('Hello World!')
        result = self._run_test('test_fail')
        self.assertEqual(1, len(result.failures))
        result_content = result.failures[0][1]
        self.assertContainsRe(result_content, '(?m)^(?:Text attachment: )?log(?:$|: )')
        self.assertContainsRe(result_content, 'this was a failing test')

    def test_error_has_log(self):
        if False:
            while True:
                i = 10
        result = self._run_test('test_error')
        self.assertEqual(1, len(result.errors))
        result_content = result.errors[0][1]
        self.assertContainsRe(result_content, '(?m)^(?:Text attachment: )?log(?:$|: )')
        self.assertContainsRe(result_content, 'this test errored')

    def test_skip_has_no_log(self):
        if False:
            print('Hello World!')
        result = self._run_test('test_skip')
        self.assertEqual(['reason'], result.skip_reasons.keys())
        skips = result.skip_reasons['reason']
        self.assertEqual(1, len(skips))
        test = skips[0]
        self.assertFalse('log' in test.getDetails())

    def test_missing_feature_has_no_log(self):
        if False:
            while True:
                i = 10
        result = self._run_test('test_missing_feature')
        self.assertEqual([missing_feature], result.skip_reasons.keys())
        skips = result.skip_reasons[missing_feature]
        self.assertEqual(1, len(skips))
        test = skips[0]
        self.assertFalse('log' in test.getDetails())

    def test_xfail_has_no_log(self):
        if False:
            i = 10
            return i + 15
        result = self._run_test('test_xfail')
        self.assertEqual(1, len(result.expectedFailures))
        result_content = result.expectedFailures[0][1]
        self.assertNotContainsRe(result_content, '(?m)^(?:Text attachment: )?log(?:$|: )')
        self.assertNotContainsRe(result_content, 'test with expected failure')

    def test_unexpected_success_has_log(self):
        if False:
            i = 10
            return i + 15
        result = self._run_test('test_unexpected_success')
        self.assertEqual(1, len(result.unexpectedSuccesses))
        test = result.unexpectedSuccesses[0]
        details = test.getDetails()
        self.assertTrue('log' in details)

class TestTestCloning(tests.TestCase):
    """Tests that test cloning of TestCases (as used by multiply_tests)."""

    def test_cloned_testcase_does_not_share_details(self):
        if False:
            print('Hello World!')
        'A TestCase cloned with clone_test does not share mutable attributes\n        such as details or cleanups.\n        '

        class Test(tests.TestCase):

            def test_foo(self):
                if False:
                    i = 10
                    return i + 15
                self.addDetail('foo', Content('text/plain', lambda : 'foo'))
        orig_test = Test('test_foo')
        cloned_test = tests.clone_test(orig_test, orig_test.id() + '(cloned)')
        orig_test.run(unittest.TestResult())
        self.assertEqual('foo', orig_test.getDetails()['foo'].iter_bytes())
        self.assertEqual(None, cloned_test.getDetails().get('foo'))

    def test_double_apply_scenario_preserves_first_scenario(self):
        if False:
            for i in range(10):
                print('nop')
        'Applying two levels of scenarios to a test preserves the attributes\n        added by both scenarios.\n        '

        class Test(tests.TestCase):

            def test_foo(self):
                if False:
                    while True:
                        i = 10
                pass
        test = Test('test_foo')
        scenarios_x = [('x=1', {'x': 1}), ('x=2', {'x': 2})]
        scenarios_y = [('y=1', {'y': 1}), ('y=2', {'y': 2})]
        suite = tests.multiply_tests(test, scenarios_x, unittest.TestSuite())
        suite = tests.multiply_tests(suite, scenarios_y, unittest.TestSuite())
        all_tests = list(tests.iter_suite_tests(suite))
        self.assertLength(4, all_tests)
        all_xys = sorted(((t.x, t.y) for t in all_tests))
        self.assertEqual([(1, 1), (1, 2), (2, 1), (2, 2)], all_xys)

@deprecated_function(deprecated_in((0, 11, 0)))
def sample_deprecated_function():
    if False:
        while True:
            i = 10
    'A deprecated function to test applyDeprecated with.'
    return 2

def sample_undeprecated_function(a_param):
    if False:
        while True:
            i = 10
    'A undeprecated function to test applyDeprecated with.'

class ApplyDeprecatedHelper(object):
    """A helper class for ApplyDeprecated tests."""

    @deprecated_method(deprecated_in((0, 11, 0)))
    def sample_deprecated_method(self, param_one):
        if False:
            print('Hello World!')
        'A deprecated method for testing with.'
        return param_one

    def sample_normal_method(self):
        if False:
            return 10
        'A undeprecated method.'

    @deprecated_method(deprecated_in((0, 10, 0)))
    def sample_nested_deprecation(self):
        if False:
            while True:
                i = 10
        return sample_deprecated_function()

class TestExtraAssertions(tests.TestCase):
    """Tests for new test assertions in bzrlib test suite"""

    def test_assert_isinstance(self):
        if False:
            print('Hello World!')
        self.assertIsInstance(2, int)
        self.assertIsInstance(u'', basestring)
        e = self.assertRaises(AssertionError, self.assertIsInstance, None, int)
        self.assertEqual(str(e), "None is an instance of <type 'NoneType'> rather than <type 'int'>")
        self.assertRaises(AssertionError, self.assertIsInstance, 23.3, int)
        e = self.assertRaises(AssertionError, self.assertIsInstance, None, int, "it's just not")
        self.assertEqual(str(e), "None is an instance of <type 'NoneType'> rather than <type 'int'>: it's just not")

    def test_assertEndsWith(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEndsWith('foo', 'oo')
        self.assertRaises(AssertionError, self.assertEndsWith, 'o', 'oo')

    def test_assertEqualDiff(self):
        if False:
            for i in range(10):
                print('nop')
        e = self.assertRaises(AssertionError, self.assertEqualDiff, '', '\n')
        self.assertEqual(str(e), 'first string is missing a final newline.\n+ \n')
        e = self.assertRaises(AssertionError, self.assertEqualDiff, '\n', '')
        self.assertEqual(str(e), 'second string is missing a final newline.\n- \n')

class TestDeprecations(tests.TestCase):

    def test_applyDeprecated_not_deprecated(self):
        if False:
            print('Hello World!')
        sample_object = ApplyDeprecatedHelper()
        self.assertRaises(AssertionError, self.applyDeprecated, deprecated_in((0, 11, 0)), sample_object.sample_normal_method)
        self.assertRaises(AssertionError, self.applyDeprecated, deprecated_in((0, 11, 0)), sample_undeprecated_function, 'a param value')
        self.assertRaises(AssertionError, self.applyDeprecated, deprecated_in((0, 10, 0)), sample_object.sample_deprecated_method, 'a param value')
        self.assertRaises(AssertionError, self.applyDeprecated, deprecated_in((0, 10, 0)), sample_deprecated_function)
        self.assertEqual('a param value', self.applyDeprecated(deprecated_in((0, 11, 0)), sample_object.sample_deprecated_method, 'a param value'))
        self.assertEqual(2, self.applyDeprecated(deprecated_in((0, 11, 0)), sample_deprecated_function))
        self.assertRaises(AssertionError, self.applyDeprecated, deprecated_in((0, 11, 0)), sample_object.sample_nested_deprecation)
        self.assertEqual(2, self.applyDeprecated(deprecated_in((0, 10, 0)), sample_object.sample_nested_deprecation))

    def test_callDeprecated(self):
        if False:
            print('Hello World!')

        def testfunc(be_deprecated, result=None):
            if False:
                while True:
                    i = 10
            if be_deprecated is True:
                symbol_versioning.warn('i am deprecated', DeprecationWarning, stacklevel=1)
            return result
        result = self.callDeprecated(['i am deprecated'], testfunc, True)
        self.assertIs(None, result)
        result = self.callDeprecated([], testfunc, False, 'result')
        self.assertEqual('result', result)
        self.callDeprecated(['i am deprecated'], testfunc, be_deprecated=True)
        self.callDeprecated([], testfunc, be_deprecated=False)

class TestWarningTests(tests.TestCase):
    """Tests for calling methods that raise warnings."""

    def test_callCatchWarnings(self):
        if False:
            print('Hello World!')

        def meth(a, b):
            if False:
                print('Hello World!')
            warnings.warn('this is your last warning')
            return a + b
        (wlist, result) = self.callCatchWarnings(meth, 1, 2)
        self.assertEqual(3, result)
        (w0,) = wlist
        self.assertIsInstance(w0, UserWarning)
        self.assertEqual('this is your last warning', str(w0))

class TestConvenienceMakers(tests.TestCaseWithTransport):
    """Test for the make_* convenience functions."""

    def test_make_branch_and_tree_with_format(self):
        if False:
            i = 10
            return i + 15
        self.make_branch_and_tree('a', format=bzrlib.bzrdir.BzrDirMetaFormat1())
        self.assertIsInstance(bzrlib.controldir.ControlDir.open('a')._format, bzrlib.bzrdir.BzrDirMetaFormat1)

    def test_make_branch_and_memory_tree(self):
        if False:
            while True:
                i = 10
        tree = self.make_branch_and_memory_tree('a')
        self.assertIsInstance(tree, bzrlib.memorytree.MemoryTree)

    def test_make_tree_for_local_vfs_backed_transport(self):
        if False:
            for i in range(10):
                print('nop')
        self.transport_server = test_server.FakeVFATServer
        self.assertFalse(self.get_url('t1').startswith('file://'))
        tree = self.make_branch_and_tree('t1')
        base = tree.bzrdir.root_transport.base
        self.assertStartsWith(base, 'file://')
        self.assertEqual(tree.bzrdir.root_transport, tree.branch.bzrdir.root_transport)
        self.assertEqual(tree.bzrdir.root_transport, tree.branch.repository.bzrdir.root_transport)

class SelfTestHelper(object):

    def run_selftest(self, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Run selftest returning its output.'
        output = StringIO()
        old_transport = bzrlib.tests.default_transport
        old_root = tests.TestCaseWithMemoryTransport.TEST_ROOT
        tests.TestCaseWithMemoryTransport.TEST_ROOT = None
        try:
            self.assertEqual(True, tests.selftest(stream=output, **kwargs))
        finally:
            bzrlib.tests.default_transport = old_transport
            tests.TestCaseWithMemoryTransport.TEST_ROOT = old_root
        output.seek(0)
        return output

class TestSelftest(tests.TestCase, SelfTestHelper):
    """Tests of bzrlib.tests.selftest."""

    def test_selftest_benchmark_parameter_invokes_test_suite__benchmark__(self):
        if False:
            print('Hello World!')
        factory_called = []

        def factory():
            if False:
                return 10
            factory_called.append(True)
            return TestUtil.TestSuite()
        out = StringIO()
        err = StringIO()
        self.apply_redirected(out, err, None, bzrlib.tests.selftest, test_suite_factory=factory)
        self.assertEqual([True], factory_called)

    def factory(self):
        if False:
            for i in range(10):
                print('nop')
        'A test suite factory.'

        class Test(tests.TestCase):

            def a(self):
                if False:
                    while True:
                        i = 10
                pass

            def b(self):
                if False:
                    print('Hello World!')
                pass

            def c(self):
                if False:
                    while True:
                        i = 10
                pass
        return TestUtil.TestSuite([Test('a'), Test('b'), Test('c')])

    def test_list_only(self):
        if False:
            for i in range(10):
                print('nop')
        output = self.run_selftest(test_suite_factory=self.factory, list_only=True)
        self.assertEqual(3, len(output.readlines()))

    def test_list_only_filtered(self):
        if False:
            print('Hello World!')
        output = self.run_selftest(test_suite_factory=self.factory, list_only=True, pattern='Test.b')
        self.assertEndsWith(output.getvalue(), 'Test.b\n')
        self.assertLength(1, output.readlines())

    def test_list_only_excludes(self):
        if False:
            while True:
                i = 10
        output = self.run_selftest(test_suite_factory=self.factory, list_only=True, exclude_pattern='Test.b')
        self.assertNotContainsRe('Test.b', output.getvalue())
        self.assertLength(2, output.readlines())

    def test_lsprof_tests(self):
        if False:
            print('Hello World!')
        self.requireFeature(features.lsprof_feature)
        results = []

        class Test(object):

            def __call__(test, result):
                if False:
                    i = 10
                    return i + 15
                test.run(result)

            def run(test, result):
                if False:
                    return 10
                results.append(result)

            def countTestCases(self):
                if False:
                    print('Hello World!')
                return 1
        self.run_selftest(test_suite_factory=Test, lsprof_tests=True)
        self.assertLength(1, results)
        self.assertIsInstance(results.pop(), ExtendedToOriginalDecorator)

    def test_random(self):
        if False:
            print('Hello World!')
        output_123 = self.run_selftest(test_suite_factory=self.factory, list_only=True, random_seed='123')
        output_234 = self.run_selftest(test_suite_factory=self.factory, list_only=True, random_seed='234')
        self.assertNotEqual(output_123, output_234)
        self.assertLength(5, output_123.readlines())
        self.assertLength(5, output_234.readlines())

    def test_random_reuse_is_same_order(self):
        if False:
            while True:
                i = 10
        expected = self.run_selftest(test_suite_factory=self.factory, list_only=True, random_seed='123')
        repeated = self.run_selftest(test_suite_factory=self.factory, list_only=True, random_seed='123')
        self.assertEqual(expected.getvalue(), repeated.getvalue())

    def test_runner_class(self):
        if False:
            i = 10
            return i + 15
        self.requireFeature(features.subunit)
        from subunit import ProtocolTestCase
        stream = self.run_selftest(runner_class=tests.SubUnitBzrRunner, test_suite_factory=self.factory)
        test = ProtocolTestCase(stream)
        result = unittest.TestResult()
        test.run(result)
        self.assertEqual(3, result.testsRun)

    def test_starting_with_single_argument(self):
        if False:
            i = 10
            return i + 15
        output = self.run_selftest(test_suite_factory=self.factory, starting_with=['bzrlib.tests.test_selftest.Test.a'], list_only=True)
        self.assertEqual('bzrlib.tests.test_selftest.Test.a\n', output.getvalue())

    def test_starting_with_multiple_argument(self):
        if False:
            for i in range(10):
                print('nop')
        output = self.run_selftest(test_suite_factory=self.factory, starting_with=['bzrlib.tests.test_selftest.Test.a', 'bzrlib.tests.test_selftest.Test.b'], list_only=True)
        self.assertEqual('bzrlib.tests.test_selftest.Test.a\nbzrlib.tests.test_selftest.Test.b\n', output.getvalue())

    def check_transport_set(self, transport_server):
        if False:
            return 10
        captured_transport = []

        def seen_transport(a_transport):
            if False:
                for i in range(10):
                    print('nop')
            captured_transport.append(a_transport)

        class Capture(tests.TestCase):

            def a(self):
                if False:
                    for i in range(10):
                        print('nop')
                seen_transport(bzrlib.tests.default_transport)

        def factory():
            if False:
                print('Hello World!')
            return TestUtil.TestSuite([Capture('a')])
        self.run_selftest(transport=transport_server, test_suite_factory=factory)
        self.assertEqual(transport_server, captured_transport[0])

    def test_transport_sftp(self):
        if False:
            i = 10
            return i + 15
        self.requireFeature(features.paramiko)
        from bzrlib.tests import stub_sftp
        self.check_transport_set(stub_sftp.SFTPAbsoluteServer)

    def test_transport_memory(self):
        if False:
            i = 10
            return i + 15
        self.check_transport_set(memory.MemoryServer)

class TestSelftestWithIdList(tests.TestCaseInTempDir, SelfTestHelper):

    def test_load_list(self):
        if False:
            print('Hello World!')
        test_id_line = '%s\n' % self.id()
        self.build_tree_contents([('test.list', test_id_line)])
        stream = self.run_selftest(load_list='test.list', list_only=True)
        self.assertEqual(test_id_line, stream.getvalue())

    def test_load_unknown(self):
        if False:
            i = 10
            return i + 15
        err = self.assertRaises(errors.NoSuchFile, self.run_selftest, load_list='missing file name', list_only=True)

class TestSubunitLogDetails(tests.TestCase, SelfTestHelper):
    _test_needs_features = [features.subunit]

    def run_subunit_stream(self, test_name):
        if False:
            for i in range(10):
                print('nop')
        from subunit import ProtocolTestCase

        def factory():
            if False:
                i = 10
                return i + 15
            return TestUtil.TestSuite([_get_test(test_name)])
        stream = self.run_selftest(runner_class=tests.SubUnitBzrRunner, test_suite_factory=factory)
        test = ProtocolTestCase(stream)
        result = testtools.TestResult()
        test.run(result)
        content = stream.getvalue()
        return (content, result)

    def test_fail_has_log(self):
        if False:
            print('Hello World!')
        (content, result) = self.run_subunit_stream('test_fail')
        self.assertEqual(1, len(result.failures))
        self.assertContainsRe(content, '(?m)^log$')
        self.assertContainsRe(content, 'this test will fail')

    def test_error_has_log(self):
        if False:
            for i in range(10):
                print('nop')
        (content, result) = self.run_subunit_stream('test_error')
        self.assertContainsRe(content, '(?m)^log$')
        self.assertContainsRe(content, 'this test errored')

    def test_skip_has_no_log(self):
        if False:
            for i in range(10):
                print('nop')
        (content, result) = self.run_subunit_stream('test_skip')
        self.assertNotContainsRe(content, '(?m)^log$')
        self.assertNotContainsRe(content, 'this test will be skipped')
        self.assertEqual(['reason'], result.skip_reasons.keys())
        skips = result.skip_reasons['reason']
        self.assertEqual(1, len(skips))
        test = skips[0]

    def test_missing_feature_has_no_log(self):
        if False:
            return 10
        (content, result) = self.run_subunit_stream('test_missing_feature')
        self.assertNotContainsRe(content, '(?m)^log$')
        self.assertNotContainsRe(content, 'missing the feature')
        self.assertEqual(['_MissingFeature\n'], result.skip_reasons.keys())
        skips = result.skip_reasons['_MissingFeature\n']
        self.assertEqual(1, len(skips))
        test = skips[0]

    def test_xfail_has_no_log(self):
        if False:
            print('Hello World!')
        (content, result) = self.run_subunit_stream('test_xfail')
        self.assertNotContainsRe(content, '(?m)^log$')
        self.assertNotContainsRe(content, 'test with expected failure')
        self.assertEqual(1, len(result.expectedFailures))
        result_content = result.expectedFailures[0][1]
        self.assertNotContainsRe(result_content, '(?m)^(?:Text attachment: )?log(?:$|: )')
        self.assertNotContainsRe(result_content, 'test with expected failure')

    def test_unexpected_success_has_log(self):
        if False:
            for i in range(10):
                print('nop')
        (content, result) = self.run_subunit_stream('test_unexpected_success')
        self.assertContainsRe(content, '(?m)^log$')
        self.assertContainsRe(content, 'test with unexpected success')
        from subunit import TestProtocolClient as _Client
        if _Client.addUnexpectedSuccess.im_func is _Client.addSuccess.im_func:
            self.expectFailure('subunit treats "unexpectedSuccess" as a plain success', self.assertEqual, 1, len(result.unexpectedSuccesses))
        self.assertEqual(1, len(result.unexpectedSuccesses))
        test = result.unexpectedSuccesses[0]

    def test_success_has_no_log(self):
        if False:
            for i in range(10):
                print('nop')
        (content, result) = self.run_subunit_stream('test_success')
        self.assertEqual(1, result.testsRun)
        self.assertNotContainsRe(content, '(?m)^log$')
        self.assertNotContainsRe(content, 'this test succeeds')

class TestRunBzr(tests.TestCase):
    out = ''
    err = ''

    def _run_bzr_core(self, argv, retcode=0, encoding=None, stdin=None, working_dir=None):
        if False:
            print('Hello World!')
        "Override _run_bzr_core to test how it is invoked by run_bzr.\n\n        Attempts to run bzr from inside this class don't actually run it.\n\n        We test how run_bzr actually invokes bzr in another location.  Here we\n        only need to test that it passes the right parameters to run_bzr.\n        "
        self.argv = list(argv)
        self.retcode = retcode
        self.encoding = encoding
        self.stdin = stdin
        self.working_dir = working_dir
        return (self.retcode, self.out, self.err)

    def test_run_bzr_error(self):
        if False:
            for i in range(10):
                print('nop')
        self.out = 'It sure does!\n'
        (out, err) = self.run_bzr_error(['^$'], ['rocks'], retcode=34)
        self.assertEqual(['rocks'], self.argv)
        self.assertEqual(34, self.retcode)
        self.assertEqual('It sure does!\n', out)
        self.assertEqual(out, self.out)
        self.assertEqual('', err)
        self.assertEqual(err, self.err)

    def test_run_bzr_error_regexes(self):
        if False:
            while True:
                i = 10
        self.out = ''
        self.err = 'bzr: ERROR: foobarbaz is not versioned'
        (out, err) = self.run_bzr_error(['bzr: ERROR: foobarbaz is not versioned'], ['file-id', 'foobarbaz'])

    def test_encoding(self):
        if False:
            for i in range(10):
                print('nop')
        'Test that run_bzr passes encoding to _run_bzr_core'
        self.run_bzr('foo bar')
        self.assertEqual(None, self.encoding)
        self.assertEqual(['foo', 'bar'], self.argv)
        self.run_bzr('foo bar', encoding='baz')
        self.assertEqual('baz', self.encoding)
        self.assertEqual(['foo', 'bar'], self.argv)

    def test_retcode(self):
        if False:
            return 10
        'Test that run_bzr passes retcode to _run_bzr_core'
        self.run_bzr('foo bar')
        self.assertEqual(0, self.retcode)
        self.assertEqual(['foo', 'bar'], self.argv)
        self.run_bzr('foo bar', retcode=1)
        self.assertEqual(1, self.retcode)
        self.assertEqual(['foo', 'bar'], self.argv)
        self.run_bzr('foo bar', retcode=None)
        self.assertEqual(None, self.retcode)
        self.assertEqual(['foo', 'bar'], self.argv)
        self.run_bzr(['foo', 'bar'], retcode=3)
        self.assertEqual(3, self.retcode)
        self.assertEqual(['foo', 'bar'], self.argv)

    def test_stdin(self):
        if False:
            for i in range(10):
                print('nop')
        self.run_bzr('foo bar', stdin='gam')
        self.assertEqual('gam', self.stdin)
        self.assertEqual(['foo', 'bar'], self.argv)
        self.run_bzr('foo bar', stdin='zippy')
        self.assertEqual('zippy', self.stdin)
        self.assertEqual(['foo', 'bar'], self.argv)

    def test_working_dir(self):
        if False:
            print('Hello World!')
        'Test that run_bzr passes working_dir to _run_bzr_core'
        self.run_bzr('foo bar')
        self.assertEqual(None, self.working_dir)
        self.assertEqual(['foo', 'bar'], self.argv)
        self.run_bzr('foo bar', working_dir='baz')
        self.assertEqual('baz', self.working_dir)
        self.assertEqual(['foo', 'bar'], self.argv)

    def test_reject_extra_keyword_arguments(self):
        if False:
            return 10
        self.assertRaises(TypeError, self.run_bzr, 'foo bar', error_regex=['error message'])

class TestRunBzrCaptured(tests.TestCaseWithTransport):

    def apply_redirected(self, stdin=None, stdout=None, stderr=None, a_callable=None, *args, **kwargs):
        if False:
            while True:
                i = 10
        self.stdin = stdin
        self.factory_stdin = getattr(bzrlib.ui.ui_factory, 'stdin', None)
        self.factory = bzrlib.ui.ui_factory
        self.working_dir = osutils.getcwd()
        stdout.write('foo\n')
        stderr.write('bar\n')
        return 0

    def test_stdin(self):
        if False:
            i = 10
            return i + 15
        self.run_bzr(['foo', 'bar'], stdin='gam')
        self.assertEqual('gam', self.stdin.read())
        self.assertTrue(self.stdin is self.factory_stdin)
        self.run_bzr(['foo', 'bar'], stdin='zippy')
        self.assertEqual('zippy', self.stdin.read())
        self.assertTrue(self.stdin is self.factory_stdin)

    def test_ui_factory(self):
        if False:
            for i in range(10):
                print('nop')
        current_factory = bzrlib.ui.ui_factory
        self.run_bzr(['foo'])
        self.assertFalse(current_factory is self.factory)
        self.assertNotEqual(sys.stdout, self.factory.stdout)
        self.assertNotEqual(sys.stderr, self.factory.stderr)
        self.assertEqual('foo\n', self.factory.stdout.getvalue())
        self.assertEqual('bar\n', self.factory.stderr.getvalue())
        self.assertIsInstance(self.factory, tests.TestUIFactory)

    def test_working_dir(self):
        if False:
            i = 10
            return i + 15
        self.build_tree(['one/', 'two/'])
        cwd = osutils.getcwd()
        self.run_bzr(['foo', 'bar'])
        self.assertEqual(cwd, self.working_dir)
        self.run_bzr(['foo', 'bar'], working_dir=None)
        self.assertEqual(cwd, self.working_dir)
        self.run_bzr(['foo', 'bar'], working_dir='one')
        self.assertNotEqual(cwd, self.working_dir)
        self.assertEndsWith(self.working_dir, 'one')
        self.assertEqual(cwd, osutils.getcwd())
        self.run_bzr(['foo', 'bar'], working_dir='two')
        self.assertNotEqual(cwd, self.working_dir)
        self.assertEndsWith(self.working_dir, 'two')
        self.assertEqual(cwd, osutils.getcwd())

class StubProcess(object):
    """A stub process for testing run_bzr_subprocess."""

    def __init__(self, out='', err='', retcode=0):
        if False:
            for i in range(10):
                print('nop')
        self.out = out
        self.err = err
        self.returncode = retcode

    def communicate(self):
        if False:
            print('Hello World!')
        return (self.out, self.err)

class TestWithFakedStartBzrSubprocess(tests.TestCaseWithTransport):
    """Base class for tests testing how we might run bzr."""

    def setUp(self):
        if False:
            return 10
        super(TestWithFakedStartBzrSubprocess, self).setUp()
        self.subprocess_calls = []

    def start_bzr_subprocess(self, process_args, env_changes=None, skip_if_plan_to_signal=False, working_dir=None, allow_plugins=False):
        if False:
            return 10
        'capture what run_bzr_subprocess tries to do.'
        self.subprocess_calls.append({'process_args': process_args, 'env_changes': env_changes, 'skip_if_plan_to_signal': skip_if_plan_to_signal, 'working_dir': working_dir, 'allow_plugins': allow_plugins})
        return self.next_subprocess

class TestRunBzrSubprocess(TestWithFakedStartBzrSubprocess):

    def assertRunBzrSubprocess(self, expected_args, process, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        'Run run_bzr_subprocess with args and kwargs using a stubbed process.\n\n        Inside TestRunBzrSubprocessCommands we use a stub start_bzr_subprocess\n        that will return static results. This assertion method populates those\n        results and also checks the arguments run_bzr_subprocess generates.\n        '
        self.next_subprocess = process
        try:
            result = self.run_bzr_subprocess(*args, **kwargs)
        except:
            self.next_subprocess = None
            for (key, expected) in expected_args.iteritems():
                self.assertEqual(expected, self.subprocess_calls[-1][key])
            raise
        else:
            self.next_subprocess = None
            for (key, expected) in expected_args.iteritems():
                self.assertEqual(expected, self.subprocess_calls[-1][key])
            return result

    def test_run_bzr_subprocess(self):
        if False:
            i = 10
            return i + 15
        'The run_bzr_helper_external command behaves nicely.'
        self.assertRunBzrSubprocess({'process_args': ['--version']}, StubProcess(), '--version')
        self.assertRunBzrSubprocess({'process_args': ['--version']}, StubProcess(), ['--version'])
        result = self.assertRunBzrSubprocess({}, StubProcess(retcode=3), '--version', retcode=None)
        result = self.assertRunBzrSubprocess({}, StubProcess(out='is free software'), '--version')
        self.assertContainsRe(result[0], 'is free software')
        self.assertRaises(AssertionError, self.assertRunBzrSubprocess, {'process_args': ['--versionn']}, StubProcess(retcode=3), '--versionn')
        result = self.assertRunBzrSubprocess({}, StubProcess(retcode=3), '--versionn', retcode=3)
        result = self.assertRunBzrSubprocess({}, StubProcess(err='unknown command', retcode=3), '--versionn', retcode=None)
        self.assertContainsRe(result[1], 'unknown command')

    def test_env_change_passes_through(self):
        if False:
            print('Hello World!')
        self.assertRunBzrSubprocess({'env_changes': {'new': 'value', 'changed': 'newvalue', 'deleted': None}}, StubProcess(), '', env_changes={'new': 'value', 'changed': 'newvalue', 'deleted': None})

    def test_no_working_dir_passed_as_None(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertRunBzrSubprocess({'working_dir': None}, StubProcess(), '')

    def test_no_working_dir_passed_through(self):
        if False:
            print('Hello World!')
        self.assertRunBzrSubprocess({'working_dir': 'dir'}, StubProcess(), '', working_dir='dir')

    def test_run_bzr_subprocess_no_plugins(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertRunBzrSubprocess({'allow_plugins': False}, StubProcess(), '')

    def test_allow_plugins(self):
        if False:
            i = 10
            return i + 15
        self.assertRunBzrSubprocess({'allow_plugins': True}, StubProcess(), '', allow_plugins=True)

class TestFinishBzrSubprocess(TestWithFakedStartBzrSubprocess):

    def test_finish_bzr_subprocess_with_error(self):
        if False:
            while True:
                i = 10
        'finish_bzr_subprocess allows specification of the desired exit code.\n        '
        process = StubProcess(err='unknown command', retcode=3)
        result = self.finish_bzr_subprocess(process, retcode=3)
        self.assertEqual('', result[0])
        self.assertContainsRe(result[1], 'unknown command')

    def test_finish_bzr_subprocess_ignoring_retcode(self):
        if False:
            print('Hello World!')
        'finish_bzr_subprocess allows the exit code to be ignored.'
        process = StubProcess(err='unknown command', retcode=3)
        result = self.finish_bzr_subprocess(process, retcode=None)
        self.assertEqual('', result[0])
        self.assertContainsRe(result[1], 'unknown command')

    def test_finish_subprocess_with_unexpected_retcode(self):
        if False:
            print('Hello World!')
        'finish_bzr_subprocess raises self.failureException if the retcode is\n        not the expected one.\n        '
        process = StubProcess(err='unknown command', retcode=3)
        self.assertRaises(self.failureException, self.finish_bzr_subprocess, process)

class _DontSpawnProcess(Exception):
    """A simple exception which just allows us to skip unnecessary steps"""

class TestStartBzrSubProcess(tests.TestCase):
    """Stub test start_bzr_subprocess."""

    def _subprocess_log_cleanup(self):
        if False:
            while True:
                i = 10
        "Inhibits the base version as we don't produce a log file."

    def _popen(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        'Override the base version to record the command that is run.\n\n        From there we can ensure it is correct without spawning a real process.\n        '
        self.check_popen_state()
        self._popen_args = args
        self._popen_kwargs = kwargs
        raise _DontSpawnProcess()

    def check_popen_state(self):
        if False:
            print('Hello World!')
        'Replace to make assertions when popen is called.'

    def test_run_bzr_subprocess_no_plugins(self):
        if False:
            while True:
                i = 10
        self.assertRaises(_DontSpawnProcess, self.start_bzr_subprocess, [])
        command = self._popen_args[0]
        self.assertEqual(sys.executable, command[0])
        self.assertEqual(self.get_bzr_path(), command[1])
        self.assertEqual(['--no-plugins'], command[2:])

    def test_allow_plugins(self):
        if False:
            i = 10
            return i + 15
        self.assertRaises(_DontSpawnProcess, self.start_bzr_subprocess, [], allow_plugins=True)
        command = self._popen_args[0]
        self.assertEqual([], command[2:])

    def test_set_env(self):
        if False:
            print('Hello World!')
        self.assertFalse('EXISTANT_ENV_VAR' in os.environ)

        def check_environment():
            if False:
                while True:
                    i = 10
            self.assertEqual('set variable', os.environ['EXISTANT_ENV_VAR'])
        self.check_popen_state = check_environment
        self.assertRaises(_DontSpawnProcess, self.start_bzr_subprocess, [], env_changes={'EXISTANT_ENV_VAR': 'set variable'})
        self.assertFalse('EXISTANT_ENV_VAR' in os.environ)

    def test_run_bzr_subprocess_env_del(self):
        if False:
            print('Hello World!')
        'run_bzr_subprocess can remove environment variables too.'
        self.assertFalse('EXISTANT_ENV_VAR' in os.environ)

        def check_environment():
            if False:
                while True:
                    i = 10
            self.assertFalse('EXISTANT_ENV_VAR' in os.environ)
        os.environ['EXISTANT_ENV_VAR'] = 'set variable'
        self.check_popen_state = check_environment
        self.assertRaises(_DontSpawnProcess, self.start_bzr_subprocess, [], env_changes={'EXISTANT_ENV_VAR': None})
        self.assertEqual('set variable', os.environ['EXISTANT_ENV_VAR'])
        del os.environ['EXISTANT_ENV_VAR']

    def test_env_del_missing(self):
        if False:
            print('Hello World!')
        self.assertFalse('NON_EXISTANT_ENV_VAR' in os.environ)

        def check_environment():
            if False:
                for i in range(10):
                    print('nop')
            self.assertFalse('NON_EXISTANT_ENV_VAR' in os.environ)
        self.check_popen_state = check_environment
        self.assertRaises(_DontSpawnProcess, self.start_bzr_subprocess, [], env_changes={'NON_EXISTANT_ENV_VAR': None})

    def test_working_dir(self):
        if False:
            for i in range(10):
                print('nop')
        'Test that we can specify the working dir for the child'
        orig_getcwd = osutils.getcwd
        orig_chdir = os.chdir
        chdirs = []

        def chdir(path):
            if False:
                print('Hello World!')
            chdirs.append(path)
        self.overrideAttr(os, 'chdir', chdir)

        def getcwd():
            if False:
                return 10
            return 'current'
        self.overrideAttr(osutils, 'getcwd', getcwd)
        self.assertRaises(_DontSpawnProcess, self.start_bzr_subprocess, [], working_dir='foo')
        self.assertEqual(['foo', 'current'], chdirs)

    def test_get_bzr_path_with_cwd_bzrlib(self):
        if False:
            while True:
                i = 10
        self.get_source_path = lambda : ''
        self.overrideAttr(os.path, 'isfile', lambda path: True)
        self.assertEqual(self.get_bzr_path(), 'bzr')

class TestActuallyStartBzrSubprocess(tests.TestCaseWithTransport):
    """Tests that really need to do things with an external bzr."""

    def test_start_and_stop_bzr_subprocess_send_signal(self):
        if False:
            return 10
        'finish_bzr_subprocess raises self.failureException if the retcode is\n        not the expected one.\n        '
        self.disable_missing_extensions_warning()
        process = self.start_bzr_subprocess(['wait-until-signalled'], skip_if_plan_to_signal=True)
        self.assertEqual('running\n', process.stdout.readline())
        result = self.finish_bzr_subprocess(process, send_signal=signal.SIGINT, retcode=3)
        self.assertEqual('', result[0])
        self.assertEqual('bzr: interrupted\n', result[1])

class TestSelftestFiltering(tests.TestCase):

    def setUp(self):
        if False:
            return 10
        super(TestSelftestFiltering, self).setUp()
        self.suite = TestUtil.TestSuite()
        self.loader = TestUtil.TestLoader()
        self.suite.addTest(self.loader.loadTestsFromModule(sys.modules['bzrlib.tests.test_selftest']))
        self.all_names = _test_ids(self.suite)

    def test_condition_id_re(self):
        if False:
            print('Hello World!')
        test_name = 'bzrlib.tests.test_selftest.TestSelftestFiltering.test_condition_id_re'
        filtered_suite = tests.filter_suite_by_condition(self.suite, tests.condition_id_re('test_condition_id_re'))
        self.assertEqual([test_name], _test_ids(filtered_suite))

    def test_condition_id_in_list(self):
        if False:
            return 10
        test_names = ['bzrlib.tests.test_selftest.TestSelftestFiltering.test_condition_id_in_list']
        id_list = tests.TestIdList(test_names)
        filtered_suite = tests.filter_suite_by_condition(self.suite, tests.condition_id_in_list(id_list))
        my_pattern = 'TestSelftestFiltering.*test_condition_id_in_list'
        re_filtered = tests.filter_suite_by_re(self.suite, my_pattern)
        self.assertEqual(_test_ids(re_filtered), _test_ids(filtered_suite))

    def test_condition_id_startswith(self):
        if False:
            return 10
        klass = 'bzrlib.tests.test_selftest.TestSelftestFiltering.'
        start1 = klass + 'test_condition_id_starts'
        start2 = klass + 'test_condition_id_in'
        test_names = [klass + 'test_condition_id_in_list', klass + 'test_condition_id_startswith']
        filtered_suite = tests.filter_suite_by_condition(self.suite, tests.condition_id_startswith([start1, start2]))
        self.assertEqual(test_names, _test_ids(filtered_suite))

    def test_condition_isinstance(self):
        if False:
            for i in range(10):
                print('nop')
        filtered_suite = tests.filter_suite_by_condition(self.suite, tests.condition_isinstance(self.__class__))
        class_pattern = 'bzrlib.tests.test_selftest.TestSelftestFiltering.'
        re_filtered = tests.filter_suite_by_re(self.suite, class_pattern)
        self.assertEqual(_test_ids(re_filtered), _test_ids(filtered_suite))

    def test_exclude_tests_by_condition(self):
        if False:
            while True:
                i = 10
        excluded_name = 'bzrlib.tests.test_selftest.TestSelftestFiltering.test_exclude_tests_by_condition'
        filtered_suite = tests.exclude_tests_by_condition(self.suite, lambda x: x.id() == excluded_name)
        self.assertEqual(len(self.all_names) - 1, filtered_suite.countTestCases())
        self.assertFalse(excluded_name in _test_ids(filtered_suite))
        remaining_names = list(self.all_names)
        remaining_names.remove(excluded_name)
        self.assertEqual(remaining_names, _test_ids(filtered_suite))

    def test_exclude_tests_by_re(self):
        if False:
            i = 10
            return i + 15
        self.all_names = _test_ids(self.suite)
        filtered_suite = tests.exclude_tests_by_re(self.suite, 'exclude_tests_by_re')
        excluded_name = 'bzrlib.tests.test_selftest.TestSelftestFiltering.test_exclude_tests_by_re'
        self.assertEqual(len(self.all_names) - 1, filtered_suite.countTestCases())
        self.assertFalse(excluded_name in _test_ids(filtered_suite))
        remaining_names = list(self.all_names)
        remaining_names.remove(excluded_name)
        self.assertEqual(remaining_names, _test_ids(filtered_suite))

    def test_filter_suite_by_condition(self):
        if False:
            while True:
                i = 10
        test_name = 'bzrlib.tests.test_selftest.TestSelftestFiltering.test_filter_suite_by_condition'
        filtered_suite = tests.filter_suite_by_condition(self.suite, lambda x: x.id() == test_name)
        self.assertEqual([test_name], _test_ids(filtered_suite))

    def test_filter_suite_by_re(self):
        if False:
            i = 10
            return i + 15
        filtered_suite = tests.filter_suite_by_re(self.suite, 'test_filter_suite_by_r')
        filtered_names = _test_ids(filtered_suite)
        self.assertEqual(filtered_names, ['bzrlib.tests.test_selftest.TestSelftestFiltering.test_filter_suite_by_re'])

    def test_filter_suite_by_id_list(self):
        if False:
            print('Hello World!')
        test_list = ['bzrlib.tests.test_selftest.TestSelftestFiltering.test_filter_suite_by_id_list']
        filtered_suite = tests.filter_suite_by_id_list(self.suite, tests.TestIdList(test_list))
        filtered_names = _test_ids(filtered_suite)
        self.assertEqual(filtered_names, ['bzrlib.tests.test_selftest.TestSelftestFiltering.test_filter_suite_by_id_list'])

    def test_filter_suite_by_id_startswith(self):
        if False:
            print('Hello World!')
        klass = 'bzrlib.tests.test_selftest.TestSelftestFiltering.'
        start1 = klass + 'test_filter_suite_by_id_starts'
        start2 = klass + 'test_filter_suite_by_id_li'
        test_list = [klass + 'test_filter_suite_by_id_list', klass + 'test_filter_suite_by_id_startswith']
        filtered_suite = tests.filter_suite_by_id_startswith(self.suite, [start1, start2])
        self.assertEqual(test_list, _test_ids(filtered_suite))

    def test_preserve_input(self):
        if False:
            return 10
        self.assertTrue(self.suite is tests.preserve_input(self.suite))
        self.assertTrue('@#$' is tests.preserve_input('@#$'))

    def test_randomize_suite(self):
        if False:
            i = 10
            return i + 15
        randomized_suite = tests.randomize_suite(self.suite)
        self.assertEqual(set(_test_ids(self.suite)), set(_test_ids(randomized_suite)))
        self.assertNotEqual(self.all_names, _test_ids(randomized_suite))
        self.assertEqual(len(self.all_names), len(_test_ids(randomized_suite)))

    def test_split_suit_by_condition(self):
        if False:
            while True:
                i = 10
        self.all_names = _test_ids(self.suite)
        condition = tests.condition_id_re('test_filter_suite_by_r')
        split_suite = tests.split_suite_by_condition(self.suite, condition)
        filtered_name = 'bzrlib.tests.test_selftest.TestSelftestFiltering.test_filter_suite_by_re'
        self.assertEqual([filtered_name], _test_ids(split_suite[0]))
        self.assertFalse(filtered_name in _test_ids(split_suite[1]))
        remaining_names = list(self.all_names)
        remaining_names.remove(filtered_name)
        self.assertEqual(remaining_names, _test_ids(split_suite[1]))

    def test_split_suit_by_re(self):
        if False:
            while True:
                i = 10
        self.all_names = _test_ids(self.suite)
        split_suite = tests.split_suite_by_re(self.suite, 'test_filter_suite_by_r')
        filtered_name = 'bzrlib.tests.test_selftest.TestSelftestFiltering.test_filter_suite_by_re'
        self.assertEqual([filtered_name], _test_ids(split_suite[0]))
        self.assertFalse(filtered_name in _test_ids(split_suite[1]))
        remaining_names = list(self.all_names)
        remaining_names.remove(filtered_name)
        self.assertEqual(remaining_names, _test_ids(split_suite[1]))

class TestCheckTreeShape(tests.TestCaseWithTransport):

    def test_check_tree_shape(self):
        if False:
            while True:
                i = 10
        files = ['a', 'b/', 'b/c']
        tree = self.make_branch_and_tree('.')
        self.build_tree(files)
        tree.add(files)
        tree.lock_read()
        try:
            self.check_tree_shape(tree, files)
        finally:
            tree.unlock()

class TestBlackboxSupport(tests.TestCase):
    """Tests for testsuite blackbox features."""

    def test_run_bzr_failure_not_caught(self):
        if False:
            print('Hello World!')
        e = self.assertRaises(AssertionError, self.run_bzr, ['assert-fail'])
        self.assertEqual('always fails', str(e))
        self.assertNotContainsRe(self.get_log(), 'Traceback')

    def test_run_bzr_user_error_caught(self):
        if False:
            return 10
        transport_server = memory.MemoryServer()
        transport_server.start_server()
        self.addCleanup(transport_server.stop_server)
        url = transport_server.get_url()
        self.permit_url(url)
        (out, err) = self.run_bzr(['log', '%s/nonexistantpath' % url], retcode=3)
        self.assertEqual(out, '')
        self.assertContainsRe(err, 'bzr: ERROR: Not a branch: ".*nonexistantpath/".\n')

class TestTestLoader(tests.TestCase):
    """Tests for the test loader."""

    def _get_loader_and_module(self):
        if False:
            i = 10
            return i + 15
        'Gets a TestLoader and a module with one test in it.'
        loader = TestUtil.TestLoader()
        module = {}

        class Stub(tests.TestCase):

            def test_foo(self):
                if False:
                    return 10
                pass

        class MyModule(object):
            pass
        MyModule.a_class = Stub
        module = MyModule()
        return (loader, module)

    def test_module_no_load_tests_attribute_loads_classes(self):
        if False:
            for i in range(10):
                print('nop')
        (loader, module) = self._get_loader_and_module()
        self.assertEqual(1, loader.loadTestsFromModule(module).countTestCases())

    def test_module_load_tests_attribute_gets_called(self):
        if False:
            while True:
                i = 10
        (loader, module) = self._get_loader_and_module()

        def load_tests(self, standard_tests, module, loader):
            if False:
                return 10
            result = loader.suiteClass()
            for test in tests.iter_suite_tests(standard_tests):
                result.addTests([test, test])
            return result
        module.__class__.load_tests = load_tests
        self.assertEqual(2, loader.loadTestsFromModule(module).countTestCases())

    def test_load_tests_from_module_name_smoke_test(self):
        if False:
            for i in range(10):
                print('nop')
        loader = TestUtil.TestLoader()
        suite = loader.loadTestsFromModuleName('bzrlib.tests.test_sampler')
        self.assertEqual(['bzrlib.tests.test_sampler.DemoTest.test_nothing'], _test_ids(suite))

    def test_load_tests_from_module_name_with_bogus_module_name(self):
        if False:
            i = 10
            return i + 15
        loader = TestUtil.TestLoader()
        self.assertRaises(ImportError, loader.loadTestsFromModuleName, 'bogus')

class TestTestIdList(tests.TestCase):

    def _create_id_list(self, test_list):
        if False:
            i = 10
            return i + 15
        return tests.TestIdList(test_list)

    def _create_suite(self, test_id_list):
        if False:
            i = 10
            return i + 15

        class Stub(tests.TestCase):

            def test_foo(self):
                if False:
                    i = 10
                    return i + 15
                pass

        def _create_test_id(id):
            if False:
                while True:
                    i = 10
            return lambda : id
        suite = TestUtil.TestSuite()
        for id in test_id_list:
            t = Stub('test_foo')
            t.id = _create_test_id(id)
            suite.addTest(t)
        return suite

    def _test_ids(self, test_suite):
        if False:
            print('Hello World!')
        'Get the ids for the tests in a test suite.'
        return [t.id() for t in tests.iter_suite_tests(test_suite)]

    def test_empty_list(self):
        if False:
            return 10
        id_list = self._create_id_list([])
        self.assertEqual({}, id_list.tests)
        self.assertEqual({}, id_list.modules)

    def test_valid_list(self):
        if False:
            print('Hello World!')
        id_list = self._create_id_list(['mod1.cl1.meth1', 'mod1.cl1.meth2', 'mod1.func1', 'mod1.cl2.meth2', 'mod1.submod1', 'mod1.submod2.cl1.meth1', 'mod1.submod2.cl2.meth2'])
        self.assertTrue(id_list.refers_to('mod1'))
        self.assertTrue(id_list.refers_to('mod1.submod1'))
        self.assertTrue(id_list.refers_to('mod1.submod2'))
        self.assertTrue(id_list.includes('mod1.cl1.meth1'))
        self.assertTrue(id_list.includes('mod1.submod1'))
        self.assertTrue(id_list.includes('mod1.func1'))

    def test_bad_chars_in_params(self):
        if False:
            return 10
        id_list = self._create_id_list(['mod1.cl1.meth1(xx.yy)'])
        self.assertTrue(id_list.refers_to('mod1'))
        self.assertTrue(id_list.includes('mod1.cl1.meth1(xx.yy)'))

    def test_module_used(self):
        if False:
            return 10
        id_list = self._create_id_list(['mod.class.meth'])
        self.assertTrue(id_list.refers_to('mod'))
        self.assertTrue(id_list.refers_to('mod.class'))
        self.assertTrue(id_list.refers_to('mod.class.meth'))

    def test_test_suite_matches_id_list_with_unknown(self):
        if False:
            print('Hello World!')
        loader = TestUtil.TestLoader()
        suite = loader.loadTestsFromModuleName('bzrlib.tests.test_sampler')
        test_list = ['bzrlib.tests.test_sampler.DemoTest.test_nothing', 'bogus']
        (not_found, duplicates) = tests.suite_matches_id_list(suite, test_list)
        self.assertEqual(['bogus'], not_found)
        self.assertEqual([], duplicates)

    def test_suite_matches_id_list_with_duplicates(self):
        if False:
            i = 10
            return i + 15
        loader = TestUtil.TestLoader()
        suite = loader.loadTestsFromModuleName('bzrlib.tests.test_sampler')
        dupes = loader.suiteClass()
        for test in tests.iter_suite_tests(suite):
            dupes.addTest(test)
            dupes.addTest(test)
        test_list = ['bzrlib.tests.test_sampler.DemoTest.test_nothing']
        (not_found, duplicates) = tests.suite_matches_id_list(dupes, test_list)
        self.assertEqual([], not_found)
        self.assertEqual(['bzrlib.tests.test_sampler.DemoTest.test_nothing'], duplicates)

class TestTestSuite(tests.TestCase):

    def test__test_suite_testmod_names(self):
        if False:
            for i in range(10):
                print('nop')
        test_list = tests._test_suite_testmod_names()
        self.assertSubset(['bzrlib.tests.blackbox', 'bzrlib.tests.per_transport', 'bzrlib.tests.test_selftest'], test_list)

    def test__test_suite_modules_to_doctest(self):
        if False:
            while True:
                i = 10
        test_list = tests._test_suite_modules_to_doctest()
        if __doc__ is None:
            self.assertEqual([], test_list)
            return
        self.assertSubset(['bzrlib.timestamp'], test_list)

    def test_test_suite(self):
        if False:
            return 10
        calls = []

        def testmod_names():
            if False:
                return 10
            calls.append('testmod_names')
            return ['bzrlib.tests.blackbox.test_branch', 'bzrlib.tests.per_transport', 'bzrlib.tests.test_selftest']
        self.overrideAttr(tests, '_test_suite_testmod_names', testmod_names)

        def doctests():
            if False:
                i = 10
                return i + 15
            calls.append('modules_to_doctest')
            if __doc__ is None:
                return []
            return ['bzrlib.timestamp']
        self.overrideAttr(tests, '_test_suite_modules_to_doctest', doctests)
        expected_test_list = ['bzrlib.tests.blackbox.test_branch.TestBranch.test_branch', 'bzrlib.tests.per_transport.TransportTests.test_abspath(LocalTransport,LocalURLServer)', 'bzrlib.tests.test_selftest.TestTestSuite.test_test_suite']
        if __doc__ is not None:
            expected_test_list.extend(['bzrlib.timestamp.format_highres_date'])
        suite = tests.test_suite()
        self.assertEqual(set(['testmod_names', 'modules_to_doctest']), set(calls))
        self.assertSubset(expected_test_list, _test_ids(suite))

    def test_test_suite_list_and_start(self):
        if False:
            i = 10
            return i + 15
        test_list = ['bzrlib.tests.test_selftest.TestTestSuite.test_test_suite']
        suite = tests.test_suite(test_list, ['bzrlib.tests.test_selftest.TestTestSuite'])
        self.assertEqual(test_list, _test_ids(suite))

class TestLoadTestIdList(tests.TestCaseInTempDir):

    def _create_test_list_file(self, file_name, content):
        if False:
            while True:
                i = 10
        fl = open(file_name, 'wt')
        fl.write(content)
        fl.close()

    def test_load_unknown(self):
        if False:
            i = 10
            return i + 15
        self.assertRaises(errors.NoSuchFile, tests.load_test_id_list, 'i_do_not_exist')

    def test_load_test_list(self):
        if False:
            print('Hello World!')
        test_list_fname = 'test.list'
        self._create_test_list_file(test_list_fname, 'mod1.cl1.meth1\nmod2.cl2.meth2\n')
        tlist = tests.load_test_id_list(test_list_fname)
        self.assertEqual(2, len(tlist))
        self.assertEqual('mod1.cl1.meth1', tlist[0])
        self.assertEqual('mod2.cl2.meth2', tlist[1])

    def test_load_dirty_file(self):
        if False:
            while True:
                i = 10
        test_list_fname = 'test.list'
        self._create_test_list_file(test_list_fname, '  mod1.cl1.meth1\n\nmod2.cl2.meth2  \nbar baz\n')
        tlist = tests.load_test_id_list(test_list_fname)
        self.assertEqual(4, len(tlist))
        self.assertEqual('mod1.cl1.meth1', tlist[0])
        self.assertEqual('', tlist[1])
        self.assertEqual('mod2.cl2.meth2', tlist[2])
        self.assertEqual('bar baz', tlist[3])

class TestFilteredByModuleTestLoader(tests.TestCase):

    def _create_loader(self, test_list):
        if False:
            for i in range(10):
                print('nop')
        id_filter = tests.TestIdList(test_list)
        loader = TestUtil.FilteredByModuleTestLoader(id_filter.refers_to)
        return loader

    def test_load_tests(self):
        if False:
            while True:
                i = 10
        test_list = ['bzrlib.tests.test_sampler.DemoTest.test_nothing']
        loader = self._create_loader(test_list)
        suite = loader.loadTestsFromModuleName('bzrlib.tests.test_sampler')
        self.assertEqual(test_list, _test_ids(suite))

    def test_exclude_tests(self):
        if False:
            return 10
        test_list = ['bogus']
        loader = self._create_loader(test_list)
        suite = loader.loadTestsFromModuleName('bzrlib.tests.test_sampler')
        self.assertEqual([], _test_ids(suite))

class TestFilteredByNameStartTestLoader(tests.TestCase):

    def _create_loader(self, name_start):
        if False:
            for i in range(10):
                print('nop')

        def needs_module(name):
            if False:
                i = 10
                return i + 15
            return name.startswith(name_start) or name_start.startswith(name)
        loader = TestUtil.FilteredByModuleTestLoader(needs_module)
        return loader

    def test_load_tests(self):
        if False:
            while True:
                i = 10
        test_list = ['bzrlib.tests.test_sampler.DemoTest.test_nothing']
        loader = self._create_loader('bzrlib.tests.test_samp')
        suite = loader.loadTestsFromModuleName('bzrlib.tests.test_sampler')
        self.assertEqual(test_list, _test_ids(suite))

    def test_load_tests_inside_module(self):
        if False:
            print('Hello World!')
        test_list = ['bzrlib.tests.test_sampler.DemoTest.test_nothing']
        loader = self._create_loader('bzrlib.tests.test_sampler.Demo')
        suite = loader.loadTestsFromModuleName('bzrlib.tests.test_sampler')
        self.assertEqual(test_list, _test_ids(suite))

    def test_exclude_tests(self):
        if False:
            return 10
        test_list = ['bogus']
        loader = self._create_loader('bogus')
        suite = loader.loadTestsFromModuleName('bzrlib.tests.test_sampler')
        self.assertEqual([], _test_ids(suite))

class TestTestPrefixRegistry(tests.TestCase):

    def _get_registry(self):
        if False:
            for i in range(10):
                print('nop')
        tp_registry = tests.TestPrefixAliasRegistry()
        return tp_registry

    def test_register_new_prefix(self):
        if False:
            return 10
        tpr = self._get_registry()
        tpr.register('foo', 'fff.ooo.ooo')
        self.assertEqual('fff.ooo.ooo', tpr.get('foo'))

    def test_register_existing_prefix(self):
        if False:
            while True:
                i = 10
        tpr = self._get_registry()
        tpr.register('bar', 'bbb.aaa.rrr')
        tpr.register('bar', 'bBB.aAA.rRR')
        self.assertEqual('bbb.aaa.rrr', tpr.get('bar'))
        self.assertThat(self.get_log(), DocTestMatches('...bar...bbb.aaa.rrr...BB.aAA.rRR', doctest.ELLIPSIS))

    def test_get_unknown_prefix(self):
        if False:
            print('Hello World!')
        tpr = self._get_registry()
        self.assertRaises(KeyError, tpr.get, 'I am not a prefix')

    def test_resolve_prefix(self):
        if False:
            return 10
        tpr = self._get_registry()
        tpr.register('bar', 'bb.aa.rr')
        self.assertEqual('bb.aa.rr', tpr.resolve_alias('bar'))

    def test_resolve_unknown_alias(self):
        if False:
            for i in range(10):
                print('nop')
        tpr = self._get_registry()
        self.assertRaises(errors.BzrCommandError, tpr.resolve_alias, 'I am not a prefix')

    def test_predefined_prefixes(self):
        if False:
            i = 10
            return i + 15
        tpr = tests.test_prefix_alias_registry
        self.assertEqual('bzrlib', tpr.resolve_alias('bzrlib'))
        self.assertEqual('bzrlib.doc', tpr.resolve_alias('bd'))
        self.assertEqual('bzrlib.utils', tpr.resolve_alias('bu'))
        self.assertEqual('bzrlib.tests', tpr.resolve_alias('bt'))
        self.assertEqual('bzrlib.tests.blackbox', tpr.resolve_alias('bb'))
        self.assertEqual('bzrlib.plugins', tpr.resolve_alias('bp'))

class TestThreadLeakDetection(tests.TestCase):
    """Ensure when tests leak threads we detect and report it"""

    class LeakRecordingResult(tests.ExtendedTestResult):

        def __init__(self):
            if False:
                for i in range(10):
                    print('nop')
            tests.ExtendedTestResult.__init__(self, StringIO(), 0, 1)
            self.leaks = []

        def _report_thread_leak(self, test, leaks, alive):
            if False:
                print('Hello World!')
            self.leaks.append((test, leaks))

    def test_testcase_without_addCleanups(self):
        if False:
            while True:
                i = 10
        "Check old TestCase instances don't break with leak detection"

        class Test(unittest.TestCase):

            def runTest(self):
                if False:
                    for i in range(10):
                        print('nop')
                pass
        result = self.LeakRecordingResult()
        test = Test()
        result.startTestRun()
        test.run(result)
        result.stopTestRun()
        self.assertEqual(result._tests_leaking_threads_count, 0)
        self.assertEqual(result.leaks, [])

    def test_thread_leak(self):
        if False:
            print('Hello World!')
        "Ensure a thread that outlives the running of a test is reported\n\n        Uses a thread that blocks on an event, and is started by the inner\n        test case. As the thread outlives the inner case's run, it should be\n        detected as a leak, but the event is then set so that the thread can\n        be safely joined in cleanup so it's not leaked for real.\n        "
        event = threading.Event()
        thread = threading.Thread(name='Leaker', target=event.wait)

        class Test(tests.TestCase):

            def test_leak(self):
                if False:
                    i = 10
                    return i + 15
                thread.start()
        result = self.LeakRecordingResult()
        test = Test('test_leak')
        self.addCleanup(thread.join)
        self.addCleanup(event.set)
        result.startTestRun()
        test.run(result)
        result.stopTestRun()
        self.assertEqual(result._tests_leaking_threads_count, 1)
        self.assertEqual(result._first_thread_leaker_id, test.id())
        self.assertEqual(result.leaks, [(test, set([thread]))])
        self.assertContainsString(result.stream.getvalue(), 'leaking threads')

    def test_multiple_leaks(self):
        if False:
            print('Hello World!')
        "Check multiple leaks are blamed on the test cases at fault\n\n        Same concept as the previous test, but has one inner test method that\n        leaks two threads, and one that doesn't leak at all.\n        "
        event = threading.Event()
        thread_a = threading.Thread(name='LeakerA', target=event.wait)
        thread_b = threading.Thread(name='LeakerB', target=event.wait)
        thread_c = threading.Thread(name='LeakerC', target=event.wait)

        class Test(tests.TestCase):

            def test_first_leak(self):
                if False:
                    for i in range(10):
                        print('nop')
                thread_b.start()

            def test_second_no_leak(self):
                if False:
                    print('Hello World!')
                pass

            def test_third_leak(self):
                if False:
                    while True:
                        i = 10
                thread_c.start()
                thread_a.start()
        result = self.LeakRecordingResult()
        first_test = Test('test_first_leak')
        third_test = Test('test_third_leak')
        self.addCleanup(thread_a.join)
        self.addCleanup(thread_b.join)
        self.addCleanup(thread_c.join)
        self.addCleanup(event.set)
        result.startTestRun()
        unittest.TestSuite([first_test, Test('test_second_no_leak'), third_test]).run(result)
        result.stopTestRun()
        self.assertEqual(result._tests_leaking_threads_count, 2)
        self.assertEqual(result._first_thread_leaker_id, first_test.id())
        self.assertEqual(result.leaks, [(first_test, set([thread_b])), (third_test, set([thread_a, thread_c]))])
        self.assertContainsString(result.stream.getvalue(), 'leaking threads')

class TestPostMortemDebugging(tests.TestCase):
    """Check post mortem debugging works when tests fail or error"""

    class TracebackRecordingResult(tests.ExtendedTestResult):

        def __init__(self):
            if False:
                return 10
            tests.ExtendedTestResult.__init__(self, StringIO(), 0, 1)
            self.postcode = None

        def _post_mortem(self, tb=None):
            if False:
                for i in range(10):
                    print('nop')
            'Record the code object at the end of the current traceback'
            tb = tb or sys.exc_info()[2]
            if tb is not None:
                next = tb.tb_next
                while next is not None:
                    tb = next
                    next = next.tb_next
                self.postcode = tb.tb_frame.f_code

        def report_error(self, test, err):
            if False:
                return 10
            pass

        def report_failure(self, test, err):
            if False:
                i = 10
                return i + 15
            pass

    def test_location_unittest_error(self):
        if False:
            print('Hello World!')
        'Needs right post mortem traceback with erroring unittest case'

        class Test(unittest.TestCase):

            def runTest(self):
                if False:
                    while True:
                        i = 10
                raise RuntimeError
        result = self.TracebackRecordingResult()
        Test().run(result)
        self.assertEqual(result.postcode, Test.runTest.func_code)

    def test_location_unittest_failure(self):
        if False:
            i = 10
            return i + 15
        'Needs right post mortem traceback with failing unittest case'

        class Test(unittest.TestCase):

            def runTest(self):
                if False:
                    for i in range(10):
                        print('nop')
                raise self.failureException
        result = self.TracebackRecordingResult()
        Test().run(result)
        self.assertEqual(result.postcode, Test.runTest.func_code)

    def test_location_bt_error(self):
        if False:
            print('Hello World!')
        'Needs right post mortem traceback with erroring bzrlib.tests case'

        class Test(tests.TestCase):

            def test_error(self):
                if False:
                    i = 10
                    return i + 15
                raise RuntimeError
        result = self.TracebackRecordingResult()
        Test('test_error').run(result)
        self.assertEqual(result.postcode, Test.test_error.func_code)

    def test_location_bt_failure(self):
        if False:
            while True:
                i = 10
        'Needs right post mortem traceback with failing bzrlib.tests case'

        class Test(tests.TestCase):

            def test_failure(self):
                if False:
                    for i in range(10):
                        print('nop')
                raise self.failureException
        result = self.TracebackRecordingResult()
        Test('test_failure').run(result)
        self.assertEqual(result.postcode, Test.test_failure.func_code)

    def test_env_var_triggers_post_mortem(self):
        if False:
            while True:
                i = 10
        'Check pdb.post_mortem is called iff BZR_TEST_PDB is set'
        import pdb
        result = tests.ExtendedTestResult(StringIO(), 0, 1)
        post_mortem_calls = []
        self.overrideAttr(pdb, 'post_mortem', post_mortem_calls.append)
        self.overrideEnv('BZR_TEST_PDB', None)
        result._post_mortem(1)
        self.overrideEnv('BZR_TEST_PDB', 'on')
        result._post_mortem(2)
        self.assertEqual([2], post_mortem_calls)

class TestRunSuite(tests.TestCase):

    def test_runner_class(self):
        if False:
            for i in range(10):
                print('nop')
        'run_suite accepts and uses a runner_class keyword argument.'

        class Stub(tests.TestCase):

            def test_foo(self):
                if False:
                    for i in range(10):
                        print('nop')
                pass
        suite = Stub('test_foo')
        calls = []

        class MyRunner(tests.TextTestRunner):

            def run(self, test):
                if False:
                    i = 10
                    return i + 15
                calls.append(test)
                return tests.ExtendedTestResult(self.stream, self.descriptions, self.verbosity)
        tests.run_suite(suite, runner_class=MyRunner, stream=StringIO())
        self.assertLength(1, calls)

class _Selftest(object):
    """Mixin for tests needing full selftest output"""

    def _inject_stream_into_subunit(self, stream):
        if False:
            i = 10
            return i + 15
        'To be overridden by subclasses that run tests out of process'

    def _run_selftest(self, **kwargs):
        if False:
            i = 10
            return i + 15
        sio = StringIO()
        self._inject_stream_into_subunit(sio)
        tests.selftest(stream=sio, stop_on_failure=False, **kwargs)
        return sio.getvalue()

class _ForkedSelftest(_Selftest):
    """Mixin for tests needing full selftest output with forked children"""
    _test_needs_features = [features.subunit]

    def _inject_stream_into_subunit(self, stream):
        if False:
            for i in range(10):
                print('nop')
        'Monkey-patch subunit so the extra output goes to stream not stdout\n\n        Some APIs need rewriting so this kind of bogus hackery can be replaced\n        by passing the stream param from run_tests down into ProtocolTestCase.\n        '
        from subunit import ProtocolTestCase
        _original_init = ProtocolTestCase.__init__

        def _init_with_passthrough(self, *args, **kwargs):
            if False:
                print('Hello World!')
            _original_init(self, *args, **kwargs)
            self._passthrough = stream
        self.overrideAttr(ProtocolTestCase, '__init__', _init_with_passthrough)

    def _run_selftest(self, **kwargs):
        if False:
            while True:
                i = 10
        if getattr(os, 'fork', None) is None:
            raise tests.TestNotApplicable("Platform doesn't support forking")
        self.overrideAttr(osutils, 'local_concurrency', lambda : 2)
        kwargs.setdefault('suite_decorators', []).append(tests.fork_decorator)
        return super(_ForkedSelftest, self)._run_selftest(**kwargs)

class TestParallelFork(_ForkedSelftest, tests.TestCase):
    """Check operation of --parallel=fork selftest option"""

    def test_error_in_child_during_fork(self):
        if False:
            for i in range(10):
                print('nop')
        'Error in a forked child during test setup should get reported'

        class Test(tests.TestCase):

            def testMethod(self):
                if False:
                    while True:
                        i = 10
                pass
        self.overrideAttr(tests, 'workaround_zealous_crypto_random', None)
        out = self._run_selftest(test_suite_factory=Test)
        self.assertContainsRe(out, 'Traceback.*:\n(?:.*\n)*.+ in fork_for_tests\n(?:.*\n)*\\s*workaround_zealous_crypto_random\\(\\)\n(?:.*\n)*TypeError:')

class TestUncollectedWarnings(_Selftest, tests.TestCase):
    """Check a test case still alive after being run emits a warning"""

    class Test(tests.TestCase):

        def test_pass(self):
            if False:
                i = 10
                return i + 15
            pass

        def test_self_ref(self):
            if False:
                i = 10
                return i + 15
            self.also_self = self.test_self_ref

        def test_skip(self):
            if False:
                while True:
                    i = 10
            self.skip("Don't need")

    def _get_suite(self):
        if False:
            return 10
        return TestUtil.TestSuite([self.Test('test_pass'), self.Test('test_self_ref'), self.Test('test_skip')])

    def _run_selftest_with_suite(self, **kwargs):
        if False:
            return 10
        old_flags = tests.selftest_debug_flags
        tests.selftest_debug_flags = old_flags.union(['uncollected_cases'])
        gc_on = gc.isenabled()
        if gc_on:
            gc.disable()
        try:
            output = self._run_selftest(test_suite_factory=self._get_suite, **kwargs)
        finally:
            if gc_on:
                gc.enable()
            tests.selftest_debug_flags = old_flags
        self.assertNotContainsRe(output, 'Uncollected test case.*test_pass')
        self.assertContainsRe(output, 'Uncollected test case.*test_self_ref')
        return output

    def test_testsuite(self):
        if False:
            while True:
                i = 10
        self._run_selftest_with_suite()

    def test_pattern(self):
        if False:
            print('Hello World!')
        out = self._run_selftest_with_suite(pattern='test_(?:pass|self_ref)$')
        self.assertNotContainsRe(out, 'test_skip')

    def test_exclude_pattern(self):
        if False:
            return 10
        out = self._run_selftest_with_suite(exclude_pattern='test_skip$')
        self.assertNotContainsRe(out, 'test_skip')

    def test_random_seed(self):
        if False:
            return 10
        self._run_selftest_with_suite(random_seed='now')

    def test_matching_tests_first(self):
        if False:
            for i in range(10):
                print('nop')
        self._run_selftest_with_suite(matching_tests_first=True, pattern='test_self_ref$')

    def test_starting_with_and_exclude(self):
        if False:
            for i in range(10):
                print('nop')
        out = self._run_selftest_with_suite(starting_with=['bt.'], exclude_pattern='test_skip$')
        self.assertNotContainsRe(out, 'test_skip')

    def test_additonal_decorator(self):
        if False:
            while True:
                i = 10
        out = self._run_selftest_with_suite(suite_decorators=[tests.TestDecorator])

class TestUncollectedWarningsSubunit(TestUncollectedWarnings):
    """Check warnings from tests staying alive are emitted with subunit"""
    _test_needs_features = [features.subunit]

    def _run_selftest_with_suite(self, **kwargs):
        if False:
            i = 10
            return i + 15
        return TestUncollectedWarnings._run_selftest_with_suite(self, runner_class=tests.SubUnitBzrRunner, **kwargs)

class TestUncollectedWarningsForked(_ForkedSelftest, TestUncollectedWarnings):
    """Check warnings from tests staying alive are emitted when forking"""

class TestEnvironHandling(tests.TestCase):

    def test_overrideEnv_None_called_twice_doesnt_leak(self):
        if False:
            print('Hello World!')
        self.assertFalse('MYVAR' in os.environ)
        self.overrideEnv('MYVAR', '42')

        class Test(tests.TestCase):

            def test_me(self):
                if False:
                    while True:
                        i = 10
                self.overrideEnv('MYVAR', None)
                self.assertEqual(None, os.environ.get('MYVAR'))
                self.overrideEnv('MYVAR', None)
                self.assertEqual(None, os.environ.get('MYVAR'))
        output = StringIO()
        result = tests.TextTestResult(output, 0, 1)
        Test('test_me').run(result)
        if not result.wasStrictlySuccessful():
            self.fail(output.getvalue())
        self.assertEqual('42', os.environ.get('MYVAR'))

class TestIsolatedEnv(tests.TestCase):
    """Test isolating tests from os.environ.

    Since we use tests that are already isolated from os.environ a bit of care
    should be taken when designing the tests to avoid bootstrap side-effects.
    The tests start an already clean os.environ which allow doing valid
    assertions about which variables are present or not and design tests around
    these assertions.
    """

    class ScratchMonkey(tests.TestCase):

        def test_me(self):
            if False:
                i = 10
                return i + 15
            pass

    def test_basics(self):
        if False:
            i = 10
            return i + 15
        self.assertTrue('BZR_HOME' in tests.isolated_environ)
        self.assertEqual(None, tests.isolated_environ['BZR_HOME'])
        self.assertFalse('BZR_HOME' in os.environ)
        self.assertTrue('LINES' in tests.isolated_environ)
        self.assertEqual('25', tests.isolated_environ['LINES'])
        self.assertEqual('25', os.environ['LINES'])

    def test_injecting_unknown_variable(self):
        if False:
            return 10
        test = self.ScratchMonkey('test_me')
        tests.override_os_environ(test, {'BZR_HOME': 'foo'})
        self.assertEqual('foo', os.environ['BZR_HOME'])
        tests.restore_os_environ(test)
        self.assertFalse('BZR_HOME' in os.environ)

    def test_injecting_known_variable(self):
        if False:
            i = 10
            return i + 15
        test = self.ScratchMonkey('test_me')
        tests.override_os_environ(test, {'LINES': '42'})
        self.assertEqual('42', os.environ['LINES'])
        tests.restore_os_environ(test)
        self.assertEqual('25', os.environ['LINES'])

    def test_deleting_variable(self):
        if False:
            print('Hello World!')
        test = self.ScratchMonkey('test_me')
        tests.override_os_environ(test, {'LINES': None})
        self.assertTrue('LINES' not in os.environ)
        tests.restore_os_environ(test)
        self.assertEqual('25', os.environ['LINES'])

class TestDocTestSuiteIsolation(tests.TestCase):
    """Test that `tests.DocTestSuite` isolates doc tests from os.environ.

    Since tests.TestCase alreay provides an isolation from os.environ, we use
    the clean environment as a base for testing. To precisely capture the
    isolation provided by tests.DocTestSuite, we use doctest.DocTestSuite to
    compare against.

    We want to make sure `tests.DocTestSuite` respect `tests.isolated_environ`,
    not `os.environ` so each test overrides it to suit its needs.

    """

    def get_doctest_suite_for_string(self, klass, string):
        if False:
            while True:
                i = 10

        class Finder(doctest.DocTestFinder):

            def find(*args, **kwargs):
                if False:
                    while True:
                        i = 10
                test = doctest.DocTestParser().get_doctest(string, {}, 'foo', 'foo.py', 0)
                return [test]
        suite = klass(test_finder=Finder())
        return suite

    def run_doctest_suite_for_string(self, klass, string):
        if False:
            while True:
                i = 10
        suite = self.get_doctest_suite_for_string(klass, string)
        output = StringIO()
        result = tests.TextTestResult(output, 0, 1)
        suite.run(result)
        return (result, output)

    def assertDocTestStringSucceds(self, klass, string):
        if False:
            print('Hello World!')
        (result, output) = self.run_doctest_suite_for_string(klass, string)
        if not result.wasStrictlySuccessful():
            self.fail(output.getvalue())

    def assertDocTestStringFails(self, klass, string):
        if False:
            while True:
                i = 10
        (result, output) = self.run_doctest_suite_for_string(klass, string)
        if result.wasStrictlySuccessful():
            self.fail(output.getvalue())

    def test_injected_variable(self):
        if False:
            for i in range(10):
                print('nop')
        self.overrideAttr(tests, 'isolated_environ', {'LINES': '42'})
        test = "\n            >>> import os\n            >>> os.environ['LINES']\n            '42'\n            "
        self.assertDocTestStringFails(doctest.DocTestSuite, test)
        self.assertDocTestStringSucceds(tests.IsolatedDocTestSuite, test)

    def test_deleted_variable(self):
        if False:
            while True:
                i = 10
        self.overrideAttr(tests, 'isolated_environ', {'LINES': None})
        test = "\n            >>> import os\n            >>> os.environ.get('LINES')\n            "
        self.assertDocTestStringFails(doctest.DocTestSuite, test)
        self.assertDocTestStringSucceds(tests.IsolatedDocTestSuite, test)

class TestSelftestExcludePatterns(tests.TestCase):

    def setUp(self):
        if False:
            return 10
        super(TestSelftestExcludePatterns, self).setUp()
        self.overrideAttr(tests, 'test_suite', self.suite_factory)

    def suite_factory(self, keep_only=None, starting_with=None):
        if False:
            print('Hello World!')
        'A test suite factory with only a few tests.'

        class Test(tests.TestCase):

            def id(self):
                if False:
                    for i in range(10):
                        print('nop')
                return self._testMethodName

            def a(self):
                if False:
                    while True:
                        i = 10
                pass

            def b(self):
                if False:
                    i = 10
                    return i + 15
                pass

            def c(self):
                if False:
                    i = 10
                    return i + 15
                pass
        return TestUtil.TestSuite([Test('a'), Test('b'), Test('c')])

    def assertTestList(self, expected, *selftest_args):
        if False:
            print('Hello World!')
        (out, err) = self.run_bzr(('selftest', '--list') + selftest_args)
        actual = out.splitlines()
        self.assertEqual(expected, actual)

    def test_full_list(self):
        if False:
            return 10
        self.assertTestList(['a', 'b', 'c'])

    def test_single_exclude(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertTestList(['b', 'c'], '-x', 'a')

    def test_mutiple_excludes(self):
        if False:
            return 10
        self.assertTestList(['c'], '-x', 'a', '-x', 'b')

class TestCounterHooks(tests.TestCase, SelfTestHelper):
    _test_needs_features = [features.subunit]

    def setUp(self):
        if False:
            i = 10
            return i + 15
        super(TestCounterHooks, self).setUp()

        class Test(tests.TestCase):

            def setUp(self):
                if False:
                    return 10
                super(Test, self).setUp()
                self.hooks = hooks.Hooks()
                self.hooks.add_hook('myhook', 'Foo bar blah', (2, 4))
                self.install_counter_hook(self.hooks, 'myhook')

            def no_hook(self):
                if False:
                    while True:
                        i = 10
                pass

            def run_hook_once(self):
                if False:
                    while True:
                        i = 10
                for hook in self.hooks['myhook']:
                    hook(self)
        self.test_class = Test

    def assertHookCalls(self, expected_calls, test_name):
        if False:
            while True:
                i = 10
        test = self.test_class(test_name)
        result = unittest.TestResult()
        test.run(result)
        self.assertTrue(hasattr(test, '_counters'))
        self.assertTrue(test._counters.has_key('myhook'))
        self.assertEqual(expected_calls, test._counters['myhook'])

    def test_no_hook(self):
        if False:
            print('Hello World!')
        self.assertHookCalls(0, 'no_hook')

    def test_run_hook_once(self):
        if False:
            i = 10
            return i + 15
        tt = features.testtools
        if tt.module.__version__ < (0, 9, 8):
            raise tests.TestSkipped('testtools-0.9.8 required for addDetail')
        self.assertHookCalls(1, 'run_hook_once')