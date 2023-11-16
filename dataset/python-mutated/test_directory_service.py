"""Test directory service implementation"""
from bzrlib import errors, transport, urlutils
from bzrlib.directory_service import AliasDirectory, DirectoryServiceRegistry, directories
from bzrlib.tests import TestCase, TestCaseWithTransport

class FooService(object):
    """A directory service that maps the name to a FILE url"""
    base = urlutils.local_path_to_url('/foo')

    def look_up(self, name, url):
        if False:
            print('Hello World!')
        return self.base + name

class TestDirectoryLookup(TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        super(TestDirectoryLookup, self).setUp()
        self.registry = DirectoryServiceRegistry()
        self.registry.register('foo:', FooService, 'Map foo URLs to http urls')

    def test_get_directory_service(self):
        if False:
            return 10
        (directory, suffix) = self.registry.get_prefix('foo:bar')
        self.assertIs(FooService, directory)
        self.assertEqual('bar', suffix)

    def test_dereference(self):
        if False:
            while True:
                i = 10
        self.assertEqual(FooService.base + 'bar', self.registry.dereference('foo:bar'))
        self.assertEqual('baz:qux', self.registry.dereference('baz:qux'))

    def test_get_transport(self):
        if False:
            return 10
        directories.register('foo:', FooService, 'Map foo URLs to http urls')
        self.addCleanup(directories.remove, 'foo:')
        self.assertEqual(FooService.base + 'bar/', transport.get_transport('foo:bar').base)

class TestAliasDirectory(TestCaseWithTransport):

    def setUp(self):
        if False:
            print('Hello World!')
        super(TestAliasDirectory, self).setUp()
        self.branch = self.make_branch('.')

    def assertAliasFromBranch(self, setter, value, alias):
        if False:
            while True:
                i = 10
        setter(value)
        self.assertEqual(value, directories.dereference(alias))

    def test_lookup_parent(self):
        if False:
            print('Hello World!')
        self.assertAliasFromBranch(self.branch.set_parent, 'http://a', ':parent')

    def test_lookup_submit(self):
        if False:
            print('Hello World!')
        self.assertAliasFromBranch(self.branch.set_submit_branch, 'http://b', ':submit')

    def test_lookup_public(self):
        if False:
            i = 10
            return i + 15
        self.assertAliasFromBranch(self.branch.set_public_branch, 'http://c', ':public')

    def test_lookup_bound(self):
        if False:
            while True:
                i = 10
        self.assertAliasFromBranch(self.branch.set_bound_location, 'http://d', ':bound')

    def test_lookup_push(self):
        if False:
            i = 10
            return i + 15
        self.assertAliasFromBranch(self.branch.set_push_location, 'http://e', ':push')

    def test_lookup_this(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.branch.base, directories.dereference(':this'))

    def test_extra_path(self):
        if False:
            while True:
                i = 10
        self.assertEqual(urlutils.join(self.branch.base, 'arg'), directories.dereference(':this/arg'))

    def test_lookup_badname(self):
        if False:
            i = 10
            return i + 15
        e = self.assertRaises(errors.InvalidLocationAlias, directories.dereference, ':booga')
        self.assertEqual('":booga" is not a valid location alias.', str(e))

    def test_lookup_badvalue(self):
        if False:
            print('Hello World!')
        e = self.assertRaises(errors.UnsetLocationAlias, directories.dereference, ':parent')
        self.assertEqual('No parent location assigned.', str(e))

    def test_register_location_alias(self):
        if False:
            return 10
        self.addCleanup(AliasDirectory.branch_aliases.remove, 'booga')
        AliasDirectory.branch_aliases.register('booga', lambda b: 'UHH?', help='Nobody knows')
        self.assertEqual('UHH?', directories.dereference(':booga'))

class TestColocatedDirectory(TestCaseWithTransport):

    def test_lookup_non_default(self):
        if False:
            while True:
                i = 10
        default = self.make_branch('.')
        non_default = default.bzrdir.create_branch(name='nondefault')
        self.assertEqual(non_default.base, directories.dereference('co:nondefault'))

    def test_lookup_default(self):
        if False:
            return 10
        default = self.make_branch('.')
        non_default = default.bzrdir.create_branch(name='nondefault')
        self.assertEqual(urlutils.join_segment_parameters(default.bzrdir.user_url, {'branch': ''}), directories.dereference('co:'))

    def test_no_such_branch(self):
        if False:
            while True:
                i = 10
        default = self.make_branch('.')
        self.assertEqual(urlutils.join_segment_parameters(default.bzrdir.user_url, {'branch': 'foo'}), directories.dereference('co:foo'))