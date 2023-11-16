import re
import textwrap
import unittest
import warnings
import importlib
import contextlib
from . import fixtures
from importlib.metadata import Distribution, PackageNotFoundError, distribution, entry_points, files, metadata, requires, version

@contextlib.contextmanager
def suppress_known_deprecation():
    if False:
        i = 10
        return i + 15
    with warnings.catch_warnings(record=True) as ctx:
        warnings.simplefilter('default')
        yield ctx

class APITests(fixtures.EggInfoPkg, fixtures.DistInfoPkg, fixtures.DistInfoPkgWithDot, fixtures.EggInfoFile, unittest.TestCase):
    version_pattern = '\\d+\\.\\d+(\\.\\d)?'

    def test_retrieves_version_of_self(self):
        if False:
            for i in range(10):
                print('nop')
        pkg_version = version('egginfo-pkg')
        assert isinstance(pkg_version, str)
        assert re.match(self.version_pattern, pkg_version)

    def test_retrieves_version_of_distinfo_pkg(self):
        if False:
            i = 10
            return i + 15
        pkg_version = version('distinfo-pkg')
        assert isinstance(pkg_version, str)
        assert re.match(self.version_pattern, pkg_version)

    def test_for_name_does_not_exist(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaises(PackageNotFoundError):
            distribution('does-not-exist')

    def test_name_normalization(self):
        if False:
            for i in range(10):
                print('nop')
        names = ('pkg.dot', 'pkg_dot', 'pkg-dot', 'pkg..dot', 'Pkg.Dot')
        for name in names:
            with self.subTest(name):
                assert distribution(name).metadata['Name'] == 'pkg.dot'

    def test_prefix_not_matched(self):
        if False:
            return 10
        prefixes = ('p', 'pkg', 'pkg.')
        for prefix in prefixes:
            with self.subTest(prefix):
                with self.assertRaises(PackageNotFoundError):
                    distribution(prefix)

    def test_for_top_level(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(distribution('egginfo-pkg').read_text('top_level.txt').strip(), 'mod')

    def test_read_text(self):
        if False:
            while True:
                i = 10
        top_level = [path for path in files('egginfo-pkg') if path.name == 'top_level.txt'][0]
        self.assertEqual(top_level.read_text(), 'mod\n')

    def test_entry_points(self):
        if False:
            i = 10
            return i + 15
        eps = entry_points()
        assert 'entries' in eps.groups
        entries = eps.select(group='entries')
        assert 'main' in entries.names
        ep = entries['main']
        self.assertEqual(ep.value, 'mod:main')
        self.assertEqual(ep.extras, [])

    def test_entry_points_distribution(self):
        if False:
            return 10
        entries = entry_points(group='entries')
        for entry in ('main', 'ns:sub'):
            ep = entries[entry]
            self.assertIn(ep.dist.name, ('distinfo-pkg', 'egginfo-pkg'))
            self.assertEqual(ep.dist.version, '1.0.0')

    def test_entry_points_unique_packages(self):
        if False:
            return 10
        alt_site_dir = self.fixtures.enter_context(fixtures.tempdir())
        self.fixtures.enter_context(self.add_sys_path(alt_site_dir))
        alt_pkg = {'distinfo_pkg-1.1.0.dist-info': {'METADATA': '\n                Name: distinfo-pkg\n                Version: 1.1.0\n                ', 'entry_points.txt': '\n                [entries]\n                main = mod:altmain\n            '}}
        fixtures.build_files(alt_pkg, alt_site_dir)
        entries = entry_points(group='entries')
        assert not any((ep.dist.name == 'distinfo-pkg' and ep.dist.version == '1.0.0' for ep in entries))
        assert 'ns:sub' not in entries

    def test_entry_points_missing_name(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaises(KeyError):
            entry_points(group='entries')['missing']

    def test_entry_points_missing_group(self):
        if False:
            while True:
                i = 10
        assert entry_points(group='missing') == ()

    def test_entry_points_dict_construction(self):
        if False:
            return 10
        with suppress_known_deprecation() as caught:
            eps = dict(entry_points(group='entries'))
        assert 'main' in eps
        assert eps['main'] == entry_points(group='entries')['main']
        expected = next(iter(caught))
        assert expected.category is DeprecationWarning
        assert 'Construction of dict of EntryPoints is deprecated' in str(expected)

    def test_entry_points_by_index(self):
        if False:
            print('Hello World!')
        '\n        Prior versions of Distribution.entry_points would return a\n        tuple that allowed access by index.\n        Capture this now deprecated use-case\n        See python/importlib_metadata#300 and bpo-44246.\n        '
        eps = distribution('distinfo-pkg').entry_points
        with suppress_known_deprecation() as caught:
            eps[0]
        expected = next(iter(caught))
        assert expected.category is DeprecationWarning
        assert 'Accessing entry points by index is deprecated' in str(expected)

    def test_entry_points_groups_getitem(self):
        if False:
            while True:
                i = 10
        with suppress_known_deprecation():
            entry_points()['entries'] == entry_points(group='entries')
            with self.assertRaises(KeyError):
                entry_points()['missing']

    def test_entry_points_groups_get(self):
        if False:
            print('Hello World!')
        with suppress_known_deprecation():
            entry_points().get('missing', 'default') == 'default'
            entry_points().get('entries', 'default') == entry_points()['entries']
            entry_points().get('missing', ()) == ()

    def test_entry_points_allows_no_attributes(self):
        if False:
            return 10
        ep = entry_points().select(group='entries', name='main')
        with self.assertRaises(AttributeError):
            ep.foo = 4

    def test_metadata_for_this_package(self):
        if False:
            while True:
                i = 10
        md = metadata('egginfo-pkg')
        assert md['author'] == 'Steven Ma'
        assert md['LICENSE'] == 'Unknown'
        assert md['Name'] == 'egginfo-pkg'
        classifiers = md.get_all('Classifier')
        assert 'Topic :: Software Development :: Libraries' in classifiers

    @staticmethod
    def _test_files(files):
        if False:
            for i in range(10):
                print('nop')
        root = files[0].root
        for file in files:
            assert file.root == root
            assert not file.hash or file.hash.value
            assert not file.hash or file.hash.mode == 'sha256'
            assert not file.size or file.size >= 0
            assert file.locate().exists()
            assert isinstance(file.read_binary(), bytes)
            if file.name.endswith('.py'):
                file.read_text()

    def test_file_hash_repr(self):
        if False:
            print('Hello World!')
        assertRegex = self.assertRegex
        util = [p for p in files('distinfo-pkg') if p.name == 'mod.py'][0]
        assertRegex(repr(util.hash), '<FileHash mode: sha256 value: .*>')

    def test_files_dist_info(self):
        if False:
            while True:
                i = 10
        self._test_files(files('distinfo-pkg'))

    def test_files_egg_info(self):
        if False:
            i = 10
            return i + 15
        self._test_files(files('egginfo-pkg'))

    def test_version_egg_info_file(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(version('egginfo-file'), '0.1')

    def test_requires_egg_info_file(self):
        if False:
            i = 10
            return i + 15
        requirements = requires('egginfo-file')
        self.assertIsNone(requirements)

    def test_requires_egg_info(self):
        if False:
            print('Hello World!')
        deps = requires('egginfo-pkg')
        assert len(deps) == 2
        assert any((dep == 'wheel >= 1.0; python_version >= "2.7"' for dep in deps))

    def test_requires_egg_info_empty(self):
        if False:
            while True:
                i = 10
        fixtures.build_files({'requires.txt': ''}, self.site_dir.joinpath('egginfo_pkg.egg-info'))
        deps = requires('egginfo-pkg')
        assert deps == []

    def test_requires_dist_info(self):
        if False:
            i = 10
            return i + 15
        deps = requires('distinfo-pkg')
        assert len(deps) == 2
        assert all(deps)
        assert 'wheel >= 1.0' in deps
        assert "pytest; extra == 'test'" in deps

    def test_more_complex_deps_requires_text(self):
        if False:
            print('Hello World!')
        requires = textwrap.dedent('\n            dep1\n            dep2\n\n            [:python_version < "3"]\n            dep3\n\n            [extra1]\n            dep4\n            dep6@ git+https://example.com/python/dep.git@v1.0.0\n\n            [extra2:python_version < "3"]\n            dep5\n            ')
        deps = sorted(Distribution._deps_from_requires_text(requires))
        expected = ['dep1', 'dep2', 'dep3; python_version < "3"', 'dep4; extra == "extra1"', 'dep5; (python_version < "3") and extra == "extra2"', 'dep6@ git+https://example.com/python/dep.git@v1.0.0 ; extra == "extra1"']
        assert deps == expected

    def test_as_json(self):
        if False:
            while True:
                i = 10
        md = metadata('distinfo-pkg').json
        assert 'name' in md
        assert md['keywords'] == ['sample', 'package']
        desc = md['description']
        assert desc.startswith('Once upon a time\nThere was')
        assert len(md['requires_dist']) == 2

    def test_as_json_egg_info(self):
        if False:
            for i in range(10):
                print('nop')
        md = metadata('egginfo-pkg').json
        assert 'name' in md
        assert md['keywords'] == ['sample', 'package']
        desc = md['description']
        assert desc.startswith('Once upon a time\nThere was')
        assert len(md['classifier']) == 2

    def test_as_json_odd_case(self):
        if False:
            while True:
                i = 10
        self.make_uppercase()
        md = metadata('distinfo-pkg').json
        assert 'name' in md
        assert len(md['requires_dist']) == 2
        assert md['keywords'] == ['SAMPLE', 'PACKAGE']

class LegacyDots(fixtures.DistInfoPkgWithDotLegacy, unittest.TestCase):

    def test_name_normalization(self):
        if False:
            return 10
        names = ('pkg.dot', 'pkg_dot', 'pkg-dot', 'pkg..dot', 'Pkg.Dot')
        for name in names:
            with self.subTest(name):
                assert distribution(name).metadata['Name'] == 'pkg.dot'

    def test_name_normalization_versionless_egg_info(self):
        if False:
            return 10
        names = ('pkg.lot', 'pkg_lot', 'pkg-lot', 'pkg..lot', 'Pkg.Lot')
        for name in names:
            with self.subTest(name):
                assert distribution(name).metadata['Name'] == 'pkg.lot'

class OffSysPathTests(fixtures.DistInfoPkgOffPath, unittest.TestCase):

    def test_find_distributions_specified_path(self):
        if False:
            while True:
                i = 10
        dists = Distribution.discover(path=[str(self.site_dir)])
        assert any((dist.metadata['Name'] == 'distinfo-pkg' for dist in dists))

    def test_distribution_at_pathlib(self):
        if False:
            for i in range(10):
                print('nop')
        dist_info_path = self.site_dir / 'distinfo_pkg-1.0.0.dist-info'
        dist = Distribution.at(dist_info_path)
        assert dist.version == '1.0.0'

    def test_distribution_at_str(self):
        if False:
            while True:
                i = 10
        dist_info_path = self.site_dir / 'distinfo_pkg-1.0.0.dist-info'
        dist = Distribution.at(str(dist_info_path))
        assert dist.version == '1.0.0'

class InvalidateCache(unittest.TestCase):

    def test_invalidate_cache(self):
        if False:
            print('Hello World!')
        importlib.invalidate_caches()