import os
import stat
import pytest
import spack.config
import spack.package_prefs
import spack.repo
import spack.util.spack_yaml as syaml
from spack.config import ConfigError
from spack.spec import CompilerSpec, Spec
from spack.version import Version

@pytest.fixture()
def configure_permissions():
    if False:
        i = 10
        return i + 15
    conf = syaml.load_config('all:\n  permissions:\n    read: group\n    write: group\n    group: all\nmpich:\n  permissions:\n    read: user\n    write: user\nmpileaks:\n  permissions:\n    write: user\n    group: mpileaks\ncallpath:\n  permissions:\n    write: world\n')
    spack.config.set('packages', conf, scope='concretize')
    yield

def concretize(abstract_spec):
    if False:
        print('Hello World!')
    return Spec(abstract_spec).concretized()

def update_packages(pkgname, section, value):
    if False:
        for i in range(10):
            print('nop')
    'Update config and reread package list'
    conf = {pkgname: {section: value}}
    spack.config.set('packages', conf, scope='concretize')

def assert_variant_values(spec, **variants):
    if False:
        i = 10
        return i + 15
    concrete = concretize(spec)
    for (variant, value) in variants.items():
        assert concrete.variants[variant].value == value

@pytest.mark.usefixtures('concretize_scope', 'mock_packages')
class TestConcretizePreferences:

    @pytest.mark.parametrize('package_name,variant_value,expected_results', [('mpileaks', '~debug~opt+shared+static', {'debug': False, 'opt': False, 'shared': True, 'static': True}), ('mpileaks', ['~debug', '~opt', '+shared', '+static'], {'debug': False, 'opt': False, 'shared': True, 'static': True}), ('mpileaks', ['+debug', '+opt', '~shared', '-static'], {'debug': True, 'opt': True, 'shared': False, 'static': False}), ('multivalue-variant', ['foo=bar,baz', 'fee=bar'], {'foo': ('bar', 'baz'), 'fee': 'bar'}), ('singlevalue-variant', ['fum=why'], {'fum': 'why'})])
    def test_preferred_variants(self, package_name, variant_value, expected_results):
        if False:
            for i in range(10):
                print('nop')
        'Test preferred variants are applied correctly'
        update_packages(package_name, 'variants', variant_value)
        assert_variant_values(package_name, **expected_results)

    def test_preferred_variants_from_wildcard(self):
        if False:
            print('Hello World!')
        "\n        Test that 'foo=*' concretizes to any value\n        "
        update_packages('multivalue-variant', 'variants', 'foo=bar')
        assert_variant_values('multivalue-variant foo=*', foo=('bar',))

    @pytest.mark.parametrize('compiler_str,spec_str', [('gcc@=4.5.0', 'mpileaks'), ('clang@=12.0.0', 'mpileaks'), ('gcc@=4.5.0', 'openmpi')])
    def test_preferred_compilers(self, compiler_str, spec_str):
        if False:
            i = 10
            return i + 15
        'Test preferred compilers are applied correctly'
        update_packages('all', 'compiler', [compiler_str])
        spec = spack.spec.Spec(spec_str).concretized()
        assert spec.compiler == CompilerSpec(compiler_str)

    @pytest.mark.only_clingo('Use case not supported by the original concretizer')
    def test_preferred_target(self, mutable_mock_repo):
        if False:
            for i in range(10):
                print('nop')
        'Test preferred targets are applied correctly'
        spec = concretize('mpich')
        default = str(spec.target)
        preferred = str(spec.target.family)
        update_packages('all', 'target', [preferred])
        spec = concretize('mpich')
        assert str(spec.target) == preferred
        spec = concretize('mpileaks')
        assert str(spec['mpileaks'].target) == preferred
        assert str(spec['mpich'].target) == preferred
        update_packages('all', 'target', [default])
        spec = concretize('mpileaks')
        assert str(spec['mpileaks'].target) == default
        assert str(spec['mpich'].target) == default

    def test_preferred_versions(self):
        if False:
            return 10
        'Test preferred package versions are applied correctly'
        update_packages('mpileaks', 'version', ['2.3'])
        spec = concretize('mpileaks')
        assert spec.version == Version('2.3')
        update_packages('mpileaks', 'version', ['2.2'])
        spec = concretize('mpileaks')
        assert spec.version == Version('2.2')

    @pytest.mark.only_clingo('This behavior is not enforced for the old concretizer')
    def test_preferred_versions_mixed_version_types(self):
        if False:
            for i in range(10):
                print('nop')
        update_packages('mixedversions', 'version', ['=2.0'])
        spec = concretize('mixedversions')
        assert spec.version == Version('2.0')

    def test_preferred_providers(self):
        if False:
            print('Hello World!')
        'Test preferred providers of virtual packages are\n        applied correctly\n        '
        update_packages('all', 'providers', {'mpi': ['mpich']})
        spec = concretize('mpileaks')
        assert 'mpich' in spec
        update_packages('all', 'providers', {'mpi': ['zmpi']})
        spec = concretize('mpileaks')
        assert 'zmpi' in spec

    def test_config_set_pkg_property_url(self, mutable_mock_repo):
        if False:
            i = 10
            return i + 15
        'Test setting an existing attribute in the package class'
        update_packages('mpileaks', 'package_attributes', {'url': 'http://www.somewhereelse.com/mpileaks-1.0.tar.gz'})
        spec = concretize('mpileaks')
        assert spec.package.fetcher.url == 'http://www.somewhereelse.com/mpileaks-2.3.tar.gz'
        update_packages('mpileaks', 'package_attributes', {})
        spec = concretize('mpileaks')
        assert spec.package.fetcher.url == 'http://www.llnl.gov/mpileaks-2.3.tar.gz'

    def test_config_set_pkg_property_new(self, mutable_mock_repo):
        if False:
            while True:
                i = 10
        'Test that you can set arbitrary attributes on the Package class'
        conf = syaml.load_config('mpileaks:\n  package_attributes:\n    v1: 1\n    v2: true\n    v3: yesterday\n    v4: "true"\n    v5:\n      x: 1\n      y: 2\n    v6:\n    - 1\n    - 2\n')
        spack.config.set('packages', conf, scope='concretize')
        spec = concretize('mpileaks')
        assert spec.package.v1 == 1
        assert spec.package.v2 is True
        assert spec.package.v3 == 'yesterday'
        assert spec.package.v4 == 'true'
        assert dict(spec.package.v5) == {'x': 1, 'y': 2}
        assert list(spec.package.v6) == [1, 2]
        update_packages('mpileaks', 'package_attributes', {})
        spec = concretize('mpileaks')
        with pytest.raises(AttributeError):
            spec.package.v1

    def test_preferred(self):
        if False:
            i = 10
            return i + 15
        ' "Test packages with some version marked as preferred=True'
        spec = Spec('python')
        spec.concretize()
        assert spec.version == Version('2.7.11')
        update_packages('python', 'version', ['3.5.0'])
        spec = Spec('python')
        spec.concretize()
        assert spec.version == Version('3.5.0')

    @pytest.mark.only_clingo('This behavior is not enforced for the old concretizer')
    def test_preferred_undefined_raises(self):
        if False:
            while True:
                i = 10
        'Preference should not specify an undefined version'
        update_packages('python', 'version', ['3.5.0.1'])
        spec = Spec('python')
        with pytest.raises(spack.config.ConfigError):
            spec.concretize()

    @pytest.mark.only_clingo('This behavior is not enforced for the old concretizer')
    def test_preferred_truncated(self):
        if False:
            return 10
        'Versions without "=" are treated as version ranges: if there is\n        a satisfying version defined in the package.py, we should use that\n        (don\'t define a new version).\n        '
        update_packages('python', 'version', ['3.5'])
        spec = Spec('python')
        spec.concretize()
        assert spec.satisfies('@3.5.1')

    def test_develop(self):
        if False:
            return 10
        'Test concretization with develop-like versions'
        spec = Spec('develop-test')
        spec.concretize()
        assert spec.version == Version('0.2.15')
        spec = Spec('develop-test2')
        spec.concretize()
        assert spec.version == Version('0.2.15')
        update_packages('develop-test', 'version', ['develop'])
        spec = Spec('develop-test')
        spec.concretize()
        assert spec.version == Version('develop')
        update_packages('develop-test2', 'version', ['0.2.15.develop'])
        spec = Spec('develop-test2')
        spec.concretize()
        assert spec.version == Version('0.2.15.develop')

    def test_external_mpi(self):
        if False:
            while True:
                i = 10
        spec = Spec('mpi')
        spec.concretize()
        assert not spec['mpi'].external
        conf = syaml.load_config('all:\n    providers:\n        mpi: [mpich]\nmpich:\n    buildable: false\n    externals:\n    - spec: mpich@3.0.4\n      prefix: /dummy/path\n')
        spack.config.set('packages', conf, scope='concretize')
        spec = Spec('mpi')
        spec.concretize()
        assert spec['mpich'].external_path == os.path.sep + os.path.join('dummy', 'path')

    def test_external_module(self, monkeypatch):
        if False:
            return 10
        'Test that packages can find externals specified by module\n\n        The specific code for parsing the module is tested elsewhere.\n        This just tests that the preference is accounted for'

        def mock_module(cmd, module):
            if False:
                return 10
            return 'prepend-path PATH /dummy/path'
        monkeypatch.setattr(spack.util.module_cmd, 'module', mock_module)
        spec = Spec('mpi')
        spec.concretize()
        assert not spec['mpi'].external
        conf = syaml.load_config('all:\n    providers:\n        mpi: [mpich]\nmpi:\n    buildable: false\n    externals:\n    - spec: mpich@3.0.4\n      modules: [dummy]\n')
        spack.config.set('packages', conf, scope='concretize')
        spec = Spec('mpi')
        spec.concretize()
        assert spec['mpich'].external_path == os.path.sep + os.path.join('dummy', 'path')

    def test_buildable_false(self):
        if False:
            i = 10
            return i + 15
        conf = syaml.load_config('libelf:\n  buildable: false\n')
        spack.config.set('packages', conf, scope='concretize')
        spec = Spec('libelf')
        assert not spack.package_prefs.is_spec_buildable(spec)
        spec = Spec('mpich')
        assert spack.package_prefs.is_spec_buildable(spec)

    def test_buildable_false_virtual(self):
        if False:
            print('Hello World!')
        conf = syaml.load_config('mpi:\n  buildable: false\n')
        spack.config.set('packages', conf, scope='concretize')
        spec = Spec('libelf')
        assert spack.package_prefs.is_spec_buildable(spec)
        spec = Spec('mpich')
        assert not spack.package_prefs.is_spec_buildable(spec)

    def test_buildable_false_all(self):
        if False:
            return 10
        conf = syaml.load_config('all:\n  buildable: false\n')
        spack.config.set('packages', conf, scope='concretize')
        spec = Spec('libelf')
        assert not spack.package_prefs.is_spec_buildable(spec)
        spec = Spec('mpich')
        assert not spack.package_prefs.is_spec_buildable(spec)

    def test_buildable_false_all_true_package(self):
        if False:
            return 10
        conf = syaml.load_config('all:\n  buildable: false\nlibelf:\n  buildable: true\n')
        spack.config.set('packages', conf, scope='concretize')
        spec = Spec('libelf')
        assert spack.package_prefs.is_spec_buildable(spec)
        spec = Spec('mpich')
        assert not spack.package_prefs.is_spec_buildable(spec)

    def test_buildable_false_all_true_virtual(self):
        if False:
            print('Hello World!')
        conf = syaml.load_config('all:\n  buildable: false\nmpi:\n  buildable: true\n')
        spack.config.set('packages', conf, scope='concretize')
        spec = Spec('libelf')
        assert not spack.package_prefs.is_spec_buildable(spec)
        spec = Spec('mpich')
        assert spack.package_prefs.is_spec_buildable(spec)

    def test_buildable_false_virtual_true_pacakge(self):
        if False:
            return 10
        conf = syaml.load_config('mpi:\n  buildable: false\nmpich:\n  buildable: true\n')
        spack.config.set('packages', conf, scope='concretize')
        spec = Spec('zmpi')
        assert not spack.package_prefs.is_spec_buildable(spec)
        spec = Spec('mpich')
        assert spack.package_prefs.is_spec_buildable(spec)

    def test_config_permissions_from_all(self, configure_permissions):
        if False:
            print('Hello World!')
        spec = Spec('zmpi')
        perms = spack.package_prefs.get_package_permissions(spec)
        assert perms == stat.S_IRWXU | stat.S_IRWXG
        dir_perms = spack.package_prefs.get_package_dir_permissions(spec)
        assert dir_perms == stat.S_IRWXU | stat.S_IRWXG | stat.S_ISGID
        group = spack.package_prefs.get_package_group(spec)
        assert group == 'all'

    def test_config_permissions_from_package(self, configure_permissions):
        if False:
            for i in range(10):
                print('nop')
        spec = Spec('mpich')
        perms = spack.package_prefs.get_package_permissions(spec)
        assert perms == stat.S_IRWXU
        dir_perms = spack.package_prefs.get_package_dir_permissions(spec)
        assert dir_perms == stat.S_IRWXU
        group = spack.package_prefs.get_package_group(spec)
        assert group == 'all'

    def test_config_permissions_differ_read_write(self, configure_permissions):
        if False:
            i = 10
            return i + 15
        spec = Spec('mpileaks')
        perms = spack.package_prefs.get_package_permissions(spec)
        assert perms == stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP
        dir_perms = spack.package_prefs.get_package_dir_permissions(spec)
        expected = stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP | stat.S_ISGID
        assert dir_perms == expected
        group = spack.package_prefs.get_package_group(spec)
        assert group == 'mpileaks'

    def test_config_perms_fail_write_gt_read(self, configure_permissions):
        if False:
            for i in range(10):
                print('nop')
        spec = Spec('callpath')
        with pytest.raises(ConfigError):
            spack.package_prefs.get_package_permissions(spec)

    @pytest.mark.regression('20040')
    def test_variant_not_flipped_to_pull_externals(self):
        if False:
            for i in range(10):
                print('nop')
        "Test that a package doesn't prefer pulling in an\n        external to using the default value of a variant.\n        "
        s = Spec('vdefault-or-external-root').concretized()
        assert '~external' in s['vdefault-or-external']
        assert 'externaltool' not in s

    @pytest.mark.regression('25585')
    def test_dependencies_cant_make_version_parent_score_better(self):
        if False:
            for i in range(10):
                print('nop')
        "Test that a package can't select a worse version for a\n        dependent because doing so it can pull-in a dependency\n        that makes the overall version score even or better and maybe\n        has a better score in some lower priority criteria.\n        "
        s = Spec('version-test-root').concretized()
        assert s.satisfies('^version-test-pkg@2.4.6')
        assert 'version-test-dependency-preferred' not in s

    @pytest.mark.regression('26598')
    def test_multivalued_variants_are_lower_priority_than_providers(self):
        if False:
            return 10
        "Test that the rule to maximize the number of values for multivalued\n        variants is considered at lower priority than selecting the default\n        provider for virtual dependencies.\n\n        This ensures that we don't e.g. select openmpi over mpich even if we\n        specified mpich as the default mpi provider, just because openmpi supports\n        more fabrics by default.\n        "
        with spack.config.override('packages:all', {'providers': {'somevirtual': ['some-virtual-preferred']}}):
            s = Spec('somevirtual').concretized()
            assert s.name == 'some-virtual-preferred'

    @pytest.mark.regression('26721,19736')
    def test_sticky_variant_accounts_for_packages_yaml(self):
        if False:
            i = 10
            return i + 15
        with spack.config.override('packages:sticky-variant', {'variants': '+allow-gcc'}):
            s = Spec('sticky-variant %gcc').concretized()
            assert s.satisfies('%gcc') and s.satisfies('+allow-gcc')