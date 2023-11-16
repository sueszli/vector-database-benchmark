import json
import os
import textwrap
from pathlib import Path
from typing import TYPE_CHECKING, Any
import pytest
from tests.lib import PipTestEnvironment, ResolverVariant, TestData, _create_test_package_with_subdirectory, create_basic_sdist_for_package, create_basic_wheel_for_package, need_svn, requirements_file
from tests.lib.local_repos import local_checkout
if TYPE_CHECKING:
    from typing import Protocol
else:
    Protocol = object

class ArgRecordingSdist:

    def __init__(self, sdist_path: Path, args_path: Path) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.sdist_path = sdist_path
        self._args_path = args_path

    def args(self) -> Any:
        if False:
            i = 10
            return i + 15
        return json.loads(self._args_path.read_text())

class ArgRecordingSdistMaker(Protocol):

    def __call__(self, name: str, **kwargs: Any) -> ArgRecordingSdist:
        if False:
            i = 10
            return i + 15
        ...

@pytest.fixture()
def arg_recording_sdist_maker(script: PipTestEnvironment) -> ArgRecordingSdistMaker:
    if False:
        return 10
    arg_writing_setup_py_prelude = textwrap.dedent('\n        import io\n        import json\n        import os\n        import sys\n\n        args_path = os.path.join(os.environ["OUTPUT_DIR"], "{name}.json")\n        with open(args_path, \'w\') as f:\n            json.dump(sys.argv, f)\n        ')
    output_dir = script.scratch_path.joinpath('args_recording_sdist_maker_output')
    output_dir.mkdir(parents=True)
    script.environ['OUTPUT_DIR'] = str(output_dir)

    def _arg_recording_sdist_maker(name: str, **kwargs: Any) -> ArgRecordingSdist:
        if False:
            for i in range(10):
                print('nop')
        sdist_path = create_basic_sdist_for_package(script, name, '0.1.0', setup_py_prelude=arg_writing_setup_py_prelude.format(name=name), **kwargs)
        args_path = output_dir / f'{name}.json'
        return ArgRecordingSdist(sdist_path, args_path)
    return _arg_recording_sdist_maker

@pytest.mark.network
def test_requirements_file(script: PipTestEnvironment) -> None:
    if False:
        return 10
    '\n    Test installing from a requirements file.\n\n    '
    (other_lib_name, other_lib_version) = ('peppercorn', '0.6')
    script.scratch_path.joinpath('initools-req.txt').write_text(textwrap.dedent(f'        INITools==0.2\n        # and something else to test out:\n        {other_lib_name}<={other_lib_version}\n        '))
    result = script.pip('install', '-r', script.scratch_path / 'initools-req.txt')
    result.did_create(script.site_packages / 'INITools-0.2.dist-info')
    result.did_create(script.site_packages / 'initools')
    assert result.files_created[script.site_packages / other_lib_name].dir
    fn = f'{other_lib_name}-{other_lib_version}.dist-info'
    assert result.files_created[script.site_packages / fn].dir

def test_schema_check_in_requirements_file(script: PipTestEnvironment) -> None:
    if False:
        i = 10
        return i + 15
    '\n    Test installing from a requirements file with an invalid vcs schema..\n\n    '
    script.scratch_path.joinpath('file-egg-req.txt').write_text('\n{}\n'.format('git://github.com/alex/django-fixture-generator.git#egg=fixture_generator'))
    with pytest.raises(AssertionError):
        script.pip('install', '-vvv', '-r', script.scratch_path / 'file-egg-req.txt')

@pytest.mark.parametrize('test_type,editable', [('rel_path', False), ('rel_path', True), ('rel_url', False), ('rel_url', True), ('embedded_rel_path', False), ('embedded_rel_path', True)])
def test_relative_requirements_file(script: PipTestEnvironment, data: TestData, test_type: str, editable: bool) -> None:
    if False:
        i = 10
        return i + 15
    '\n    Test installing from a requirements file with a relative path. For path\n    URLs, use an egg= definition.\n\n    '
    dist_info_folder = script.site_packages / 'FSPkg-0.1.dev0.dist-info'
    egg_link_file = script.site_packages / 'FSPkg.egg-link'
    package_folder = script.site_packages / 'fspkg'
    full_rel_path = os.path.relpath(data.packages.joinpath('FSPkg'), script.scratch_path)
    full_rel_url = 'file:' + full_rel_path + '#egg=FSPkg'
    embedded_rel_path = script.scratch_path.joinpath(full_rel_path)
    req_path = {'rel_path': full_rel_path, 'rel_url': full_rel_url, 'embedded_rel_path': os.fspath(embedded_rel_path)}[test_type]
    req_path = req_path.replace(os.path.sep, '/')
    if not editable:
        with requirements_file(req_path + '\n', script.scratch_path) as reqs_file:
            result = script.pip('install', '-vvv', '-r', reqs_file.name, cwd=script.scratch_path)
            result.did_create(dist_info_folder)
            result.did_create(package_folder)
    else:
        with requirements_file('-e ' + req_path + '\n', script.scratch_path) as reqs_file:
            result = script.pip('install', '-vvv', '-r', reqs_file.name, cwd=script.scratch_path)
            result.did_create(egg_link_file)

@pytest.mark.xfail
@pytest.mark.network
@need_svn
def test_multiple_requirements_files(script: PipTestEnvironment, tmpdir: Path) -> None:
    if False:
        i = 10
        return i + 15
    '\n    Test installing from multiple nested requirements files.\n\n    '
    (other_lib_name, other_lib_version) = ('six', '1.16.0')
    script.scratch_path.joinpath('initools-req.txt').write_text(textwrap.dedent('\n            -e {}@10#egg=INITools\n            -r {}-req.txt\n        ').format(local_checkout('svn+http://svn.colorstudy.com/INITools', tmpdir), other_lib_name))
    script.scratch_path.joinpath(f'{other_lib_name}-req.txt').write_text(f'{other_lib_name}<={other_lib_version}')
    result = script.pip('install', '-r', script.scratch_path / 'initools-req.txt')
    assert result.files_created[script.site_packages / other_lib_name].dir
    fn = f'{other_lib_name}-{other_lib_version}.dist-info'
    assert result.files_created[script.site_packages / fn].dir
    result.did_create(script.venv / 'src' / 'initools')

def test_package_in_constraints_and_dependencies(script: PipTestEnvironment, data: TestData) -> None:
    if False:
        print('Hello World!')
    script.scratch_path.joinpath('constraints.txt').write_text('TopoRequires2==0.0.1\nTopoRequires==0.0.1')
    result = script.pip('install', '--no-index', '-f', data.find_links, '-c', script.scratch_path / 'constraints.txt', 'TopoRequires2')
    assert 'installed TopoRequires-0.0.1' in result.stdout

def test_multiple_constraints_files(script: PipTestEnvironment, data: TestData) -> None:
    if False:
        print('Hello World!')
    script.scratch_path.joinpath('outer.txt').write_text('-c inner.txt')
    script.scratch_path.joinpath('inner.txt').write_text('Upper==1.0')
    result = script.pip('install', '--no-index', '-f', data.find_links, '-c', script.scratch_path / 'outer.txt', 'Upper')
    assert 'installed Upper-1.0' in result.stdout

def test_respect_order_in_requirements_file(script: PipTestEnvironment, data: TestData) -> None:
    if False:
        print('Hello World!')
    script.scratch_path.joinpath('frameworks-req.txt').write_text(textwrap.dedent('        parent\n        child\n        simple\n        '))
    result = script.pip('install', '--no-index', '-f', data.find_links, '-r', script.scratch_path / 'frameworks-req.txt')
    downloaded = [line for line in result.stdout.split('\n') if 'Processing' in line]
    assert 'parent' in downloaded[0], f'First download should be "parent" but was "{downloaded[0]}"'
    assert 'child' in downloaded[1], f'Second download should be "child" but was "{downloaded[1]}"'
    assert 'simple' in downloaded[2], f'Third download should be "simple" but was "{downloaded[2]}"'

def test_install_local_editable_with_extras(script: PipTestEnvironment, data: TestData) -> None:
    if False:
        return 10
    to_install = data.packages.joinpath('LocalExtras')
    res = script.pip_install_local('-e', f'{to_install}[bar]', allow_stderr_warning=True)
    res.did_update(script.site_packages / 'easy-install.pth')
    res.did_create(script.site_packages / 'LocalExtras.egg-link')
    res.did_create(script.site_packages / 'simple')

def test_install_collected_dependencies_first(script: PipTestEnvironment) -> None:
    if False:
        for i in range(10):
            print('nop')
    result = script.pip_install_local('toporequires2')
    text = [line for line in result.stdout.split('\n') if 'Installing' in line][0]
    assert text.endswith('toporequires2')

@pytest.mark.network
def test_install_local_editable_with_subdirectory(script: PipTestEnvironment) -> None:
    if False:
        print('Hello World!')
    version_pkg_path = _create_test_package_with_subdirectory(script, 'version_subdir')
    result = script.pip('install', '-e', '{uri}#egg=version_subpkg&subdirectory=version_subdir'.format(uri=f'git+{version_pkg_path.as_uri()}'))
    result.assert_installed('version-subpkg', sub_dir='version_subdir')

@pytest.mark.network
def test_install_local_with_subdirectory(script: PipTestEnvironment) -> None:
    if False:
        while True:
            i = 10
    version_pkg_path = _create_test_package_with_subdirectory(script, 'version_subdir')
    result = script.pip('install', '{uri}#egg=version_subpkg&subdirectory=version_subdir'.format(uri=f'git+{version_pkg_path.as_uri()}'))
    result.assert_installed('version_subpkg.py', editable=False)

@pytest.mark.usefixtures('enable_user_site')
def test_wheel_user_with_prefix_in_pydistutils_cfg(script: PipTestEnvironment, data: TestData) -> None:
    if False:
        i = 10
        return i + 15
    if os.name == 'posix':
        user_filename = '.pydistutils.cfg'
    else:
        user_filename = 'pydistutils.cfg'
    user_cfg = os.path.join(os.path.expanduser('~'), user_filename)
    script.scratch_path.joinpath('bin').mkdir()
    with open(user_cfg, 'w') as cfg:
        cfg.write(textwrap.dedent(f'\n            [install]\n            prefix={script.scratch_path}'))
    result = script.pip('install', '--user', '--no-index', '-f', data.find_links, 'requiresupper')
    assert 'installed requiresupper' in result.stdout

def test_constraints_not_installed_by_default(script: PipTestEnvironment, data: TestData) -> None:
    if False:
        print('Hello World!')
    script.scratch_path.joinpath('c.txt').write_text('requiresupper')
    result = script.pip('install', '--no-index', '-f', data.find_links, '-c', script.scratch_path / 'c.txt', 'Upper')
    assert 'requiresupper' not in result.stdout

def test_constraints_only_causes_error(script: PipTestEnvironment, data: TestData) -> None:
    if False:
        for i in range(10):
            print('nop')
    script.scratch_path.joinpath('c.txt').write_text('requiresupper')
    result = script.pip('install', '--no-index', '-f', data.find_links, '-c', script.scratch_path / 'c.txt', expect_error=True)
    assert 'installed requiresupper' not in result.stdout

def test_constraints_local_editable_install_causes_error(script: PipTestEnvironment, data: TestData, resolver_variant: ResolverVariant) -> None:
    if False:
        i = 10
        return i + 15
    script.scratch_path.joinpath('constraints.txt').write_text('singlemodule==0.0.0')
    to_install = data.src.joinpath('singlemodule')
    result = script.pip('install', '--no-index', '-f', data.find_links, '-c', script.scratch_path / 'constraints.txt', '-e', to_install, expect_error=True)
    assert 'Cannot install singlemodule 0.0.1' in result.stderr, str(result)

@pytest.mark.network
def test_constraints_local_editable_install_pep518(script: PipTestEnvironment, data: TestData) -> None:
    if False:
        return 10
    to_install = data.src.joinpath('pep518-3.0')
    script.pip('download', 'setuptools', 'wheel', '-d', data.packages)
    script.pip('install', '--no-index', '-f', data.find_links, '-e', to_install)

def test_constraints_local_install_causes_error(script: PipTestEnvironment, data: TestData, resolver_variant: ResolverVariant) -> None:
    if False:
        for i in range(10):
            print('nop')
    script.scratch_path.joinpath('constraints.txt').write_text('singlemodule==0.0.0')
    to_install = data.src.joinpath('singlemodule')
    result = script.pip('install', '--no-index', '-f', data.find_links, '-c', script.scratch_path / 'constraints.txt', to_install, expect_error=True)
    assert 'Cannot install singlemodule 0.0.1' in result.stderr, str(result)

def test_constraints_constrain_to_local_editable(script: PipTestEnvironment, data: TestData, resolver_variant: ResolverVariant) -> None:
    if False:
        for i in range(10):
            print('nop')
    to_install = data.src.joinpath('singlemodule')
    script.scratch_path.joinpath('constraints.txt').write_text(f'-e {to_install.as_uri()}#egg=singlemodule')
    result = script.pip('install', '--no-index', '-f', data.find_links, '-c', script.scratch_path / 'constraints.txt', 'singlemodule', allow_stderr_warning=True, expect_error=resolver_variant == 'resolvelib')
    if resolver_variant == 'resolvelib':
        assert 'Editable requirements are not allowed as constraints' in result.stderr
    else:
        assert 'Running setup.py develop for singlemodule' in result.stdout

def test_constraints_constrain_to_local(script: PipTestEnvironment, data: TestData, resolver_variant: ResolverVariant) -> None:
    if False:
        i = 10
        return i + 15
    to_install = data.src.joinpath('singlemodule')
    script.scratch_path.joinpath('constraints.txt').write_text(f'{to_install.as_uri()}#egg=singlemodule')
    result = script.pip('install', '--no-index', '-f', data.find_links, '-c', script.scratch_path / 'constraints.txt', 'singlemodule', allow_stderr_warning=True)
    assert 'Building wheel for singlemodule' in result.stdout

def test_constrained_to_url_install_same_url(script: PipTestEnvironment, data: TestData) -> None:
    if False:
        print('Hello World!')
    to_install = data.src.joinpath('singlemodule')
    constraints = f'{to_install.as_uri()}#egg=singlemodule'
    script.scratch_path.joinpath('constraints.txt').write_text(constraints)
    result = script.pip('install', '--no-index', '-f', data.find_links, '-c', script.scratch_path / 'constraints.txt', to_install, allow_stderr_warning=True)
    assert 'Building wheel for singlemodule' in result.stdout, str(result)

def test_double_install_spurious_hash_mismatch(script: PipTestEnvironment, tmpdir: Path, data: TestData) -> None:
    if False:
        print('Hello World!')
    "Make sure installing the same hashed sdist twice doesn't throw hash\n    mismatch errors.\n\n    Really, this is a test that we disable reads from the wheel cache in\n    hash-checking mode. Locally, implicitly built wheels of sdists obviously\n    have different hashes from the original archives. Comparing against those\n    causes spurious mismatch errors.\n\n    "
    with requirements_file('simple==1.0 --hash=sha256:393043e672415891885c9a2a0929b1af95fb866d6ca016b42d2e6ce53619b653', tmpdir) as reqs_file:
        result = script.pip_install_local('--find-links', data.find_links, '-r', reqs_file.resolve())
        assert 'Successfully installed simple-1.0' in str(result)
        script.pip('uninstall', '-y', 'simple')
        result = script.pip_install_local('--find-links', data.find_links, '-r', reqs_file.resolve())
        assert 'Successfully installed simple-1.0' in str(result)

def test_install_with_extras_from_constraints(script: PipTestEnvironment, data: TestData, resolver_variant: ResolverVariant) -> None:
    if False:
        for i in range(10):
            print('nop')
    to_install = data.packages.joinpath('LocalExtras')
    script.scratch_path.joinpath('constraints.txt').write_text(f'{to_install.as_uri()}#egg=LocalExtras[bar]')
    result = script.pip_install_local('-c', script.scratch_path / 'constraints.txt', 'LocalExtras', allow_stderr_warning=True, expect_error=resolver_variant == 'resolvelib')
    if resolver_variant == 'resolvelib':
        assert 'Constraints cannot have extras' in result.stderr
    else:
        result.did_create(script.site_packages / 'simple')

def test_install_with_extras_from_install(script: PipTestEnvironment) -> None:
    if False:
        while True:
            i = 10
    create_basic_wheel_for_package(script, name='LocalExtras', version='0.0.1', extras={'bar': ['simple'], 'baz': ['singlemodule']})
    script.scratch_path.joinpath('constraints.txt').write_text('LocalExtras')
    result = script.pip_install_local('--find-links', script.scratch_path, '-c', script.scratch_path / 'constraints.txt', 'LocalExtras[baz]')
    result.did_create(script.site_packages / 'singlemodule.py')

def test_install_with_extras_joined(script: PipTestEnvironment, data: TestData, resolver_variant: ResolverVariant) -> None:
    if False:
        while True:
            i = 10
    to_install = data.packages.joinpath('LocalExtras')
    script.scratch_path.joinpath('constraints.txt').write_text(f'{to_install.as_uri()}#egg=LocalExtras[bar]')
    result = script.pip_install_local('-c', script.scratch_path / 'constraints.txt', 'LocalExtras[baz]', allow_stderr_warning=True, expect_error=resolver_variant == 'resolvelib')
    if resolver_variant == 'resolvelib':
        assert 'Constraints cannot have extras' in result.stderr
    else:
        result.did_create(script.site_packages / 'simple')
        result.did_create(script.site_packages / 'singlemodule.py')

def test_install_with_extras_editable_joined(script: PipTestEnvironment, data: TestData, resolver_variant: ResolverVariant) -> None:
    if False:
        for i in range(10):
            print('nop')
    to_install = data.packages.joinpath('LocalExtras')
    script.scratch_path.joinpath('constraints.txt').write_text(f'-e {to_install.as_uri()}#egg=LocalExtras[bar]')
    result = script.pip_install_local('-c', script.scratch_path / 'constraints.txt', 'LocalExtras[baz]', allow_stderr_warning=True, expect_error=resolver_variant == 'resolvelib')
    if resolver_variant == 'resolvelib':
        assert 'Editable requirements are not allowed as constraints' in result.stderr
    else:
        result.did_create(script.site_packages / 'simple')
        result.did_create(script.site_packages / 'singlemodule.py')

def test_install_distribution_full_union(script: PipTestEnvironment, data: TestData) -> None:
    if False:
        return 10
    to_install = data.packages.joinpath('LocalExtras')
    result = script.pip_install_local(to_install, f'{to_install}[bar]', f'{to_install}[baz]')
    assert 'Building wheel for LocalExtras' in result.stdout
    result.did_create(script.site_packages / 'simple')
    result.did_create(script.site_packages / 'singlemodule.py')

def test_install_distribution_duplicate_extras(script: PipTestEnvironment, data: TestData) -> None:
    if False:
        i = 10
        return i + 15
    to_install = data.packages.joinpath('LocalExtras')
    package_name = f'{to_install}[bar]'
    with pytest.raises(AssertionError):
        result = script.pip_install_local(package_name, package_name)
        expected = f'Double requirement given: {package_name}'
        assert expected in result.stderr

def test_install_distribution_union_with_constraints(script: PipTestEnvironment, data: TestData, resolver_variant: ResolverVariant) -> None:
    if False:
        return 10
    to_install = data.packages.joinpath('LocalExtras')
    script.scratch_path.joinpath('constraints.txt').write_text(f'{to_install}[bar]')
    result = script.pip_install_local('-c', script.scratch_path / 'constraints.txt', f'{to_install}[baz]', allow_stderr_warning=True, expect_error=resolver_variant == 'resolvelib')
    if resolver_variant == 'resolvelib':
        msg = 'Unnamed requirements are not allowed as constraints'
        assert msg in result.stderr
    else:
        assert 'Building wheel for LocalExtras' in result.stdout
        result.did_create(script.site_packages / 'singlemodule.py')

def test_install_distribution_union_with_versions(script: PipTestEnvironment, data: TestData, resolver_variant: ResolverVariant) -> None:
    if False:
        return 10
    to_install_001 = data.packages.joinpath('LocalExtras')
    to_install_002 = data.packages.joinpath('LocalExtras-0.0.2')
    result = script.pip_install_local(f'{to_install_001}[bar]', f'{to_install_002}[baz]', expect_error=resolver_variant == 'resolvelib')
    if resolver_variant == 'resolvelib':
        assert 'Cannot install localextras[bar]' in result.stderr
        assert 'localextras[bar] 0.0.1 depends on localextras 0.0.1' in result.stdout
        assert 'localextras[baz] 0.0.2 depends on localextras 0.0.2' in result.stdout
    else:
        assert 'Successfully installed LocalExtras-0.0.1 simple-3.0 singlemodule-0.0.1' in result.stdout

@pytest.mark.xfail
def test_install_distribution_union_conflicting_extras(script: PipTestEnvironment, data: TestData) -> None:
    if False:
        while True:
            i = 10
    to_install = data.packages.joinpath('LocalExtras-0.0.2')
    result = script.pip_install_local(to_install, f'{to_install}[bar]', expect_error=True)
    assert 'installed' not in result.stdout
    assert 'Conflict' in result.stderr

def test_install_unsupported_wheel_link_with_marker(script: PipTestEnvironment) -> None:
    if False:
        return 10
    script.scratch_path.joinpath('with-marker.txt').write_text(textwrap.dedent('            {url}; {req}\n        ').format(url='https://github.com/a/b/c/asdf-1.5.2-cp27-none-xyz.whl', req='sys_platform == "xyz"'))
    result = script.pip('install', '-r', script.scratch_path / 'with-marker.txt')
    assert 'Ignoring asdf: markers \'sys_platform == "xyz"\' don\'t match your environment' in result.stdout
    assert len(result.files_created) == 0

def test_install_unsupported_wheel_file(script: PipTestEnvironment, data: TestData) -> None:
    if False:
        i = 10
        return i + 15
    path = data.packages.joinpath('simple.dist-0.1-py1-none-invalid.whl')
    script.scratch_path.joinpath('wheel-file.txt').write_text(f'{path}\n')
    result = script.pip('install', '-r', script.scratch_path / 'wheel-file.txt', expect_error=True, expect_stderr=True)
    assert 'simple.dist-0.1-py1-none-invalid.whl is not a supported wheel on this platform' in result.stderr
    assert len(result.files_created) == 0

def test_config_settings_local_to_package(script: PipTestEnvironment, common_wheels: Path, arg_recording_sdist_maker: ArgRecordingSdistMaker) -> None:
    if False:
        print('Hello World!')
    pyproject_toml = textwrap.dedent('\n        [build-system]\n        requires = ["setuptools"]\n        build-backend = "setuptools.build_meta"\n        ')
    simple0_sdist = arg_recording_sdist_maker('simple0', extra_files={'pyproject.toml': pyproject_toml}, depends=['foo'])
    foo_sdist = arg_recording_sdist_maker('foo', extra_files={'pyproject.toml': pyproject_toml})
    simple1_sdist = arg_recording_sdist_maker('simple1', extra_files={'pyproject.toml': pyproject_toml}, depends=['bar'])
    bar_sdist = arg_recording_sdist_maker('bar', extra_files={'pyproject.toml': pyproject_toml}, depends=['simple3'])
    simple3_sdist = arg_recording_sdist_maker('simple3', extra_files={'pyproject.toml': pyproject_toml})
    simple2_sdist = arg_recording_sdist_maker('simple2', extra_files={'pyproject.toml': pyproject_toml})
    reqs_file = script.scratch_path.joinpath('reqs.txt')
    reqs_file.write_text(textwrap.dedent('\n            simple0 --config-settings "--build-option=--verbose"\n            foo --config-settings "--build-option=--quiet"\n            simple1 --config-settings "--build-option=--verbose"\n            simple2\n            '))
    script.pip('install', '--no-index', '-f', script.scratch_path, '-f', common_wheels, '-r', reqs_file)
    simple0_args = simple0_sdist.args()
    assert '--verbose' in simple0_args
    foo_args = foo_sdist.args()
    assert '--quiet' in foo_args
    simple1_args = simple1_sdist.args()
    assert '--verbose' in simple1_args
    bar_args = bar_sdist.args()
    assert '--verbose' not in bar_args
    simple3_args = simple3_sdist.args()
    assert '--verbose' not in simple3_args
    simple2_args = simple2_sdist.args()
    assert '--verbose' not in simple2_args