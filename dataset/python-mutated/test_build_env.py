import os
import sys
from textwrap import dedent
from typing import Optional
import pytest
from pip._internal.build_env import BuildEnvironment, _get_system_sitepackages
from tests.lib import PipTestEnvironment, TestPipResult, create_basic_wheel_for_package, make_test_finder

def indent(text: str, prefix: str) -> str:
    if False:
        for i in range(10):
            print('nop')
    return '\n'.join(((prefix if line else '') + line for line in text.split('\n')))

def run_with_build_env(script: PipTestEnvironment, setup_script_contents: str, test_script_contents: Optional[str]=None) -> TestPipResult:
    if False:
        print('Hello World!')
    build_env_script = script.scratch_path / 'build_env.py'
    build_env_script.write_text(dedent('\n            import subprocess\n            import sys\n\n            from pip._internal.build_env import BuildEnvironment\n            from pip._internal.index.collector import LinkCollector\n            from pip._internal.index.package_finder import PackageFinder\n            from pip._internal.models.search_scope import SearchScope\n            from pip._internal.models.selection_prefs import (\n                SelectionPreferences\n            )\n            from pip._internal.network.session import PipSession\n            from pip._internal.utils.temp_dir import global_tempdir_manager\n\n            link_collector = LinkCollector(\n                session=PipSession(),\n                search_scope=SearchScope.create([{scratch!r}], [], False),\n            )\n            selection_prefs = SelectionPreferences(\n                allow_yanked=True,\n            )\n            finder = PackageFinder.create(\n                link_collector=link_collector,\n                selection_prefs=selection_prefs,\n            )\n\n            with global_tempdir_manager():\n                build_env = BuildEnvironment()\n            '.format(scratch=str(script.scratch_path))) + indent(dedent(setup_script_contents), '    ') + indent(dedent('\n                if len(sys.argv) > 1:\n                    with build_env:\n                        subprocess.check_call((sys.executable, sys.argv[1]))\n                '), '    '))
    args = ['python', os.fspath(build_env_script)]
    if test_script_contents is not None:
        test_script = script.scratch_path / 'test.py'
        test_script.write_text(dedent(test_script_contents))
        args.append(os.fspath(test_script))
    return script.run(*args)

def test_build_env_allow_empty_requirements_install() -> None:
    if False:
        i = 10
        return i + 15
    finder = make_test_finder()
    build_env = BuildEnvironment()
    for prefix in ('normal', 'overlay'):
        build_env.install_requirements(finder, [], prefix, kind='Installing build dependencies')

def test_build_env_allow_only_one_install(script: PipTestEnvironment) -> None:
    if False:
        return 10
    create_basic_wheel_for_package(script, 'foo', '1.0')
    create_basic_wheel_for_package(script, 'bar', '1.0')
    finder = make_test_finder(find_links=[os.fspath(script.scratch_path)])
    build_env = BuildEnvironment()
    for prefix in ('normal', 'overlay'):
        build_env.install_requirements(finder, ['foo'], prefix, kind=f'installing foo in {prefix}')
        with pytest.raises(AssertionError):
            build_env.install_requirements(finder, ['bar'], prefix, kind=f'installing bar in {prefix}')
        with pytest.raises(AssertionError):
            build_env.install_requirements(finder, [], prefix, kind=f'installing in {prefix}')

def test_build_env_requirements_check(script: PipTestEnvironment) -> None:
    if False:
        print('Hello World!')
    create_basic_wheel_for_package(script, 'foo', '2.0')
    create_basic_wheel_for_package(script, 'bar', '1.0')
    create_basic_wheel_for_package(script, 'bar', '3.0')
    create_basic_wheel_for_package(script, 'other', '0.5')
    script.pip_install_local('-f', script.scratch_path, 'foo', 'bar', 'other')
    run_with_build_env(script, "\n        r = build_env.check_requirements(['foo', 'bar', 'other'])\n        assert r == (set(), {'foo', 'bar', 'other'}), repr(r)\n\n        r = build_env.check_requirements(['foo>1.0', 'bar==3.0'])\n        assert r == (set(), {'foo>1.0', 'bar==3.0'}), repr(r)\n\n        r = build_env.check_requirements(['foo>3.0', 'bar>=2.5'])\n        assert r == (set(), {'foo>3.0', 'bar>=2.5'}), repr(r)\n        ")
    run_with_build_env(script, "\n        build_env.install_requirements(finder, ['foo', 'bar==3.0'], 'normal',\n                                       kind='installing foo in normal')\n\n        r = build_env.check_requirements(['foo', 'bar', 'other'])\n        assert r == (set(), {'other'}), repr(r)\n\n        r = build_env.check_requirements(['foo>1.0', 'bar==3.0'])\n        assert r == (set(), set()), repr(r)\n\n        r = build_env.check_requirements(['foo>3.0', 'bar>=2.5'])\n        assert r == ({('foo==2.0', 'foo>3.0')}, set()), repr(r)\n        ")
    run_with_build_env(script, "\n        build_env.install_requirements(finder, ['foo', 'bar==3.0'], 'normal',\n                                       kind='installing foo in normal')\n        build_env.install_requirements(finder, ['bar==1.0'], 'overlay',\n                                       kind='installing foo in overlay')\n\n        r = build_env.check_requirements(['foo', 'bar', 'other'])\n        assert r == (set(), {'other'}), repr(r)\n\n        r = build_env.check_requirements(['foo>1.0', 'bar==3.0'])\n        assert r == ({('bar==1.0', 'bar==3.0')}, set()), repr(r)\n\n        r = build_env.check_requirements(['foo>3.0', 'bar>=2.5'])\n        assert r == ({('bar==1.0', 'bar>=2.5'), ('foo==2.0', 'foo>3.0')},             set()), repr(r)\n        ")
    run_with_build_env(script, '\n        build_env.install_requirements(\n            finder,\n            ["bar==3.0"],\n            "normal",\n            kind="installing bar in normal",\n        )\n        r = build_env.check_requirements(\n            [\n                "bar==2.0; python_version < \'3.0\'",\n                "bar==3.0; python_version >= \'3.0\'",\n                "foo==4.0; extra == \'dev\'",\n            ],\n        )\n        assert r == (set(), set()), repr(r)\n        ')

def test_build_env_overlay_prefix_has_priority(script: PipTestEnvironment) -> None:
    if False:
        print('Hello World!')
    create_basic_wheel_for_package(script, 'pkg', '2.0')
    create_basic_wheel_for_package(script, 'pkg', '4.3')
    result = run_with_build_env(script, "\n        build_env.install_requirements(finder, ['pkg==2.0'], 'overlay',\n                                       kind='installing pkg==2.0 in overlay')\n        build_env.install_requirements(finder, ['pkg==4.3'], 'normal',\n                                       kind='installing pkg==4.3 in normal')\n        ", "\n        print(__import__('pkg').__version__)\n        ")
    assert result.stdout.strip() == '2.0', str(result)
if sys.version_info < (3, 12):
    BUILD_ENV_ERROR_DEBUG_CODE = "\n            from distutils.sysconfig import get_python_lib\n            print(\n                f'imported `pkg` from `{pkg.__file__}`',\n                file=sys.stderr)\n            print('system sites:\\n  ' + '\\n  '.join(sorted({\n                            get_python_lib(plat_specific=0),\n                            get_python_lib(plat_specific=1),\n                    })), file=sys.stderr)\n    "
else:
    BUILD_ENV_ERROR_DEBUG_CODE = "\n            from sysconfig import get_paths\n            paths = get_paths()\n            print(\n                f'imported `pkg` from `{pkg.__file__}`',\n                file=sys.stderr)\n            print('system sites:\\n  ' + '\\n  '.join(sorted({\n                            paths['platlib'],\n                            paths['purelib'],\n                    })), file=sys.stderr)\n    "

@pytest.mark.usefixtures('enable_user_site')
def test_build_env_isolation(script: PipTestEnvironment) -> None:
    if False:
        for i in range(10):
            print('nop')
    pkg_whl = create_basic_wheel_for_package(script, 'pkg', '1.0')
    script.pip_install_local(pkg_whl)
    script.pip_install_local('--ignore-installed', '--user', pkg_whl)
    target = script.scratch_path / 'pth_install'
    script.pip_install_local('-t', target, pkg_whl)
    (script.site_packages_path / 'build_requires.pth').write_text(str(target) + '\n')
    target = script.scratch_path / 'pypath_install'
    script.pip_install_local('-t', target, pkg_whl)
    script.environ['PYTHONPATH'] = target
    system_sites = _get_system_sitepackages()
    assert system_sites
    run_with_build_env(script, '', f"""\n        import sys\n\n        try:\n            import pkg\n        except ImportError:\n            pass\n        else:\n            {BUILD_ENV_ERROR_DEBUG_CODE}\n            print('sys.path:\\n  ' + '\\n  '.join(sys.path), file=sys.stderr)\n            sys.exit(1)\n        # second check: direct check of exclusion of system site packages\n        import os\n\n        normalized_path = [os.path.normcase(path) for path in sys.path]\n        for system_path in {system_sites!r}:\n            assert system_path not in normalized_path,             f"{{system_path}} found in {{normalized_path}}"\n        """)