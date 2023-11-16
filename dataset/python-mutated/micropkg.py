"""A collection of CLI commands for working with Kedro micro-packages."""
from __future__ import annotations
import logging
import re
import shutil
import sys
import tarfile
import tempfile
from importlib import import_module
from pathlib import Path
from typing import Any, Iterable, Iterator
import click
from build.util import project_wheel_metadata
from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name
from rope.base.project import Project
from rope.contrib import generate
from rope.refactor.move import MoveModule
from rope.refactor.rename import Rename
from setuptools.discovery import FlatLayoutPackageFinder
from kedro.framework.cli.pipeline import _assert_pkg_name_ok, _check_pipeline_name, _get_artifacts_to_package, _sync_dirs
from kedro.framework.cli.utils import KedroCliError, _clean_pycache, call, command_with_verbosity, env_option, python_call
from kedro.framework.startup import ProjectMetadata
_PYPROJECT_TOML_TEMPLATE = '\n[build-system]\nrequires = ["setuptools"]\nbuild-backend = "setuptools.build_meta"\n\n[project]\nname = "{name}"\nversion = "{version}"\ndescription = "Micro-package `{name}`"\ndependencies = {install_requires}\n\n[tool.setuptools.packages]\nfind = {{}}\n'
logger = logging.getLogger(__name__)

class _EquivalentRequirement(Requirement):
    """Parse a requirement according to PEP 508.

    This class overrides __eq__ to be backwards compatible with pkg_resources.Requirement
    while making __str__ and __hash__ use the non-canonicalized name
    as agreed in https://github.com/pypa/packaging/issues/644,

    Implementation taken from https://github.com/pypa/packaging/pull/696/
    """

    def _iter_parts(self, name: str) -> Iterator[str]:
        if False:
            print('Hello World!')
        yield name
        if self.extras:
            formatted_extras = ','.join(sorted(self.extras))
            yield f'[{formatted_extras}]'
        if self.specifier:
            yield str(self.specifier)
        if self.url:
            yield f'@ {self.url}'
            if self.marker:
                yield ' '
        if self.marker:
            yield f'; {self.marker}'

    def __str__(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return ''.join(self._iter_parts(self.name))

    def __hash__(self) -> int:
        if False:
            print('Hello World!')
        return hash((self.__class__.__name__, *self._iter_parts(canonicalize_name(self.name))))

    def __eq__(self, other: Any) -> bool:
        if False:
            return 10
        return canonicalize_name(self.name) == canonicalize_name(other.name) and self.extras == other.extras and (self.specifier == other.specifier) and (self.url == other.url) and (self.marker == other.marker)

def _check_module_path(ctx, param, value):
    if False:
        print('Hello World!')
    if value and (not re.match('^[\\w.]+$', value)):
        message = 'The micro-package location you provided is not a valid Python module path'
        raise KedroCliError(message)
    return value

@click.group(name='Kedro')
def micropkg_cli():
    if False:
        print('Hello World!')
    pass

@micropkg_cli.group()
def micropkg():
    if False:
        i = 10
        return i + 15
    'Commands for working with micro-packages.'

@command_with_verbosity(micropkg, 'pull')
@click.argument('package_path', nargs=1, required=False)
@click.option('--all', '-a', 'all_flag', is_flag=True, help='Pull and unpack all micro-packages in the `pyproject.toml` package manifest section.')
@env_option(help='Environment to install the micro-package configuration to. Defaults to `base`.')
@click.option('--alias', type=str, default='', help='Rename the package.')
@click.option('-d', '--destination', type=click.Path(file_okay=False, dir_okay=False), default=None, help='Module location where to unpack under.')
@click.option('--fs-args', type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True, resolve_path=True), default=None, help='Location of a configuration file for the fsspec filesystem used to pull the package.')
@click.pass_obj
def pull_package(metadata: ProjectMetadata, package_path, env, alias, destination, fs_args, all_flag, **kwargs) -> None:
    if False:
        print('Hello World!')
    'Pull and unpack a modular pipeline and other micro-packages in your project.'
    if not package_path and (not all_flag):
        click.secho("Please specify a package path or add '--all' to pull all micro-packages in the 'pyproject.toml' package manifest section.")
        sys.exit(1)
    if all_flag:
        _pull_packages_from_manifest(metadata)
        return
    _pull_package(package_path, metadata, env=env, alias=alias, destination=destination, fs_args=fs_args)
    as_alias = f" as '{alias}'" if alias else ''
    message = f'Micro-package {package_path} pulled and unpacked{as_alias}!'
    click.secho(message, fg='green')

def _pull_package(package_path: str, metadata: ProjectMetadata, env: str=None, alias: str=None, destination: str=None, fs_args: str=None):
    if False:
        while True:
            i = 10
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir).resolve()
        _unpack_sdist(package_path, temp_dir_path, fs_args)
        contents = [member for member in temp_dir_path.iterdir() if member.is_dir()]
        if len(contents) != 1:
            raise KedroCliError(f'Invalid sdist was extracted: exactly one directory was expected, got {contents}')
        project_root_dir = contents[0]
        library_meta = project_wheel_metadata(project_root_dir)
        packages = [package for package in FlatLayoutPackageFinder().find(project_root_dir) if '.' not in package]
        if len(packages) != 1:
            raise KedroCliError(f'Invalid package contents: exactly one package was expected, got {packages}')
        package_name = packages[0]
        package_reqs = _get_all_library_reqs(library_meta)
        if package_reqs:
            requirements_txt = metadata.source_dir / 'requirements.txt'
            _append_package_reqs(requirements_txt, package_reqs, package_name)
        _clean_pycache(temp_dir_path)
        _install_files(metadata, package_name, project_root_dir, env, alias, destination)

def _pull_packages_from_manifest(metadata: ProjectMetadata) -> None:
    if False:
        return 10
    import anyconfig
    config_dict = anyconfig.load(metadata.config_file)
    config_dict = config_dict['tool']['kedro']
    build_specs = config_dict.get('micropkg', {}).get('pull')
    if not build_specs:
        click.secho("Nothing to pull. Please update the 'pyproject.toml' package manifest section.", fg='yellow')
        return
    for (package_path, specs) in build_specs.items():
        if 'alias' in specs:
            _assert_pkg_name_ok(specs['alias'].split('.')[-1])
        _pull_package(package_path, metadata, **specs)
        click.secho(f"Pulled and unpacked '{package_path}'!")
    click.secho('Micro-packages pulled and unpacked!', fg='green')

def _package_micropkgs_from_manifest(metadata: ProjectMetadata) -> None:
    if False:
        return 10
    import anyconfig
    config_dict = anyconfig.load(metadata.config_file)
    config_dict = config_dict['tool']['kedro']
    build_specs = config_dict.get('micropkg', {}).get('package')
    if not build_specs:
        click.secho("Nothing to package. Please update the 'pyproject.toml' package manifest section.", fg='yellow')
        return
    for (package_name, specs) in build_specs.items():
        if 'alias' in specs:
            _assert_pkg_name_ok(specs['alias'])
        _package_micropkg(package_name, metadata, **specs)
        click.secho(f"Packaged '{package_name}' micro-package!")
    click.secho('Micro-packages packaged!', fg='green')

@command_with_verbosity(micropkg, 'package')
@env_option(help='Environment where the micro-package configuration lives. Defaults to `base`.')
@click.option('--alias', type=str, default='', callback=_check_pipeline_name, help='Alternative name to package under.')
@click.option('-d', '--destination', type=click.Path(resolve_path=True, file_okay=False), help='Location where to create the source distribution file. Defaults to `dist/`.')
@click.option('--all', '-a', 'all_flag', is_flag=True, help='Package all micro-packages in the `pyproject.toml` package manifest section.')
@click.argument('module_path', nargs=1, required=False, callback=_check_module_path)
@click.pass_obj
def package_micropkg(metadata: ProjectMetadata, module_path, env, alias, destination, all_flag, **kwargs):
    if False:
        while True:
            i = 10
    'Package up a modular pipeline or micro-package as a Python source distribution.'
    if not module_path and (not all_flag):
        click.secho("Please specify a micro-package name or add '--all' to package all micro-packages in the 'pyproject.toml' package manifest section.")
        sys.exit(1)
    if all_flag:
        _package_micropkgs_from_manifest(metadata)
        return
    result_path = _package_micropkg(module_path, metadata, alias=alias, destination=destination, env=env)
    as_alias = f" as '{alias}'" if alias else ''
    message = f"'{metadata.package_name}.{module_path}' packaged{as_alias}! Location: {result_path}"
    click.secho(message, fg='green')

def _get_fsspec_filesystem(location: str, fs_args: str | None):
    if False:
        print('Hello World!')
    import anyconfig
    import fsspec
    from kedro.io.core import get_protocol_and_path
    (protocol, _) = get_protocol_and_path(location)
    fs_args_config = anyconfig.load(fs_args) if fs_args else {}
    try:
        return fsspec.filesystem(protocol, **fs_args_config)
    except Exception as exc:
        click.secho(str(exc), fg='red')
        click.secho("Trying to use 'pip download'...", fg='red')
        return None

def _is_within_directory(directory, target):
    if False:
        print('Hello World!')
    abs_directory = directory.resolve()
    abs_target = target.resolve()
    return abs_directory in abs_target.parents

def safe_extract(tar, path):
    if False:
        i = 10
        return i + 15
    for member in tar.getmembers():
        member_path = path / member.name
        if not _is_within_directory(path, member_path):
            raise Exception('Failed to safely extract tar file.')
    tar.extractall(path)

def _unpack_sdist(location: str, destination: Path, fs_args: str | None) -> None:
    if False:
        i = 10
        return i + 15
    filesystem = _get_fsspec_filesystem(location, fs_args)
    if location.endswith('.tar.gz') and filesystem and filesystem.exists(location):
        with filesystem.open(location) as fs_file:
            with tarfile.open(fileobj=fs_file, mode='r:gz') as tar_file:
                safe_extract(tar_file, destination)
    else:
        python_call('pip', ['download', '--no-deps', '--no-binary', ':all:', '--dest', str(destination), location])
        sdist_file = list(destination.glob('*.tar.gz'))
        if len(sdist_file) != 1:
            file_names = [sf.name for sf in sdist_file]
            raise KedroCliError(f'More than 1 or no sdist files found: {file_names}. There has to be exactly one source distribution file.')
        with tarfile.open(sdist_file[0], 'r:gz') as fs_file:
            safe_extract(fs_file, destination)

def _rename_files(conf_source: Path, old_name: str, new_name: str):
    if False:
        i = 10
        return i + 15
    config_files_to_rename = (each for each in conf_source.rglob('*') if each.is_file() and old_name in each.name)
    for config_file in config_files_to_rename:
        new_config_name = config_file.name.replace(old_name, new_name)
        config_file.rename(config_file.parent / new_config_name)

def _refactor_code_for_unpacking(project: Project, package_path: Path, tests_path: Path, alias: str | None, destination: str | None, project_metadata: ProjectMetadata) -> tuple[Path, Path]:
    if False:
        while True:
            i = 10
    'This is the reverse operation of `_refactor_code_for_package`, i.e\n    we go from:\n    <temp_dir>  # also the root of the Rope project\n    |__ <micro_package>  # or <alias>\n        |__ __init__.py\n    |__ tests  # only tests for <micro_package>\n        |__ __init__.py\n        |__ tests.py\n\n    to:\n    <temp_dir>  # also the root of the Rope project\n    |__ <package_name>\n        |__ __init__.py\n        |__ <path_to_micro_package>\n            |__ __init__.py\n            |__ <micro_package>\n                |__ __init__.py\n    |__ tests\n        |__ __init__.py\n        |__ <path_to_micro_package>\n            |__ __init__.py\n            |__ <micro_package>\n                |__ __init__.py\n    '

    def _move_package_with_conflicting_name(target: Path, original_name: str, desired_name: str=None) -> Path:
        if False:
            i = 10
            return i + 15
        _rename_package(project, original_name, 'tmp_name')
        full_path = _create_nested_package(project, target)
        _move_package(project, 'tmp_name', target.as_posix())
        desired_name = desired_name or original_name
        _rename_package(project, (target / 'tmp_name').as_posix(), desired_name)
        return full_path
    package_name = package_path.stem
    package_target = Path(project_metadata.package_name)
    tests_target = Path('tests')
    if destination:
        destination_path = Path(destination)
        package_target = package_target / destination_path
        tests_target = tests_target / destination_path
    if alias and alias != package_name:
        _rename_package(project, package_name, alias)
        package_name = alias
    if package_name == project_metadata.package_name:
        full_path = _move_package_with_conflicting_name(package_target, package_name)
    else:
        full_path = _create_nested_package(project, package_target)
        _move_package(project, package_name, package_target.as_posix())
    refactored_package_path = full_path / package_name
    if not tests_path.exists():
        return (refactored_package_path, tests_path)
    full_path = _move_package_with_conflicting_name(tests_target, original_name='tests', desired_name=package_name)
    refactored_tests_path = full_path / package_name
    return (refactored_package_path, refactored_tests_path)

def _install_files(project_metadata: ProjectMetadata, package_name: str, source_path: Path, env: str=None, alias: str=None, destination: str=None):
    if False:
        return 10
    env = env or 'base'
    (package_source, test_source, conf_source) = _get_package_artifacts(source_path, package_name)
    if conf_source.is_dir() and alias:
        _rename_files(conf_source, package_name, alias)
    module_path = alias or package_name
    if destination:
        module_path = f'{destination}.{module_path}'
    (package_dest, test_dest, conf_dest) = _get_artifacts_to_package(project_metadata, module_path=module_path, env=env)
    if conf_source.is_dir():
        _sync_dirs(conf_source, conf_dest)
        shutil.rmtree(str(conf_source))
    project = Project(source_path)
    (refactored_package_source, refactored_test_source) = _refactor_code_for_unpacking(project, package_source, test_source, alias, destination, project_metadata)
    project.close()
    if refactored_test_source.is_dir():
        _sync_dirs(refactored_test_source, test_dest)
    if refactored_package_source.is_dir():
        _sync_dirs(refactored_package_source, package_dest)

def _find_config_files(source_config_dir: Path, glob_patterns: list[str]) -> list[tuple[Path, str]]:
    if False:
        print('Hello World!')
    config_files: list[tuple[Path, str]] = []
    if source_config_dir.is_dir():
        config_files = [(path, path.parent.relative_to(source_config_dir).as_posix()) for glob_pattern in glob_patterns for path in source_config_dir.glob(glob_pattern) if path.is_file()]
    return config_files

def _get_default_version(metadata: ProjectMetadata, micropkg_module_path: str) -> str:
    if False:
        return 10
    try:
        micropkg_module = import_module(f'{metadata.package_name}.{micropkg_module_path}')
        return micropkg_module.__version__
    except (AttributeError, ModuleNotFoundError):
        logger.warning("Micropackage version not found in '%s.%s', will take the top-level one in '%s'", metadata.package_name, micropkg_module_path, metadata.package_name)
        project_module = import_module(f'{metadata.package_name}')
        return project_module.__version__

def _package_micropkg(micropkg_module_path: str, metadata: ProjectMetadata, alias: str=None, destination: str=None, env: str=None) -> Path:
    if False:
        for i in range(10):
            print('nop')
    micropkg_name = micropkg_module_path.split('.')[-1]
    package_dir = metadata.source_dir / metadata.package_name
    env = env or 'base'
    (package_source, package_tests, package_conf) = _get_artifacts_to_package(metadata, module_path=micropkg_module_path, env=env)
    configs_to_package = _find_config_files(package_conf, [f'**/parameters_{micropkg_name}.yml', f'**/{micropkg_name}/**/*', f'parameters*/**/{micropkg_name}.yml', f'parameters*/**/{micropkg_name}/**/*'])
    source_paths = (package_source, package_tests, configs_to_package)
    _validate_dir(package_source)
    destination = Path(destination) if destination else metadata.project_path / 'dist'
    version = _get_default_version(metadata, micropkg_module_path)
    _generate_sdist_file(micropkg_name=micropkg_name, destination=destination.resolve(), source_paths=source_paths, version=version, metadata=metadata, alias=alias)
    _clean_pycache(package_dir)
    _clean_pycache(metadata.project_path)
    return destination

def _validate_dir(path: Path) -> None:
    if False:
        i = 10
        return i + 15
    if not path.is_dir():
        raise KedroCliError(f"Directory '{path}' doesn't exist.")
    if not list(path.iterdir()):
        raise KedroCliError(f"'{path}' is an empty directory.")

def _get_sdist_name(name, version):
    if False:
        return 10
    return f'{name}-{version}.tar.gz'

def _sync_path_list(source: list[tuple[Path, str]], target: Path) -> None:
    if False:
        while True:
            i = 10
    for (source_path, suffix) in source:
        target_with_suffix = (target / suffix).resolve()
        _sync_dirs(source_path, target_with_suffix)

def _drop_comment(line):
    if False:
        while True:
            i = 10
    return line.partition(' #')[0]

def _make_install_requires(requirements_txt: Path) -> list[str]:
    if False:
        print('Hello World!')
    'Parses each line of requirements.txt into a version specifier valid to put in\n    install_requires.\n    Matches pkg_resources.parse_requirements'
    if not requirements_txt.exists():
        return []
    return [str(_EquivalentRequirement(_drop_comment(requirement_line))) for requirement_line in requirements_txt.read_text().splitlines() if requirement_line and (not requirement_line.startswith('#'))]

def _create_nested_package(project: Project, package_path: Path) -> Path:
    if False:
        i = 10
        return i + 15
    packages = package_path.parts
    parent = generate.create_package(project, packages[0])
    nested_path = Path(project.address) / packages[0]
    for package in packages[1:]:
        parent = generate.create_package(project, package, sourcefolder=parent)
        nested_path = nested_path / package
    return nested_path

def _move_package(project: Project, source: str, target: str) -> None:
    if False:
        i = 10
        return i + 15
    '\n    Move a Python package, refactoring relevant imports along the way.\n    A target of empty string means moving to the root of the `project`.\n\n    Args:\n        project: rope.base.Project holding the scope of the refactoring.\n        source: Name of the Python package to be moved. Can be a fully\n            qualified module path relative to the `project` root, e.g.\n            "package.pipelines.pipeline" or "package/pipelines/pipeline".\n        target: Destination of the Python package to be moved. Can be a fully\n            qualified module path relative to the `project` root, e.g.\n            "package.pipelines.pipeline" or "package/pipelines/pipeline".\n    '
    src_folder = project.get_module(source).get_resource()
    target_folder = project.get_module(target).get_resource()
    change = MoveModule(project, src_folder).get_changes(dest=target_folder)
    project.do(change)

def _rename_package(project: Project, old_name: str, new_name: str) -> None:
    if False:
        return 10
    '\n    Rename a Python package, refactoring relevant imports along the way,\n    as well as references in comments.\n\n    Args:\n        project: rope.base.Project holding the scope of the refactoring.\n        old_name: Old module name. Can be a fully qualified module path,\n            e.g. "package.pipelines.pipeline" or "package/pipelines/pipeline",\n            relative to the `project` root.\n        new_name: New module name. Can\'t be a fully qualified module path.\n    '
    folder = project.get_folder(old_name)
    change = Rename(project, folder).get_changes(new_name, docs=True)
    project.do(change)

def _refactor_code_for_package(project: Project, package_path: Path, tests_path: Path, alias: str | None, project_metadata: ProjectMetadata) -> None:
    if False:
        for i in range(10):
            print('nop')
    'In order to refactor the imports properly, we need to recreate\n    the same nested structure as in the project. Therefore, we create:\n    <temp_dir>  # also the root of the Rope project\n    |__ <package_name>\n        |__ __init__.py\n        |__ <path_to_micro_package>\n            |__ __init__.py\n            |__ <micro_package>\n                |__ __init__.py\n    |__ tests\n        |__ __init__.py\n        |__ path_to_micro_package\n            |__ __init__.py\n            |__ <micro_package>\n                |__ __init__.py\n    We then move <micro_package> outside of package src to top level ("")\n    in temp_dir, and rename folder & imports if alias provided.\n\n    For tests, we need to extract all the contents of <micro_package>\n    at into top-level `tests` folder. This is not possible in one go with\n    the Rope API, so we have to do it in a bit of a hacky way.\n    We rename <micro_package> to a `tmp_name` and move it at top-level ("")\n    in temp_dir. We remove the old `tests` folder and rename `tmp_name` to `tests`.\n\n    The final structure should be:\n    <temp_dir>  # also the root of the Rope project\n    |__ <micro_package>  # or <alias>\n        |__ __init__.py\n    |__ tests  # only tests for <micro_package>\n        |__ __init__.py\n        |__ test.py\n    '

    def _move_package_with_conflicting_name(target: Path, conflicting_name: str):
        if False:
            for i in range(10):
                print('nop')
        tmp_name = 'tmp_name'
        tmp_module = target.parent / tmp_name
        _rename_package(project, target.as_posix(), tmp_name)
        _move_package(project, tmp_module.as_posix(), '')
        shutil.rmtree(Path(project.address) / conflicting_name)
        _rename_package(project, tmp_name, conflicting_name)
    package_target = package_path.relative_to(project_metadata.source_dir)
    full_path = _create_nested_package(project, package_target)
    _sync_dirs(package_path, full_path, overwrite=True)
    if tests_path.exists():
        tests_target = tests_path.relative_to(project_metadata.source_dir)
        full_path = _create_nested_package(project, tests_target)
        _sync_dirs(tests_path, full_path, overwrite=True)
    micro_package_name = package_target.stem
    if micro_package_name == project_metadata.package_name:
        _move_package_with_conflicting_name(package_target, micro_package_name)
    else:
        _move_package(project, package_target.as_posix(), '')
        shutil.rmtree(Path(project.address) / project_metadata.package_name)
    if alias:
        _rename_package(project, micro_package_name, alias)
    if tests_path.exists():
        _move_package_with_conflicting_name(tests_target, 'tests')

def _generate_sdist_file(micropkg_name: str, destination: Path, source_paths: tuple[Path, Path, list[tuple[Path, str]]], version: str, metadata: ProjectMetadata, alias: str=None) -> None:
    if False:
        print('Hello World!')
    package_name = alias or micropkg_name
    (package_source, tests_source, conf_source) = source_paths
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir).resolve()
        project = Project(temp_dir_path)
        _refactor_code_for_package(project, package_source, tests_source, alias, metadata)
        project.close()
        (_, _, conf_target) = _get_package_artifacts(temp_dir_path, package_name)
        _sync_path_list(conf_source, conf_target)
        if conf_target.is_dir() and alias:
            _rename_files(conf_target, micropkg_name, alias)
        try:
            install_requires = _make_install_requires(package_source / 'requirements.txt')
        except Exception as exc:
            click.secho('FAILED', fg='red')
            cls = exc.__class__
            raise KedroCliError(f'{cls.__module__}.{cls.__qualname__}: {exc}') from exc
        _generate_manifest_file(temp_dir_path)
        _generate_pyproject_file(package_name, version, install_requires, temp_dir_path)
        package_file = destination / _get_sdist_name(name=package_name, version=version)
        if package_file.is_file():
            click.secho(f'Package file {package_file} will be overwritten!', fg='yellow')
        call([sys.executable, '-m', 'build', '--sdist', '--outdir', str(destination)], cwd=temp_dir)

def _generate_manifest_file(output_dir: Path):
    if False:
        i = 10
        return i + 15
    manifest_file = output_dir / 'MANIFEST.in'
    manifest_file.write_text('\n        global-include README.md\n        global-include config/parameters*\n        global-include config/**/parameters*\n        global-include config/parameters*/**\n        global-include config/parameters*/**/*\n        ')

def _generate_pyproject_file(package_name: str, version: str, install_requires: list[str], output_dir: Path) -> Path:
    if False:
        return 10
    pyproject_file = output_dir / 'pyproject.toml'
    pyproject_file_context = {'name': package_name, 'version': version, 'install_requires': install_requires}
    pyproject_file.write_text(_PYPROJECT_TOML_TEMPLATE.format(**pyproject_file_context))
    return pyproject_file

def _get_package_artifacts(source_path: Path, package_name: str) -> tuple[Path, Path, Path]:
    if False:
        print('Hello World!')
    'From existing package, returns in order:\n    source_path, tests_path, config_path\n    '
    artifacts = (source_path / package_name, source_path / 'tests', source_path / package_name / 'config')
    return artifacts

def _append_package_reqs(requirements_txt: Path, package_reqs: list[str], package_name: str) -> None:
    if False:
        print('Hello World!')
    'Appends micro-package requirements to project level requirements.txt'
    incoming_reqs = _safe_parse_requirements(package_reqs)
    if requirements_txt.is_file():
        existing_reqs = _safe_parse_requirements(requirements_txt.read_text())
        reqs_to_add = set(incoming_reqs) - set(existing_reqs)
        if not reqs_to_add:
            return
        sorted_reqs = sorted((str(req) for req in reqs_to_add))
        sep = '\n'
        with open(requirements_txt, 'a', encoding='utf-8') as file:
            file.write(f'\n\n# Additional requirements from micro-package `{package_name}`:\n')
            file.write(sep.join(sorted_reqs))
        click.secho(f"Added the following requirements from micro-package '{package_name}' to requirements.txt:\n{sep.join(sorted_reqs)}")
    else:
        click.secho('No project requirements.txt found. Copying contents from project requirements.txt...')
        sorted_reqs = sorted((str(req) for req in incoming_reqs))
        sep = '\n'
        with open(requirements_txt, 'a', encoding='utf-8') as file:
            file.write(sep.join(sorted_reqs))
    click.secho("Use 'kedro build-reqs' to compile and 'pip install -r src/requirements.lock' to install the updated list of requirements.")

def _get_all_library_reqs(metadata):
    if False:
        return 10
    'Get all library requirements from metadata, leaving markers intact.'
    return [str(_EquivalentRequirement(dep_str)) for dep_str in metadata.get_all('Requires-Dist', [])]

def _safe_parse_requirements(requirements: str | Iterable[str]) -> set[_EquivalentRequirement]:
    if False:
        for i in range(10):
            print('nop')
    'Safely parse a requirement or set of requirements. This avoids blowing up when it\n    encounters a requirement it cannot parse (e.g. `-r requirements.txt`). This way\n    we can still extract all the parseable requirements out of a set containing some\n    unparseable requirements.\n    '
    parseable_requirements = set()
    if isinstance(requirements, str):
        requirements = requirements.splitlines()
    for requirement_line in requirements:
        if requirement_line and (not requirement_line.startswith('#')) and (not requirement_line.startswith('-e')):
            try:
                parseable_requirements.add(_EquivalentRequirement(_drop_comment(requirement_line)))
            except InvalidRequirement:
                continue
    return parseable_requirements