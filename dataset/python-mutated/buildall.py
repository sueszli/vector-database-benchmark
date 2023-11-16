"""
Build all of the packages in a given directory.
"""
import dataclasses
import shutil
import subprocess
import sys
from collections import defaultdict
from collections.abc import Iterable
from datetime import datetime
from functools import total_ordering
from graphlib import TopologicalSorter
from pathlib import Path
from queue import PriorityQueue, Queue
from threading import Lock, Thread
from time import perf_counter, sleep
from typing import Any
from pyodide_lock import PyodideLockSpec
from pyodide_lock.spec import PackageSpec as PackageLockSpec
from rich.live import Live
from rich.progress import BarColumn, Progress, TimeElapsedColumn
from rich.spinner import Spinner
from rich.table import Table
from . import build_env, recipe
from .buildpkg import needs_rebuild
from .common import extract_wheel_metadata_file, find_matching_wheels, find_missing_executables, repack_zip_archive
from .io import MetaConfig, _BuildSpecTypes
from .logger import console_stdout, logger
from .pywasmcross import BuildArgs

class BuildError(Exception):

    def __init__(self, returncode: int) -> None:
        if False:
            return 10
        self.returncode = returncode
        super().__init__()

@total_ordering
@dataclasses.dataclass(eq=False, repr=False)
class BasePackage:
    pkgdir: Path
    name: str
    version: str
    disabled: bool
    meta: MetaConfig
    package_type: _BuildSpecTypes
    run_dependencies: list[str]
    host_dependencies: list[str]
    executables_required: list[str]
    dependencies: set[str]
    unbuilt_host_dependencies: set[str]
    host_dependents: set[str]
    unvendored_tests: Path | None = None
    file_name: str | None = None
    install_dir: str = 'site'
    _queue_idx: int | None = None

    def __lt__(self, other: Any) -> bool:
        if False:
            return 10
        return len(self.host_dependents) > len(other.host_dependents)

    def __eq__(self, other: Any) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return len(self.host_dependents) == len(other.host_dependents)

    def __repr__(self) -> str:
        if False:
            print('Hello World!')
        return f'{type(self).__name__}({self.name})'

    def needs_rebuild(self) -> bool:
        if False:
            return 10
        return needs_rebuild(self.pkgdir, self.pkgdir / 'build', self.meta.source)

    def build(self, build_args: BuildArgs) -> None:
        if False:
            print('Hello World!')
        raise NotImplementedError()

    def dist_artifact_path(self) -> Path:
        if False:
            return 10
        raise NotImplementedError()

    def tests_path(self) -> Path | None:
        if False:
            i = 10
            return i + 15
        return None

@dataclasses.dataclass
class Package(BasePackage):

    def __init__(self, pkgdir: Path, config: MetaConfig):
        if False:
            return 10
        self.pkgdir = pkgdir
        self.meta = config.copy(deep=True)
        self.name = self.meta.package.name
        self.version = self.meta.package.version
        self.disabled = self.meta.package.disabled
        self.package_type = self.meta.build.package_type
        assert self.name == pkgdir.name, f'{self.name} != {pkgdir.name}'
        self.run_dependencies = self.meta.requirements.run
        self.host_dependencies = self.meta.requirements.host
        self.executables_required = self.meta.requirements.executable
        self.dependencies = set(self.run_dependencies + self.host_dependencies)
        self.unbuilt_host_dependencies = set(self.host_dependencies)
        self.host_dependents = set()

    def dist_artifact_path(self) -> Path:
        if False:
            return 10
        dist_dir = self.pkgdir / 'dist'
        if self.package_type in ('shared_library', 'cpython_module'):
            candidates = list(dist_dir.glob('*.zip'))
        else:
            candidates = list(find_matching_wheels(dist_dir.glob('*.whl'), build_env.pyodide_tags()))
        if len(candidates) != 1:
            raise RuntimeError(f'Unexpected number of wheels/archives {len(candidates)} when building {self.name}')
        return candidates[0]

    def tests_path(self) -> Path | None:
        if False:
            return 10
        tests = list((self.pkgdir / 'dist').glob('*-tests.tar'))
        assert len(tests) <= 1
        if tests:
            return tests[0]
        return None

    def build(self, build_args: BuildArgs) -> None:
        if False:
            return 10
        p = subprocess.run([sys.executable, '-m', 'pyodide_build', 'buildpkg', str(self.pkgdir / 'meta.yaml'), f'--cflags={build_args.cflags}', f'--cxxflags={build_args.cxxflags}', f'--ldflags={build_args.ldflags}', f'--target-install-dir={build_args.target_install_dir}', f'--host-install-dir={build_args.host_install_dir}', '--force-rebuild'], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if p.returncode != 0:
            logger.error(f'Error building {self.name}. Printing build logs.')
            logfile = self.pkgdir / 'build.log'
            if logfile.is_file():
                logger.error(logfile.read_text(encoding='utf-8'))
            else:
                logger.error('ERROR: No build log found.')
            logger.error('ERROR: cancelling buildall')
            raise BuildError(p.returncode)

class PackageStatus:

    def __init__(self, *, name: str, idx: int, thread: int, total_packages: int) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.pkg_name = name
        self.prefix = f'[{idx}/{total_packages}] (thread {thread})'
        self.status = Spinner('dots', style='red', speed=0.2)
        self.table = Table.grid(padding=1)
        self.table.add_row(f'{self.prefix} building {self.pkg_name}', self.status)
        self.finished = False

    def finish(self, success: bool, elapsed_time: float) -> None:
        if False:
            return 10
        time = datetime.utcfromtimestamp(elapsed_time)
        if time.minute == 0:
            minutes = ''
        else:
            minutes = f'{time.minute}m '
        timestr = f'{minutes}{time.second}s'
        status = 'built' if success else 'failed'
        done_message = f'{self.prefix} {status} {self.pkg_name} in {timestr}'
        self.finished = True
        if success:
            logger.success(done_message)
        else:
            logger.error(done_message)

    def __rich__(self):
        if False:
            while True:
                i = 10
        return self.table

class ReplProgressFormatter:

    def __init__(self, num_packages: int) -> None:
        if False:
            return 10
        self.progress = Progress('[progress.description]{task.description}', BarColumn(), '{task.completed}/{task.total} [progress.percentage]{task.percentage:>3.0f}%', 'Time elapsed:', TimeElapsedColumn())
        self.task = self.progress.add_task('Building packages...', total=num_packages)
        self.packages: list[PackageStatus] = []
        self.reset_grid()

    def reset_grid(self):
        if False:
            return 10
        'Empty out the rendered grids.'
        self.top_grid = Table.grid()
        for package in self.packages:
            self.top_grid.add_row(package)
        self.main_grid = Table.grid()
        self.main_grid.add_row(self.top_grid)
        self.main_grid.add_row(self.progress)

    def add_package(self, *, name: str, idx: int, thread: int, total_packages: int) -> PackageStatus:
        if False:
            for i in range(10):
                print('nop')
        status = PackageStatus(name=name, idx=idx, thread=thread, total_packages=total_packages)
        self.packages.append(status)
        self.reset_grid()
        return status

    def remove_package(self, pkg: PackageStatus) -> None:
        if False:
            while True:
                i = 10
        self.packages.remove(pkg)
        self.reset_grid()

    def update_progress_bar(self):
        if False:
            print('Hello World!')
        'Step the progress bar by one (to show that a package finished)'
        self.progress.update(self.task, advance=1)

    def __rich__(self):
        if False:
            while True:
                i = 10
        return self.main_grid

def _validate_package_map(pkg_map: dict[str, BasePackage]) -> bool:
    if False:
        for i in range(10):
            print('nop')
    for (pkg_name, pkg) in pkg_map.items():
        for runtime_dep_name in pkg.run_dependencies:
            runtime_dep = pkg_map[runtime_dep_name]
            if runtime_dep.package_type == 'static_library':
                raise ValueError(f'{pkg_name} has an invalid dependency: {runtime_dep_name}. Static libraries must be a host dependency.')
    missing_executables = defaultdict(list)
    for (name, pkg) in pkg_map.items():
        for exe in find_missing_executables(pkg.executables_required):
            missing_executables[exe].append(name)
    if missing_executables:
        error_msg = 'The following executables are missing in the host system:\n'
        for (executable, pkgs) in missing_executables.items():
            error_msg += f"- {executable} (required by: {', '.join(pkgs)})\n"
        raise RuntimeError(error_msg)
    return True

def _parse_package_query(query: list[str] | str | None) -> tuple[set[str], set[str]]:
    if False:
        for i in range(10):
            print('nop')
    '\n    Parse a package query string into a list of requested packages and a list of\n    disabled packages.\n\n    Parameters\n    ----------\n    query\n        A list of packages to build, this can be a comma separated string.\n\n    Returns\n    -------\n    A tuple of two lists, the first list contains requested packages, the second\n    list contains disabled packages.\n\n    Examples\n    --------\n    >>> _parse_package_query(None)\n    (set(), set())\n    >>> requested, disabled = _parse_package_query("a,b,c")\n    >>> requested == {\'a\', \'b\', \'c\'}, disabled == set()\n    (True, True)\n    >>> requested, disabled = _parse_package_query("a,b,!c")\n    >>> requested == {\'a\', \'b\'}, disabled == {\'c\'}\n    (True, True)\n    >>> requested, disabled = _parse_package_query(["a", "b", "!c"])\n    >>> requested == {\'a\', \'b\'}, disabled == {\'c\'}\n    (True, True)\n    '
    if not query:
        query = []
    if isinstance(query, str):
        query = [el.strip() for el in query.split(',')]
    requested = set()
    disabled = set()
    for name in query:
        if not name:
            continue
        if name.startswith('!'):
            disabled.add(name[1:])
        else:
            requested.add(name)
    return (requested, disabled)

def generate_dependency_graph(packages_dir: Path, requested: set[str], disabled: set[str] | None=None) -> dict[str, BasePackage]:
    if False:
        return 10
    'This generates a dependency graph for given packages.\n\n    A node in the graph is a BasePackage object defined above, which maintains\n    a list of dependencies and also dependents. That is, each node stores both\n    incoming and outgoing edges.\n\n    The dependencies and dependents are stored via their name, and we have a\n    lookup table pkg_map: Dict[str, BasePackage] to look up the corresponding\n    BasePackage object. The function returns pkg_map, which contains all\n    packages in the graph as its values.\n\n    Parameters\n    ----------\n    packages_dir\n        A directory that contains packages\n    requested\n        A set of packages to build\n    disabled\n        A set of packages to not build\n\n    Returns\n    -------\n    A dictionary mapping package names to BasePackage objects\n    '
    pkg: BasePackage
    pkgname: str
    pkg_map: dict[str, BasePackage] = {}
    if not disabled:
        disabled = set()
    graph = {}
    all_recipes = recipe.load_all_recipes(packages_dir)
    no_numpy_dependents = 'no-numpy-dependents' in requested
    requested.discard('no-numpy-dependents')
    packages = requested.copy()
    while packages:
        pkgname = packages.pop()
        if pkgname not in all_recipes:
            raise ValueError(f'No metadata file found for the following package: {pkgname}')
        pkg = Package(packages_dir / pkgname, all_recipes[pkgname])
        pkg_map[pkgname] = pkg
        graph[pkgname] = pkg.dependencies
        for dep in pkg.dependencies:
            if pkg_map.get(dep) is None:
                packages.add(dep)
    for pkgname in TopologicalSorter(graph).static_order():
        pkg = pkg_map[pkgname]
        if pkgname in disabled:
            pkg.disabled = True
            continue
        if no_numpy_dependents and 'numpy' in pkg.dependencies:
            pkg.disabled = True
            continue
        for dep in pkg.dependencies:
            if pkg_map[dep].disabled:
                pkg.disabled = True
                break
    requested_with_deps = requested.copy()
    disabled_packages = set()
    for pkgname in reversed(list(TopologicalSorter(graph).static_order())):
        pkg = pkg_map[pkgname]
        if pkg.disabled:
            requested_with_deps.discard(pkgname)
            disabled_packages.add(pkgname)
            continue
        if pkgname not in requested_with_deps:
            continue
        requested_with_deps.update(pkg.dependencies)
        for dep in pkg.host_dependencies:
            pkg_map[dep].host_dependents.add(pkg.name)
    pkg_map = {name: pkg_map[name] for name in requested_with_deps}
    _validate_package_map(pkg_map)
    if disabled_packages:
        logger.warning(f"The following packages are disabled: {', '.join(disabled_packages)}")
    return pkg_map

def job_priority(pkg: BasePackage) -> int:
    if False:
        return 10
    if pkg.name == 'numpy':
        return 0
    else:
        return 1

def format_name_list(l: list[str]) -> str:
    if False:
        for i in range(10):
            print('nop')
    '\n    >>> format_name_list(["regex"])\n    \'regex\'\n    >>> format_name_list(["regex", "parso"])\n    \'regex and parso\'\n    >>> format_name_list(["regex", "parso", "jedi"])\n    \'regex, parso, and jedi\'\n    '
    if len(l) == 1:
        return l[0]
    most = l[:-1]
    if len(most) > 1:
        most = [x + ',' for x in most]
    return ' '.join(most) + ' and ' + l[-1]

def mark_package_needs_build(pkg_map: dict[str, BasePackage], pkg: BasePackage, needs_build: set[str]) -> None:
    if False:
        return 10
    '\n    Helper for generate_needs_build_set. Modifies needs_build in place.\n    Recursively add pkg and all of its dependencies to needs_build.\n    '
    if pkg.name in needs_build:
        return
    needs_build.add(pkg.name)
    for dep in pkg.host_dependents:
        mark_package_needs_build(pkg_map, pkg_map[dep], needs_build)

def generate_needs_build_set(pkg_map: dict[str, BasePackage]) -> set[str]:
    if False:
        i = 10
        return i + 15
    '\n    Generate the set of packages that need to be rebuilt.\n\n    This consists of:\n    1. packages whose source files have changed since they were last built\n       according to needs_rebuild, and\n    2. packages which depend on case 1 packages.\n    '
    needs_build: set[str] = set()
    for pkg in pkg_map.values():
        if pkg.needs_rebuild():
            mark_package_needs_build(pkg_map, pkg, needs_build)
    return needs_build

def build_from_graph(pkg_map: dict[str, BasePackage], build_args: BuildArgs, n_jobs: int=1, force_rebuild: bool=False) -> None:
    if False:
        return 10
    '\n    This builds packages in pkg_map in parallel, building at most n_jobs\n    packages at once.\n\n    We have a priority queue of packages we are ready to build (build_queue),\n    where a package is ready to build if all its dependencies are built. The\n    priority is based on the number of dependents --- we prefer to build\n    packages with more dependents first.\n\n    To build packages in parallel, we use a thread pool of n_jobs many\n    threads listening to build_queue. When the thread is free, it takes an\n    item off build_queue and builds it. Once the package is built, it sends the\n    package to the built_queue. The main thread listens to the built_queue and\n    checks if any of the dependents are ready to be built. If so, it adds the\n    package to the build queue.\n    '
    build_queue: PriorityQueue[tuple[int, BasePackage]] = PriorityQueue()
    if force_rebuild:
        needs_build = set(pkg_map.keys())
    else:
        needs_build = generate_needs_build_set(pkg_map)
    already_built = set(pkg_map.keys()).difference(needs_build)
    for pkg_name in needs_build:
        pkg_map[pkg_name].unbuilt_host_dependencies.difference_update(already_built)
    if already_built:
        logger.info(f'The following packages are already built: [bold]{format_name_list(sorted(already_built))}[/bold]')
    if not needs_build:
        logger.success('All packages already built. Quitting.')
        return
    logger.info(f'Building the following packages: [bold]{format_name_list(sorted(needs_build))}[/bold]')
    for pkg_name in needs_build:
        pkg = pkg_map[pkg_name]
        if len(pkg.unbuilt_host_dependencies) == 0:
            build_queue.put((job_priority(pkg), pkg))
    built_queue: Queue[BasePackage | Exception] = Queue()
    thread_lock = Lock()
    queue_idx = 1
    building_rust_pkg = False
    progress_formatter = ReplProgressFormatter(len(needs_build))

    def builder(n: int) -> None:
        if False:
            i = 10
            return i + 15
        nonlocal queue_idx, building_rust_pkg
        while True:
            (_, pkg) = build_queue.get()
            with thread_lock:
                if pkg.meta.is_rust_package():
                    if building_rust_pkg:
                        build_queue.put((job_priority(pkg), pkg))
                        sleep(0.1)
                        continue
                    building_rust_pkg = True
                pkg._queue_idx = queue_idx
                queue_idx += 1
            pkg_status = progress_formatter.add_package(name=pkg.name, idx=pkg._queue_idx, thread=n, total_packages=len(needs_build))
            t0 = perf_counter()
            success = True
            try:
                pkg.build(build_args)
            except Exception as e:
                built_queue.put(e)
                success = False
                return
            finally:
                pkg_status.finish(success, perf_counter() - t0)
                progress_formatter.remove_package(pkg_status)
            built_queue.put(pkg)
            with thread_lock:
                if pkg.meta.is_rust_package():
                    building_rust_pkg = False
            sleep(0.01)
    for n in range(0, n_jobs):
        Thread(target=builder, args=(n + 1,), daemon=True).start()
    num_built = len(already_built)
    with Live(progress_formatter, console=console_stdout):
        while num_built < len(pkg_map):
            match built_queue.get():
                case BuildError() as err:
                    raise SystemExit(err.returncode)
                case Exception() as err:
                    raise err
                case a_package:
                    assert not isinstance(a_package, Exception)
                    pkg = a_package
            num_built += 1
            progress_formatter.update_progress_bar()
            for _dependent in pkg.host_dependents:
                dependent = pkg_map[_dependent]
                dependent.unbuilt_host_dependencies.remove(pkg.name)
                if len(dependent.unbuilt_host_dependencies) == 0:
                    build_queue.put((job_priority(dependent), dependent))

def generate_packagedata(output_dir: Path, pkg_map: dict[str, BasePackage]) -> dict[str, PackageLockSpec]:
    if False:
        while True:
            i = 10
    packages: dict[str, PackageLockSpec] = {}
    for (name, pkg) in pkg_map.items():
        if not pkg.file_name or pkg.package_type == 'static_library':
            continue
        if not Path(output_dir, pkg.file_name).exists():
            continue
        pkg_entry = PackageLockSpec(name=name, version=pkg.version, file_name=pkg.file_name, install_dir=pkg.install_dir, package_type=pkg.package_type)
        pkg_entry.update_sha256(output_dir / pkg.file_name)
        pkg_type = pkg.package_type
        if pkg_type in ('shared_library', 'cpython_module'):
            pkg_entry.shared_library = True
            pkg_entry.install_dir = 'stdlib' if pkg_type == 'cpython_module' else 'dynlib'
        pkg_entry.depends = [x.lower() for x in pkg.run_dependencies]
        if pkg.package_type not in ('static_library', 'shared_library'):
            pkg_entry.imports = pkg.meta.package.top_level if pkg.meta.package.top_level else [name]
        packages[name.lower()] = pkg_entry
        if pkg.unvendored_tests:
            packages[name.lower()].unvendored_tests = True
            pkg_entry = PackageLockSpec(name=name + '-tests', version=pkg.version, depends=[name.lower()], file_name=pkg.unvendored_tests.name, install_dir=pkg.install_dir)
            pkg_entry.update_sha256(output_dir / pkg.unvendored_tests.name)
            packages[name.lower() + '-tests'] = pkg_entry
    packages = dict(sorted(packages.items()))
    return packages

def generate_lockfile(output_dir: Path, pkg_map: dict[str, BasePackage]) -> PyodideLockSpec:
    if False:
        while True:
            i = 10
    'Generate the package.json file'
    from . import __version__
    [platform, _, arch] = build_env.platform().rpartition('_')
    info = {'arch': arch, 'platform': platform, 'version': __version__, 'python': sys.version.partition(' ')[0]}
    packages = generate_packagedata(output_dir, pkg_map)
    return PyodideLockSpec(info=info, packages=packages)

def copy_packages_to_dist_dir(packages: Iterable[BasePackage], output_dir: Path, compression_level: int=6, metadata_files: bool=False) -> None:
    if False:
        print('Hello World!')
    for pkg in packages:
        if pkg.package_type == 'static_library':
            continue
        dist_artifact_path = pkg.dist_artifact_path()
        shutil.copy(dist_artifact_path, output_dir)
        repack_zip_archive(output_dir / dist_artifact_path.name, compression_level=compression_level)
        if metadata_files and dist_artifact_path.suffix == '.whl':
            extract_wheel_metadata_file(dist_artifact_path, output_dir / f'{dist_artifact_path.name}.metadata')
        test_path = pkg.tests_path()
        if test_path:
            shutil.copy(test_path, output_dir)

def build_packages(packages_dir: Path, targets: str, build_args: BuildArgs, n_jobs: int=1, force_rebuild: bool=False) -> dict[str, BasePackage]:
    if False:
        print('Hello World!')
    (requested, disabled) = _parse_package_query(targets)
    requested_packages = recipe.load_recipes(packages_dir, requested)
    pkg_map = generate_dependency_graph(packages_dir, set(requested_packages.keys()), disabled)
    build_from_graph(pkg_map, build_args, n_jobs, force_rebuild)
    for pkg in pkg_map.values():
        assert isinstance(pkg, Package)
        if pkg.package_type == 'static_library':
            continue
        pkg.file_name = pkg.dist_artifact_path().name
        pkg.unvendored_tests = pkg.tests_path()
    return pkg_map

def copy_logs(pkg_map: dict[str, BasePackage], log_dir: Path) -> None:
    if False:
        i = 10
        return i + 15
    '\n    Copy build logs of packages to the log directory.\n    Parameters\n    ----------\n    pkg_map\n        A dictionary mapping package names to package objects.\n    log_dir\n        The directory to copy the logs to.\n    '
    log_dir.mkdir(exist_ok=True, parents=True)
    logger.info(f'Copying build logs to {log_dir}')
    for pkg in pkg_map.values():
        log_file = pkg.pkgdir / 'build.log'
        if log_file.exists():
            shutil.copy(log_file, log_dir / f'{pkg.name}.log')
        else:
            logger.warning(f'Warning: {pkg.name} has no build log')

def install_packages(pkg_map: dict[str, BasePackage], output_dir: Path, compression_level: int=6, metadata_files: bool=False) -> None:
    if False:
        print('Hello World!')
    '\n    Install packages into the output directory.\n    - copies build artifacts (wheel, zip, ...) to the output directory\n    - create pyodide_lock.json\n\n\n    pkg_map\n        package map created from build_packages\n\n    output_dir\n        output directory to install packages into\n    '
    output_dir.mkdir(exist_ok=True, parents=True)
    logger.info(f'Copying built packages to {output_dir}')
    copy_packages_to_dist_dir(pkg_map.values(), output_dir, compression_level=compression_level, metadata_files=metadata_files)
    lockfile_path = output_dir / 'pyodide-lock.json'
    logger.info(f'Writing pyodide-lock.json to {lockfile_path}')
    package_data = generate_lockfile(output_dir, pkg_map)
    package_data.to_json(lockfile_path)

def set_default_build_args(build_args: BuildArgs) -> BuildArgs:
    if False:
        while True:
            i = 10
    args = dataclasses.replace(build_args)
    if args.cflags is None:
        args.cflags = build_env.get_build_flag('SIDE_MODULE_CFLAGS')
    if args.cxxflags is None:
        args.cxxflags = build_env.get_build_flag('SIDE_MODULE_CXXFLAGS')
    if args.ldflags is None:
        args.ldflags = build_env.get_build_flag('SIDE_MODULE_LDFLAGS')
    if args.target_install_dir is None:
        args.target_install_dir = build_env.get_build_flag('TARGETINSTALLDIR')
    if args.host_install_dir is None:
        args.host_install_dir = build_env.get_build_flag('HOSTINSTALLDIR')
    if args.compression_level is None:
        args.compression_level = int(build_env.get_build_flag('PYODIDE_ZIP_COMPRESSION_LEVEL'))
    return args