import json
import logging
import textwrap
from pathlib import Path
from typing import Collection, Dict, List, NamedTuple, Optional, Set, Tuple
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
try:
    from importlib import metadata
except ImportError:
    import importlib_metadata as metadata
from pipx.constants import WINDOWS
from pipx.util import PipxError, run_subprocess
logger = logging.getLogger(__name__)

class VenvInspectInformation(NamedTuple):
    distributions: Collection[metadata.Distribution]
    env: Dict[str, str]
    bin_path: Path

class VenvMetadata(NamedTuple):
    apps: List[str]
    app_paths: List[Path]
    apps_of_dependencies: List[str]
    app_paths_of_dependencies: Dict[str, List[Path]]
    package_version: str
    python_version: str

def get_dist(package: str, distributions: Collection[metadata.Distribution]) -> Optional[metadata.Distribution]:
    if False:
        while True:
            i = 10
    'Find matching distribution in the canonicalized sense.'
    for dist in distributions:
        if canonicalize_name(dist.metadata['name']) == canonicalize_name(package):
            return dist
    return None

def get_package_dependencies(dist: metadata.Distribution, extras: Set[str], env: Dict[str, str]) -> List[Requirement]:
    if False:
        print('Hello World!')
    eval_env = env.copy()
    if not extras:
        extras.add('')
    dependencies = []
    for req in map(Requirement, dist.requires or []):
        if not req.marker:
            dependencies.append(req)
        else:
            for extra in extras:
                eval_env['extra'] = extra
                if req.marker.evaluate(eval_env):
                    dependencies.append(req)
                    break
    return dependencies

def get_apps(dist: metadata.Distribution, bin_path: Path) -> List[str]:
    if False:
        return 10
    apps = set()
    sections = {'console_scripts', 'gui_scripts'}
    for ep in dist.entry_points:
        if ep.group not in sections:
            continue
        if (bin_path / ep.name).exists():
            apps.add(ep.name)
        if WINDOWS and (bin_path / (ep.name + '.exe')).exists():
            apps.add(ep.name + '.exe')
    for path in dist.files or []:
        if Path(path).parts[0] != '..':
            continue
        dist_file_path = Path(dist.locate_file(path))
        try:
            if dist_file_path.parent.samefile(bin_path):
                apps.add(path.name)
        except FileNotFoundError:
            pass
    inst_files = dist.read_text('installed-files.txt') or ''
    for line in inst_files.splitlines():
        entry = line.split(',')[0]
        inst_file_path = Path(dist.locate_file(entry)).resolve()
        try:
            if inst_file_path.parent.samefile(bin_path):
                apps.add(inst_file_path.name)
        except FileNotFoundError:
            pass
    return sorted(apps)

def _dfs_package_apps(dist: metadata.Distribution, package_req: Requirement, venv_inspect_info: VenvInspectInformation, app_paths_of_dependencies: Dict[str, List[Path]], dep_visited: Optional[Dict[str, bool]]=None) -> Dict[str, List[Path]]:
    if False:
        return 10
    if dep_visited is None:
        dep_visited = {canonicalize_name(package_req.name): True}
    dependencies = get_package_dependencies(dist, package_req.extras, venv_inspect_info.env)
    for dep_req in dependencies:
        dep_name = canonicalize_name(dep_req.name)
        if dep_name in dep_visited:
            continue
        dep_dist = get_dist(dep_req.name, venv_inspect_info.distributions)
        if dep_dist is None:
            raise PipxError(f'Pipx Internal Error: cannot find package {dep_req.name!r} metadata.')
        app_names = get_apps(dep_dist, venv_inspect_info.bin_path)
        if app_names:
            app_paths_of_dependencies[dep_name] = [venv_inspect_info.bin_path / app for app in app_names]
        dep_visited[dep_name] = True
        app_paths_of_dependencies = _dfs_package_apps(dep_dist, dep_req, venv_inspect_info, app_paths_of_dependencies, dep_visited)
    return app_paths_of_dependencies

def _windows_extra_app_paths(app_paths: List[Path]) -> List[Path]:
    if False:
        for i in range(10):
            print('nop')
    app_paths_output = app_paths.copy()
    for app_path in app_paths:
        win_app_path = app_path.parent / (app_path.stem + '-script.py')
        if win_app_path.exists():
            app_paths_output.append(win_app_path)
        win_app_path = app_path.parent / (app_path.stem + '.exe.manifest')
        if win_app_path.exists():
            app_paths_output.append(win_app_path)
    return app_paths_output

def fetch_info_in_venv(venv_python_path: Path) -> Tuple[List[str], Dict[str, str], str]:
    if False:
        return 10
    command_str = textwrap.dedent('\n        import json\n        import os\n        import platform\n        import sys\n\n        impl_ver = sys.implementation.version\n        implementation_version = "{0.major}.{0.minor}.{0.micro}".format(impl_ver)\n        if impl_ver.releaselevel != "final":\n            implementation_version = "{}{}{}".format(\n                implementation_version,\n                impl_ver.releaselevel[0],\n                impl_ver.serial,\n            )\n\n        sys_path = sys.path\n        try:\n            sys_path.remove("")\n        except ValueError:\n            pass\n\n        print(\n            json.dumps(\n                {\n                    "sys_path": sys_path,\n                    "python_version": "{0.major}.{0.minor}.{0.micro}".format(sys.version_info),\n                    "environment": {\n                        "implementation_name": sys.implementation.name,\n                        "implementation_version": implementation_version,\n                        "os_name": os.name,\n                        "platform_machine": platform.machine(),\n                        "platform_release": platform.release(),\n                        "platform_system": platform.system(),\n                        "platform_version": platform.version(),\n                        "python_full_version": platform.python_version(),\n                        "platform_python_implementation": platform.python_implementation(),\n                        "python_version": ".".join(platform.python_version_tuple()[:2]),\n                        "sys_platform": sys.platform,\n                    },\n                }\n            )\n        )\n        ')
    venv_info = json.loads(run_subprocess([venv_python_path, '-c', command_str], capture_stderr=False, log_cmd_str='<fetch_info_in_venv commands>').stdout)
    return (venv_info['sys_path'], venv_info['environment'], f"Python {venv_info['python_version']}")

def inspect_venv(root_package_name: str, root_package_extras: Set[str], venv_bin_path: Path, venv_python_path: Path) -> VenvMetadata:
    if False:
        while True:
            i = 10
    app_paths_of_dependencies: Dict[str, List[Path]] = {}
    apps_of_dependencies: List[str] = []
    root_req = Requirement(root_package_name)
    root_req.extras = root_package_extras
    (venv_sys_path, venv_env, venv_python_version) = fetch_info_in_venv(venv_python_path)
    distributions = tuple(metadata.distributions(path=venv_sys_path))
    venv_inspect_info = VenvInspectInformation(bin_path=venv_bin_path, env=venv_env, distributions=distributions)
    root_dist = get_dist(root_req.name, venv_inspect_info.distributions)
    if root_dist is None:
        raise PipxError(f'Pipx Internal Error: cannot find package {root_req.name!r} metadata.')
    app_paths_of_dependencies = _dfs_package_apps(root_dist, root_req, venv_inspect_info, app_paths_of_dependencies)
    apps = get_apps(root_dist, venv_bin_path)
    app_paths = [venv_bin_path / app for app in apps]
    if WINDOWS:
        app_paths = _windows_extra_app_paths(app_paths)
    for dep in app_paths_of_dependencies:
        apps_of_dependencies += [dep_path.name for dep_path in app_paths_of_dependencies[dep]]
        if WINDOWS:
            app_paths_of_dependencies[dep] = _windows_extra_app_paths(app_paths_of_dependencies[dep])
    venv_metadata = VenvMetadata(apps=apps, app_paths=app_paths, apps_of_dependencies=apps_of_dependencies, app_paths_of_dependencies=app_paths_of_dependencies, package_version=root_dist.version, python_version=venv_python_version)
    return venv_metadata