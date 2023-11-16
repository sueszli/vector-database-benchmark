import importlib.util
import os
from collections import namedtuple
from typing import Any, List, Optional
from pip._vendor import tomli
from pip._vendor.packaging.requirements import InvalidRequirement, Requirement
from pip._internal.exceptions import InstallationError, InvalidPyProjectBuildRequires, MissingPyProjectBuildRequires

def _is_list_of_str(obj: Any) -> bool:
    if False:
        return 10
    return isinstance(obj, list) and all((isinstance(item, str) for item in obj))

def make_pyproject_path(unpacked_source_directory: str) -> str:
    if False:
        while True:
            i = 10
    return os.path.join(unpacked_source_directory, 'pyproject.toml')
BuildSystemDetails = namedtuple('BuildSystemDetails', ['requires', 'backend', 'check', 'backend_path'])

def load_pyproject_toml(use_pep517: Optional[bool], pyproject_toml: str, setup_py: str, req_name: str) -> Optional[BuildSystemDetails]:
    if False:
        i = 10
        return i + 15
    "Load the pyproject.toml file.\n\n    Parameters:\n        use_pep517 - Has the user requested PEP 517 processing? None\n                     means the user hasn't explicitly specified.\n        pyproject_toml - Location of the project's pyproject.toml file\n        setup_py - Location of the project's setup.py file\n        req_name - The name of the requirement we're processing (for\n                   error reporting)\n\n    Returns:\n        None if we should use the legacy code path, otherwise a tuple\n        (\n            requirements from pyproject.toml,\n            name of PEP 517 backend,\n            requirements we should check are installed after setting\n                up the build environment\n            directory paths to import the backend from (backend-path),\n                relative to the project root.\n        )\n    "
    has_pyproject = os.path.isfile(pyproject_toml)
    has_setup = os.path.isfile(setup_py)
    if not has_pyproject and (not has_setup):
        raise InstallationError(f"{req_name} does not appear to be a Python project: neither 'setup.py' nor 'pyproject.toml' found.")
    if has_pyproject:
        with open(pyproject_toml, encoding='utf-8') as f:
            pp_toml = tomli.loads(f.read())
        build_system = pp_toml.get('build-system')
    else:
        build_system = None
    if has_pyproject and (not has_setup):
        if use_pep517 is not None and (not use_pep517):
            raise InstallationError('Disabling PEP 517 processing is invalid: project does not have a setup.py')
        use_pep517 = True
    elif build_system and 'build-backend' in build_system:
        if use_pep517 is not None and (not use_pep517):
            raise InstallationError('Disabling PEP 517 processing is invalid: project specifies a build backend of {} in pyproject.toml'.format(build_system['build-backend']))
        use_pep517 = True
    elif use_pep517 is None:
        use_pep517 = has_pyproject or not importlib.util.find_spec('setuptools') or (not importlib.util.find_spec('wheel'))
    assert use_pep517 is not None
    if not use_pep517:
        return None
    if build_system is None:
        build_system = {'requires': ['setuptools>=40.8.0', 'wheel'], 'build-backend': 'setuptools.build_meta:__legacy__'}
    assert build_system is not None
    if 'requires' not in build_system:
        raise MissingPyProjectBuildRequires(package=req_name)
    requires = build_system['requires']
    if not _is_list_of_str(requires):
        raise InvalidPyProjectBuildRequires(package=req_name, reason='It is not a list of strings.')
    for requirement in requires:
        try:
            Requirement(requirement)
        except InvalidRequirement as error:
            raise InvalidPyProjectBuildRequires(package=req_name, reason=f'It contains an invalid requirement: {requirement!r}') from error
    backend = build_system.get('build-backend')
    backend_path = build_system.get('backend-path', [])
    check: List[str] = []
    if backend is None:
        backend = 'setuptools.build_meta:__legacy__'
        check = ['setuptools>=40.8.0']
    return BuildSystemDetails(requires, backend, check, backend_path)