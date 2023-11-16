import sys
from distutils.version import LooseVersion
from importlib import import_module
from typing import Dict, Optional, Union
from importlib_metadata import distributions
from pycaret.internal.logging import get_logger, redirect_output
logger = get_logger()
INSTALLED_MODULES = None

def _try_import_and_get_module_version(modname: str) -> Optional[Union[LooseVersion, bool]]:
    if False:
        for i in range(10):
            print('nop')
    'Returns False if module is not installed, None if version is not available'
    try:
        if modname in sys.modules:
            mod = sys.modules[modname]
        elif logger:
            with redirect_output(logger):
                mod = import_module(modname)
        else:
            mod = import_module(modname)
        try:
            ver = mod.__version__
        except AttributeError:
            ver = None
    except ImportError:
        ver = False
    if ver:
        ver = LooseVersion(ver)
    return ver

def get_installed_modules() -> Dict[str, Optional[LooseVersion]]:
    if False:
        while True:
            i = 10
    '\n    Get installed modules and their versions from pip metadata.\n    '
    global INSTALLED_MODULES
    if not INSTALLED_MODULES:
        module_versions = {}
        for dist in distributions():
            for pkg in (dist.read_text('top_level.txt') or '').split():
                try:
                    ver = LooseVersion(dist.metadata['Version'])
                except Exception:
                    ver = None
                module_versions[pkg] = ver
        INSTALLED_MODULES = module_versions
    return INSTALLED_MODULES

def _get_module_version(modname: str) -> Optional[Union[LooseVersion, bool]]:
    if False:
        while True:
            i = 10
    'Will cache the version in INSTALLED_MODULES\n\n    Returns False if module is not installed.'
    installed_modules = get_installed_modules()
    if modname not in installed_modules:
        installed_modules[modname] = _try_import_and_get_module_version(modname)
    return installed_modules[modname]

def get_module_version(modname: str) -> Optional[LooseVersion]:
    if False:
        print('Hello World!')
    'Raises a ValueError if module is not installed'
    version = _get_module_version(modname)
    if version is False:
        raise ValueError(f"Module '{modname}' is not installed.")
    return version

def is_module_installed(modname: str) -> bool:
    if False:
        while True:
            i = 10
    try:
        get_module_version(modname)
        return True
    except ValueError:
        return False

def _check_soft_dependencies(package: str, severity: str='error', extra: Optional[str]='all_extras', install_name: Optional[str]=None) -> bool:
    if False:
        print('Hello World!')
    'Check if all soft dependencies are installed and raise appropriate error message\n    when not.\n\n    Parameters\n    ----------\n    package : str\n        Package to check\n    severity : str, optional\n        Whether to raise an error ("error") or just a warning message ("warning"),\n        by default "error"\n    extra : Optional[str], optional\n        The \'extras\' that will install this package, by default "all_extras".\n        If None, it means that the dependency is not available in optional\n        requirements file and must be installed by the user on their own.\n    install_name : Optional[str], optional\n        The package name to install, by default None\n        If none, the name in `package` argument is used\n\n    Returns\n    -------\n    bool\n        If error is set to "warning", returns True if package can be imported or False\n        if it can not be imported\n\n    Raises\n    ------\n    ModuleNotFoundError\n        User friendly error with suggested action to install all required soft\n        dependencies\n    RuntimeError\n        Is the severity argument is not one of the allowed values\n    '
    install_name = install_name or package
    package_available = is_module_installed(package)
    if package_available:
        ver = get_module_version(package)
        logger.info('Soft dependency imported: {k}: {stat}'.format(k=package, stat=str(ver)))
    else:
        msg = f"\n'{package}' is a soft dependency and not included in the pycaret installation. Please run: `pip install {install_name}` to install."
        if extra is not None:
            msg += f'\nAlternately, you can install this by running `pip install pycaret[{extra}]`'
        if severity == 'error':
            logger.exception(f'{msg}')
            raise ModuleNotFoundError(msg)
        elif severity == 'warning':
            logger.warning(f'{msg}')
            package_available = False
        else:
            raise RuntimeError(f'Error in calling _check_soft_dependencies, severity argument must be "error" or "warning", found "{severity}".')
    return package_available