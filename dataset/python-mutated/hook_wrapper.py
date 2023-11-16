"""
Hooks Wrapper Class
"""
import importlib
import logging
from pathlib import Path
from typing import Dict, List, Optional, cast
from .exceptions import HookPackageExecuteFunctionalityException, InvalidHookWrapperException
from .hook_config import HookPackageConfig
LOG = logging.getLogger(__name__)
INTERNAL_PACKAGES_ROOT = Path(__file__).parent / '..' / '..' / 'hook_packages'

class IacHookWrapper:
    """IacHookWrapper

    An IacHookWrapper instance, upon instantiation, looks up the hook package with the specified hook package ID.
    It provides the "prepare" method, which generates an IaC metadata and output the location of the metadata file.

    Example:
    ```
    hook = IacHookWrapper("terraform")
    metadata_loc = hook.prepare("path/to/iac_project", "path/to/output", True)
    ```
    """
    _hook_name: str
    _config: Optional[HookPackageConfig]

    def __init__(self, hook_name: str):
        if False:
            for i in range(10):
                print('nop')
        '\n        Parameters\n        ----------\n        hook_name: str\n            Hook name\n        '
        self._hook_name = hook_name
        self._config = None
        self._load_hook_package(hook_name)

    def prepare(self, output_dir_path: str, iac_project_path: Optional[str]=None, debug: bool=False, aws_profile: Optional[str]=None, aws_region: Optional[str]=None, skip_prepare_infra: bool=False, plan_file: Optional[str]=None, project_root_dir: Optional[str]=None) -> str:
        if False:
            return 10
        '\n        Run the prepare hook to generate the IaC Metadata file.\n\n        Parameters\n        ----------\n        output_dir_path: str\n            the path where the hook can create the generated Metadata files. Required\n        iac_project_path: str\n            the path where the hook can find the TF application. Default value in current work directory.\n        debug: bool\n            True/False flag to tell the hooks if should print debugging logs or not. Default is False.\n        aws_profile: str\n            AWS profile to use. Default is None (use default profile)\n        aws_region: str\n            AWS region to use. Default is None (use default region)\n        skip_prepare_infra: bool\n            Flag to skip prepare hook if we already have the metadata file. Default is False.\n        plan_file: Optional[str]\n            Provided plan file to use instead of generating one from the hook\n        project_root_dir: Optional[str]\n            The Project root directory that contains the application directory, src code, and other modules\n        Returns\n        -------\n        str\n            Path to the generated IaC Metadata file\n        '
        LOG.info('Executing prepare hook of hook "%s"', self._hook_name)
        params = {'IACProjectPath': iac_project_path if iac_project_path else str(Path.cwd()), 'OutputDirPath': output_dir_path, 'Debug': debug, 'SkipPrepareInfra': skip_prepare_infra}
        if aws_profile:
            params['Profile'] = aws_profile
        if aws_region:
            params['Region'] = aws_region
        if plan_file:
            params['PlanFile'] = plan_file
        if project_root_dir:
            params['ProjectRootDir'] = project_root_dir
        output = self._execute('prepare', params)
        metadata_file_loc = None
        iac_applications: Dict[str, Dict] = output.get('iac_applications', {})
        if iac_applications and len(iac_applications) == 1:
            main_application = list(iac_applications.values())[0]
            metadata_file_loc = main_application.get('metadata_file')
        if not metadata_file_loc:
            raise InvalidHookWrapperException('Metadata file path not found in the prepare hook output')
        LOG.debug('Metadata file location - %s', metadata_file_loc)
        return cast(str, metadata_file_loc)

    def _load_hook_package(self, hook_name: str) -> None:
        if False:
            return 10
        'Find and load hook package config with given hook name\n\n        Parameters\n        ----------\n        hook_name: str\n            Hook name\n        '
        LOG.debug('Looking for internal hook package')
        for child in INTERNAL_PACKAGES_ROOT.iterdir():
            if child.name == hook_name:
                LOG.debug('Loaded internal hook package "%s"', hook_name)
                self._config = HookPackageConfig(child)
                return
        raise InvalidHookWrapperException(f'Cannot locate hook package with hook_name "{hook_name}"')

    def _execute(self, functionality_key: str, params: Optional[Dict]=None) -> Dict:
        if False:
            return 10
        '\n        Execute a functionality with given key\n\n        Parameters\n        ----------\n        functionality_key: str\n            The key of the functionality\n        params: Dict\n            A dict of parameters to pass into the execution\n\n        Returns\n        -------\n        Dict\n            the output from the execution\n        '
        if not self._config:
            raise InvalidHookWrapperException('Config is missing. You must instantiate a hook with a valid config')
        if functionality_key not in self._config.functionalities:
            raise HookPackageExecuteFunctionalityException(f'Functionality "{functionality_key}" is not defined in the hook package')
        functionality = self._config.functionalities[functionality_key]
        if functionality.entry_method:
            return _execute_as_module(functionality.module, functionality.method, params)
        raise InvalidHookWrapperException(f'Functionality "{functionality_key}" is missing an "entry_method"')

def _execute_as_module(module: str, method: str, params: Optional[Dict]=None) -> Dict:
    if False:
        print('Hello World!')
    '\n    Execute a module/method with given module and given method\n\n    Parameters\n    ----------\n    module: str\n        the module where the method lives in\n    method: str\n        the name of the method to execute\n    params: Dict\n        A dict of parameters to pass into the execution\n\n    Returns\n    -------\n    Dict\n        the output from the execution\n    '
    try:
        mod = importlib.import_module(module)
    except ImportError as e:
        raise InvalidHookWrapperException(f'Import error - HookFunctionality module "{module}"') from e
    if not hasattr(mod, method):
        raise InvalidHookWrapperException(f'HookFunctionality module "{module}" has no method "{method}"')
    result = getattr(mod, method)(params)
    return cast(Dict, result)

def get_available_hook_packages_ids() -> List[str]:
    if False:
        while True:
            i = 10
    '\n    return a list of available hook names.\n\n    Returns\n    -------\n    List\n        The available hook names.\n    '
    LOG.debug('Return available internal hook packages')
    hook_packages_ids = []
    for child in INTERNAL_PACKAGES_ROOT.iterdir():
        if child.is_dir() and child.name[0].isalpha():
            hook_packages_ids.append(child.name)
    return hook_packages_ids