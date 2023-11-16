"""SyncFlow base class """
import logging
from abc import ABC, abstractmethod
from enum import Enum
from os import environ
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, Any, Dict, List, NamedTuple, Optional, Set, cast
from boto3.session import Session
from samcli.lib.build.app_builder import ApplicationBuildResult
from samcli.lib.providers.provider import ResourceIdentifier, Stack, get_resource_by_id
from samcli.lib.sync.exceptions import MissingLockException, MissingPhysicalResourceError
from samcli.lib.utils.boto_utils import get_boto_client_provider_from_session_with_config
from samcli.lib.utils.lock_distributor import LockChain, LockDistributor
from samcli.lib.utils.resources import RESOURCES_WITH_LOCAL_PATHS
if TYPE_CHECKING:
    from samcli.commands.build.build_context import BuildContext
    from samcli.commands.deploy.deploy_context import DeployContext
    from samcli.commands.sync.sync_context import SyncContext
LOG = logging.getLogger(__name__)

def get_default_retry_config() -> Optional[Dict]:
    if False:
        return 10
    '\n    Returns a default retry config if nothing is overriden by environment variables\n    '
    if environ.get('AWS_MAX_ATTEMPTS') or environ.get('AWS_RETRY_MODE'):
        return None
    return {'max_attempts': 10, 'mode': 'standard'}

class ApiCallTypes(Enum):
    """API call stages that can be locked on"""
    BUILD = 'Build'
    UPDATE_FUNCTION_CONFIGURATION = 'UpdateFunctionConfiguration'
    UPDATE_FUNCTION_CODE = 'UpdateFunctionCode'

class ResourceAPICall(NamedTuple):
    """Named tuple for a resource and its potential API calls"""
    shared_resource: str
    api_calls: List[ApiCallTypes]

class SyncFlow(ABC):
    """Base class for a SyncFlow"""
    _log_name: str
    _build_context: 'BuildContext'
    _deploy_context: 'DeployContext'
    _sync_context: 'SyncContext'
    _stacks: Optional[List[Stack]]
    _session: Optional[Session]
    _physical_id_mapping: Dict[str, str]
    _locks: Optional[Dict[str, Lock]]
    _local_sha: Optional[str]
    _application_build_result: Optional[ApplicationBuildResult]

    def __init__(self, build_context: 'BuildContext', deploy_context: 'DeployContext', sync_context: 'SyncContext', physical_id_mapping: Dict[str, str], log_name: str, stacks: Optional[List[Stack]]=None, application_build_result: Optional[ApplicationBuildResult]=None):
        if False:
            print('Hello World!')
        '\n        Parameters\n        ----------\n        build_context : BuildContext\n            BuildContext used for build related parameters\n        deploy_context : BuildContext\n            DeployContext used for this deploy related parameters\n        sync_context: SyncContext\n            SyncContext object that obtains sync information.\n        physical_id_mapping : Dict[str, str]\n            Mapping between resource logical identifier and physical identifier\n        log_name : str\n            Name to be used for logging purposes\n        stacks : List[Stack], optional\n            List of stacks containing a root stack and optional nested stacks\n         application_build_result: Optional[ApplicationBuildResult]\n            Pre-build ApplicationBuildResult which can be re-used during SyncFlows\n        '
        self._build_context = build_context
        self._deploy_context = deploy_context
        self._sync_context = sync_context
        self._log_name = log_name
        self._stacks = stacks
        self._session = None
        self._physical_id_mapping = physical_id_mapping
        self._locks = None
        self._local_sha = None
        self._application_build_result = application_build_result

    def set_up(self) -> None:
        if False:
            return 10
        'Clients and other expensives setups should be handled here instead of constructor'
        pass

    def _get_session(self) -> Session:
        if False:
            return 10
        if not self._session:
            self._session = Session(profile_name=self._deploy_context.profile, region_name=self._deploy_context.region)
        return self._session

    def _boto_client(self, client_name: str):
        if False:
            return 10
        default_retry_config = get_default_retry_config()
        if not default_retry_config:
            LOG.debug("Creating boto client (%s) with user's retry config", client_name)
            return get_boto_client_provider_from_session_with_config(self._get_session())(client_name)
        LOG.debug('Creating boto client (%s) with default retry config', client_name)
        return get_boto_client_provider_from_session_with_config(self._get_session(), retries=default_retry_config)(client_name)

    @property
    @abstractmethod
    def sync_state_identifier(self) -> str:
        if False:
            i = 10
            return i + 15
        '\n        Sync state is the unique identifier for each sync flow\n        We store the identifier in sync state toml file as key\n        '
        raise NotImplementedError('sync_state_identifier')

    @abstractmethod
    def gather_resources(self) -> None:
        if False:
            while True:
                i = 10
        'Local operations that need to be done before comparison and syncing with remote\n        Ex: Building lambda functions\n        '
        raise NotImplementedError('gather_resources')

    def _update_local_hash(self) -> None:
        if False:
            i = 10
            return i + 15
        'Updates the latest local hash of the sync flow which then can be used for comparison for next run'
        if not self._local_sha:
            LOG.debug('%sNo local hash is configured, skipping to update local hash', self.log_prefix)
            return
        self._sync_context.update_resource_sync_state(self.sync_state_identifier, self._local_sha)

    def compare_local(self) -> bool:
        if False:
            return 10
        'Comparison between local resource and its local stored state.\n        If the resources are identical, sync and gather dependencies will be skipped.\n        Simply return False if there is no comparison needed.\n        Ex: Comparing local Lambda function artifact with stored SHA256\n\n        Returns\n        -------\n        bool\n            Return True if current resource and cached are in sync. Skipping rest of the execution.\n            Return False otherwise.\n        '
        stored_sha = self._sync_context.get_resource_latest_sync_hash(self.sync_state_identifier)
        LOG.debug('%sLocal SHA: %s Stored SHA: %s', self.log_prefix, self._local_sha, stored_sha)
        if self._local_sha and stored_sha and (self._local_sha == stored_sha):
            return True
        return False

    @abstractmethod
    def compare_remote(self) -> bool:
        if False:
            i = 10
            return i + 15
        'Comparison between local and remote resources.\n        This can be used for optimization if comparison is a lot faster than sync.\n        If the resources are identical, sync and gather dependencies will be skipped.\n        Simply return False if there is no comparison needed.\n        Ex: Comparing local Lambda function artifact with remote SHA256\n\n        Returns\n        -------\n        bool\n            Return True if local and remote are in sync. Skipping rest of the execution.\n            Return False otherwise.\n        '
        raise NotImplementedError('compare_remote')

    @abstractmethod
    def sync(self) -> None:
        if False:
            while True:
                i = 10
        'Step that syncs local resources with remote.\n        Ex: Call UpdateFunctionCode for Lambda Functions\n        '
        raise NotImplementedError('sync')

    @abstractmethod
    def gather_dependencies(self) -> List['SyncFlow']:
        if False:
            i = 10
            return i + 15
        'Gather a list of SyncFlows that should be executed after the current change.\n        This can be sync flows for other resources that depends on the current one.\n        Ex: Update Lambda functions if a layer sync flow creates a new version.\n\n        Returns\n        ------\n        List[SyncFlow]\n            List of sync flows that need to be executed after the current one finishes.\n        '
        raise NotImplementedError('update_dependencies')

    @abstractmethod
    def _get_resource_api_calls(self) -> List[ResourceAPICall]:
        if False:
            while True:
                i = 10
        'Get resources and their associating API calls. This is used for locking purposes.\n        Returns\n        -------\n        Dict[str, List[str]]\n            Key as resource logical ID\n            Value as list of api calls that the resource can make\n        '
        raise NotImplementedError('_get_resource_api_calls')

    def has_locks(self) -> bool:
        if False:
            i = 10
            return i + 15
        'Check if a sync flow has locks and needs to enter a lock context\n        Returns\n        -------\n        bool\n            whether or not a sync flow contains locks\n        '
        return bool(self._locks)

    def get_lock_keys(self) -> Set[str]:
        if False:
            for i in range(10):
                print('nop')
        'Get a list of function + API calls that can be used as keys for LockDistributor\n\n        Returns\n        -------\n        Set[str]\n            Set of keys for all resources and their API calls\n        '
        lock_keys = set()
        for resource_api_calls in self._get_resource_api_calls():
            for api_call in resource_api_calls.api_calls:
                lock_keys.add(SyncFlow._get_lock_key(resource_api_calls.shared_resource, api_call))
        return lock_keys

    def set_locks_with_distributor(self, distributor: LockDistributor):
        if False:
            while True:
                i = 10
        'Set locks to be used with a LockDistributor. Keys should be generated using get_lock_keys().\n\n        Parameters\n        ----------\n        distributor : LockDistributor\n            Lock distributor\n        '
        self.set_locks_with_dict(distributor.get_locks(self.get_lock_keys()))

    def set_locks_with_dict(self, locks: Dict[str, Lock]):
        if False:
            return 10
        'Set locks to be used. Keys should be generated using get_lock_keys().\n\n        Parameters\n        ----------\n        locks : Dict[str, Lock]\n            Dict of locks with keys from get_lock_keys()\n        '
        self._locks = locks

    @staticmethod
    def _get_lock_key(logical_id: str, api_call: ApiCallTypes) -> str:
        if False:
            print('Hello World!')
        'Get a single lock key for a pair of resource and API call.\n\n        Parameters\n        ----------\n        logical_id : str\n            Logical ID of a resource.\n        api_call : str\n            API call the resource will use.\n\n        Returns\n        -------\n        str\n            String key created with logical ID and API call name.\n        '
        return f'{logical_id}_{api_call.value}'

    def _get_lock_chain(self) -> LockChain:
        if False:
            while True:
                i = 10
        'Return a LockChain object for all the locks\n\n        Returns\n        -------\n        Optional[LockChain]\n            A LockChain object containing all locks. None if there are no locks.\n        '
        if self._locks:
            return LockChain(self._locks)
        raise MissingLockException('Missing Locks for LockChain')

    def _get_resource(self, resource_identifier: str) -> Optional[Dict[str, Any]]:
        if False:
            for i in range(10):
                print('nop')
        'Get a resource dict with resource identifier\n\n        Parameters\n        ----------\n        resource_identifier : str\n            Resource identifier\n\n        Returns\n        -------\n        Optional[Dict[str, Any]]\n            Resource dict containing its template fields.\n        '
        return get_resource_by_id(self._stacks, ResourceIdentifier(resource_identifier)) if self._stacks else None

    def get_physical_id(self, resource_identifier: str) -> str:
        if False:
            for i in range(10):
                print('nop')
        'Get the physical ID of a resource using physical_id_mapping. This does not directly check with remote.\n\n        Parameters\n        ----------\n        resource_identifier : str\n            Resource identifier\n\n        Returns\n        -------\n        str\n            Resource physical ID\n\n        Raises\n        ------\n        MissingPhysicalResourceError\n            Resource does not exist in the physical ID mapping.\n            This could mean remote and local templates are not in sync.\n        '
        physical_id = self._physical_id_mapping.get(resource_identifier)
        if not physical_id:
            raise MissingPhysicalResourceError(resource_identifier)
        return physical_id

    @abstractmethod
    def _equality_keys(self) -> Any:
        if False:
            return 10
        'This method needs to be overridden to distinguish between multiple instances of SyncFlows\n        If the return values of two instances are the same, then those two instances will be assumed to be equal.\n\n        Returns\n        -------\n        Any\n            Anything that can be hashed and compared with "=="\n        '
        raise NotImplementedError('_equality_keys is not implemented.')

    def __hash__(self) -> int:
        if False:
            while True:
                i = 10
        return hash((type(self), self._equality_keys()))

    def __eq__(self, o: object) -> bool:
        if False:
            print('Hello World!')
        if type(o) is not type(self):
            return False
        return cast(bool, self._equality_keys() == cast(SyncFlow, o)._equality_keys())

    @property
    def log_name(self) -> str:
        if False:
            while True:
                i = 10
        '\n        Returns\n        -------\n        str\n            Human readable name/identifier for logging purposes\n        '
        return self._log_name

    @property
    def log_prefix(self) -> str:
        if False:
            while True:
                i = 10
        '\n        Returns\n        -------\n        str\n            Log prefix to be used for logging.\n        '
        return f'SyncFlow [{self.log_name}]: '

    def execute(self) -> List['SyncFlow']:
        if False:
            return 10
        'Execute the sync flow and returns a list of dependent sync flows.\n        Skips sync() and gather_dependencies() if compare() is True\n\n        Returns\n        -------\n        List[SyncFlow]\n            A list of dependent sync flows\n        '
        dependencies: List['SyncFlow'] = list()
        LOG.debug('%sSetting Up', self.log_prefix)
        self.set_up()
        LOG.debug('%sGathering Resources', self.log_prefix)
        self.gather_resources()
        LOG.debug('%sComparing with Remote', self.log_prefix)
        if not self.compare_local() and (not self.compare_remote()):
            LOG.debug('%sSyncing', self.log_prefix)
            self.sync()
            LOG.debug('%sUpdating local hash of the sync flow', self.log_prefix)
            self._update_local_hash()
            LOG.debug('%sGathering Dependencies', self.log_prefix)
            dependencies = self.gather_dependencies()
        else:
            LOG.info("%sSkipping resource update as the content didn't change", self.log_prefix)
        LOG.debug('%sFinished', self.log_prefix)
        return dependencies

def get_definition_path(resource: Dict, identifier: str, use_base_dir: bool, base_dir: str, stacks: List[Stack]) -> Optional[Path]:
    if False:
        return 10
    "\n    A helper method used by non-function sync flows to resolve definition file path\n    that are relative to the child stack to absolute path for nested stacks\n\n    Parameters\n    -------\n    resource: Dict\n        The resource's template dict\n    identifier: str\n        The logical ID identifier of the resource\n    use_base_dir: bool\n        Whether or not the base_dir option was used\n    base_dir: str\n        Base directory if provided, otherwise the root template directory\n    stacks: List[Stack]\n        The list of stacks for the application\n\n    Returns\n    -------\n    Optional[Path]\n        A resolved absolute path for the definition file\n    "
    definition_field_names = RESOURCES_WITH_LOCAL_PATHS.get(resource.get('Type', ''))
    if not definition_field_names:
        LOG.error("Couldn't find definition field name for resource %s", identifier)
        return None
    definition_field_name = definition_field_names[0]
    LOG.debug('Found definition field name as %s', definition_field_name)
    properties = resource.get('Properties', {})
    definition_file = properties.get(definition_field_name)
    definition_path = None
    if definition_file:
        definition_path = Path(base_dir).joinpath(definition_file)
        if not use_base_dir:
            child_stack = Stack.get_stack_by_full_path(ResourceIdentifier(identifier).stack_path, stacks)
            if child_stack:
                definition_path = Path(child_stack.location).parent.joinpath(definition_file)
    return definition_path