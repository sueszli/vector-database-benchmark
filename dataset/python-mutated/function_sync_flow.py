"""Base SyncFlow for Lambda Function"""
import logging
import time
from abc import ABC
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, cast
from botocore.client import BaseClient
from samcli.lib.build.app_builder import ApplicationBuildResult
from samcli.lib.providers.provider import Function, Stack
from samcli.lib.providers.sam_function_provider import SamFunctionProvider
from samcli.lib.sync.flows.alias_version_sync_flow import AliasVersionSyncFlow
from samcli.lib.sync.sync_flow import SyncFlow
from samcli.local.lambdafn.exceptions import FunctionNotFound
if TYPE_CHECKING:
    from samcli.commands.build.build_context import BuildContext
    from samcli.commands.deploy.deploy_context import DeployContext
    from samcli.commands.sync.sync_context import SyncContext
LOG = logging.getLogger(__name__)
FUNCTION_SLEEP = 1

class FunctionSyncFlow(SyncFlow, ABC):
    _function_identifier: str
    _function_provider: SamFunctionProvider
    _function: Function
    _lambda_client: Any
    _lambda_waiter: Any
    _lambda_waiter_config: Dict[str, Any]

    def __init__(self, function_identifier: str, build_context: 'BuildContext', deploy_context: 'DeployContext', sync_context: 'SyncContext', physical_id_mapping: Dict[str, str], stacks: List[Stack], application_build_result: Optional[ApplicationBuildResult]):
        if False:
            for i in range(10):
                print('nop')
        '\n        Parameters\n        ----------\n        function_identifier : str\n            Function resource identifier that need to be synced.\n        build_context : BuildContext\n            BuildContext\n        deploy_context : DeployContext\n            DeployContext\n        sync_context: SyncContext\n            SyncContext object that obtains sync information.\n        physical_id_mapping : Dict[str, str]\n            Physical ID Mapping\n        stacks : Optional[List[Stack]]\n            Stacks\n        application_build_result: Optional[ApplicationBuildResult]\n            Pre-build ApplicationBuildResult which can be re-used during SyncFlows\n        '
        super().__init__(build_context, deploy_context, sync_context, physical_id_mapping, log_name='Lambda Function ' + function_identifier, stacks=stacks, application_build_result=application_build_result)
        self._function_identifier = function_identifier
        self._function_provider = self._build_context.function_provider
        self._function = cast(Function, self._function_provider.get(self._function_identifier))
        self._lambda_client = None
        self._lambda_waiter = None
        self._lambda_waiter_config = {'Delay': 1, 'MaxAttempts': 60}

    def set_up(self) -> None:
        if False:
            return 10
        super().set_up()
        self._lambda_client = self._boto_client('lambda')
        self._lambda_waiter = self._lambda_client.get_waiter('function_updated')

    @property
    def sync_state_identifier(self) -> str:
        if False:
            i = 10
            return i + 15
        '\n        Sync state is the unique identifier for each sync flow\n        In sync state toml file we will store\n        Key as ZipFunctionSyncFlow:FunctionLogicalId\n        Value as function ZIP hash\n        '
        return self.__class__.__name__ + ':' + self._function_identifier

    def gather_dependencies(self) -> List[SyncFlow]:
        if False:
            for i in range(10):
                print('nop')
        'Gathers alias and versions related to a function.\n        Currently only handles serverless function AutoPublishAlias field\n        since a manually created function version resource behaves statically in a stack.\n        Redeploying a version resource through CFN will not create a new version.\n        '
        LOG.debug('%sWaiting on Remote Function Update', self.log_prefix)
        self._lambda_waiter.wait(FunctionName=self.get_physical_id(self._function_identifier), WaiterConfig=self._lambda_waiter_config)
        LOG.debug('%sRemote Function Updated', self.log_prefix)
        sync_flows: List[SyncFlow] = list()
        function_resource = self._get_resource(self._function_identifier)
        if not function_resource:
            raise FunctionNotFound(f'Unable to find function {self._function_identifier}')
        auto_publish_alias_name = function_resource.get('Properties', dict()).get('AutoPublishAlias', None)
        if auto_publish_alias_name:
            sync_flows.append(AliasVersionSyncFlow(self._function_identifier, auto_publish_alias_name, self._build_context, self._deploy_context, self._sync_context, self._physical_id_mapping, self._stacks))
            LOG.debug('%sCreated Alias and Version SyncFlow', self.log_prefix)
        return sync_flows

    def _equality_keys(self):
        if False:
            while True:
                i = 10
        return self._function_identifier

class FunctionUpdateStatus(Enum):
    """Function update return types"""
    SUCCESS = 'Successful'
    FAILED = 'Failed'
    IN_PROGRESS = 'InProgress'

def wait_for_function_update_complete(lambda_client: BaseClient, physical_id: str) -> None:
    if False:
        for i in range(10):
            print('nop')
    '\n    Checks on cloud side to wait for the function update status to be complete\n\n    Parameters\n    ----------\n    lambda_client : boto.core.BaseClient\n        Lambda client that performs get_function API call.\n    physical_id : str\n        Physical identifier of the function resource\n    '
    status = FunctionUpdateStatus.IN_PROGRESS.value
    while status == FunctionUpdateStatus.IN_PROGRESS.value:
        response = lambda_client.get_function(FunctionName=physical_id)
        status = response.get('Configuration', {}).get('LastUpdateStatus', '')
        if status == FunctionUpdateStatus.IN_PROGRESS.value:
            time.sleep(FUNCTION_SLEEP)
    LOG.debug('Function update status on %s is now %s on cloud.', physical_id, status)