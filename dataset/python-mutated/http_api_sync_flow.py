"""SyncFlow for HttpApi"""
import logging
from typing import TYPE_CHECKING, Dict, List
from samcli.lib.providers.exceptions import MissingLocalDefinition
from samcli.lib.providers.provider import ResourceIdentifier, Stack
from samcli.lib.sync.flows.generic_api_sync_flow import GenericApiSyncFlow
if TYPE_CHECKING:
    from samcli.commands.build.build_context import BuildContext
    from samcli.commands.deploy.deploy_context import DeployContext
    from samcli.commands.sync.sync_context import SyncContext
LOG = logging.getLogger(__name__)

class HttpApiSyncFlow(GenericApiSyncFlow):
    """SyncFlow for HttpApi's"""

    def __init__(self, httpapi_identifier: str, build_context: 'BuildContext', deploy_context: 'DeployContext', sync_context: 'SyncContext', physical_id_mapping: Dict[str, str], stacks: List[Stack]):
        if False:
            while True:
                i = 10
        '\n        Parameters\n        ----------\n        httpapi_identifier : str\n            HttpApi resource identifier that needs to have associated HttpApi updated.\n        build_context : BuildContext\n            BuildContext used for build related parameters\n        deploy_context : BuildContext\n            DeployContext used for this deploy related parameters\n        sync_context: SyncContext\n            SyncContext object that obtains sync information.\n        physical_id_mapping : Dict[str, str]\n            Mapping between resource logical identifier and physical identifier\n        stacks : List[Stack], optional\n            List of stacks containing a root stack and optional nested stacks\n        '
        super().__init__(httpapi_identifier, build_context, deploy_context, sync_context, physical_id_mapping, log_name='HttpApi ' + httpapi_identifier, stacks=stacks)

    def set_up(self) -> None:
        if False:
            i = 10
            return i + 15
        super().set_up()
        self._api_client = self._boto_client('apigatewayv2')

    def sync(self) -> None:
        if False:
            while True:
                i = 10
        api_physical_id = self.get_physical_id(self._api_identifier)
        if self._definition_uri is None:
            raise MissingLocalDefinition(ResourceIdentifier(self._api_identifier), 'DefinitionUri')
        if self._swagger_body:
            LOG.debug('%sTrying to import HttpAPI through client', self.log_prefix)
            response = self._api_client.reimport_api(ApiId=api_physical_id, Body=self._swagger_body.decode())
            LOG.debug('%sImport HttpApi Result: %s', self.log_prefix, response)
        else:
            LOG.debug('%sEmpty OpenApi definition, skipping the sync for %s', self.log_prefix, self._api_identifier)