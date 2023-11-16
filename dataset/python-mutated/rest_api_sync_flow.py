"""SyncFlow for RestApi"""
import logging
from typing import TYPE_CHECKING, Dict, List, Optional, Set, cast
from botocore.exceptions import ClientError
from samcli.lib.providers.exceptions import MissingLocalDefinition
from samcli.lib.providers.provider import ResourceIdentifier, Stack, get_resource_by_id, get_resource_ids_by_type
from samcli.lib.sync.flows.generic_api_sync_flow import GenericApiSyncFlow
from samcli.lib.utils.colors import Colored
from samcli.lib.utils.resources import AWS_APIGATEWAY_DEPLOYMENT, AWS_APIGATEWAY_STAGE, AWS_SERVERLESS_API
if TYPE_CHECKING:
    from samcli.commands.build.build_context import BuildContext
    from samcli.commands.deploy.deploy_context import DeployContext
    from samcli.commands.sync.sync_context import SyncContext
LOG = logging.getLogger(__name__)

class RestApiSyncFlow(GenericApiSyncFlow):
    """SyncFlow for RestApi's"""

    def __init__(self, restapi_identifier: str, build_context: 'BuildContext', deploy_context: 'DeployContext', sync_context: 'SyncContext', physical_id_mapping: Dict[str, str], stacks: List[Stack]):
        if False:
            return 10
        '\n        Parameters\n        ----------\n        restapi_identifier : str\n            RestApi resource identifier that needs to have associated RestApi updated.\n        build_context : BuildContext\n            BuildContext used for build related parameters\n        deploy_context : BuildContext\n            DeployContext used for this deploy related parameters\n        sync_context: SyncContext\n            SyncContext object that obtains sync information.\n        physical_id_mapping : Dict[str, str]\n            Mapping between resource logical identifier and physical identifier\n        stacks : List[Stack], optional\n            List of stacks containing a root stack and optional nested stacks\n        '
        super().__init__(restapi_identifier, build_context, deploy_context, sync_context, physical_id_mapping, log_name='RestApi ' + restapi_identifier, stacks=stacks)
        self._api_physical_id = ''

    def set_up(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().set_up()
        self._api_client = self._boto_client('apigateway')
        self._api_physical_id = self.get_physical_id(self._api_identifier)

    def sync(self) -> None:
        if False:
            while True:
                i = 10
        if self._definition_uri is None:
            raise MissingLocalDefinition(ResourceIdentifier(self._api_identifier), 'DefinitionUri')
        self._update_api()
        new_dep_id = self._create_deployment()
        stages = self._collect_stages()
        prev_dep_ids = self._update_stages(stages, new_dep_id)
        self._delete_deployments(prev_dep_ids)

    def _update_api(self) -> None:
        if False:
            while True:
                i = 10
        '\n        Update the API content\n        '
        LOG.debug('%sTrying to update RestAPI through client', self.log_prefix)
        response_put = cast(Dict, self._api_client.put_rest_api(restApiId=self._api_physical_id, mode='overwrite', body=self._swagger_body))
        LOG.debug('%sPut RestApi Result: %s', self.log_prefix, response_put)

    def _create_deployment(self) -> Optional[str]:
        if False:
            while True:
                i = 10
        '\n        Create a deployment using the updated API and record the created deployment ID\n\n        Returns\n        ----------\n        Optional[str]: The newly created deployment ID\n        '
        LOG.debug('%sTrying to create a deployment through client', self.log_prefix)
        response_dep = cast(Dict, self._api_client.create_deployment(restApiId=self._api_physical_id, description='Created by SAM Sync'))
        new_dep_id = response_dep.get('id')
        LOG.debug('%sCreate Deployment Result: %s', self.log_prefix, response_dep)
        return new_dep_id

    def _collect_stages(self) -> Set[str]:
        if False:
            i = 10
            return i + 15
        '\n        Collect all stages needed to be updated\n\n        Returns\n        ----------\n        Set[str]: The set of stage names to be updated\n        '
        api_resource = get_resource_by_id(self._stacks, ResourceIdentifier(self._api_identifier))
        stage_resources = get_resource_ids_by_type(self._stacks, AWS_APIGATEWAY_STAGE)
        deployment_resources = get_resource_ids_by_type(self._stacks, AWS_APIGATEWAY_DEPLOYMENT)
        stages = set()
        if api_resource:
            if api_resource.get('Type') == AWS_SERVERLESS_API:
                stage_name = api_resource.get('Properties', {}).get('StageName')
                if stage_name:
                    stages.add(cast(str, stage_name))
                if stage_name != 'Stage':
                    response_sta = cast(Dict, self._api_client.get_stages(restApiId=self._api_physical_id))
                    for item in response_sta.get('item'):
                        if item.get('stageName') == 'Stage':
                            stages.add('Stage')
        for stage_resource in stage_resources:
            stage_dict = get_resource_by_id(self._stacks, stage_resource)
            if not stage_dict:
                continue
            rest_api_id = stage_dict.get('Properties', {}).get('RestApiId')
            dep_id = stage_dict.get('Properties', {}).get('DeploymentId')
            if dep_id is None:
                continue
            for deployment_resource in deployment_resources:
                if deployment_resource.resource_iac_id == dep_id and rest_api_id == self._api_identifier:
                    stages.add(cast(str, stage_dict.get('Properties', {}).get('StageName')))
                    break
        return stages

    def _update_stages(self, stages: Set[str], deployment_id: Optional[str]) -> Set[str]:
        if False:
            i = 10
            return i + 15
        '\n        Update all the relevant stages\n\n        Parameters\n        ----------\n        stages: Set[str]\n            The set of stage names to be updated\n        deployment_id: Optional[str]\n            The newly created deployment ID to be used in the stages\n        Returns\n        ----------\n        Set[str]: A set of previous deployment IDs to be cleaned up\n        '
        prev_dep_ids = set()
        for stage in stages:
            response_get = cast(Dict, self._api_client.get_stage(restApiId=self._api_physical_id, stageName=stage))
            prev_dep_id = response_get.get('deploymentId')
            if prev_dep_id:
                prev_dep_ids.add(cast(str, prev_dep_id))
            LOG.debug('%sTrying to update the stage %s through client', self.log_prefix, stage)
            response_upd = cast(Dict, self._api_client.update_stage(restApiId=self._api_physical_id, stageName=stage, patchOperations=[{'op': 'replace', 'path': '/deploymentId', 'value': deployment_id}]))
            LOG.debug('%sUpdate Stage Result: %s', self.log_prefix, response_upd)
            self._api_client.flush_stage_cache(restApiId=self._api_physical_id, stageName=stage)
            self._api_client.flush_stage_authorizers_cache(restApiId=self._api_physical_id, stageName=stage)
        return prev_dep_ids

    def _delete_deployments(self, prev_deployment_ids: Set[str]) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Delete the previous deployment\n\n        Parameters\n        ----------\n        prev_deployment_ids: Set[str]\n            A set of previous deployment IDs to be cleaned up\n        '
        for prev_dep_id in prev_deployment_ids:
            try:
                LOG.debug('%sTrying to delete the previous deployment %s through client', self.log_prefix, prev_dep_id)
                response_del = cast(Dict, self._api_client.delete_deployment(restApiId=self._api_physical_id, deploymentId=prev_dep_id))
                LOG.debug('%sDelete Deployment Result: %s', self.log_prefix, response_del)
            except ClientError:
                LOG.warning(Colored().yellow('Delete deployment for %s failed, it may be due to the it being used by another stage. please check the console to see if you have other stages that needs to be updated.'), prev_dep_id)