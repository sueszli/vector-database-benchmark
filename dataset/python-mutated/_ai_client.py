import json
import logging
import os
import tempfile
from typing import Any, Optional, Union
import yaml
from azure.ai.resources._restclient.v2022_10_01 import AzureMachineLearningWorkspaces as ServiceClient100122
from azure.ai.resources._utils._ai_client_utils import find_config_file_path, get_config_info
from azure.ai.resources._utils._open_ai_utils import build_open_ai_protocol
from azure.ai.resources._utils._str_utils import build_connection_id
from azure.ai.resources.constants._common import DEFAULT_OPEN_AI_CONNECTION_NAME, DEFAULT_CONTENT_SAFETY_CONNECTION_NAME
from azure.ai.resources.entities.mlindex import Index as MLIndexAsset
from azure.ai.resources.operations import ACSOutputConfig, ACSSource, GitSource, IndexDataSource, LocalSource
from azure.ai.ml import MLClient
from azure.ai.ml._restclient.v2023_06_01_preview import AzureMachineLearningWorkspaces as ServiceClient062023Preview
from azure.ai.ml._utils._experimental import experimental
from azure.ai.ml.entities._credentials import ManagedIdentityConfiguration, UserIdentityConfiguration
from azure.core.credentials import TokenCredential
from .._project_scope import OperationScope
from .._user_agent import USER_AGENT
from azure.ai.resources.operations import AIResourceOperations, ConnectionOperations, DeploymentOperations, MLIndexOperations, PFOperations, ProjectOperations, DataOperations, ModelOperations
from azure.ai.resources.operations._ingest_data_to_index import ingest_data_to_index
module_logger = logging.getLogger(__name__)
from azure.ai.resources._telemetry import ActivityType, monitor_with_activity, monitor_with_telemetry_mixin, get_appinsights_log_handler, OpsLogger
ops_logger = OpsLogger(__name__)
logger = ops_logger.package_logger

@experimental
class AIClient:

    def __init__(self, credential: TokenCredential, subscription_id: Optional[str]=None, resource_group_name: Optional[str]=None, ai_resource_name: Optional[str]=None, project_name: Optional[str]=None, **kwargs: Any):
        if False:
            print('Hello World!')
        self._add_user_agent(kwargs)
        properties = {'subscription_id': subscription_id, 'resource_group_name': resource_group_name}
        if ai_resource_name:
            properties.update({'ai_resource_name': ai_resource_name})
        if project_name:
            properties.update({'project_name': project_name})
        user_agent = USER_AGENT
        enable_telemetry = kwargs.pop('enable_telemetry', True)
        app_insights_handler = get_appinsights_log_handler(user_agent, **{'properties': properties}, enable_telemetry=enable_telemetry)
        app_insights_handler_kwargs = {'app_insights_handler': app_insights_handler}
        self._scope = OperationScope(subscription_id=subscription_id, resource_group_name=resource_group_name, ai_resource_name=ai_resource_name, project_name=project_name)
        self._credential = credential
        self._ml_client = MLClient(credential=credential, subscription_id=subscription_id, resource_group_name=resource_group_name, workspace_name=project_name, **kwargs)
        if project_name:
            ai_resource_name = ai_resource_name or self._ml_client.workspaces.get(project_name).workspace_hub.split('/')[-1]
        self._ai_resource_ml_client = MLClient(credential=credential, subscription_id=subscription_id, resource_group_name=resource_group_name, workspace_name=ai_resource_name, **kwargs)
        self._service_client_06_2023_preview = ServiceClient062023Preview(credential=self._credential, subscription_id=subscription_id, **kwargs)
        self._projects = ProjectOperations(resource_group_name=resource_group_name, service_client=self._service_client_06_2023_preview, ml_client=self._ml_client, **app_insights_handler_kwargs)
        self._connections = ConnectionOperations(self._ai_resource_ml_client, **app_insights_handler_kwargs)
        self._mlindexes = MLIndexOperations(self._ml_client, **app_insights_handler_kwargs)
        self._ai_resources = AIResourceOperations(self._ml_client, **app_insights_handler_kwargs)
        self._deployments = DeploymentOperations(self._ml_client, self._connections, **app_insights_handler_kwargs)
        self._data = DataOperations(self._ml_client)
        self._models = ModelOperations(self._ml_client)

    @classmethod
    def from_config(cls, credential: TokenCredential, *, path: Optional[Union[os.PathLike, str]]=None, file_name=None, **kwargs) -> 'AIClient':
        if False:
            while True:
                i = 10
        config_file_path = find_config_file_path(path, file_name)
        config_info = get_config_info(config_file_path)
        return AIClient(credential=credential, subscription_id=config_info['subscription_id'], resource_group_name=config_info['resource_group_name'], project_name=config_info['project_name'])

    @property
    def ai_resources(self) -> AIResourceOperations:
        if False:
            for i in range(10):
                print('nop')
        'A collection of AI resource-related operations.\n\n        :return: AI Resource operations\n        :rtype: AIResourceOperations\n        '
        return self._ai_resources

    @property
    def projects(self) -> ProjectOperations:
        if False:
            print('Hello World!')
        'A collection of project-related operations.\n\n        :return: Project operations\n        :rtype: ProjectOperations\n        '
        return self._projects

    @property
    def connections(self) -> ConnectionOperations:
        if False:
            for i in range(10):
                print('nop')
        "A collection of connection-related operations.\n        NOTE: Unlike other operation handles, the connections handle\n        is scoped to the AIClient's AI Resource, and not the project.\n        SDK support for project-scoped connections does not exist yet.\n\n        :return: Connections operations\n        :rtype: ConnectionsOperations\n        "
        return self._connections

    @property
    def indexes(self) -> MLIndexOperations:
        if False:
            return 10
        'A collection of ML index-related operations.\n\n        :return: ML index operations\n        :rtype: MLIndexOperations\n        '
        return self._mlindexes

    @property
    def pf(self) -> PFOperations:
        if False:
            while True:
                i = 10
        'A collection of PF operation-related operations.\n\n        :return: PF Operation operations\n        :rtype: PFOperations\n        '
        return self._pf

    @property
    def data(self) -> DataOperations:
        if False:
            return 10
        'A collection of data-related operations.\n\n        :return: Data operations\n        :rtype: DataOperations\n        '
        return self._data

    @property
    def deployments(self) -> DeploymentOperations:
        if False:
            print('Hello World!')
        'A collection of deployment-related operations.\n\n        :return: Deployment operations\n        :rtype: DeploymentOperations\n        '
        return self._deployments

    @property
    def models(self) -> ModelOperations:
        if False:
            print('Hello World!')
        'A collection of model-related operations.\n\n        :return: Model operations\n        :rtype: ModelOperations\n        '
        return self._models

    @property
    def subscription_id(self) -> str:
        if False:
            while True:
                i = 10
        'Get the subscription Id of a MLClient object.\n\n        :return: An Azure subscription Id.\n        :rtype: str\n        '
        return self._scope.subscription_id

    @property
    def resource_group_name(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        'Get the resource group name of a MLClient object.\n\n        :return: An Azure resource group name.\n        :rtype: str\n        '
        return self._scope.resource_group_name

    @property
    def project_name(self) -> Optional[str]:
        if False:
            print('Hello World!')
        'The workspace where workspace dependent operations will be executed in.\n\n        :return: Default workspace name.\n        :rtype: str\n        '
        return self._scope.project_name

    @property
    def tracking_uri(self):
        if False:
            return 10
        project = self.projects._service_client.workspaces.get(self._scope.resource_group_name, self._scope.project_name, api_version='2023-04-01-preview')
        return project.ml_flow_tracking_uri

    def build_index_on_cloud(self, *, output_index_name: str, vector_store: str, data_source_url: str=None, chunk_size: int=None, chunk_overlap: int=None, input_glob: str=None, max_sample_files: int=None, chunk_prepend_summary: bool=None, document_path_replacement_regex: str=None, embeddings_model='text-embedding-ada-002', aoai_connection_id: str=DEFAULT_OPEN_AI_CONNECTION_NAME, acs_config: ACSOutputConfig=None, input_source: Union[IndexDataSource, str], identity: Optional[Union[ManagedIdentityConfiguration, UserIdentityConfiguration]]=None, _dry_run: bool=False):
        if False:
            print('Hello World!')
        'TODO _summary_\n\n        Args:\n            output_index_name (str): _description_\n            vector_store (str): _description_\n            embeddings_model (str): _description_\n            aoai_connection_id (str): _description_\n            input_config (IndexDataSource): _description_\n            acs_config (ACSOutputConfig): _description_\n            data_source_url (str, optional): _description_. Defaults to None.\n            chunk_size (int, optional): _description_. Defaults to None.\n            chunk_overlap (int, optional): _description_. Defaults to None.\n            input_glob (str, optional): _description_. Defaults to None.\n            max_sample_files (int, optional): _description_. Defaults to None.\n            chunk_prepend_summary (bool, optional): _description_. Defaults to None.\n            document_path_replacement_regex (str, optional): _description_. Defaults to None.\n\n        Returns:\n            _type_: _description_\n        '
        from azure.ai.resources._index._dataindex.data_index import index_data
        from azure.ai.resources._index._dataindex.entities import CitationRegex, Data, DataIndex, Embedding, IndexSource, IndexStore
        from azure.ai.resources._index._embeddings import EmbeddingsContainer
        if isinstance(input_source, ACSSource):
            from azure.ai.resources._index._utils.connections import get_connection_by_id_v2, get_target_from_connection
            mlindex_config = {}
            connection_args = {'connection_type': 'workspace_connection', 'connection': {'id': aoai_connection_id}}
            mlindex_config['embeddings'] = EmbeddingsContainer.from_uri(build_open_ai_protocol(embeddings_model), **connection_args).get_metadata()
            mlindex_config['index'] = {'kind': 'acs', 'connection_type': 'workspace_connection', 'connection': {'id': input_source.acs_connection_id}, 'index': input_source.acs_index_name, 'endpoint': get_target_from_connection(get_connection_by_id_v2(input_source.acs_connection_id, credential=self._ml_client._credential)), 'engine': 'azure-sdk', 'field_mapping': {'content': input_source.acs_content_key, 'title': input_source.acs_title_key, 'metadata': input_source.acs_metadata_key}}
            if input_source.acs_embedding_key is not None:
                mlindex_config['index']['embedding'] = input_source.acs_embedding_key
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_file = os.path.join(temp_dir, 'MLIndex')
                with open(temp_file, 'w') as f:
                    yaml.dump(mlindex_config, f)
                mlindex = MLIndexAsset(name=output_index_name, path=temp_dir)
                return self.indexes.create_or_update(mlindex)
        if document_path_replacement_regex:
            document_path_replacement_regex = json.loads(document_path_replacement_regex)
        data_index = DataIndex(name=output_index_name, source=IndexSource(input_data=Data(type='uri_folder', path='<This will be replaced later>'), input_glob=input_glob, chunk_size=chunk_size, chunk_overlap=chunk_overlap, citation_url=data_source_url, citation_url_replacement_regex=CitationRegex(match_pattern=document_path_replacement_regex['match_pattern'], replacement_pattern=document_path_replacement_regex['replacement_pattern']) if document_path_replacement_regex else None), embedding=Embedding(model=build_open_ai_protocol(embeddings_model), connection=build_connection_id(aoai_connection_id, self._scope)), index=IndexStore(type='acs', connection=build_connection_id(acs_config.acs_connection_id, self._scope), name=acs_config.acs_index_name) if acs_config is not None else IndexStore(type='faiss'), path=f'azureml://datastores/workspaceblobstore/paths/indexes/{output_index_name}/{{name}}')
        if isinstance(input_source, GitSource):
            from azure.ai.ml.dsl import pipeline
            ml_registry = MLClient(credential=self._ml_client._credential, registry_name='azureml')
            git_clone_component = ml_registry.components.get('llm_rag_git_clone', label='latest')

            @pipeline(default_compute='serverless')
            def git_to_index(git_url, branch_name='', git_connection_id=''):
                if False:
                    i = 10
                    return i + 15
                git_clone = git_clone_component(git_repository=git_url, branch_name=branch_name)
                git_clone.environment_variables['AZUREML_WORKSPACE_CONNECTION_ID_GIT'] = git_connection_id
                index_job = index_data(description=data_index.description, data_index=data_index, input_data_override=git_clone.outputs.output_data, ml_client=self._ml_client)
                return index_job.outputs
            git_index_job = git_to_index(git_url=input_source.git_url, branch_name=input_source.git_branch_name, git_connection_id=input_source.git_connection_id)
            git_index_job.settings.force_rerun = True
            return self._ml_client.jobs.create_or_update(git_index_job, identity=identity)
        if isinstance(input_source, LocalSource):
            data_index.source.input_data = Data(type='uri_folder', path=input_source.input_data.path)
            return self._ml_client.data.index_data(data_index=data_index, identity=identity)
        elif isinstance(input_source, str):
            data_index.source.input_data = Data(type='uri_folder', path=input_source)
            return self._ml_client.data.index_data(data_index=data_index, identity=identity)
        else:
            raise ValueError(f'Unsupported input source type {type(input_source)}')

    def get_default_aoai_connection(self):
        if False:
            i = 10
            return i + 15
        "Retrieves the default Azure Open AI connection associated with this AIClient's project,\n        creating it if it does not already exist.\n\n        :return: A Connection to Azure Open AI\n        :rtype: ~azure.ai.resources.entities.AzureOpenAIConnection\n        "
        return self._connections.get(DEFAULT_OPEN_AI_CONNECTION_NAME)

    def get_default_content_safety_connection(self):
        if False:
            for i in range(10):
                print('nop')
        "Retrieves a default Azure AI Service connection associated with this AIClient's project,\n        creating it if the connection does not already exist.\n        This particular AI Service connection is linked to an Azure Content Safety service.\n\n        :return: A Connection to an Azure AI Service\n        :rtype: ~azure.ai.resources.entities.AzureAIServiceConnection\n        "
        return self._connections.get(DEFAULT_CONTENT_SAFETY_CONNECTION_NAME)

    def _add_user_agent(self, kwargs) -> None:
        if False:
            print('Hello World!')
        user_agent = kwargs.pop('user_agent', None)
        user_agent = f'{user_agent} {USER_AGENT}' if user_agent else USER_AGENT
        kwargs.setdefault('user_agent', user_agent)