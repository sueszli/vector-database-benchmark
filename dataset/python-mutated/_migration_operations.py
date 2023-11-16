from typing import TYPE_CHECKING
from msrest import Serializer
from azure.core.exceptions import ClientAuthenticationError, HttpResponseError, ResourceExistsError, ResourceNotFoundError, map_error
from azure.core.pipeline import PipelineResponse
from azure.core.pipeline.transport import HttpResponse
from azure.core.rest import HttpRequest
from azure.core.tracing.decorator import distributed_trace
from azure.mgmt.core.exceptions import ARMErrorFormat
from .. import models as _models
from .._vendor import _convert_request
if TYPE_CHECKING:
    from typing import Any, Callable, Dict, Optional, TypeVar
    T = TypeVar('T')
    ClsType = Optional[Callable[[PipelineResponse[HttpRequest, HttpResponse], T, Dict[str, Any]], Any]]
_SERIALIZER = Serializer()
_SERIALIZER.client_side_validation = False

def build_start_migration_request(**kwargs):
    if False:
        return 10
    migration = kwargs.pop('migration', None)
    timeout = kwargs.pop('timeout', '00:01:00')
    collection_id = kwargs.pop('collection_id', None)
    workspace_id = kwargs.pop('workspace_id', None)
    _url = kwargs.pop('template_url', '/modelregistry/v1.0/meta/migration')
    _query_parameters = kwargs.pop('params', {})
    if migration is not None:
        _query_parameters['migration'] = _SERIALIZER.query('migration', migration, 'str')
    if timeout is not None:
        _query_parameters['timeout'] = _SERIALIZER.query('timeout', timeout, 'str')
    if collection_id is not None:
        _query_parameters['collectionId'] = _SERIALIZER.query('collection_id', collection_id, 'str')
    if workspace_id is not None:
        _query_parameters['workspaceId'] = _SERIALIZER.query('workspace_id', workspace_id, 'str')
    return HttpRequest(method='POST', url=_url, params=_query_parameters, **kwargs)

class MigrationOperations(object):
    """MigrationOperations operations.

    You should not instantiate this class directly. Instead, you should create a Client instance that
    instantiates it for you and attaches it as an attribute.

    :ivar models: Alias to model classes used in this operation group.
    :type models: ~azure.mgmt.machinelearningservices.models
    :param client: Client for service requests.
    :param config: Configuration of service client.
    :param serializer: An object model serializer.
    :param deserializer: An object model deserializer.
    """
    models = _models

    def __init__(self, client, config, serializer, deserializer):
        if False:
            for i in range(10):
                print('nop')
        self._client = client
        self._serialize = serializer
        self._deserialize = deserializer
        self._config = config

    @distributed_trace
    def start_migration(self, migration=None, timeout='00:01:00', collection_id=None, workspace_id=None, **kwargs):
        if False:
            print('Hello World!')
        'start_migration.\n\n        :param migration:\n        :type migration: str\n        :param timeout:\n        :type timeout: str\n        :param collection_id:\n        :type collection_id: str\n        :param workspace_id:\n        :type workspace_id: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: None, or the result of cls(response)\n        :rtype: None\n        :raises: ~azure.core.exceptions.HttpResponseError\n        '
        cls = kwargs.pop('cls', None)
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError}
        error_map.update(kwargs.pop('error_map', {}))
        request = build_start_migration_request(migration=migration, timeout=timeout, collection_id=collection_id, workspace_id=workspace_id, template_url=self.start_migration.metadata['url'])
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)
        pipeline_response = self._client._pipeline.run(request, stream=False, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            raise HttpResponseError(response=response, error_format=ARMErrorFormat)
        if cls:
            return cls(pipeline_response, None, {})
    start_migration.metadata = {'url': '/modelregistry/v1.0/meta/migration'}