from typing import TYPE_CHECKING
from msrest import Serializer
from azure.core.exceptions import ClientAuthenticationError, HttpResponseError, ResourceExistsError, ResourceNotFoundError, map_error
from azure.core.pipeline import PipelineResponse
from azure.core.pipeline.transport import HttpResponse
from azure.core.rest import HttpRequest
from azure.core.tracing.decorator import distributed_trace
from azure.mgmt.core.exceptions import ARMErrorFormat
from .. import models as _models
from .._vendor import _convert_request, _format_url_section
if TYPE_CHECKING:
    from typing import Any, Callable, Dict, Optional, TypeVar
    T = TypeVar('T')
    ClsType = Optional[Callable[[PipelineResponse[HttpRequest, HttpResponse], T, Dict[str, Any]], Any]]
_SERIALIZER = Serializer()
_SERIALIZER.client_side_validation = False

def build_query_by_id_request(id, subscription_id, resource_group_name, workspace_name, **kwargs):
    if False:
        while True:
            i = 10
    accept = 'application/json, text/json'
    _url = kwargs.pop('template_url', '/modelregistry/v1.0/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.MachineLearningServices/workspaces/{workspaceName}/extensiveModels/{id}')
    path_format_arguments = {'id': _SERIALIZER.url('id', id, 'str'), 'subscriptionId': _SERIALIZER.url('subscription_id', subscription_id, 'str'), 'resourceGroupName': _SERIALIZER.url('resource_group_name', resource_group_name, 'str'), 'workspaceName': _SERIALIZER.url('workspace_name', workspace_name, 'str')}
    _url = _format_url_section(_url, **path_format_arguments)
    _header_parameters = kwargs.pop('headers', {})
    _header_parameters['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='GET', url=_url, headers=_header_parameters, **kwargs)

class ExtensiveModelOperations(object):
    """ExtensiveModelOperations operations.

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
            while True:
                i = 10
        self._client = client
        self._serialize = serializer
        self._deserialize = deserializer
        self._config = config

    @distributed_trace
    def query_by_id(self, id, subscription_id, resource_group_name, workspace_name, **kwargs):
        if False:
            return 10
        'query_by_id.\n\n        :param id:\n        :type id: str\n        :param subscription_id:\n        :type subscription_id: str\n        :param resource_group_name:\n        :type resource_group_name: str\n        :param workspace_name:\n        :type workspace_name: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: ExtensiveModel, or the result of cls(response)\n        :rtype: ~azure.mgmt.machinelearningservices.models.ExtensiveModel\n        :raises: ~azure.core.exceptions.HttpResponseError\n        '
        cls = kwargs.pop('cls', None)
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError}
        error_map.update(kwargs.pop('error_map', {}))
        request = build_query_by_id_request(id=id, subscription_id=subscription_id, resource_group_name=resource_group_name, workspace_name=workspace_name, template_url=self.query_by_id.metadata['url'])
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)
        pipeline_response = self._client._pipeline.run(request, stream=False, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            raise HttpResponseError(response=response, error_format=ARMErrorFormat)
        deserialized = self._deserialize('ExtensiveModel', pipeline_response)
        if cls:
            return cls(pipeline_response, deserialized, {})
        return deserialized
    query_by_id.metadata = {'url': '/modelregistry/v1.0/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.MachineLearningServices/workspaces/{workspaceName}/extensiveModels/{id}'}