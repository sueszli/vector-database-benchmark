from typing import TYPE_CHECKING
import warnings
from azure.core.exceptions import ClientAuthenticationError, HttpResponseError, ResourceExistsError, ResourceNotFoundError, map_error
from azure.core.pipeline import PipelineResponse
from azure.core.pipeline.transport import HttpRequest, HttpResponse
from azure.mgmt.core.exceptions import ARMErrorFormat
from .. import models as _models
if TYPE_CHECKING:
    from typing import Any, Callable, Dict, Generic, Optional, TypeVar
    T = TypeVar('T')
    ClsType = Optional[Callable[[PipelineResponse[HttpRequest, HttpResponse], T, Dict[str, Any]], Any]]

class OperationStatusesOperations(object):
    """OperationStatusesOperations operations.

    You should not instantiate this class directly. Instead, you should create a Client instance that
    instantiates it for you and attaches it as an attribute.

    :ivar models: Alias to model classes used in this operation group.
    :type models: ~communication_service_management_client.models
    :param client: Client for service requests.
    :param config: Configuration of service client.
    :param serializer: An object model serializer.
    :param deserializer: An object model deserializer.
    """
    models = _models

    def __init__(self, client, config, serializer, deserializer):
        if False:
            return 10
        self._client = client
        self._serialize = serializer
        self._deserialize = deserializer
        self._config = config

    def get(self, location, operation_id, **kwargs):
        if False:
            while True:
                i = 10
        'Get Operation Status.\n\n        Gets the current status of an async operation.\n\n        :param location: The Azure region.\n        :type location: str\n        :param operation_id: The ID of an ongoing async operation.\n        :type operation_id: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: OperationStatus, or the result of cls(response)\n        :rtype: ~communication_service_management_client.models.OperationStatus\n        :raises: ~azure.core.exceptions.HttpResponseError\n        '
        cls = kwargs.pop('cls', None)
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError}
        error_map.update(kwargs.pop('error_map', {}))
        api_version = '2020-08-20'
        accept = 'application/json'
        url = self.get.metadata['url']
        path_format_arguments = {'location': self._serialize.url('location', location, 'str'), 'operationId': self._serialize.url('operation_id', operation_id, 'str')}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        query_parameters['api-version'] = self._serialize.query('api_version', api_version, 'str')
        header_parameters = {}
        header_parameters['Accept'] = self._serialize.header('accept', accept, 'str')
        request = self._client.get(url, query_parameters, header_parameters)
        pipeline_response = self._client._pipeline.run(request, stream=False, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            error = self._deserialize(_models.ErrorResponse, response)
            raise HttpResponseError(response=response, model=error, error_format=ARMErrorFormat)
        deserialized = self._deserialize('OperationStatus', pipeline_response)
        if cls:
            return cls(pipeline_response, deserialized, {})
        return deserialized
    get.metadata = {'url': '/providers/Microsoft.Communication/locations/{location}/operationStatuses/{operationId}'}