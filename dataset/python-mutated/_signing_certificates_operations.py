from typing import TYPE_CHECKING
import warnings
from azure.core.exceptions import ClientAuthenticationError, HttpResponseError, ResourceExistsError, ResourceNotFoundError, map_error
from azure.core.pipeline import PipelineResponse
from azure.core.pipeline.transport import HttpRequest, HttpResponse
from .. import models as _models
if TYPE_CHECKING:
    from typing import Any, Callable, Dict, Generic, Optional, TypeVar
    T = TypeVar('T')
    ClsType = Optional[Callable[[PipelineResponse[HttpRequest, HttpResponse], T, Dict[str, Any]], Any]]

class SigningCertificatesOperations(object):
    """SigningCertificatesOperations operations.

    You should not instantiate this class directly. Instead, you should create a Client instance that
    instantiates it for you and attaches it as an attribute.

    :ivar models: Alias to model classes used in this operation group.
    :type models: ~azure.security.attestation._generated.models
    :param client: Client for service requests.
    :param config: Configuration of service client.
    :param serializer: An object model serializer.
    :param deserializer: An object model deserializer.
    """
    models = _models

    def __init__(self, client, config, serializer, deserializer):
        if False:
            print('Hello World!')
        self._client = client
        self._serialize = serializer
        self._deserialize = deserializer
        self._config = config

    def get(self, **kwargs):
        if False:
            print('Hello World!')
        'Retrieves the attestation signing keys in use by the attestation service.\n\n        Retrieves metadata signing certificates in use by the attestation service.\n\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: JSONWebKeySet, or the result of cls(response)\n        :rtype: ~azure.security.attestation._generated.models.JSONWebKeySet\n        :raises: ~azure.core.exceptions.HttpResponseError\n        '
        cls = kwargs.pop('cls', None)
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError}
        error_map.update(kwargs.pop('error_map', {}))
        accept = 'application/jwk+json, application/json'
        url = self.get.metadata['url']
        path_format_arguments = {'instanceUrl': self._serialize.url('self._config.instance_url', self._config.instance_url, 'str', skip_quote=True)}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        header_parameters = {}
        header_parameters['Accept'] = self._serialize.header('accept', accept, 'str')
        request = self._client.get(url, query_parameters, header_parameters)
        pipeline_response = self._client._pipeline.run(request, stream=False, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            error = self._deserialize.failsafe_deserialize(_models.CloudError, response)
            raise HttpResponseError(response=response, model=error)
        deserialized = self._deserialize('JSONWebKeySet', pipeline_response)
        if cls:
            return cls(pipeline_response, deserialized, {})
        return deserialized
    get.metadata = {'url': '/certs'}