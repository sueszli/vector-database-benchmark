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

class AttestationOperations(object):
    """AttestationOperations operations.

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

    def attest_open_enclave(self, request, **kwargs):
        if False:
            i = 10
            return i + 15
        'Attest to an SGX enclave.\n\n        Processes an OpenEnclave report , producing an artifact. The type of artifact produced is\n        dependent upon attestation policy.\n\n        :param request: Request object containing the quote.\n        :type request: ~azure.security.attestation._generated.models.AttestOpenEnclaveRequest\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: AttestationResponse, or the result of cls(response)\n        :rtype: ~azure.security.attestation._generated.models.AttestationResponse\n        :raises: ~azure.core.exceptions.HttpResponseError\n        '
        cls = kwargs.pop('cls', None)
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError}
        error_map.update(kwargs.pop('error_map', {}))
        api_version = '2020-10-01'
        content_type = kwargs.pop('content_type', 'application/json')
        accept = 'application/json'
        url = self.attest_open_enclave.metadata['url']
        path_format_arguments = {'instanceUrl': self._serialize.url('self._config.instance_url', self._config.instance_url, 'str', skip_quote=True)}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        query_parameters['api-version'] = self._serialize.query('api_version', api_version, 'str')
        header_parameters = {}
        header_parameters['Content-Type'] = self._serialize.header('content_type', content_type, 'str')
        header_parameters['Accept'] = self._serialize.header('accept', accept, 'str')
        body_content_kwargs = {}
        body_content = self._serialize.body(request, 'AttestOpenEnclaveRequest')
        body_content_kwargs['content'] = body_content
        request = self._client.post(url, query_parameters, header_parameters, **body_content_kwargs)
        pipeline_response = self._client._pipeline.run(request, stream=False, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            error = self._deserialize.failsafe_deserialize(_models.CloudError, response)
            raise HttpResponseError(response=response, model=error)
        deserialized = self._deserialize('AttestationResponse', pipeline_response)
        if cls:
            return cls(pipeline_response, deserialized, {})
        return deserialized
    attest_open_enclave.metadata = {'url': '/attest/OpenEnclave'}

    def attest_sgx_enclave(self, request, **kwargs):
        if False:
            while True:
                i = 10
        'Attest to an SGX enclave.\n\n        Processes an SGX enclave quote, producing an artifact. The type of artifact produced is\n        dependent upon attestation policy.\n\n        :param request: Request object containing the quote.\n        :type request: ~azure.security.attestation._generated.models.AttestSgxEnclaveRequest\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: AttestationResponse, or the result of cls(response)\n        :rtype: ~azure.security.attestation._generated.models.AttestationResponse\n        :raises: ~azure.core.exceptions.HttpResponseError\n        '
        cls = kwargs.pop('cls', None)
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError}
        error_map.update(kwargs.pop('error_map', {}))
        api_version = '2020-10-01'
        content_type = kwargs.pop('content_type', 'application/json')
        accept = 'application/json'
        url = self.attest_sgx_enclave.metadata['url']
        path_format_arguments = {'instanceUrl': self._serialize.url('self._config.instance_url', self._config.instance_url, 'str', skip_quote=True)}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        query_parameters['api-version'] = self._serialize.query('api_version', api_version, 'str')
        header_parameters = {}
        header_parameters['Content-Type'] = self._serialize.header('content_type', content_type, 'str')
        header_parameters['Accept'] = self._serialize.header('accept', accept, 'str')
        body_content_kwargs = {}
        body_content = self._serialize.body(request, 'AttestSgxEnclaveRequest')
        body_content_kwargs['content'] = body_content
        request = self._client.post(url, query_parameters, header_parameters, **body_content_kwargs)
        pipeline_response = self._client._pipeline.run(request, stream=False, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            error = self._deserialize.failsafe_deserialize(_models.CloudError, response)
            raise HttpResponseError(response=response, model=error)
        deserialized = self._deserialize('AttestationResponse', pipeline_response)
        if cls:
            return cls(pipeline_response, deserialized, {})
        return deserialized
    attest_sgx_enclave.metadata = {'url': '/attest/SgxEnclave'}

    def attest_tpm(self, data=None, **kwargs):
        if False:
            return 10
        'Attest a Virtualization-based Security (VBS) enclave.\n\n        Processes attestation evidence from a VBS enclave, producing an attestation result. The\n        attestation result produced is dependent upon the attestation policy.\n\n        :param data: Protocol data containing artifacts for attestation.\n        :type data: bytes\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: TpmAttestationResponse, or the result of cls(response)\n        :rtype: ~azure.security.attestation._generated.models.TpmAttestationResponse\n        :raises: ~azure.core.exceptions.HttpResponseError\n        '
        cls = kwargs.pop('cls', None)
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError}
        error_map.update(kwargs.pop('error_map', {}))
        _request = _models.TpmAttestationRequest(data=data)
        api_version = '2020-10-01'
        content_type = kwargs.pop('content_type', 'application/json')
        accept = 'application/json'
        url = self.attest_tpm.metadata['url']
        path_format_arguments = {'instanceUrl': self._serialize.url('self._config.instance_url', self._config.instance_url, 'str', skip_quote=True)}
        url = self._client.format_url(url, **path_format_arguments)
        query_parameters = {}
        query_parameters['api-version'] = self._serialize.query('api_version', api_version, 'str')
        header_parameters = {}
        header_parameters['Content-Type'] = self._serialize.header('content_type', content_type, 'str')
        header_parameters['Accept'] = self._serialize.header('accept', accept, 'str')
        body_content_kwargs = {}
        body_content = self._serialize.body(_request, 'TpmAttestationRequest')
        body_content_kwargs['content'] = body_content
        request = self._client.post(url, query_parameters, header_parameters, **body_content_kwargs)
        pipeline_response = self._client._pipeline.run(request, stream=False, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            error = self._deserialize.failsafe_deserialize(_models.CloudError, response)
            raise HttpResponseError(response=response, model=error)
        deserialized = self._deserialize('TpmAttestationResponse', pipeline_response)
        if cls:
            return cls(pipeline_response, deserialized, {})
        return deserialized
    attest_tpm.metadata = {'url': '/attest/Tpm'}