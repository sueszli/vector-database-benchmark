from typing import Any, Callable, Dict, IO, Optional, TypeVar, Union, overload
from azure.core.exceptions import ClientAuthenticationError, HttpResponseError, ResourceExistsError, ResourceNotFoundError, ResourceNotModifiedError, map_error
from azure.core.pipeline import PipelineResponse
from azure.core.pipeline.transport import HttpResponse
from azure.core.rest import HttpRequest
from azure.core.tracing.decorator import distributed_trace
from azure.core.utils import case_insensitive_dict
from azure.mgmt.core.exceptions import ARMErrorFormat
from .. import models as _models
from ..._serialization import Serializer
from .._vendor import DataBoxManagementClientMixinABC, _convert_request, _format_url_section
T = TypeVar('T')
ClsType = Optional[Callable[[PipelineResponse[HttpRequest, HttpResponse], T, Dict[str, Any]], Any]]
_SERIALIZER = Serializer()
_SERIALIZER.client_side_validation = False

def build_mitigate_request(job_name: str, resource_group_name: str, subscription_id: str, **kwargs: Any) -> HttpRequest:
    if False:
        while True:
            i = 10
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: str = kwargs.pop('api_version', _params.pop('api-version', '2022-09-01'))
    content_type: Optional[str] = kwargs.pop('content_type', _headers.pop('Content-Type', None))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DataBox/jobs/{jobName}/mitigate')
    path_format_arguments = {'jobName': _SERIALIZER.url('job_name', job_name, 'str', max_length=24, min_length=3, pattern='^[-\\w\\.]+$'), 'subscriptionId': _SERIALIZER.url('subscription_id', subscription_id, 'str'), 'resourceGroupName': _SERIALIZER.url('resource_group_name', resource_group_name, 'str')}
    _url: str = _format_url_section(_url, **path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    if content_type is not None:
        _headers['Content-Type'] = _SERIALIZER.header('content_type', content_type, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='POST', url=_url, params=_params, headers=_headers, **kwargs)

class DataBoxManagementClientOperationsMixin(DataBoxManagementClientMixinABC):

    @overload
    def mitigate(self, job_name: str, resource_group_name: str, mitigate_job_request: _models.MitigateJobRequest, *, content_type: str='application/json', **kwargs: Any) -> None:
        if False:
            print('Hello World!')
        'Request to mitigate for a given job.\n\n        :param job_name: The name of the job Resource within the specified resource group. job names\n         must be between 3 and 24 characters in length and use any alphanumeric and underscore only.\n         Required.\n        :type job_name: str\n        :param resource_group_name: The Resource Group Name. Required.\n        :type resource_group_name: str\n        :param mitigate_job_request: Mitigation Request. Required.\n        :type mitigate_job_request: ~azure.mgmt.databox.v2022_09_01.models.MitigateJobRequest\n        :keyword content_type: Body Parameter content-type. Content type parameter for JSON body.\n         Default value is "application/json".\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: None or the result of cls(response)\n        :rtype: None\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '

    @overload
    def mitigate(self, job_name: str, resource_group_name: str, mitigate_job_request: IO, *, content_type: str='application/json', **kwargs: Any) -> None:
        if False:
            print('Hello World!')
        'Request to mitigate for a given job.\n\n        :param job_name: The name of the job Resource within the specified resource group. job names\n         must be between 3 and 24 characters in length and use any alphanumeric and underscore only.\n         Required.\n        :type job_name: str\n        :param resource_group_name: The Resource Group Name. Required.\n        :type resource_group_name: str\n        :param mitigate_job_request: Mitigation Request. Required.\n        :type mitigate_job_request: IO\n        :keyword content_type: Body Parameter content-type. Content type parameter for binary body.\n         Default value is "application/json".\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: None or the result of cls(response)\n        :rtype: None\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '

    @distributed_trace
    def mitigate(self, job_name: str, resource_group_name: str, mitigate_job_request: Union[_models.MitigateJobRequest, IO], **kwargs: Any) -> None:
        if False:
            print('Hello World!')
        "Request to mitigate for a given job.\n\n        :param job_name: The name of the job Resource within the specified resource group. job names\n         must be between 3 and 24 characters in length and use any alphanumeric and underscore only.\n         Required.\n        :type job_name: str\n        :param resource_group_name: The Resource Group Name. Required.\n        :type resource_group_name: str\n        :param mitigate_job_request: Mitigation Request. Is either a MitigateJobRequest type or a IO\n         type. Required.\n        :type mitigate_job_request: ~azure.mgmt.databox.v2022_09_01.models.MitigateJobRequest or IO\n        :keyword content_type: Body Parameter content-type. Known values are: 'application/json'.\n         Default value is None.\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: None or the result of cls(response)\n        :rtype: None\n        :raises ~azure.core.exceptions.HttpResponseError:\n        "
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})
        _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', '2022-09-01'))
        content_type: Optional[str] = kwargs.pop('content_type', _headers.pop('Content-Type', None))
        cls: ClsType[None] = kwargs.pop('cls', None)
        content_type = content_type or 'application/json'
        _json = None
        _content = None
        if isinstance(mitigate_job_request, (IO, bytes)):
            _content = mitigate_job_request
        else:
            _json = self._serialize.body(mitigate_job_request, 'MitigateJobRequest')
        request = build_mitigate_request(job_name=job_name, resource_group_name=resource_group_name, subscription_id=self._config.subscription_id, api_version=api_version, content_type=content_type, json=_json, content=_content, template_url=self.mitigate.metadata['url'], headers=_headers, params=_params)
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)
        _stream = False
        pipeline_response: PipelineResponse = self._client._pipeline.run(request, stream=_stream, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [204]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            error = self._deserialize.failsafe_deserialize(_models.ApiError, pipeline_response)
            raise HttpResponseError(response=response, model=error, error_format=ARMErrorFormat)
        if cls:
            return cls(pipeline_response, None, {})
    mitigate.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.DataBox/jobs/{jobName}/mitigate'}