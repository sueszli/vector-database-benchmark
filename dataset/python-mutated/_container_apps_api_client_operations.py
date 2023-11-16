from typing import Any, Callable, Dict, Optional, TypeVar
from azure.core.exceptions import ClientAuthenticationError, HttpResponseError, ResourceExistsError, ResourceNotFoundError, ResourceNotModifiedError, map_error
from azure.core.pipeline import PipelineResponse
from azure.core.pipeline.transport import HttpResponse
from azure.core.rest import HttpRequest
from azure.core.tracing.decorator import distributed_trace
from azure.core.utils import case_insensitive_dict
from azure.mgmt.core.exceptions import ARMErrorFormat
from .. import models as _models
from .._serialization import Serializer
from .._vendor import ContainerAppsAPIClientMixinABC, _convert_request
T = TypeVar('T')
ClsType = Optional[Callable[[PipelineResponse[HttpRequest, HttpResponse], T, Dict[str, Any]], Any]]
_SERIALIZER = Serializer()
_SERIALIZER.client_side_validation = False

def build_job_execution_request(resource_group_name: str, job_name: str, job_execution_name: str, subscription_id: str, **kwargs: Any) -> HttpRequest:
    if False:
        print('Hello World!')
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: str = kwargs.pop('api_version', _params.pop('api-version', '2023-05-01'))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.App/jobs/{jobName}/executions/{jobExecutionName}')
    path_format_arguments = {'subscriptionId': _SERIALIZER.url('subscription_id', subscription_id, 'str', min_length=1), 'resourceGroupName': _SERIALIZER.url('resource_group_name', resource_group_name, 'str', max_length=90, min_length=1), 'jobName': _SERIALIZER.url('job_name', job_name, 'str', pattern='^[-\\w\\._\\(\\)]+$'), 'jobExecutionName': _SERIALIZER.url('job_execution_name', job_execution_name, 'str', pattern='^[-\\w\\._\\(\\)]+$')}
    _url: str = _url.format(**path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='GET', url=_url, params=_params, headers=_headers, **kwargs)

class ContainerAppsAPIClientOperationsMixin(ContainerAppsAPIClientMixinABC):

    @distributed_trace
    def job_execution(self, resource_group_name: str, job_name: str, job_execution_name: str, **kwargs: Any) -> _models.JobExecution:
        if False:
            i = 10
            return i + 15
        'Get details of a single job execution.\n\n        Get details of a single job execution.\n\n        :param resource_group_name: The name of the resource group. The name is case insensitive.\n         Required.\n        :type resource_group_name: str\n        :param job_name: Job Name. Required.\n        :type job_name: str\n        :param job_execution_name: Job execution name. Required.\n        :type job_execution_name: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: JobExecution or the result of cls(response)\n        :rtype: ~azure.mgmt.appcontainers.models.JobExecution\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        cls: ClsType[_models.JobExecution] = kwargs.pop('cls', None)
        request = build_job_execution_request(resource_group_name=resource_group_name, job_name=job_name, job_execution_name=job_execution_name, subscription_id=self._config.subscription_id, api_version=api_version, template_url=self.job_execution.metadata['url'], headers=_headers, params=_params)
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)
        _stream = False
        pipeline_response: PipelineResponse = self._client._pipeline.run(request, stream=_stream, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            error = self._deserialize.failsafe_deserialize(_models.DefaultErrorResponse, pipeline_response)
            raise HttpResponseError(response=response, model=error, error_format=ARMErrorFormat)
        deserialized = self._deserialize('JobExecution', pipeline_response)
        if cls:
            return cls(pipeline_response, deserialized, {})
        return deserialized
    job_execution.metadata = {'url': '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.App/jobs/{jobName}/executions/{jobExecutionName}'}