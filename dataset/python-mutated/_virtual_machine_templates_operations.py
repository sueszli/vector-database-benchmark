import sys
from typing import Any, AsyncIterable, Callable, Dict, Optional, TypeVar
import urllib.parse
from azure.core.async_paging import AsyncItemPaged, AsyncList
from azure.core.exceptions import ClientAuthenticationError, HttpResponseError, ResourceExistsError, ResourceNotFoundError, ResourceNotModifiedError, map_error
from azure.core.pipeline import PipelineResponse
from azure.core.pipeline.transport import AsyncHttpResponse
from azure.core.rest import HttpRequest
from azure.core.tracing.decorator import distributed_trace
from azure.core.tracing.decorator_async import distributed_trace_async
from azure.core.utils import case_insensitive_dict
from azure.mgmt.core.exceptions import ARMErrorFormat
from ... import models as _models
from ..._vendor import _convert_request
from ...operations._virtual_machine_templates_operations import build_get_request, build_list_request
if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal
T = TypeVar('T')
ClsType = Optional[Callable[[PipelineResponse[HttpRequest, AsyncHttpResponse], T, Dict[str, Any]], Any]]

class VirtualMachineTemplatesOperations:
    """
    .. warning::
        **DO NOT** instantiate this class directly.

        Instead, you should access the following operations through
        :class:`~azure.mgmt.vmwarecloudsimple.aio.VMwareCloudSimple`'s
        :attr:`virtual_machine_templates` attribute.
    """
    models = _models

    def __init__(self, *args, **kwargs) -> None:
        if False:
            i = 10
            return i + 15
        input_args = list(args)
        self._client = input_args.pop(0) if input_args else kwargs.pop('client')
        self._config = input_args.pop(0) if input_args else kwargs.pop('config')
        self._serialize = input_args.pop(0) if input_args else kwargs.pop('serializer')
        self._deserialize = input_args.pop(0) if input_args else kwargs.pop('deserializer')

    @distributed_trace
    def list(self, pc_name: str, region_id: str, resource_pool_name: str, **kwargs: Any) -> AsyncIterable['_models.VirtualMachineTemplate']:
        if False:
            i = 10
            return i + 15
        'Implements list of available VM templates.\n\n        Returns list of virtual machine templates in region for private cloud.\n\n        :param pc_name: The private cloud name. Required.\n        :type pc_name: str\n        :param region_id: The region Id (westus, eastus). Required.\n        :type region_id: str\n        :param resource_pool_name: Resource pool used to derive vSphere cluster which contains VM\n         templates. Required.\n        :type resource_pool_name: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: An iterator like instance of either VirtualMachineTemplate or the result of\n         cls(response)\n        :rtype:\n         ~azure.core.async_paging.AsyncItemPaged[~azure.mgmt.vmwarecloudsimple.models.VirtualMachineTemplate]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        cls = kwargs.pop('cls', None)
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})

        def prepare_request(next_link=None):
            if False:
                print('Hello World!')
            if not next_link:
                request = build_list_request(pc_name=pc_name, region_id=region_id, subscription_id=self._config.subscription_id, resource_pool_name=resource_pool_name, api_version=api_version, template_url=self.list.metadata['url'], headers=_headers, params=_params)
                request = _convert_request(request)
                request.url = self._client.format_url(request.url)
            else:
                _parsed_next_link = urllib.parse.urlparse(next_link)
                _next_request_params = case_insensitive_dict({key: [urllib.parse.quote(v) for v in value] for (key, value) in urllib.parse.parse_qs(_parsed_next_link.query).items()})
                _next_request_params['api-version'] = self._config.api_version
                request = HttpRequest('GET', urllib.parse.urljoin(next_link, _parsed_next_link.path), params=_next_request_params)
                request = _convert_request(request)
                request.url = self._client.format_url(request.url)
                request.method = 'GET'
            return request

        async def extract_data(pipeline_response):
            deserialized = self._deserialize('VirtualMachineTemplateListResponse', pipeline_response)
            list_of_elem = deserialized.value
            if cls:
                list_of_elem = cls(list_of_elem)
            return (deserialized.next_link or None, AsyncList(list_of_elem))

        async def get_next(next_link=None):
            request = prepare_request(next_link)
            pipeline_response = await self._client._pipeline.run(request, stream=False, **kwargs)
            response = pipeline_response.http_response
            if response.status_code not in [200]:
                map_error(status_code=response.status_code, response=response, error_map=error_map)
                error = self._deserialize.failsafe_deserialize(_models.CSRPError, pipeline_response)
                raise HttpResponseError(response=response, model=error, error_format=ARMErrorFormat)
            return pipeline_response
        return AsyncItemPaged(get_next, extract_data)
    list.metadata = {'url': '/subscriptions/{subscriptionId}/providers/Microsoft.VMwareCloudSimple/locations/{regionId}/privateClouds/{pcName}/virtualMachineTemplates'}

    @distributed_trace_async
    async def get(self, region_id: str, pc_name: str, virtual_machine_template_name: str, **kwargs: Any) -> _models.VirtualMachineTemplate:
        """Implements virtual machine template GET method.

        Returns virtual machine templates by its name.

        :param region_id: The region Id (westus, eastus). Required.
        :type region_id: str
        :param pc_name: The private cloud name. Required.
        :type pc_name: str
        :param virtual_machine_template_name: virtual machine template id (vsphereId). Required.
        :type virtual_machine_template_name: str
        :keyword callable cls: A custom type or function that will be passed the direct response
        :return: VirtualMachineTemplate or the result of cls(response)
        :rtype: ~azure.mgmt.vmwarecloudsimple.models.VirtualMachineTemplate
        :raises ~azure.core.exceptions.HttpResponseError:
        """
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        cls = kwargs.pop('cls', None)
        request = build_get_request(region_id=region_id, pc_name=pc_name, virtual_machine_template_name=virtual_machine_template_name, subscription_id=self._config.subscription_id, api_version=api_version, template_url=self.get.metadata['url'], headers=_headers, params=_params)
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)
        pipeline_response = await self._client._pipeline.run(request, stream=False, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            error = self._deserialize.failsafe_deserialize(_models.CSRPError, pipeline_response)
            raise HttpResponseError(response=response, model=error, error_format=ARMErrorFormat)
        deserialized = self._deserialize('VirtualMachineTemplate', pipeline_response)
        if cls:
            return cls(pipeline_response, deserialized, {})
        return deserialized
    get.metadata = {'url': '/subscriptions/{subscriptionId}/providers/Microsoft.VMwareCloudSimple/locations/{regionId}/privateClouds/{pcName}/virtualMachineTemplates/{virtualMachineTemplateName}'}