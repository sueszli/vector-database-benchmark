from enum import Enum
from typing import Optional, Union, TypeVar, Dict, Any, Sequence
from azure.core import CaseInsensitiveEnumMeta
from azure.core.polling.base_polling import LongRunningOperation, LROBasePolling, OperationFailed, BadResponse, OperationResourcePolling, LocationPolling, StatusCheckPolling, _as_json, _is_empty
from azure.core.pipeline import PipelineResponse
from azure.core.pipeline.transport import HttpRequest as LegacyHttpRequest, HttpResponse as LegacyHttpResponse, AsyncHttpResponse as LegacyAsyncHttpResponse
from azure.core.rest import HttpRequest, HttpResponse, AsyncHttpResponse
ResponseType = Union[HttpResponse, AsyncHttpResponse]
PipelineResponseType = PipelineResponse[HttpRequest, ResponseType]
HttpRequestType = Union[LegacyHttpRequest, HttpRequest]
AllHttpResponseType = Union[LegacyHttpResponse, HttpResponse, LegacyAsyncHttpResponse, AsyncHttpResponse]
HttpRequestTypeVar = TypeVar('HttpRequestTypeVar', bound=HttpRequestType)
AllHttpResponseTypeVar = TypeVar('AllHttpResponseTypeVar', bound=AllHttpResponseType)

class _LroOption(str, Enum, metaclass=CaseInsensitiveEnumMeta):
    """Known LRO options from Swagger."""
    FINAL_STATE_VIA = 'final-state-via'

class _FinalStateViaOption(str, Enum, metaclass=CaseInsensitiveEnumMeta):
    """Possible final-state-via options."""
    AZURE_ASYNC_OPERATION_FINAL_STATE = 'azure-async-operation'
    LOCATION_FINAL_STATE = 'location'

class AzureAsyncOperationPolling(OperationResourcePolling[HttpRequestTypeVar, AllHttpResponseTypeVar]):
    """Implements a operation resource polling, typically from Azure-AsyncOperation."""

    def __init__(self, lro_options: Optional[Dict[str, Any]]=None) -> None:
        if False:
            return 10
        super(AzureAsyncOperationPolling, self).__init__(operation_location_header='azure-asyncoperation')
        self._lro_options = lro_options or {}

    def get_final_get_url(self, pipeline_response: PipelineResponse[HttpRequestTypeVar, AllHttpResponseTypeVar]) -> Optional[str]:
        if False:
            while True:
                i = 10
        'If a final GET is needed, returns the URL.\n\n        :param ~azure.core.pipeline.PipelineResponse pipeline_response: The pipeline response object.\n        :return: The URL to poll for the final GET.\n        :rtype: str\n        '
        if self._lro_options.get(_LroOption.FINAL_STATE_VIA) == _FinalStateViaOption.AZURE_ASYNC_OPERATION_FINAL_STATE and self._request.method == 'POST':
            return None
        return super(AzureAsyncOperationPolling, self).get_final_get_url(pipeline_response)

class BodyContentPolling(LongRunningOperation[HttpRequestTypeVar, AllHttpResponseTypeVar]):
    """Poll based on the body content.

    Implement a ARM resource poller (using provisioning state).
    """
    _initial_response: PipelineResponse[HttpRequestTypeVar, AllHttpResponseTypeVar]
    'Store the initial response.'

    def can_poll(self, pipeline_response: PipelineResponse[HttpRequestTypeVar, AllHttpResponseTypeVar]) -> bool:
        if False:
            while True:
                i = 10
        'Answer if this polling method could be used.\n\n        :param ~azure.core.pipeline.PipelineResponse pipeline_response: The pipeline response object.\n        :return: True if this polling method could be used.\n        :rtype: bool\n        '
        response = pipeline_response.http_response
        return response.request.method in ['PUT', 'PATCH']

    def get_polling_url(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        'Return the polling URL.\n        :return: The polling URL.\n        :rtype: str\n        '
        return self._initial_response.http_response.request.url

    def get_final_get_url(self, pipeline_response: Any) -> None:
        if False:
            return 10
        'If a final GET is needed, returns the URL.\n\n        :param ~azure.core.pipeline.PipelineResponse pipeline_response: The pipeline response object.\n        :return: The URL to poll for the final GET.\n        :rtype: str\n        '
        return None

    def set_initial_status(self, pipeline_response: PipelineResponse[HttpRequestTypeVar, AllHttpResponseTypeVar]) -> str:
        if False:
            i = 10
            return i + 15
        'Process first response after initiating long running operation.\n\n        :param ~azure.core.pipeline.PipelineResponse pipeline_response: initial REST call response.\n        :return: Status string.\n        :rtype: str\n        '
        self._initial_response = pipeline_response
        response = pipeline_response.http_response
        if response.status_code == 202:
            return 'InProgress'
        if response.status_code == 201:
            status = self._get_provisioning_state(response)
            return status or 'InProgress'
        if response.status_code == 200:
            status = self._get_provisioning_state(response)
            return status or 'Succeeded'
        if response.status_code == 204:
            return 'Succeeded'
        raise OperationFailed('Invalid status found')

    @staticmethod
    def _get_provisioning_state(response: AllHttpResponseTypeVar) -> Optional[str]:
        if False:
            print('Hello World!')
        "Attempt to get provisioning state from resource.\n\n        :param azure.core.pipeline.transport.HttpResponse response: latest REST call response.\n        :returns: Status if found, else 'None'.\n        :rtype: str or None\n        "
        if _is_empty(response):
            return None
        body = _as_json(response)
        return body.get('properties', {}).get('provisioningState')

    def get_status(self, pipeline_response: PipelineResponse[HttpRequestTypeVar, AllHttpResponseTypeVar]) -> str:
        if False:
            i = 10
            return i + 15
        'Process the latest status update retrieved from the same URL as\n        the previous request.\n\n        :param ~azure.core.pipeline.PipelineResponse pipeline_response: latest REST call response.\n        :return: Status string.\n        :rtype: str\n        :raises: BadResponse if status not 200 or 204.\n        '
        response = pipeline_response.http_response
        if _is_empty(response):
            raise BadResponse('The response from long running operation does not contain a body.')
        status = self._get_provisioning_state(response)
        return status or 'Succeeded'

class ARMPolling(LROBasePolling):

    def __init__(self, timeout: float=30, lro_algorithms: Optional[Sequence[LongRunningOperation[HttpRequestTypeVar, AllHttpResponseTypeVar]]]=None, lro_options: Optional[Dict[str, Any]]=None, path_format_arguments: Optional[Dict[str, str]]=None, **operation_config: Any) -> None:
        if False:
            i = 10
            return i + 15
        lro_algorithms = lro_algorithms or [AzureAsyncOperationPolling(lro_options=lro_options), LocationPolling(), BodyContentPolling(), StatusCheckPolling()]
        super(ARMPolling, self).__init__(timeout=timeout, lro_algorithms=lro_algorithms, lro_options=lro_options, path_format_arguments=path_format_arguments, **operation_config)
__all__ = ['AzureAsyncOperationPolling', 'BodyContentPolling', 'ARMPolling']