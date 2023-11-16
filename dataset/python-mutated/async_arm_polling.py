from typing import Optional, Dict, Any, Sequence
from azure.core.polling.base_polling import LocationPolling, StatusCheckPolling, LongRunningOperation
from azure.core.polling.async_base_polling import AsyncLROBasePolling
from .arm_polling import AzureAsyncOperationPolling, BodyContentPolling, HttpRequestTypeVar, AllHttpResponseTypeVar

class AsyncARMPolling(AsyncLROBasePolling):

    def __init__(self, timeout: float=30, lro_algorithms: Optional[Sequence[LongRunningOperation[HttpRequestTypeVar, AllHttpResponseTypeVar]]]=None, lro_options: Optional[Dict[str, Any]]=None, path_format_arguments: Optional[Dict[str, str]]=None, **operation_config: Any) -> None:
        if False:
            while True:
                i = 10
        lro_algorithms = lro_algorithms or [AzureAsyncOperationPolling(lro_options=lro_options), LocationPolling(), BodyContentPolling(), StatusCheckPolling()]
        super(AsyncLROBasePolling, self).__init__(timeout=timeout, lro_algorithms=lro_algorithms, lro_options=lro_options, path_format_arguments=path_format_arguments, **operation_config)
__all__ = ['AsyncARMPolling']