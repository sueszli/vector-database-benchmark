"""Http Logging Policy for Azure SDK"""
import json
import logging
import time
from typing import Optional, TypeVar
from azure.core.pipeline import PipelineRequest, PipelineResponse
from azure.core.pipeline.policies import HttpLoggingPolicy
from .http_constants import HttpHeaders
HTTPResponseType = TypeVar('HTTPResponseType', covariant=True)
HTTPRequestType = TypeVar('HTTPRequestType', covariant=True)

def _format_error(payload: str) -> str:
    if False:
        i = 10
        return i + 15
    output = json.loads(payload)
    return output['message'].replace('\r', ' ')

class CosmosHttpLoggingPolicy(HttpLoggingPolicy):

    def __init__(self, logger: Optional[logging.Logger]=None, *, enable_diagnostics_logging: Optional[bool]=False, **kwargs):
        if False:
            print('Hello World!')
        self._enable_diagnostics_logging = enable_diagnostics_logging
        super().__init__(logger, **kwargs)
        if self._enable_diagnostics_logging:
            cosmos_disallow_list = ['Authorization', 'ProxyAuthorization']
            cosmos_allow_list = [v for (k, v) in HttpHeaders.__dict__.items() if not k.startswith('_') and k not in cosmos_disallow_list]
            self.allowed_header_names = set(cosmos_allow_list)

    def on_request(self, request):
        if False:
            print('Hello World!')
        super().on_request(request)
        if self._enable_diagnostics_logging:
            request.context['start_time'] = time.time()

    def on_response(self, request: PipelineRequest[HTTPRequestType], response: PipelineResponse[HTTPRequestType, HTTPResponseType]) -> None:
        if False:
            while True:
                i = 10
        super().on_response(request, response)
        if self._enable_diagnostics_logging:
            http_response = response.http_response
            options = response.context.options
            logger = request.context.setdefault('logger', options.pop('logger', self.logger))
            try:
                logger.info('Elapsed time in seconds: {}'.format(time.time() - request.context.get('start_time')))
                if http_response.status_code >= 400:
                    logger.info('Response error message: %r', _format_error(http_response.text()))
            except Exception as err:
                logger.warning('Failed to log request: %s', repr(err))