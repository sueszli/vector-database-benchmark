from __future__ import annotations
import logging
from typing import Callable, List, Type
from django.http import HttpRequest
from django.http.response import HttpResponseBase
from sentry.silo import SiloMode
logger = logging.getLogger(__name__)
from sentry.middleware.integrations.classifications import BaseClassification, IntegrationClassification, PluginClassification
ResponseHandler = Callable[[HttpRequest], HttpResponseBase]

class IntegrationControlMiddleware:
    classifications: List[Type[BaseClassification]] = [IntegrationClassification, PluginClassification]
    '\n    Classifications to determine whether request must be parsed, sorted in priority order.\n    getsentry expands this list on django initialization.\n    '

    def __init__(self, get_response: ResponseHandler):
        if False:
            for i in range(10):
                print('nop')
        self.get_response = get_response

    def _should_operate(self, request: HttpRequest) -> bool:
        if False:
            return 10
        '\n        Determines whether this middleware will operate or just pass the request along.\n        '
        return SiloMode.get_current_mode() == SiloMode.CONTROL

    @classmethod
    def register_classifications(cls, classifications: List[Type[BaseClassification]]):
        if False:
            i = 10
            return i + 15
        '\n        Add new classifications for middleware to determine request parsing dynamically.\n        Used in getsentry to expand scope of parsing.\n        '
        cls.classifications += classifications

    def __call__(self, request: HttpRequest):
        if False:
            while True:
                i = 10
        if not self._should_operate(request):
            return self.get_response(request)
        for classification in self.classifications:
            _cls = classification(response_handler=self.get_response)
            if _cls.should_operate(request):
                return _cls.get_response(request)
        return self.get_response(request)