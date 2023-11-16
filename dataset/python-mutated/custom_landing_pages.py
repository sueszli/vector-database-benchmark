"""Controllers for custom landing pages."""
from __future__ import annotations
from core.controllers import acl_decorators
from core.controllers import base
from typing import Dict

class FractionLandingRedirectPage(base.BaseHandler[Dict[str, str], Dict[str, str]]):
    """The handler redirecting to the Fractions landing page."""
    URL_PATH_ARGS_SCHEMAS: Dict[str, str] = {}
    HANDLER_ARGS_SCHEMAS: Dict[str, Dict[str, str]] = {'GET': {}}

    @acl_decorators.open_access
    def get(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Handles GET requests.'
        self.redirect('/math/fractions')

class TopicLandingRedirectPage(base.BaseHandler[Dict[str, str], Dict[str, str]]):
    """The handler redirecting the old landing page URL to the new one."""
    URL_PATH_ARGS_SCHEMAS = {'topic': {'schema': {'type': 'basestring'}}}
    HANDLER_ARGS_SCHEMAS: Dict[str, Dict[str, str]] = {'GET': {}}

    @acl_decorators.open_access
    def get(self, topic: str) -> None:
        if False:
            i = 10
            return i + 15
        'Handles GET requests.\n\n        Args:\n            topic: str. Topic of page to be redirected to.\n        '
        self.redirect('/math/%s' % topic)