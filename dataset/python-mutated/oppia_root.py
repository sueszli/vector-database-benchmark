"""Controllers for the oppia root page."""
from __future__ import annotations
from core.controllers import acl_decorators
from core.controllers import base
from typing import Dict

class OppiaRootPage(base.BaseHandler[Dict[str, str], Dict[str, str]]):
    """Renders oppia root page (unified entry point) for all routes registered
    with angular router.
    """

    @acl_decorators.open_access
    def get(self, **kwargs: Dict[str, str]) -> None:
        if False:
            return 10
        'Handles GET requests.'
        self.render_template('oppia-root.mainpage.html')

class OppiaLightweightRootPage(base.BaseHandler[Dict[str, str], Dict[str, str]]):
    """Renders lightweight oppia root page (unified entry point) for all routes
    registered with angular router.
    """

    @acl_decorators.open_access
    def get(self, **kwargs: Dict[str, str]) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Handles GET requests.'
        if self.request.cookies.get('dir') == 'rtl':
            self.render_template('lightweight-oppia-root.mainpage.html')
            return
        if self.request.cookies.get('dir') == 'ltr':
            self.render_template('index.html', template_is_aot_compiled=True)
            return
        if self.request.get('dir') == 'rtl':
            self.render_template('lightweight-oppia-root.mainpage.html')
            return
        self.render_template('index.html', template_is_aot_compiled=True)