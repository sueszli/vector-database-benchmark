"""Sample plugin which renders custom panels on certain pages."""
from part.views import PartDetail
from plugin import InvenTreePlugin
from plugin.mixins import PanelMixin, SettingsMixin
from stock.views import StockLocationDetail

class CustomPanelSample(PanelMixin, SettingsMixin, InvenTreePlugin):
    """A sample plugin which renders some custom panels."""
    NAME = 'CustomPanelExample'
    SLUG = 'samplepanel'
    TITLE = 'Custom Panel Example'
    DESCRIPTION = 'An example plugin demonstrating how custom panels can be added to the user interface'
    VERSION = '0.1'
    SETTINGS = {'ENABLE_HELLO_WORLD': {'name': 'Enable Hello World', 'description': 'Enable a custom hello world panel on every page', 'default': False, 'validator': bool}, 'ENABLE_BROKEN_PANEL': {'name': 'Enable Broken Panel', 'description': 'Enable a panel with rendering issues', 'default': False, 'validator': bool}}

    def get_panel_context(self, view, request, context):
        if False:
            print('Hello World!')
        'Returns enriched context.'
        ctx = super().get_panel_context(view, request, context)
        if isinstance(view, StockLocationDetail):
            ctx['location'] = view.get_object()
        return ctx

    def get_custom_panels(self, view, request):
        if False:
            while True:
                i = 10
        'You can decide at run-time which custom panels you want to display!\n\n        - Display on every page\n        - Only on a single page or set of pages\n        - Only for a specific instance (e.g. part)\n        - Based on the user viewing the page!\n        '
        panels = [{'title': 'No Content'}]
        if self.get_setting('ENABLE_HELLO_WORLD'):
            content = "\n            <strong>Hello world!</strong>\n            <hr>\n            <div class='alert-alert-block alert-info'>\n                <em>We can render custom content using the templating system!</em>\n            </div>\n            <hr>\n            <table class='table table-striped'>\n                <tr><td><strong>Path</strong></td><td>{{ request.path }}</tr>\n                <tr><td><strong>User</strong></td><td>{{ user.username }}</tr>\n            </table>\n            "
            panels.append({'title': 'Hello World', 'icon': 'fas fa-boxes', 'content': content, 'description': 'A simple panel which renders hello world', 'javascript': 'console.log("Hello world, from a custom panel!");'})
        if self.get_setting('ENABLE_BROKEN_PANEL'):
            panels.append({'title': 'Broken Panel', 'icon': 'fas fa-times-circle', 'content': '{% tag_not_loaded %}', 'description': 'This panel is broken', 'javascript': '{% another_bad_tag %}'})
        if isinstance(view, PartDetail):
            panels.append({'title': 'Custom Part Panel', 'icon': 'fas fa-shapes', 'content': '<em>This content only appears on the PartDetail page, you know!</em>'})
        if isinstance(view, StockLocationDetail):
            try:
                loc = view.get_object()
                if not loc.get_descendants(include_self=False).exists():
                    panels.append({'title': 'Childless Location', 'icon': 'fa-user', 'content_template': 'panel_demo/childless.html'})
            except Exception:
                pass
        return panels