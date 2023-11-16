from unittest.mock import MagicMock
from django.template import engines
from sentry.plugins.base.v2 import Plugin2
from sentry.testutils.cases import PluginTestCase

class SamplePlugin(Plugin2):

    def get_actions(self, request, group):
        if False:
            i = 10
            return i + 15
        return [('Example Action', f'http://example.com?id={group.id}')]

    def get_annotations(self, group):
        if False:
            i = 10
            return i + 15
        return [{'label': 'Example Tag', 'url': f'http://example.com?id={group.id}'}, {'label': 'Example Two'}]

    def is_enabled(self, project=None):
        if False:
            while True:
                i = 10
        return True

class GetActionsTest(PluginTestCase):
    plugin = SamplePlugin
    TEMPLATE = engines['django'].from_string('\n        {% load sentry_plugins %}\n        {% for k, v in group|get_actions:request %}\n            <span>{{ k }} - {{ v }}</span>\n        {% endfor %}\n    ')

    def test_includes_v2_plugins(self):
        if False:
            while True:
                i = 10
        group = self.create_group()
        result = self.TEMPLATE.render(context={'group': group}, request=MagicMock())
        assert f'<span>Example Action - http://example.com?id={group.id}</span>' in result

class GetAnnotationsTest(PluginTestCase):
    plugin = SamplePlugin
    TEMPLATE = engines['django'].from_string('\n        {% load sentry_plugins %}\n        {% for a in group|get_annotations:request %}\n            <span>{{ a.label }} - {{ a.url }}</span>\n        {% endfor %}\n    ')

    def test_includes_v2_plugins(self):
        if False:
            while True:
                i = 10
        group = self.create_group()
        result = self.TEMPLATE.render(context={'group': group}, request=MagicMock())
        assert f'<span>Example Tag - http://example.com?id={group.id}</span>' in result
        assert '<span>Example Two - None</span>' in result