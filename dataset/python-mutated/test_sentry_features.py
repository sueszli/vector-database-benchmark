from django.template import engines
from sentry.testutils.cases import TestCase

class FeaturesTest(TestCase):
    TEMPLATE = engines['django'].from_string('\n        {% load sentry_features %}\n        {% feature auth:register %}\n            <span>register</span>\n        {% else %}\n            <span>nope</span>\n        {% endfeature %}\n    ')

    def test_enabled(self):
        if False:
            while True:
                i = 10
        with self.feature('auth:register'):
            result = self.TEMPLATE.render()
            assert '<span>register</span>' in result

    def test_disabled(self):
        if False:
            return 10
        with self.feature({'auth:register': False}):
            result = self.TEMPLATE.render()
            assert '<span>nope</span>' in result