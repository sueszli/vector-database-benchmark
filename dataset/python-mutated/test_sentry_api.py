from django.template import engines
from sentry.testutils.cases import TestCase

class SerializeDetailedOrgTest(TestCase):
    TEMPLATE = engines['django'].from_string('\n        {% load sentry_api %}\n        {% serialize_detailed_org org %}\n    ')

    def test_escapes_js(self):
        if False:
            while True:
                i = 10
        org = self.create_organization(name='<script>alert(1);</script>')
        result = self.TEMPLATE.render(context={'org': org})
        assert '<script>' not in result
        assert '\\u003cscript\\u003ealert(1);\\u003c/script\\u003e' in result