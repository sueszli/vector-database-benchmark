import unittest
from fixtures.schema_validation import invalid_schema
from sentry.api.validators.sentry_apps.schema import validate_component

class TestOpenInSchemaValidation(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.schema = {'type': 'stacktrace-link', 'uri': '/sentry/issue'}

    def test_valid_schema(self):
        if False:
            i = 10
            return i + 15
        validate_component(self.schema)

    @invalid_schema
    def test_missing_uri(self):
        if False:
            print('Hello World!')
        del self.schema['uri']
        validate_component(self.schema)