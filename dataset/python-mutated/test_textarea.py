from __future__ import annotations
import unittest
from typing import Any
from fixtures.schema_validation import invalid_schema
from sentry.api.validators.sentry_apps.schema import validate_component

class TestTextareaSchemaValidation(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.schema: dict[str, Any] = {'type': 'textarea', 'name': 'title', 'label': 'Title'}

    def test_valid_schema(self):
        if False:
            while True:
                i = 10
        validate_component(self.schema)

    def test_with_a_valid_default(self):
        if False:
            i = 10
            return i + 15
        self.schema['default'] = 'issue.title'
        validate_component(self.schema)

    @invalid_schema
    def test_missing_name(self):
        if False:
            return 10
        del self.schema['name']
        validate_component(self.schema)

    @invalid_schema
    def test_missing_label(self):
        if False:
            print('Hello World!')
        del self.schema['label']
        validate_component(self.schema)

    @invalid_schema
    def test_invalid_label_type(self):
        if False:
            i = 10
            return i + 15
        self.schema['label'] = 1
        validate_component(self.schema)

    @invalid_schema
    def test_invalid_name_type(self):
        if False:
            print('Hello World!')
        self.schema['name'] = 1
        validate_component(self.schema)

    @invalid_schema
    def test_invalid_default_option(self):
        if False:
            while True:
                i = 10
        self.schema['default'] = 'issue.id'
        validate_component(self.schema)