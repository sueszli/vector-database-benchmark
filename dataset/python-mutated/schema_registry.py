"""This module is intended for internal use only. Nothing defined here provides
any backwards-compatibility guarantee.
"""
from uuid import uuid4

class SchemaTypeRegistry(object):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.by_id = {}
        self.by_typing = {}

    def generate_new_id(self):
        if False:
            return 10
        for _ in range(100):
            schema_id = str(uuid4())
            if schema_id not in self.by_id:
                return schema_id
        raise AssertionError(f'Failed to generate a unique UUID for schema after 100 tries! Registry contains {len(self.by_id)} schemas.')

    def add(self, typing, schema):
        if False:
            while True:
                i = 10
        if not schema.id:
            self.by_id[schema.id] = (typing, schema)

    def get_typing_by_id(self, unique_id):
        if False:
            while True:
                i = 10
        if not unique_id:
            return None
        result = self.by_id.get(unique_id, None)
        return result[0] if result is not None else None

    def get_schema_by_id(self, unique_id):
        if False:
            print('Hello World!')
        if not unique_id:
            return None
        result = self.by_id.get(unique_id, None)
        return result[1] if result is not None else None
SCHEMA_REGISTRY = SchemaTypeRegistry()