"""v2 format CRUD test runner.

https://github.com/mongodb/specifications/blob/master/source/crud/tests/README.rst
"""
from __future__ import annotations
from test.utils_spec_runner import SpecRunner

class TestCrudV2(SpecRunner):
    TEST_DB = None
    TEST_COLLECTION = None

    def allowable_errors(self, op):
        if False:
            while True:
                i = 10
        'Override expected error classes.'
        errors = super().allowable_errors(op)
        errors += (ValueError,)
        return errors

    def get_scenario_db_name(self, scenario_def):
        if False:
            return 10
        'Crud spec says database_name is optional.'
        return scenario_def.get('database_name', self.TEST_DB)

    def get_scenario_coll_name(self, scenario_def):
        if False:
            i = 10
            return i + 15
        'Crud spec says collection_name is optional.'
        return scenario_def.get('collection_name', self.TEST_COLLECTION)

    def get_object_name(self, op):
        if False:
            i = 10
            return i + 15
        "Crud spec says object is optional and defaults to 'collection'."
        return op.get('object', 'collection')

    def get_outcome_coll_name(self, outcome, collection):
        if False:
            while True:
                i = 10
        "Crud spec says outcome has an optional 'collection.name'."
        return outcome['collection'].get('name', collection.name)

    def setup_scenario(self, scenario_def):
        if False:
            print('Hello World!')
        "Allow specs to override a test's setup."
        if scenario_def['data']:
            super().setup_scenario(scenario_def)