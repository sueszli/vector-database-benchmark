from os import path
import unittest
from ..util import LanghostTest

class ComponentDependenciesTest(LanghostTest):

    def test_component_provider_resolution(self):
        if False:
            return 10
        self.run_test(program=path.join(self.base_path(), 'component_provider_resolution'), expected_resource_count=4)

    def register_resource(self, _ctx, _dry_run, ty, name, _resource, _dependencies, _parent, _custom, protect, _provider, _property_deps, _delete_before_replace, _ignore_changes, _version, _import, _replace_on_changes, _providers, source_position):
        if False:
            while True:
                i = 10
        if name == 'combined-mine':
            self.assertTrue(protect)
            self.assertEqual(_provider, '')
        elif name == 'combined-other':
            self.assertTrue(protect)
            self.assertEqual(_provider, 'prov1::prov1')
        return {'urn': name, 'id': name, 'object': {'outprop': 'qux'}}