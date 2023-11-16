from os import path
import unittest
from ..util import LanghostTest

class InvalidPropertyDependencyTest(LanghostTest):

    def test_invalid_property_dependency(self):
        if False:
            for i in range(10):
                print('nop')
        self.run_test(program=path.join(self.base_path(), 'invalid_property_dependency'), expected_bail=True, expected_resource_count=1)

    def register_resource(self, _ctx, _dry_run, ty, name, _resource, _dependencies, _parent, _custom, protect, _provider, _property_deps, _delete_before_replace, _ignore_changes, _version, _import, _replace_on_changes, _providers, source_position):
        if False:
            return 10
        self.assertEqual(ty, 'test:index:MyResource')
        if name == 'resA':
            self.assertListEqual(_dependencies, [])
            self.assertDictEqual(_property_deps, {})
        else:
            self.fail(f'unexpected resource: {name} ({ty})')
        return {'urn': name, 'id': name, 'object': {'outprop': 'qux'}}