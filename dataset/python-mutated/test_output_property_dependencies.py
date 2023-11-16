from os import path
import unittest
from ..util import LanghostTest

class OutputPropertyDependenciesTest(LanghostTest):

    def test_output_property_dependencies(self):
        if False:
            print('Hello World!')
        self.run_test(program=path.join(self.base_path(), 'output_property_dependencies'), expected_resource_count=2)

    def register_resource(self, _ctx, _dry_run, ty, name, _resource, _dependencies, _parent, _custom, protect, _provider, _property_deps, _delete_before_replace, _ignore_changes, _version, _import, _replace_on_changes, _providers, source_position):
        if False:
            print('Hello World!')
        self.assertEqual(ty, 'test:index:MyResource')
        if name == 'resA':
            return {'urn': name, 'id': name, 'object': {'outProp': 'qux'}, 'propertyDependencies': {'outProp': ['resB']}}
        elif name == 'resC':
            self.assertListEqual(_dependencies, ['resA', 'resB'], msg=f'{name}._dependencies')
            self.assertDictEqual(_property_deps, {'inProp': ['resA', 'resB']}, msg=f'{name}._property_deps')
        return {'urn': name, 'id': name, 'object': {'outProp': 'qux'}}