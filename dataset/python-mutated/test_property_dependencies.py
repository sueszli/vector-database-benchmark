from os import path
import unittest
from ..util import LanghostTest

class PropertyDependenciesTest(LanghostTest):

    def test_property_dependencies(self):
        if False:
            return 10
        self.run_test(program=path.join(self.base_path(), 'property_dependencies'), expected_resource_count=5)

    def register_resource(self, _ctx, _dry_run, ty, name, _resource, _dependencies, _parent, _custom, protect, _provider, _property_deps, _delete_before_replace, _ignore_changes, _version, _import, _replace_on_changes, _providers, source_position):
        if False:
            i = 10
            return i + 15
        self.assertEqual(ty, 'test:index:MyResource')
        if name == 'resA':
            self.assertListEqual(_dependencies, [], msg=f'{name}._dependencies')
            self.assertDictEqual(_property_deps, {}, msg=f'{name}._property_deps')
        elif name == 'resB':
            self.assertListEqual(_dependencies, ['resA'], msg=f'{name}._dependencies')
            self.assertDictEqual(_property_deps, {}, msg=f'{name}._property_deps')
        elif name == 'resC':
            self.assertListEqual(_dependencies, ['resA', 'resB'], msg=f'{name}._dependencies')
            self.assertDictEqual(_property_deps, {'propA': ['resA'], 'propB': ['resB'], 'propC': []}, msg=f'{name}._property_deps')
        elif name == 'resD':
            self.assertListEqual(_dependencies, ['resA', 'resB', 'resC'], msg=f'{name}._dependencies')
            self.assertDictEqual(_property_deps, {'propA': ['resA', 'resB'], 'propB': ['resC'], 'propC': []}, msg=f'{name}._property_deps')
        elif name == 'resE':
            self.assertListEqual(_dependencies, ['resA', 'resB', 'resC', 'resD'], msg=f'{name}._dependencies')
            self.assertDictEqual(_property_deps, {'propA': ['resC'], 'propB': ['resA', 'resB'], 'propC': []}, msg=f'{name}._property_deps')
        return {'urn': name, 'id': name, 'object': {'outprop': 'qux'}}