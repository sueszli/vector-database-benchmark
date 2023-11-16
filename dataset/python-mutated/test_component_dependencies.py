from os import path
import unittest
from ..util import LanghostTest

class ComponentDependenciesTest(LanghostTest):

    def test_component_dependencies(self):
        if False:
            i = 10
            return i + 15
        self.run_test(program=path.join(self.base_path(), 'component_dependencies'), expected_resource_count=16)

    def register_resource(self, _ctx, _dry_run, ty, name, _resource, _dependencies, _parent, _custom, protect, _provider, _property_deps, _delete_before_replace, _ignore_changes, _version, _import, _replace_on_changes, _providers, source_position):
        if False:
            while True:
                i = 10
        if name == 'resD':
            self.assertListEqual(_dependencies, ['resA'], msg=f'{name}._dependencies')
            self.assertDictEqual(_property_deps, {'propA': ['resA']}, msg=f'{name}._property_deps')
        elif name == 'resE':
            self.assertListEqual(_dependencies, ['resD'], msg=f'{name}._dependencies')
            self.assertDictEqual(_property_deps, {'propA': ['resD']}, msg=f'{name}._property_deps')
        elif name == 'resF':
            self.assertListEqual(_dependencies, ['resA'], msg=f'{name}._dependencies')
            self.assertDictEqual(_property_deps, {'propA': ['resA']}, msg=f'{name}._property_deps')
        elif name == 'resG':
            self.assertListEqual(_dependencies, ['resB', 'resD', 'resE'], msg=f'{name}._dependencies')
            self.assertDictEqual(_property_deps, {'propA': ['resB', 'resD', 'resE']}, msg=f'{name}._property_deps')
        elif name == 'resH':
            self.assertListEqual(_dependencies, ['resD', 'resE'], msg=f'{name}._dependencies')
            self.assertDictEqual(_property_deps, {'propA': ['resD', 'resE']}, msg=f'{name}._property_deps')
        elif name == 'resI':
            self.assertListEqual(_dependencies, ['resG'], msg=f'{name}._dependencies')
            self.assertDictEqual(_property_deps, {'propA': ['resG']}, msg=f'{name}._property_deps')
        elif name == 'resJ':
            self.assertListEqual(_dependencies, ['resD', 'resE'], msg=f'{name}._dependencies')
            self.assertDictEqual(_property_deps, {}, msg=f'{name}._property_deps')
        elif name == 'second':
            self.assertListEqual(_dependencies, ['firstChild'], msg=f'{name}._dependencies')
        return {'urn': name, 'id': name, 'object': {'outprop': 'qux'}}