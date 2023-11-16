from os import path
import unittest
from ..util import LanghostTest

class RemoteComponentDependenciesTest(LanghostTest):

    def test_remote_component_dependencies(self):
        if False:
            print('Hello World!')
        self.run_test(program=path.join(self.base_path(), 'remote_component_dependencies'), expected_resource_count=3)

    def register_resource(self, _ctx, _dry_run, ty, name, _resource, _dependencies, _parent, _custom, protect, _provider, _property_deps, _delete_before_replace, _ignore_changes, _version, _import, _replace_on_changes, _providers, source_position):
        if False:
            while True:
                i = 10
        if name == 'resA':
            self.assertEqual(len(_dependencies), 0, 'resA dependencies')
        elif name == 'resB':
            self.assertEqual(len(_dependencies), 1, 'resB dependencies')
        elif name == 'resC':
            self.assertEqual(len(_dependencies), 1, 'resC dependencies')
            self.assertEqual(_parent, 'resA', 'resC parent')
        else:
            assert False
        return {'urn': name, 'id': name, 'object': {'outprop': 'qux'}}