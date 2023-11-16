from os import path
from ..util import LanghostTest

class OutputNestedTest(LanghostTest):

    def test_output_nested(self):
        if False:
            for i in range(10):
                print('nop')
        self.run_test(program=path.join(self.base_path(), 'output_nested'), expected_resource_count=3)

    def register_resource(self, _ctx, _dry_run, ty, name, _resource, _dependencies, _parent, _custom, protect, _provider, _property_deps, _delete_before_replace, _ignore_changes, _version, _import, _replace_on_changes, _providers, source_position):
        if False:
            while True:
                i = 10
        nested_numbers = None
        if name == 'testResource1':
            self.assertEqual(ty, 'test:index:MyResource')
            nested_numbers = {'foo': {'bar': 9}, 'baz': 1}
        elif name == 'testResource2':
            self.assertEqual(ty, 'test:index:MyResource')
            nested_numbers = {'foo': {'bar': 99}, 'baz': 1}
        elif name == 'sumResource':
            self.assertEqual(ty, 'test:index:SumResource')
            self.assertEqual(_resource['sum'], 10)
            nested_numbers = _resource['sum']
        return {'urn': self.make_urn(ty, name), 'id': name, 'object': {'nested_numbers': nested_numbers}}