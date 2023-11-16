from os import path
from ..util import LanghostTest

class TestFirstClassProviderInvoke(LanghostTest):
    """
    Tests that Invoke passes provider references to the engine, both with and without the presence of a "parent" from
    which to derive a provider.
    """

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.prov_id = None
        self.prov_urn = None

    def test_first_class_provider_invoke(self):
        if False:
            i = 10
            return i + 15
        self.run_test(program=path.join(self.base_path(), 'first_class_provider_invoke'), expected_resource_count=4)

    def invoke(self, _ctx, token, args, provider, _version):
        if False:
            for i in range(10):
                print('nop')
        if token == 'test:index:MyFunction':
            self.assertDictEqual({'value': 9000}, args)
            self.assertEqual(f'{self.prov_urn}::{self.prov_id}', provider)
        elif token == 'test:index:MyFunctionWithParent':
            self.assertDictEqual({'value': 41}, args)
            self.assertEqual(f'{self.prov_urn}::{self.prov_id}', provider)
        else:
            self.fail(f'unexpected token: {token}')
        return ([], {'value': args['value'] + 1})

    def register_resource(self, _ctx, _dry_run, ty, name, _resource, _dependencies, _parent, _custom, protect, _provider, _property_deps, _delete_before_replace, _ignore_changes, _version, _import, _replace_on_changes, _providers, source_position):
        if False:
            i = 10
            return i + 15
        if name == 'testprov':
            self.assertEqual('pulumi:providers:test', ty)
            self.prov_urn = self.make_urn(ty, name)
            self.prov_id = name
            return {'urn': self.prov_urn, 'id': self.prov_id}
        if name == 'resourceA':
            self.assertEqual('test:index:MyResource', ty)
            self.assertEqual(_resource['value'], 9001)
            return {'urn': self.make_urn(ty, name), 'id': name, 'object': _resource}
        if name == 'resourceB':
            self.assertEqual('test:index:MyComponent', ty)
            return {'urn': self.make_urn(ty, name)}
        if name == 'resourceC':
            self.assertEqual('test:index:MyResource', ty)
            self.assertEqual(_resource['value'], 42)
            return {'urn': self.make_urn(ty, name), 'id': name, 'object': _resource}
        self.fail(f'unexpected resource: {name}')