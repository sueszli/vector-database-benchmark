from os import path
from pulumi.runtime import rpc
from ..util import LanghostTest

class FirstClassProviderUnknown(LanghostTest):
    """
    Tests that when a first class provider's ID isn't known in a preview, the language host passes a provider reference
    to the engine using the rpc UNKNOWN sentinel in place of the ID.
    """

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.prov_id = None
        self.prov_urn = None

    def test_first_class_provider_unknown(self):
        if False:
            for i in range(10):
                print('nop')
        self.run_test(program=path.join(self.base_path(), 'first_class_provider_unknown'), expected_resource_count=2)

    def register_resource(self, _ctx, _dry_run, ty, name, _resource, _dependencies, _parent, _custom, protect, _provider, _property_deps, _delete_before_replace, _ignore_changes, _version, _import, _replace_on_changes, _providers, source_position):
        if False:
            i = 10
            return i + 15
        if name == 'testprov':
            self.assertEqual('pulumi:providers:test', ty)
            self.prov_urn = self.make_urn(ty, name)
            if _dry_run:
                return {'urn': self.prov_urn}
            self.prov_id = name
            return {'urn': self.prov_urn, 'id': self.prov_id}
        if name == 'res':
            self.assertEqual('test:index:MyResource', ty)
            if _dry_run:
                self.assertEqual(f'{self.prov_urn}::{rpc.UNKNOWN}', _provider)
            else:
                self.assertEqual(f'{self.prov_urn}::{self.prov_id}', _provider)
            return {'urn': self.make_urn(ty, name), 'id': name, 'object': _resource}
        self.fail(f'unknown resource: {name} ({ty})')