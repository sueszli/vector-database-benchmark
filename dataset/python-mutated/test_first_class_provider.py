from os import path
from ..util import LanghostTest

class FirstClassProviderTest(LanghostTest):
    """
    Tests that resources created with their 'provider' ResourceOption set pass a provider reference
    to the Pulumi engine.
    """
    prov_urn = None
    prov_id = None

    def test_first_class_provider(self):
        if False:
            print('Hello World!')
        self.run_test(program=path.join(self.base_path(), 'first_class_provider'), expected_resource_count=2)

    def register_resource(self, _ctx, _dry_run, ty, name, _resource, _dependencies, _parent, _custom, protect, _provider, _property_deps, _delete_before_replace, _ignore_changes, _version, _import, _replace_on_changes, _providers, source_position):
        if False:
            return 10
        if name == 'testprov':
            self.assertEqual('pulumi:providers:test', ty)
            self.assertEqual('', _provider)
            self.prov_urn = self.make_urn(ty, name)
            self.prov_id = 'testid'
            return {'urn': self.prov_urn, 'id': self.prov_id}
        if name == 'testres':
            self.assertEqual('test:index:Resource', ty)
            self.assertIsNotNone(self.prov_urn)
            self.assertIsNotNone(self.prov_id)
            self.assertEqual(f'{self.prov_urn}::{self.prov_id}', _provider)
            return {'urn': self.make_urn(ty, name)}
        self.fail(f'unexpected resource: {name} ({ty})')