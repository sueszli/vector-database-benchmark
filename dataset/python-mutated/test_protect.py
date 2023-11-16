from os import path
from ..util import LanghostTest

class ProtectTest(LanghostTest):
    """
    Tests that protected resources correctly pass the "protect" boolean to the engine.
    """

    def test_protect(self):
        if False:
            while True:
                i = 10
        self.run_test(program=path.join(self.base_path(), 'protect'), expected_resource_count=1)

    def register_resource(self, _ctx, _dry_run, ty, name, _resource, _dependencies, _parent, _custom, protect, _provider, _property_deps, _delete_before_replace, _ignore_changes, _version, _import, _replace_on_changes, _providers, source_position):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual('foo', name)
        self.assertTrue(protect)
        return {'urn': self.make_urn(ty, name)}