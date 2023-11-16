from os import path
from ..util import LanghostTest

class DeleteBeforeReplaceTest(LanghostTest):
    """
    Tests that DBRed resources correctly pass the "DBR" boolean to the engine.
    """

    def test_delete_before_replace(self):
        if False:
            return 10
        self.run_test(program=path.join(self.base_path(), 'delete_before_replace'), expected_resource_count=1)

    def register_resource(self, _ctx, _dry_run, ty, name, _resource, _dependencies, _parent, _custom, protect, _provider, _property_deps, _delete_before_replace, _ignore_changes, _version, _import, _replace_on_changes, _providers, source_position):
        if False:
            i = 10
            return i + 15
        self.assertEqual('foo', name)
        self.assertTrue(_delete_before_replace)
        return {'urn': self.make_urn(ty, name)}