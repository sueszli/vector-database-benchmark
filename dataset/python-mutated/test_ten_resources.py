from os import path
from ..util import LanghostTest

class TenResourcesTest(LanghostTest):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.seen = {}

    def test_ten_resources(self):
        if False:
            while True:
                i = 10
        self.run_test(program=path.join(self.base_path(), 'ten_resources'), expected_resource_count=10)

    def register_resource(self, _ctx, _dry_run, ty, name, _resource, _dependencies, _parent, _custom, protect, _provider, _property_deps, _delete_before_replace, _ignore_changes, _version, _import, _replace_on_changes, _providers, source_position):
        if False:
            return 10
        self.assertEqual('test:index:MyResource', ty)
        if not _dry_run:
            self.assertIsNone(self.seen.get(name), 'Got multiple resources with the same name: ' + name)
            self.seen[name] = True
        return {'urn': self.make_urn(ty, name)}