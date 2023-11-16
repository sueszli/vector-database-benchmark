from os import path
from ..util import LanghostTest

class TestReplaceOnChanges(LanghostTest):
    """
    Tests that Pulumi resources can accept replace_on_changes resource options.
    """

    def test_replace_on_changes(self):
        if False:
            while True:
                i = 10
        self.run_test(program=path.join(self.base_path(), 'replace_on_changes'), expected_resource_count=1)

    def register_resource(self, _ctx, _dry_run, ty, name, _resource, _dependencies, _parent, _custom, protect, _provider, _property_deps, _delete_before_replace, _ignore_changes, _version, _import, _replace_on_changes, _providers, source_position):
        if False:
            return 10
        print(f'register_resource args: {locals()}')
        self.assertEqual('testResource', name)
        self.assertListEqual(_replace_on_changes, ['a', 'b'])
        return {'urn': self.make_urn(ty, name)}