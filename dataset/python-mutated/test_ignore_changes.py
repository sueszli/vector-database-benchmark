from os import path
from ..util import LanghostTest

class TestIgnoreChanges(LanghostTest):
    """
    Tests that Pulumi resources can accept ignore_changes resource options.
    """

    def test_ignore_changes(self):
        if False:
            i = 10
            return i + 15
        self.run_test(program=path.join(self.base_path(), 'ignore_changes'), expected_resource_count=1)

    def register_resource(self, _ctx, _dry_run, ty, name, _resource, _dependencies, _parent, _custom, protect, _provider, _property_deps, _delete_before_replace, _ignore_changes, _version, _import, _replace_on_changes, _providers, source_position):
        if False:
            return 10
        self.assertListEqual(_ignore_changes, ['ignoredProperty', 'ignored_property_other'])
        return {'urn': self.make_urn(ty, name), 'id': name, 'object': _resource}