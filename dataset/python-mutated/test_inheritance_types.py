from os import path
from ..util import LanghostTest

class InheritanceTypesTest(LanghostTest):

    def test_inheritance_types(self):
        if False:
            return 10
        self.run_test(program=path.join(self.base_path(), 'inheritance_types'), expected_resource_count=1)

    def register_resource(self, _ctx, _dry_run, ty, name, _resource, _dependencies, _parent, _custom, protect, _provider, _property_deps, _delete_before_replace, _ignore_changes, _version, _import, _replace_on_changes, _providers, source_position):
        if False:
            return 10
        self.assertEqual('test:index:MyResource', ty)
        return {'urn': self.make_urn(ty, name), 'id': name, 'object': {'foo': {'bar': 'hello', 'baz': 'world'}}}

    def register_resource_outputs(self, _ctx, _dry_run, _urn, ty, _name, _resource, outputs):
        if False:
            while True:
                i = 10
        self.assertEqual('pulumi:pulumi:Stack', ty)
        self.assertEqual({'combined_values': 'hello world'}, outputs)