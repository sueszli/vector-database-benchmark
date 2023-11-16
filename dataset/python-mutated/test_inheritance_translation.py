from os import path
from ..util import LanghostTest

class InheritanceTranslationTest(LanghostTest):

    def test_inheritance_translation(self):
        if False:
            while True:
                i = 10
        self.run_test(program=path.join(self.base_path(), 'inheritance_translation'), expected_resource_count=4)

    def register_resource(self, _ctx, _dry_run, ty, name, _resource, _dependencies, _parent, _custom, protect, _provider, _property_deps, _delete_before_replace, _ignore_changes, _version, _import, _replace_on_changes, _providers, source_position):
        if False:
            while True:
                i = 10
        self.assertEqual('test:index:MyResource', ty)
        return {'urn': self.make_urn(ty, name), 'id': name, 'object': {'someValue': 'hello', 'anotherValue': 'world'}}

    def register_resource_outputs(self, _ctx, _dry_run, _urn, ty, _name, _resource, outputs):
        if False:
            return 10
        self.assertEqual('pulumi:pulumi:Stack', ty)
        self.assertEqual({'r1.some_value': 'hello', 'r1.another_value': 'world', 'r1.combined_values': 'hello world', 'r2.some_value': 'hello', 'r2.another_value': 'world', 'r2.combined_values': 'hello world', 'r2.new_value': 'hello world!', 'r3.some_value': 'hello', 'r3.another_value': 'world', 'r3.combined_values': 'hello world', 'r4.some_value': 'hello', 'r4.another_value': 'world', 'r4.combined_values': 'hello world', 'r4.new_value': 'hello world!'}, outputs)