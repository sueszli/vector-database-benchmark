from os import path
from ..util import LanghostTest

class PropertyRenamingTest(LanghostTest):
    """
    Tests that Pulumi resources can override translate_input_property and translate_output_property
    in order to control the naming of their own properties.
    """

    def test_property_renaming(self):
        if False:
            i = 10
            return i + 15
        self.run_test(program=path.join(self.base_path(), 'property_renaming'), expected_resource_count=1)

    def register_resource(self, _ctx, _dry_run, ty, name, _resource, _dependencies, _parent, _custom, protect, _provider, _property_deps, _delete_before_replace, _ignore_changes, _version, _import, _replace_on_changes, _providers, source_position):
        if False:
            while True:
                i = 10
        self.assertEqual('test:index:TranslatedResource', ty)
        self.assertEqual('res', name)
        self.assertIn('engineProp', _resource)
        self.assertEqual('some string', _resource['engineProp'])
        self.assertIn('recursiveProp', _resource)
        self.assertDictEqual({'recursiveKey': 'value'}, _resource['recursiveProp'])
        return {'urn': self.make_urn(ty, name), 'id': name, 'object': {'engineProp': 'some string', 'engineOutputProp': 'some output string', 'recursiveProp': {'recursiveKey': 'value', 'recursiveOutput': 'some other output'}}}

    def register_resource_outputs(self, _ctx, _dry_run, _urn, ty, _name, _resource, outputs):
        if False:
            print('Hello World!')
        self.assertEqual(ty, 'pulumi:pulumi:Stack')
        self.assertDictEqual({'transformed_prop': 'some string', 'engine_output_prop': 'some output string', 'recursive_prop': {'recursive_key': 'value', 'recursive_output': 'some other output'}}, outputs)