from os import path
from ..util import LanghostTest

class InputValuesForOutputsTest(LanghostTest):
    """
    """

    def test_input_values_for_outputs(self):
        if False:
            i = 10
            return i + 15
        self.run_test(program=path.join(self.base_path(), 'input_values_for_outputs'), expected_resource_count=1)

    def register_resource(self, _ctx, _dry_run, ty, name, _resource, _dependencies, _parent, _custom, protect, _provider, _property_deps, _delete_before_replace, _ignore_changes, _version, _import, _replace_on_changes, _providers, source_position):
        if False:
            print('Hello World!')
        return {'urn': self.make_urn(ty, name), 'id': name, 'object': {}}