from os import path
from ..util import LanghostTest

class StackOutputTest(LanghostTest):
    """
    Test that tests Pulumi's ability to register resource outputs.
    """

    def test_stack_outputs(self):
        if False:
            i = 10
            return i + 15
        self.run_test(program=path.join(self.base_path(), 'stack_output'), expected_resource_count=0)

    def register_resource_outputs(self, _ctx, _dry_run, _urn, ty, _name, _resource, outputs):
        if False:
            print('Hello World!')
        self.assertEqual(ty, 'pulumi:pulumi:Stack')
        self.assertDictEqual({'string': 'pulumi', 'number': 1.0, 'boolean': True, 'list': [], 'list_with_none': [None], 'list_of_lists': [[], []], 'list_of_outputs': [[1], [2]], 'set': ['val'], 'dict': {'a': 1.0}, 'output': 1.0, 'class': {'num': 1.0}, 'recursive': {'a': 1.0, 'b': 2.0}, 'duplicate_output_0': {'num': 1.0}, 'duplicate_output_1': {'num': 1.0}}, outputs)