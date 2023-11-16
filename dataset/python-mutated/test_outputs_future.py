from os import path
from ..util import LanghostTest

class TestOutputsFuture(LanghostTest):

    def test_outputs_future(self):
        if False:
            while True:
                i = 10
        self.run_test(program=path.join(self.base_path(), 'outputs_future'), expected_resource_count=0)

    def invoke(self, _ctx, token, args, _provider, _version):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual('test:index:MyFunction', token)
        self.assertDictEqual({'value': 41}, args)
        return ([], {'value': args['value'] + 1})

    def register_resource_outputs(self, _ctx, _dry_run, _urn, ty, _name, _resource, outputs):
        if False:
            i = 10
            return i + 15
        self.assertEqual(ty, 'pulumi:pulumi:Stack')
        self.assertDictEqual({'value': 42}, outputs)