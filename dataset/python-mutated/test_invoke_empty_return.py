from os import path
from ..util import LanghostTest

class TestInvokeEmptyReturn(LanghostTest):

    def test_invoke_emptyReturn(self):
        if False:
            i = 10
            return i + 15
        self.run_test(program=path.join(self.base_path(), 'invoke_empty_return'), expected_resource_count=0)

    def invoke(self, _ctx, token, _args, provider, _version):
        if False:
            while True:
                i = 10
        self.assertEqual('test:index:MyFunction', token)
        self.assertEqual('', provider)
        return ([], {})