from os import path
from ..util import LanghostTest

class TestInvokeFuture(LanghostTest):

    def test_invoke_future(self):
        if False:
            i = 10
            return i + 15
        self.run_test(program=path.join(self.base_path(), 'invoke_future'), expected_resource_count=0)

    def invoke(self, _ctx, token, args, provider, _version):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual('test:index:MyFunction', token)
        self.assertEqual('', provider)
        return ([], {'value': 42})