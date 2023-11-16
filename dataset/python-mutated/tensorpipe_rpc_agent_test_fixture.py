import torch.distributed.rpc as rpc
from torch.testing._internal.distributed.rpc.rpc_agent_test_fixture import RpcAgentTestFixture
from torch.testing._internal.common_distributed import tp_transports

class TensorPipeRpcAgentTestFixture(RpcAgentTestFixture):

    @property
    def rpc_backend(self):
        if False:
            while True:
                i = 10
        return rpc.backend_registry.BackendType['TENSORPIPE']

    @property
    def rpc_backend_options(self):
        if False:
            for i in range(10):
                print('nop')
        return rpc.backend_registry.construct_rpc_backend_options(self.rpc_backend, init_method=self.init_method, _transports=tp_transports())

    def get_shutdown_error_regex(self):
        if False:
            i = 10
            return i + 15
        error_regexes = ['.*']
        return '|'.join([f'({error_str})' for error_str in error_regexes])

    def get_timeout_error_regex(self):
        if False:
            print('Hello World!')
        return 'RPC ran for more than'