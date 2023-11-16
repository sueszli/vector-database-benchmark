import os
from abc import ABC, abstractmethod
import torch.testing._internal.dist_utils

class RpcAgentTestFixture(ABC):

    @property
    def world_size(self) -> int:
        if False:
            while True:
                i = 10
        return 4

    @property
    def init_method(self):
        if False:
            i = 10
            return i + 15
        use_tcp_init = os.environ.get('RPC_INIT_WITH_TCP', None)
        if use_tcp_init == '1':
            master_addr = os.environ['MASTER_ADDR']
            master_port = os.environ['MASTER_PORT']
            return f'tcp://{master_addr}:{master_port}'
        else:
            return self.file_init_method

    @property
    def file_init_method(self):
        if False:
            print('Hello World!')
        return torch.testing._internal.dist_utils.INIT_METHOD_TEMPLATE.format(file_name=self.file_name)

    @property
    @abstractmethod
    def rpc_backend(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    @property
    @abstractmethod
    def rpc_backend_options(self):
        if False:
            return 10
        pass

    def setup_fault_injection(self, faulty_messages, messages_to_delay):
        if False:
            i = 10
            return i + 15
        'Method used by dist_init to prepare the faulty agent.\n\n        Does nothing for other agents.\n        '
        pass

    @abstractmethod
    def get_shutdown_error_regex(self):
        if False:
            i = 10
            return i + 15
        '\n        Return various error message we may see from RPC agents while running\n        tests that check for failures. This function is used to match against\n        possible errors to ensure failures were raised properly.\n        '
        pass

    @abstractmethod
    def get_timeout_error_regex(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns a partial string indicating the error we should receive when an\n        RPC has timed out. Useful for use with assertRaisesRegex() to ensure we\n        have the right errors during timeout.\n        '
        pass