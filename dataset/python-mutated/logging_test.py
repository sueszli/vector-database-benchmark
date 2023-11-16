import torch.distributed.elastic.utils.logging as logging
from torch.testing._internal.common_utils import run_tests, TestCase
log = logging.get_logger()

class LoggingTest(TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        super().setUp()
        self.clazz_log = logging.get_logger()

    def test_logger_name(self):
        if False:
            for i in range(10):
                print('nop')
        local_log = logging.get_logger()
        name_override_log = logging.get_logger('foobar')
        self.assertEqual(__name__, log.name)
        self.assertEqual(__name__, self.clazz_log.name)
        self.assertEqual(__name__, local_log.name)
        self.assertEqual('foobar', name_override_log.name)

    def test_derive_module_name(self):
        if False:
            print('Hello World!')
        module_name = logging._derive_module_name(depth=1)
        self.assertEqual(__name__, module_name)
if __name__ == '__main__':
    run_tests()