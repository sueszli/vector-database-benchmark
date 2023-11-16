from bzrlib import tests
from bzrlib.builtins import cmd_init_repository
from bzrlib.tests.transport_util import TestCaseWithConnectionHookedTransport

class TestInitRepository(TestCaseWithConnectionHookedTransport):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super(TestInitRepository, self).setUp()
        self.start_logging_connections()

    def test_init_repository(self):
        if False:
            return 10
        cmd = cmd_init_repository()
        cmd.outf = tests.StringIOWrapper()
        cmd.run(self.get_url())
        self.assertEqual(1, len(self.connections))