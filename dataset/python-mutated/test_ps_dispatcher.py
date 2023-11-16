import unittest
from paddle.incubate.distributed.fleet.parameter_server.ir.ps_dispatcher import HashName, PSDispatcher, RoundRobin

class TestPsDispatcher(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.points = ['127.0.0.1:1001', '127.0.0.1:1002', '127.0.0.1:1003', '127.0.0.1:1004']

    def test_base(self):
        if False:
            print('Hello World!')
        base = PSDispatcher(self.points)
        self.assertEqual(len(base.eps), 4)
        base.reset()
        with self.assertRaises(NotImplementedError):
            base.dispatch([])

    def test_hash(self):
        if False:
            for i in range(10):
                print('nop')

        class Var:

            def __init__(self, index):
                if False:
                    print('Hello World!')
                self._name = f'var_{index}'

            def name(self):
                if False:
                    for i in range(10):
                        print('nop')
                return self._name
        xx = HashName(self.points)
        self.assertEqual(len(xx.eps), 4)
        xx.reset()
        vars = []
        for i in range(4):
            vars.append(Var(i))
        eplist = xx.dispatch(vars)
        self.assertEqual(len(eplist), 4)

    def test_round_rodin(self):
        if False:
            while True:
                i = 10

        class Var:

            def __init__(self, index):
                if False:
                    while True:
                        i = 10
                self._name = f'var_{index}'

            def name(self):
                if False:
                    return 10
                return self._name
        xx = RoundRobin(self.points)
        self.assertEqual(len(xx.eps), 4)
        xx.reset()
        vars = []
        for i in range(4):
            vars.append(Var(i))
        eplist = xx.dispatch(vars)
        self.assertEqual(len(eplist), 4)
if __name__ == '__main__':
    unittest.main()