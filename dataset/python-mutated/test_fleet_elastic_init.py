import unittest
from paddle.distributed.fleet.elastic import enable_elastic, launch_elastic
from paddle.distributed.fleet.launch_utils import DistributeMode

class TestElasticInit(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')

        class Argument:
            elastic_server = '127.0.0.1:2379'
            job_id = 'test_job_id_123'
            np = '2:4'
        self.args = Argument()

    def test_enable_elastic(self):
        if False:
            while True:
                i = 10
        result = enable_elastic(self.args, DistributeMode.COLLECTIVE)
        self.assertEqual(result, True)

    def test_launch_elastic(self):
        if False:
            for i in range(10):
                print('nop')
        try:
            launch_elastic(self.args, DistributeMode.COLLECTIVE)
        except Exception as e:
            pass
if __name__ == '__main__':
    unittest.main()