import unittest
from app import services

class TestWebInfoHunter(unittest.TestCase):

    def test_run_wih(self):
        if False:
            for i in range(10):
                print('nop')
        sites = ['https://www.freebuf.com', 'https://www.qq.com/']
        results = services.run_wih(sites)
        for result in results:
            print(result)
        self.assertTrue(len(results) > 2)