import unittest
from bigdl.ppml.fl.algorithms.psi import PSI
from bigdl.ppml.fl.fl_server import FLServer
from bigdl.ppml.fl import *
from bigdl.ppml.fl.utils import FLTest

class TestPSI(FLTest):

    def setUp(self) -> None:
        if False:
            return 10
        self.fl_server = FLServer()
        self.fl_server.set_port(self.port)
        self.fl_server.build()
        self.fl_server.start()

    def tearDown(self) -> None:
        if False:
            return 10
        self.fl_server.stop()

    def test_psi_get_salt(self):
        if False:
            print('Hello World!')
        init_fl_context(1, self.target)
        psi = PSI()
        salt = psi.get_salt()
        assert isinstance(salt, str)

    def test_psi_pipeline(self):
        if False:
            i = 10
            return i + 15
        init_fl_context(1, self.target)
        psi = PSI()
        salt = psi.get_salt()
        key = ['k1', 'k2']
        psi.upload_set(key, salt)
        intersection = psi.download_intersection()
        assert isinstance(intersection, list)
        self.assertEqual(len(intersection), 2)
if __name__ == '__main__':
    unittest.main()