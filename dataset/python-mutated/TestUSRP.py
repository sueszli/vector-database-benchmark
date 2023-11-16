from multiprocessing.connection import Pipe
import sys
from urh.util import util
util.set_shared_library_path()
from urh.dev.native.USRP import USRP
from urh.dev.native.lib import usrp
import unittest

class TestUSRP(unittest.TestCase):

    def test_cython_wrapper(self):
        if False:
            for i in range(10):
                print('nop')
        print(usrp.find_devices(''))
        usrp.set_tx(False)
        return_code = usrp.open('addr=192.168.10.2')
        print('open', return_code)
        usrp.setup_stream()
        print('Made rx_streame handler')
        print(usrp.get_device_representation())
        print('Set sample rate', usrp.set_sample_rate(2000000.0))
        print('Set freq', usrp.set_center_freq(433920000.0))
        print('Set bandwidth', usrp.set_bandwidth(1000000.0))
        print('Set gain', usrp.set_rf_gain(0.5))
        buffer = bytearray()
        num_samples = 32768 // 2
        usrp.start_stream(num_samples)
        (parent_conn, child_conn) = Pipe()
        for i in range(500):
            usrp.recv_stream(child_conn, num_samples)
            received_bytes = parent_conn.recv_bytes()
            print(i)
            buffer.extend(received_bytes)
        f = open('/tmp/test.complex', 'wb')
        f.write(buffer)
        f.close()
        usrp.destroy_stream()
        print('Freed rx streamer handler')
        return_code = usrp.close()
        print('close', return_code)
if __name__ == '__main__':
    unittest.main()