import time
import unittest
import os
import tempfile
import numpy as np
from urh.util import util
util.set_shared_library_path()
from urh.dev.native.lib import hackrf
from urh.dev.native.HackRF import HackRF

class TestHackRF(unittest.TestCase):

    def callback_fun(self, buffer):
        if False:
            print('Hello World!')
        print(buffer)
        for i in range(0, len(buffer), 4):
            try:
                r = np.fromstring(buffer[i:i + 2], dtype=np.float16) / 32767.5
                i = np.fromstring(buffer[i + 2:i + 4], dtype=np.float16) / 32767.5
            except ValueError:
                continue
            if r and i:
                print(r, i)
        return 0

    def test_fromstring(self):
        if False:
            while True:
                i = 10
        buffer = b'\xfe\xfe\xfe\xfe\xfe\xfe\xfe\xfd\xff\xfd\xfe\xfe\xfe\xfe\xfe\xfe\xfe\xfe\xfd\xfe\xfd\xfe\xff\xfe\xfe\xfe\xfe\xfe\xfe\xfe\xfd\xfe'
        r = np.empty(len(buffer) // 2, dtype=np.float32)
        i = np.empty(len(buffer) // 2, dtype=np.float32)
        c = np.empty(len(buffer) // 2, dtype=np.complex64)
        unpacked = np.frombuffer(buffer, dtype=[('r', np.uint8), ('i', np.uint8)])
        ru = unpacked['r'] / 128.0
        iu = unpacked['i'] / 128.0
        c.real = ru
        c.imag = iu
        print(c)

    def test_fromstring2(self):
        if False:
            print('Hello World!')
        buffer = b'\xfe\xfe\xfe\xfe\xfe\xfe\xfe\xfd\xff\xfd\xfe\xfe\xfe\xfe\xfe\xfe\xfe\xfe\xfd\xfe\xfd\xfe\xff\xfe\xfe\xfe\xfe\xfe\xfe\xfe\xfd\xfe'
        c = np.empty(len(buffer) // 2, dtype=np.complex64)
        unpacked = np.frombuffer(buffer, dtype='<h')
        print(unpacked)
        f = 1.0 / 32767.5
        for i in range(0, len(unpacked) - 1, 2):
            c[i] = complex(float(unpacked[i] * f), float(unpacked[i + 1] * f))
        print(c)

    def test_hackrf_class_recv(self):
        if False:
            return 10
        hfc = HackRF(433920000.0, 1000000.0, 1000000.0, 20)
        hfc.start_rx_mode()
        i = 0
        TIME_TOTAL = 5
        while i < TIME_TOTAL:
            print('{0}/{1}'.format(i + 1, TIME_TOTAL))
            time.sleep(1)
            i += 1
        print('{0:,}'.format(hfc.current_recv_index))
        hfc.received_data.tofile(os.path.join(tempfile.gettempdir(), 'hackrf.complex'))
        print('Wrote Data')
        hfc.stop_rx_mode('Finished test')

    def test_hackrf_class_send(self):
        if False:
            for i in range(10):
                print('nop')
        hfc = HackRF(433920000.0, 1000000.0, 1000000.0, 20)
        hfc.start_tx_mode(np.fromfile(os.path.join(tempfile.gettempdir(), 'hackrf.complex'), dtype=np.complex64), repeats=1)
        while not hfc.sending_finished:
            print('Repeat: {0} Current Sample: {1}/{2}'.format(hfc.current_sending_repeat + 1, hfc.current_sent_sample, len(hfc.samples_to_send)))
            time.sleep(1)
        hfc.stop_tx_mode('Test finished')

    def test_hackrf_pack_unpack(self):
        if False:
            print('Hello World!')
        arr = np.array([-128, -128, -0.5, -0.5, -3, -3, 127, 127], dtype=np.int8)
        self.assertEqual(arr[0], -128)
        self.assertEqual(arr[1], -128)
        self.assertEqual(arr[-1], 127)
        received = arr.tostring()
        self.assertEqual(len(received), len(arr))
        self.assertEqual(np.int8(received[0]), -128)
        self.assertEqual(np.int8(received[1]), -128)
        unpacked = HackRF.bytes_to_iq(received, len(received) // 2)
        self.assertEqual(unpacked[0], complex(-1, -1))
        self.assertAlmostEqual(unpacked[1], complex(0, 0), places=1)
        self.assertAlmostEqual(unpacked[2], complex(0, 0), places=1)
        self.assertEqual(unpacked[3], complex(1, 1))
        packed = HackRF.iq_to_bytes(unpacked)
        self.assertEqual(received, packed)

    def test_c_api(self):
        if False:
            print('Hello World!')

        def callback(n):
            if False:
                return 10
            print('called')
            return np.array([1], dtype=np.complex64)
        print('init', hackrf.init())
        print('open', hackrf.open())
        print('start_tx', hackrf.start_tx_mode(callback))
        time.sleep(1)
        print('stop_tx', hackrf.stop_tx_mode())
        print('close', hackrf.close())
        print('exit', hackrf.exit())

    def test_device_list(self):
        if False:
            print('Hello World!')
        print(hackrf.get_device_list())
if __name__ == '__main__':
    unittest.main()