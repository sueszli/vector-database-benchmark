import unittest
import time
from multiprocessing.connection import Connection, Pipe
import numpy as np
from multiprocessing import Process
from urh.dev.native.SDRPlay import SDRPlay
from urh.util import util
import ctypes
util.set_shared_library_path()
from urh.dev.native.lib import sdrplay

def recv(conn: Connection):
    if False:
        i = 10
        return i + 15
    while True:
        t = time.time()
        result = SDRPlay.bytes_to_iq(conn.recv_bytes())
        print('UNPACK', time.time() - t)

class TestSDRPlay(unittest.TestCase):

    def test_c_wrapper(self):
        if False:
            return 10

        def pycallback(data):
            if False:
                return 10
            arr = np.asarray(data)
        print(sdrplay.get_api_version())
        print(sdrplay.get_devices())
        print(sdrplay.set_device_index(0))
        (parent_conn, child_conn) = Pipe()
        p = Process(target=recv, args=(parent_conn,))
        p.daemon = True
        p.start()
        null_ptr = ctypes.POINTER(ctypes.c_voidp)()
        print('Init stream', sdrplay.init_stream(50, 2000000.0, 433920000.0, 2000000.0, 500, child_conn))
        time.sleep(2)
        print('settings sample rate')
        print('Set sample rate', sdrplay.set_sample_rate(2000000.0))
        time.sleep(1)
        p.terminate()
        p.join()