import unittest
from threading import Thread
from impacket.ImpactPacket import TCP, ImpactPacketException

class TestTCP(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.frame = '\x124\x00P\x00\x00\x00\x01\x00\x00\x00\x00`\x00\x00\x00\x8d\\\x00\x00\x02\x00\x00\x00'

    def test_01(self):
        if False:
            while True:
                i = 10
        'Test TCP options parsing hangs'

        class it_hangs(Thread):

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                Thread.__init__(self)

            def run(self):
                if False:
                    print('Hello World!')
                try:
                    frame = '\x124\x00P\x00\x00\x00\x01\x00\x00\x00\x00`\x00\x00\x00\x8d\\\x00\x00\x02\x00\x00\x00'
                    TCP(frame)
                except ImpactPacketException as e:
                    if str(e) != "'TCP Option length is too low'":
                        raise e
                except Exception:
                    pass
        thread_hangs = it_hangs()
        thread_hangs.daemon = True
        thread_hangs.start()
        thread_hangs.join(1.0)
        self.assertEqual(thread_hangs.is_alive(), False)
if __name__ == '__main__':
    unittest.main(verbosity=1)