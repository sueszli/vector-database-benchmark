import os
import random
import select
import unittest
from test.support import cpython_only
if not hasattr(select, 'devpoll'):
    raise unittest.SkipTest('test works only on Solaris OS family')

def find_ready_matching(ready, flag):
    if False:
        print('Hello World!')
    match = []
    for (fd, mode) in ready:
        if mode & flag:
            match.append(fd)
    return match

class DevPollTests(unittest.TestCase):

    def test_devpoll1(self):
        if False:
            i = 10
            return i + 15
        p = select.devpoll()
        NUM_PIPES = 12
        MSG = b' This is a test.'
        MSG_LEN = len(MSG)
        readers = []
        writers = []
        r2w = {}
        w2r = {}
        for i in range(NUM_PIPES):
            (rd, wr) = os.pipe()
            p.register(rd)
            p.modify(rd, select.POLLIN)
            p.register(wr, select.POLLOUT)
            readers.append(rd)
            writers.append(wr)
            r2w[rd] = wr
            w2r[wr] = rd
        bufs = []
        while writers:
            ready = p.poll()
            ready_writers = find_ready_matching(ready, select.POLLOUT)
            if not ready_writers:
                self.fail('no pipes ready for writing')
            wr = random.choice(ready_writers)
            os.write(wr, MSG)
            ready = p.poll()
            ready_readers = find_ready_matching(ready, select.POLLIN)
            if not ready_readers:
                self.fail('no pipes ready for reading')
            self.assertEqual([w2r[wr]], ready_readers)
            rd = ready_readers[0]
            buf = os.read(rd, MSG_LEN)
            self.assertEqual(len(buf), MSG_LEN)
            bufs.append(buf)
            os.close(r2w[rd])
            os.close(rd)
            p.unregister(r2w[rd])
            p.unregister(rd)
            writers.remove(r2w[rd])
        self.assertEqual(bufs, [MSG] * NUM_PIPES)

    def test_timeout_overflow(self):
        if False:
            print('Hello World!')
        pollster = select.devpoll()
        (w, r) = os.pipe()
        pollster.register(w)
        pollster.poll(-1)
        self.assertRaises(OverflowError, pollster.poll, -2)
        self.assertRaises(OverflowError, pollster.poll, -1 << 31)
        self.assertRaises(OverflowError, pollster.poll, -1 << 64)
        pollster.poll(0)
        pollster.poll(1)
        pollster.poll(1 << 30)
        self.assertRaises(OverflowError, pollster.poll, 1 << 31)
        self.assertRaises(OverflowError, pollster.poll, 1 << 63)
        self.assertRaises(OverflowError, pollster.poll, 1 << 64)

    def test_close(self):
        if False:
            while True:
                i = 10
        open_file = open(__file__, 'rb')
        self.addCleanup(open_file.close)
        fd = open_file.fileno()
        devpoll = select.devpoll()
        self.assertIsInstance(devpoll.fileno(), int)
        self.assertFalse(devpoll.closed)
        devpoll.close()
        self.assertTrue(devpoll.closed)
        self.assertRaises(ValueError, devpoll.fileno)
        devpoll.close()
        self.assertRaises(ValueError, devpoll.modify, fd, select.POLLIN)
        self.assertRaises(ValueError, devpoll.poll)
        self.assertRaises(ValueError, devpoll.register, fd, select.POLLIN)
        self.assertRaises(ValueError, devpoll.unregister, fd)

    def test_fd_non_inheritable(self):
        if False:
            return 10
        devpoll = select.devpoll()
        self.addCleanup(devpoll.close)
        self.assertEqual(os.get_inheritable(devpoll.fileno()), False)

    def test_events_mask_overflow(self):
        if False:
            print('Hello World!')
        pollster = select.devpoll()
        (w, r) = os.pipe()
        pollster.register(w)
        self.assertRaises(ValueError, pollster.register, 0, -1)
        self.assertRaises(OverflowError, pollster.register, 0, 1 << 64)
        self.assertRaises(ValueError, pollster.modify, 1, -1)
        self.assertRaises(OverflowError, pollster.modify, 1, 1 << 64)

    @cpython_only
    def test_events_mask_overflow_c_limits(self):
        if False:
            i = 10
            return i + 15
        from _testcapi import USHRT_MAX
        pollster = select.devpoll()
        (w, r) = os.pipe()
        pollster.register(w)
        self.assertRaises(OverflowError, pollster.register, 0, USHRT_MAX + 1)
        self.assertRaises(OverflowError, pollster.modify, 1, USHRT_MAX + 1)
if __name__ == '__main__':
    unittest.main()