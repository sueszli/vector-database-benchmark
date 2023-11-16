import zmq
from zmq.tests import BaseZMQTestCase, GreenTest, have_gevent

class TestReqRep(BaseZMQTestCase):

    def test_basic(self):
        if False:
            i = 10
            return i + 15
        (s1, s2) = self.create_bound_pair(zmq.REQ, zmq.REP)
        msg1 = b'message 1'
        msg2 = self.ping_pong(s1, s2, msg1)
        assert msg1 == msg2

    def test_multiple(self):
        if False:
            i = 10
            return i + 15
        (s1, s2) = self.create_bound_pair(zmq.REQ, zmq.REP)
        for i in range(10):
            msg1 = i * b' '
            msg2 = self.ping_pong(s1, s2, msg1)
            assert msg1 == msg2

    def test_bad_send_recv(self):
        if False:
            for i in range(10):
                print('nop')
        (s1, s2) = self.create_bound_pair(zmq.REQ, zmq.REP)
        if zmq.zmq_version() != '2.1.8':
            for copy in (True, False):
                self.assertRaisesErrno(zmq.EFSM, s1.recv, copy=copy)
                self.assertRaisesErrno(zmq.EFSM, s2.send, b'asdf', copy=copy)
        msg1 = b'asdf'
        msg2 = self.ping_pong(s1, s2, msg1)
        assert msg1 == msg2

    def test_json(self):
        if False:
            print('Hello World!')
        (s1, s2) = self.create_bound_pair(zmq.REQ, zmq.REP)
        o = dict(a=10, b=list(range(10)))
        self.ping_pong_json(s1, s2, o)

    def test_pyobj(self):
        if False:
            i = 10
            return i + 15
        (s1, s2) = self.create_bound_pair(zmq.REQ, zmq.REP)
        o = dict(a=10, b=range(10))
        self.ping_pong_pyobj(s1, s2, o)

    def test_large_msg(self):
        if False:
            for i in range(10):
                print('nop')
        (s1, s2) = self.create_bound_pair(zmq.REQ, zmq.REP)
        msg1 = 10000 * b'X'
        for i in range(10):
            msg2 = self.ping_pong(s1, s2, msg1)
            assert msg1 == msg2
if have_gevent:

    class TestReqRepGreen(GreenTest, TestReqRep):
        pass