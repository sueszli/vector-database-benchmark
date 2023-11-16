import time
import zmq
import numpy

class probe_manager(object):

    def __init__(self):
        if False:
            return 10
        self.zmq_context = zmq.Context()
        self.poller = zmq.Poller()
        self.interfaces = []

    def add_socket(self, address, data_type, callback_func):
        if False:
            return 10
        socket = self.zmq_context.socket(zmq.SUB)
        socket.setsockopt(zmq.SUBSCRIBE, b'')
        socket.setsockopt(zmq.RCVTIMEO, 100)
        socket.connect(address)
        self.interfaces.append((socket, data_type, callback_func))
        self.poller.register(socket, zmq.POLLIN)
        time.sleep(0.5)

    def watcher(self):
        if False:
            return 10
        poll = dict(self.poller.poll(1000))
        for i in self.interfaces:
            if poll.get(i[0]) == zmq.POLLIN:
                msg_packed = i[0].recv()
                msg_unpacked = numpy.frombuffer(msg_packed, numpy.dtype(i[1]))
                i[2](msg_unpacked)