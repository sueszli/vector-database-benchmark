import zmq
import errno
import pickle
from zmq.eventloop import ioloop, zmqstream
import zmq.utils.monitor
import functools, sys, logging
from threading import Thread, Event
from . import utils
import weakref, logging

class ZmqWrapper:
    _thread: Thread = None
    _ioloop: ioloop.IOLoop = None
    _start_event: Event = None
    _ioloop_block: Event = None

    @staticmethod
    def initialize():
        if False:
            i = 10
            return i + 15
        if ZmqWrapper._thread is None:
            ZmqWrapper._thread = Thread(target=ZmqWrapper._run_io_loop, name='ZMQIOLoop', daemon=True)
            ZmqWrapper._start_event = Event()
            ZmqWrapper._ioloop_block = Event()
            ZmqWrapper._ioloop_block.set()
            ZmqWrapper._thread.start()
            ZmqWrapper._start_event.wait()

    @staticmethod
    def close():
        if False:
            while True:
                i = 10
        if ZmqWrapper._thread is not None:
            ZmqWrapper._ioloop_block.set()
            ZmqWrapper._ioloop.add_callback(ZmqWrapper._ioloop.stop)
            ZmqWrapper._thread = None
            ZmqWrapper._ioloop = None
            print('ZMQ IOLoop is now closed')

    @staticmethod
    def get_timer(secs, callback, start=True):
        if False:
            return 10
        utils.debug_log('Adding PeriodicCallback', secs)
        pc = ioloop.PeriodicCallback(callback, secs * 1000.0)
        if start:
            pc.start()
        return pc

    @staticmethod
    def _run_io_loop():
        if False:
            for i in range(10):
                print('nop')
        if 'asyncio' in sys.modules:
            import asyncio
            asyncio.set_event_loop(asyncio.new_event_loop())
        ZmqWrapper._ioloop = ioloop.IOLoop()
        ZmqWrapper._ioloop.make_current()
        while ZmqWrapper._thread is not None:
            try:
                ZmqWrapper._start_event.set()
                utils.debug_log('starting ioloop...')
                ZmqWrapper._ioloop.start()
            except zmq.ZMQError as ex:
                if ex.errno == errno.EINTR:
                    logging.exception('Cannot start IOLoop - ZMQError')
                    continue
                else:
                    raise

    @staticmethod
    def _io_loop_call(has_result, f, *kargs, **kwargs):
        if False:
            for i in range(10):
                print('nop')

        class Result:

            def __init__(self, val=None):
                if False:
                    while True:
                        i = 10
                self.val = val

        def wrapper(f, r, *kargs, **kwargs):
            if False:
                while True:
                    i = 10
            try:
                r.val = f(*kargs, **kwargs)
                ZmqWrapper._ioloop_block.set()
            except:
                logging.exception('Error in call scheduled in ioloop')
        if has_result:
            if not ZmqWrapper._ioloop_block.is_set():
                print('Previous blocking call on IOLoop is not yet complete!')
            ZmqWrapper._ioloop_block.clear()
            r = Result()
            f_wrapped = functools.partial(wrapper, f, r, *kargs, **kwargs)
            ZmqWrapper._ioloop.add_callback(f_wrapped)
            utils.debug_log('Waiting for call on ioloop', f, verbosity=5)
            ZmqWrapper._ioloop_block.wait()
            utils.debug_log('Call on ioloop done', f, verbosity=5)
            return r.val
        else:
            f_wrapped = functools.partial(f, *kargs, **kwargs)
            ZmqWrapper._ioloop.add_callback(f_wrapped)

    class Publication:

        def __init__(self, port, host='*', block_until_connected=True):
            if False:
                i = 10
                return i + 15
            self._socket = None
            self._mon_socket = None
            self._mon_stream = None
            ZmqWrapper.initialize()
            utils.debug_log('Creating Publication', port, verbosity=1)
            ZmqWrapper._io_loop_call(block_until_connected, self._start_srv, port, host)

        def _start_srv(self, port, host):
            if False:
                i = 10
                return i + 15
            context = zmq.Context()
            self._socket = context.socket(zmq.PUB)
            utils.debug_log('Binding socket', (host, port), verbosity=5)
            self._socket.bind('tcp://%s:%d' % (host, port))
            utils.debug_log('Bound socket', (host, port), verbosity=5)
            self._mon_socket = self._socket.get_monitor_socket(zmq.EVENT_CONNECTED | zmq.EVENT_DISCONNECTED)
            self._mon_stream = zmqstream.ZMQStream(self._mon_socket)
            self._mon_stream.on_recv(self._on_mon)

        def close(self):
            if False:
                return 10
            if self._socket:
                ZmqWrapper._io_loop_call(False, self._socket.close)

        def _send_multipart(self, parts):
            if False:
                i = 10
                return i + 15
            return self._socket.send_multipart(parts)

        def send_obj(self, obj, topic=''):
            if False:
                return 10
            ZmqWrapper._io_loop_call(False, self._send_multipart, [topic.encode(), pickle.dumps(obj)])

        def _on_mon(self, msg):
            if False:
                print('Hello World!')
            ev = zmq.utils.monitor.parse_monitor_message(msg)
            event = ev['event']
            endpoint = ev['endpoint']
            if event == zmq.EVENT_CONNECTED:
                utils.debug_log('Subscriber connect event', endpoint, verbosity=1)
            elif event == zmq.EVENT_DISCONNECTED:
                utils.debug_log('Subscriber disconnect event', endpoint, verbosity=1)

    class Subscription:

        def __init__(self, port, topic='', callback=None, host='localhost'):
            if False:
                return 10
            self._socket = None
            self._stream = None
            self.topic = None
            ZmqWrapper.initialize()
            utils.debug_log('Creating Subscription', port, verbosity=1)
            ZmqWrapper._io_loop_call(False, self._add_sub, port, topic=topic, callback=callback, host=host)

        def close(self):
            if False:
                while True:
                    i = 10
            if self._socket:
                ZmqWrapper._io_loop_call(False, self._socket.close)

        def _add_sub(self, port, topic, callback, host):
            if False:
                i = 10
                return i + 15

            def callback_wrapper(weak_callback, msg):
                if False:
                    for i in range(10):
                        print('nop')
                [topic, obj_s] = msg
                try:
                    if weak_callback and weak_callback():
                        weak_callback()(pickle.loads(obj_s))
                except Exception as ex:
                    logging.exception('Error in subscription callback')
                    raise
            context = zmq.Context()
            self.topic = topic.encode()
            self._socket = context.socket(zmq.SUB)
            utils.debug_log('Subscriber connecting...', (host, port), verbosity=1)
            self._socket.connect('tcp://%s:%d' % (host, port))
            utils.debug_log('Subscriber connected!', (host, port), verbosity=1)
            if topic != '':
                self._socket.setsockopt(zmq.SUBSCRIBE, self.topic)
            if callback is not None:
                self._stream = zmqstream.ZMQStream(self._socket)
                wr_cb = weakref.WeakMethod(callback)
                wrapper = functools.partial(callback_wrapper, wr_cb)
                self._stream.on_recv(wrapper)

        def _receive_obj(self):
            if False:
                for i in range(10):
                    print('nop')
            [topic, obj_s] = self._socket.recv_multipart()
            if topic != self.topic:
                raise ValueError('Expected topic: %s, Received topic: %s' % (topic, self.topic))
            return pickle.loads(obj_s)

        def receive_obj(self):
            if False:
                print('Hello World!')
            return ZmqWrapper._io_loop_call(True, self._receive_obj)

        def _get_socket_identity(self):
            if False:
                return 10
            ep_id = self._socket.getsockopt(zmq.LAST_ENDPOINT)
            return ep_id

        def get_socket_identity(self):
            if False:
                return 10
            return ZmqWrapper._io_loop_call(True, self._get_socket_identity)

    class ClientServer:

        def __init__(self, port, is_server, callback=None, host=None):
            if False:
                i = 10
                return i + 15
            self._socket = None
            self._stream = None
            ZmqWrapper.initialize()
            utils.debug_log('Creating ClientServer', (is_server, port), verbosity=1)
            ZmqWrapper._io_loop_call(True, self._connect, port, is_server, callback, host)

        def close(self):
            if False:
                print('Hello World!')
            if self._socket:
                ZmqWrapper._io_loop_call(False, self._socket.close)

        def _connect(self, port, is_server, callback, host):
            if False:
                for i in range(10):
                    print('nop')

            def callback_wrapper(callback, msg):
                if False:
                    return 10
                utils.debug_log('Server received request...', verbosity=6)
                [obj_s] = msg
                try:
                    ret = callback(self, pickle.loads(obj_s))
                    self._socket.send_multipart([pickle.dumps((ret, None))])
                except Exception as ex:
                    logging.exception('ClientServer call raised exception')
                    self._socket.send_multipart([pickle.dumps((None, ex))])
                utils.debug_log('Server sent response', verbosity=6)
            context = zmq.Context()
            if is_server:
                host = host or '127.0.0.1'
                self._socket = context.socket(zmq.REP)
                utils.debug_log('Binding socket', (host, port), verbosity=5)
                self._socket.bind('tcp://%s:%d' % (host, port))
                utils.debug_log('Bound socket', (host, port), verbosity=5)
            else:
                host = host or 'localhost'
                self._socket = context.socket(zmq.REQ)
                self._socket.setsockopt(zmq.REQ_CORRELATE, 1)
                self._socket.setsockopt(zmq.REQ_RELAXED, 1)
                utils.debug_log('Client connecting...', verbosity=1)
                self._socket.connect('tcp://%s:%d' % (host, port))
                utils.debug_log('Client connected!', verbosity=1)
            if callback is not None:
                self._stream = zmqstream.ZMQStream(self._socket)
                wrapper = functools.partial(callback_wrapper, callback)
                self._stream.on_recv(wrapper)

        def send_obj(self, obj):
            if False:
                return 10
            ZmqWrapper._io_loop_call(False, self._socket.send_multipart, [pickle.dumps(obj)])

        def receive_obj(self):
            if False:
                for i in range(10):
                    print('nop')
            [obj_s] = ZmqWrapper._io_loop_call(True, self._socket.recv_multipart)
            return pickle.loads(obj_s)

        def request(self, req_obj):
            if False:
                print('Hello World!')
            utils.debug_log('Client sending request...', verbosity=6)
            self.send_obj(req_obj)
            r = self.receive_obj()
            utils.debug_log('Client received response', verbosity=6)
            return r