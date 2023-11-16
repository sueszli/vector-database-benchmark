import itertools
import logging
import signal
import threading
from . import base_namespace
from . import packet
default_logger = logging.getLogger('socketio.client')
reconnecting_clients = []

def signal_handler(sig, frame):
    if False:
        return 10
    'SIGINT handler.\n\n    Notify any clients that are in a reconnect loop to abort. Other\n    disconnection tasks are handled at the engine.io level.\n    '
    for client in reconnecting_clients[:]:
        client._reconnect_abort.set()
    if callable(original_signal_handler):
        return original_signal_handler(sig, frame)
    else:
        return signal.default_int_handler(sig, frame)
original_signal_handler = None

class BaseClient:
    reserved_events = ['connect', 'connect_error', 'disconnect', '__disconnect_final']

    def __init__(self, reconnection=True, reconnection_attempts=0, reconnection_delay=1, reconnection_delay_max=5, randomization_factor=0.5, logger=False, serializer='default', json=None, handle_sigint=True, **kwargs):
        if False:
            print('Hello World!')
        global original_signal_handler
        if handle_sigint and original_signal_handler is None and (threading.current_thread() == threading.main_thread()):
            original_signal_handler = signal.signal(signal.SIGINT, signal_handler)
        self.reconnection = reconnection
        self.reconnection_attempts = reconnection_attempts
        self.reconnection_delay = reconnection_delay
        self.reconnection_delay_max = reconnection_delay_max
        self.randomization_factor = randomization_factor
        self.handle_sigint = handle_sigint
        engineio_options = kwargs
        engineio_options['handle_sigint'] = handle_sigint
        engineio_logger = engineio_options.pop('engineio_logger', None)
        if engineio_logger is not None:
            engineio_options['logger'] = engineio_logger
        if serializer == 'default':
            self.packet_class = packet.Packet
        elif serializer == 'msgpack':
            from . import msgpack_packet
            self.packet_class = msgpack_packet.MsgPackPacket
        else:
            self.packet_class = serializer
        if json is not None:
            self.packet_class.json = json
            engineio_options['json'] = json
        self.eio = self._engineio_client_class()(**engineio_options)
        self.eio.on('connect', self._handle_eio_connect)
        self.eio.on('message', self._handle_eio_message)
        self.eio.on('disconnect', self._handle_eio_disconnect)
        if not isinstance(logger, bool):
            self.logger = logger
        else:
            self.logger = default_logger
            if self.logger.level == logging.NOTSET:
                if logger:
                    self.logger.setLevel(logging.INFO)
                else:
                    self.logger.setLevel(logging.ERROR)
                self.logger.addHandler(logging.StreamHandler())
        self.connection_url = None
        self.connection_headers = None
        self.connection_auth = None
        self.connection_transports = None
        self.connection_namespaces = []
        self.socketio_path = None
        self.sid = None
        self.connected = False
        self.namespaces = {}
        self.handlers = {}
        self.namespace_handlers = {}
        self.callbacks = {}
        self._binary_packet = None
        self._connect_event = None
        self._reconnect_task = None
        self._reconnect_abort = None

    def is_asyncio_based(self):
        if False:
            for i in range(10):
                print('nop')
        return False

    def on(self, event, handler=None, namespace=None):
        if False:
            return 10
        "Register an event handler.\n\n        :param event: The event name. It can be any string. The event names\n                      ``'connect'``, ``'message'`` and ``'disconnect'`` are\n                      reserved and should not be used.\n        :param handler: The function that should be invoked to handle the\n                        event. When this parameter is not given, the method\n                        acts as a decorator for the handler function.\n        :param namespace: The Socket.IO namespace for the event. If this\n                          argument is omitted the handler is associated with\n                          the default namespace.\n\n        Example usage::\n\n            # as a decorator:\n            @sio.on('connect')\n            def connect_handler():\n                print('Connected!')\n\n            # as a method:\n            def message_handler(msg):\n                print('Received message: ', msg)\n                sio.send( 'response')\n            sio.on('message', message_handler)\n\n        The ``'connect'`` event handler receives no arguments. The\n        ``'message'`` handler and handlers for custom event names receive the\n        message payload as only argument. Any values returned from a message\n        handler will be passed to the client's acknowledgement callback\n        function if it exists. The ``'disconnect'`` handler does not take\n        arguments.\n        "
        namespace = namespace or '/'

        def set_handler(handler):
            if False:
                for i in range(10):
                    print('nop')
            if namespace not in self.handlers:
                self.handlers[namespace] = {}
            self.handlers[namespace][event] = handler
            return handler
        if handler is None:
            return set_handler
        set_handler(handler)

    def event(self, *args, **kwargs):
        if False:
            return 10
        "Decorator to register an event handler.\n\n        This is a simplified version of the ``on()`` method that takes the\n        event name from the decorated function.\n\n        Example usage::\n\n            @sio.event\n            def my_event(data):\n                print('Received data: ', data)\n\n        The above example is equivalent to::\n\n            @sio.on('my_event')\n            def my_event(data):\n                print('Received data: ', data)\n\n        A custom namespace can be given as an argument to the decorator::\n\n            @sio.event(namespace='/test')\n            def my_event(data):\n                print('Received data: ', data)\n        "
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            return self.on(args[0].__name__)(args[0])
        else:

            def set_handler(handler):
                if False:
                    print('Hello World!')
                return self.on(handler.__name__, *args, **kwargs)(handler)
            return set_handler

    def register_namespace(self, namespace_handler):
        if False:
            while True:
                i = 10
        'Register a namespace handler object.\n\n        :param namespace_handler: An instance of a :class:`Namespace`\n                                  subclass that handles all the event traffic\n                                  for a namespace.\n        '
        if not isinstance(namespace_handler, base_namespace.BaseClientNamespace):
            raise ValueError('Not a namespace instance')
        if self.is_asyncio_based() != namespace_handler.is_asyncio_based():
            raise ValueError('Not a valid namespace class for this client')
        namespace_handler._set_client(self)
        self.namespace_handlers[namespace_handler.namespace] = namespace_handler

    def get_sid(self, namespace=None):
        if False:
            print('Hello World!')
        'Return the ``sid`` associated with a connection.\n\n        :param namespace: The Socket.IO namespace. If this argument is omitted\n                          the handler is associated with the default\n                          namespace. Note that unlike previous versions, the\n                          current version of the Socket.IO protocol uses\n                          different ``sid`` values per namespace.\n\n        This method returns the ``sid`` for the requested namespace as a\n        string.\n        '
        return self.namespaces.get(namespace or '/')

    def transport(self):
        if False:
            while True:
                i = 10
        "Return the name of the transport used by the client.\n\n        The two possible values returned by this function are ``'polling'``\n        and ``'websocket'``.\n        "
        return self.eio.transport()

    def _generate_ack_id(self, namespace, callback):
        if False:
            while True:
                i = 10
        'Generate a unique identifier for an ACK packet.'
        namespace = namespace or '/'
        if namespace not in self.callbacks:
            self.callbacks[namespace] = {0: itertools.count(1)}
        id = next(self.callbacks[namespace][0])
        self.callbacks[namespace][id] = callback
        return id

    def _handle_eio_connect(self):
        if False:
            return 10
        raise NotImplementedError()

    def _handle_eio_message(self, data):
        if False:
            while True:
                i = 10
        raise NotImplementedError()

    def _handle_eio_disconnect(self):
        if False:
            print('Hello World!')
        raise NotImplementedError()

    def _engineio_client_class(self):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError()