import logging
from . import manager
from . import base_namespace
from . import packet
default_logger = logging.getLogger('socketio.server')

class BaseServer:
    reserved_events = ['connect', 'disconnect']

    def __init__(self, client_manager=None, logger=False, serializer='default', json=None, async_handlers=True, always_connect=False, namespaces=None, **kwargs):
        if False:
            i = 10
            return i + 15
        engineio_options = kwargs
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
        engineio_options['async_handlers'] = False
        self.eio = self._engineio_server_class()(**engineio_options)
        self.eio.on('connect', self._handle_eio_connect)
        self.eio.on('message', self._handle_eio_message)
        self.eio.on('disconnect', self._handle_eio_disconnect)
        self.environ = {}
        self.handlers = {}
        self.namespace_handlers = {}
        self.not_handled = object()
        self._binary_packet = {}
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
        if client_manager is None:
            client_manager = manager.Manager()
        self.manager = client_manager
        self.manager.set_server(self)
        self.manager_initialized = False
        self.async_handlers = async_handlers
        self.always_connect = always_connect
        self.namespaces = namespaces or ['/']
        self.async_mode = self.eio.async_mode

    def is_asyncio_based(self):
        if False:
            i = 10
            return i + 15
        return False

    def on(self, event, handler=None, namespace=None):
        if False:
            i = 10
            return i + 15
        "Register an event handler.\n\n        :param event: The event name. It can be any string. The event names\n                      ``'connect'``, ``'message'`` and ``'disconnect'`` are\n                      reserved and should not be used.\n        :param handler: The function that should be invoked to handle the\n                        event. When this parameter is not given, the method\n                        acts as a decorator for the handler function.\n        :param namespace: The Socket.IO namespace for the event. If this\n                          argument is omitted the handler is associated with\n                          the default namespace.\n\n        Example usage::\n\n            # as a decorator:\n            @sio.on('connect', namespace='/chat')\n            def connect_handler(sid, environ):\n                print('Connection request')\n                if environ['REMOTE_ADDR'] in blacklisted:\n                    return False  # reject\n\n            # as a method:\n            def message_handler(sid, msg):\n                print('Received message: ', msg)\n                sio.send(sid, 'response')\n            socket_io.on('message', namespace='/chat', handler=message_handler)\n\n        The handler function receives the ``sid`` (session ID) for the\n        client as first argument. The ``'connect'`` event handler receives the\n        WSGI environment as a second argument, and can return ``False`` to\n        reject the connection. The ``'message'`` handler and handlers for\n        custom event names receive the message payload as a second argument.\n        Any values returned from a message handler will be passed to the\n        client's acknowledgement callback function if it exists. The\n        ``'disconnect'`` handler does not take a second argument.\n        "
        namespace = namespace or '/'

        def set_handler(handler):
            if False:
                print('Hello World!')
            if namespace not in self.handlers:
                self.handlers[namespace] = {}
            self.handlers[namespace][event] = handler
            return handler
        if handler is None:
            return set_handler
        set_handler(handler)

    def event(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        "Decorator to register an event handler.\n\n        This is a simplified version of the ``on()`` method that takes the\n        event name from the decorated function.\n\n        Example usage::\n\n            @sio.event\n            def my_event(data):\n                print('Received data: ', data)\n\n        The above example is equivalent to::\n\n            @sio.on('my_event')\n            def my_event(data):\n                print('Received data: ', data)\n\n        A custom namespace can be given as an argument to the decorator::\n\n            @sio.event(namespace='/test')\n            def my_event(data):\n                print('Received data: ', data)\n        "
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            return self.on(args[0].__name__)(args[0])
        else:

            def set_handler(handler):
                if False:
                    return 10
                return self.on(handler.__name__, *args, **kwargs)(handler)
            return set_handler

    def register_namespace(self, namespace_handler):
        if False:
            i = 10
            return i + 15
        'Register a namespace handler object.\n\n        :param namespace_handler: An instance of a :class:`Namespace`\n                                  subclass that handles all the event traffic\n                                  for a namespace.\n        '
        if not isinstance(namespace_handler, base_namespace.BaseServerNamespace):
            raise ValueError('Not a namespace instance')
        if self.is_asyncio_based() != namespace_handler.is_asyncio_based():
            raise ValueError('Not a valid namespace class for this server')
        namespace_handler._set_server(self)
        self.namespace_handlers[namespace_handler.namespace] = namespace_handler

    def rooms(self, sid, namespace=None):
        if False:
            i = 10
            return i + 15
        'Return the rooms a client is in.\n\n        :param sid: Session ID of the client.\n        :param namespace: The Socket.IO namespace for the event. If this\n                          argument is omitted the default namespace is used.\n        '
        namespace = namespace or '/'
        return self.manager.get_rooms(sid, namespace)

    def transport(self, sid):
        if False:
            print('Hello World!')
        "Return the name of the transport used by the client.\n\n        The two possible values returned by this function are ``'polling'``\n        and ``'websocket'``.\n\n        :param sid: The session of the client.\n        "
        return self.eio.transport(sid)

    def get_environ(self, sid, namespace=None):
        if False:
            print('Hello World!')
        'Return the WSGI environ dictionary for a client.\n\n        :param sid: The session of the client.\n        :param namespace: The Socket.IO namespace. If this argument is omitted\n                          the default namespace is used.\n        '
        eio_sid = self.manager.eio_sid_from_sid(sid, namespace or '/')
        return self.environ.get(eio_sid)

    def _handle_eio_connect(self):
        if False:
            while True:
                i = 10
        raise NotImplementedError()

    def _handle_eio_message(self, data):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError()

    def _handle_eio_disconnect(self):
        if False:
            print('Hello World!')
        raise NotImplementedError()

    def _engineio_server_class(self):
        if False:
            while True:
                i = 10
        raise NotImplementedError('Must be implemented in subclasses')