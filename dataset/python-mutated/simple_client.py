from threading import Event
from socketio import Client
from socketio.exceptions import SocketIOError, TimeoutError, DisconnectedError

class SimpleClient:
    """A Socket.IO client.

    This class implements a simple, yet fully compliant Socket.IO web client
    with support for websocket and long-polling transports.

    Th positional and keyword arguments given in the constructor are passed
    to the underlying :func:`socketio.Client` object.
    """

    def __init__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        self.client_args = args
        self.client_kwargs = kwargs
        self.client = None
        self.namespace = '/'
        self.connected_event = Event()
        self.connected = False
        self.input_event = Event()
        self.input_buffer = []

    def connect(self, url, headers={}, auth=None, transports=None, namespace='/', socketio_path='socket.io', wait_timeout=5):
        if False:
            return 10
        "Connect to a Socket.IO server.\n\n        :param url: The URL of the Socket.IO server. It can include custom\n                    query string parameters if required by the server. If a\n                    function is provided, the client will invoke it to obtain\n                    the URL each time a connection or reconnection is\n                    attempted.\n        :param headers: A dictionary with custom headers to send with the\n                        connection request. If a function is provided, the\n                        client will invoke it to obtain the headers dictionary\n                        each time a connection or reconnection is attempted.\n        :param auth: Authentication data passed to the server with the\n                     connection request, normally a dictionary with one or\n                     more string key/value pairs. If a function is provided,\n                     the client will invoke it to obtain the authentication\n                     data each time a connection or reconnection is attempted.\n        :param transports: The list of allowed transports. Valid transports\n                           are ``'polling'`` and ``'websocket'``. If not\n                           given, the polling transport is connected first,\n                           then an upgrade to websocket is attempted.\n        :param namespace: The namespace to connect to as a string. If not\n                          given, the default namespace ``/`` is used.\n        :param socketio_path: The endpoint where the Socket.IO server is\n                              installed. The default value is appropriate for\n                              most cases.\n        :param wait_timeout: How long the client should wait for the\n                             connection to be established. The default is 5\n                             seconds.\n        "
        if self.connected:
            raise RuntimeError('Already connected')
        self.namespace = namespace
        self.input_buffer = []
        self.input_event.clear()
        self.client = Client(*self.client_args, **self.client_kwargs)

        @self.client.event(namespace=self.namespace)
        def connect():
            if False:
                for i in range(10):
                    print('nop')
            self.connected = True
            self.connected_event.set()

        @self.client.event(namespace=self.namespace)
        def disconnect():
            if False:
                print('Hello World!')
            self.connected_event.clear()

        @self.client.event(namespace=self.namespace)
        def __disconnect_final():
            if False:
                return 10
            self.connected = False
            self.connected_event.set()

        @self.client.on('*', namespace=self.namespace)
        def on_event(event, *args):
            if False:
                i = 10
                return i + 15
            self.input_buffer.append([event, *args])
            self.input_event.set()
        self.client.connect(url, headers=headers, auth=auth, transports=transports, namespaces=[namespace], socketio_path=socketio_path, wait_timeout=wait_timeout)

    @property
    def sid(self):
        if False:
            i = 10
            return i + 15
        'The session ID received from the server.\n\n        The session ID is not guaranteed to remain constant throughout the life\n        of the connection, as reconnections can cause it to change.\n        '
        return self.client.get_sid(self.namespace) if self.client else None

    @property
    def transport(self):
        if False:
            for i in range(10):
                print('nop')
        'The name of the transport currently in use.\n\n        The transport is returned as a string and can be one of ``polling``\n        and ``websocket``.\n        '
        return self.client.transport if self.client else ''

    def emit(self, event, data=None):
        if False:
            return 10
        "Emit an event to the server.\n\n        :param event: The event name. It can be any string. The event names\n                      ``'connect'``, ``'message'`` and ``'disconnect'`` are\n                      reserved and should not be used.\n        :param data: The data to send to the server. Data can be of\n                     type ``str``, ``bytes``, ``list`` or ``dict``. To send\n                     multiple arguments, use a tuple where each element is of\n                     one of the types indicated above.\n\n        This method schedules the event to be sent out and returns, without\n        actually waiting for its delivery. In cases where the client needs to\n        ensure that the event was received, :func:`socketio.SimpleClient.call`\n        should be used instead.\n        "
        while True:
            self.connected_event.wait()
            if not self.connected:
                raise DisconnectedError()
            try:
                return self.client.emit(event, data, namespace=self.namespace)
            except SocketIOError:
                pass

    def call(self, event, data=None, timeout=60):
        if False:
            return 10
        "Emit an event to the server and wait for a response.\n\n        This method issues an emit and waits for the server to provide a\n        response or acknowledgement. If the response does not arrive before the\n        timeout, then a ``TimeoutError`` exception is raised.\n\n        :param event: The event name. It can be any string. The event names\n                      ``'connect'``, ``'message'`` and ``'disconnect'`` are\n                      reserved and should not be used.\n        :param data: The data to send to the server. Data can be of\n                     type ``str``, ``bytes``, ``list`` or ``dict``. To send\n                     multiple arguments, use a tuple where each element is of\n                     one of the types indicated above.\n        :param timeout: The waiting timeout. If the timeout is reached before\n                        the server acknowledges the event, then a\n                        ``TimeoutError`` exception is raised.\n        "
        while True:
            self.connected_event.wait()
            if not self.connected:
                raise DisconnectedError()
            try:
                return self.client.call(event, data, namespace=self.namespace, timeout=timeout)
            except SocketIOError:
                pass

    def receive(self, timeout=None):
        if False:
            print('Hello World!')
        'Wait for an event from the server.\n\n        :param timeout: The waiting timeout. If the timeout is reached before\n                        the server acknowledges the event, then a\n                        ``TimeoutError`` exception is raised.\n\n        The return value is a list with the event name as the first element. If\n        the server included arguments with the event, they are returned as\n        additional list elements.\n        '
        while not self.input_buffer:
            if not self.connected_event.wait(timeout=timeout):
                raise TimeoutError()
            if not self.connected:
                raise DisconnectedError()
            if not self.input_event.wait(timeout=timeout):
                raise TimeoutError()
            self.input_event.clear()
        return self.input_buffer.pop(0)

    def disconnect(self):
        if False:
            while True:
                i = 10
        'Disconnect from the server.'
        if self.connected:
            self.client.disconnect()
            self.client = None
            self.connected = False

    def __enter__(self):
        if False:
            i = 10
            return i + 15
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if False:
            i = 10
            return i + 15
        self.disconnect()