import deeplake
from deeplake.util.threading import terminate_thread
import socketserver
from typing import Optional, Callable, Dict
import inspect
import threading
import queue
import multiprocessing.connection
import atexit
import time
import uuid
_DISCONNECT_MESSAGE = '!@_dIsCoNNect'

def _get_free_port() -> int:
    if False:
        return 10
    with socketserver.TCPServer(('localhost', 0), None) as s:
        return s.server_address[1]

class Server(object):

    def __init__(self, callback):
        if False:
            i = 10
            return i + 15
        self.callback = callback
        self.start()
        atexit.register(self.stop)

    def start(self):
        if False:
            while True:
                i = 10
        if getattr(self, '_connect_thread', None):
            return
        self.port = _get_free_port()
        self._listener = multiprocessing.connection.Listener(('localhost', self.port))
        self._connections = {}
        self._connect_thread = threading.Thread(target=self._connect_loop, daemon=True)
        self._connect_thread.start()

    def _connect_loop(self):
        if False:
            print('Hello World!')
        try:
            while True:
                try:
                    connection = self._listener.accept()
                    key = str(uuid.uuid4())
                    thread = threading.Thread(target=self._receive_loop, args=(key,))
                    self._connections[key] = (connection, thread)
                    thread.start()
                except Exception:
                    time.sleep(0.1)
        except Exception:
            pass

    def _receive_loop(self, key):
        if False:
            i = 10
            return i + 15
        try:
            while True:
                connection = self._connections[key][0]
                try:
                    msg = connection.recv()
                except ConnectionAbortedError:
                    return
                if msg == _DISCONNECT_MESSAGE:
                    self._connections.pop(key)
                    connection.close()
                    return
                self.callback(msg)
        except Exception:
            pass

    def stop(self):
        if False:
            for i in range(10):
                print('nop')
        if self._connect_thread:
            terminate_thread(self._connect_thread)
            self._connect_thread = None
        timer = 0
        while self._connections:
            if timer >= 5:
                for (connection, thread) in self._connections.values():
                    terminate_thread(thread)
                    connection.close()
                self._connections.clear()
            else:
                timer += 1
                time.sleep(1)
        self._listener.close()

    @property
    def url(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return f'http://localhost:{self.port}/'

class Client(object):

    def __init__(self, port):
        if False:
            while True:
                i = 10
        self.port = port
        self._buffer = []
        self._client = None
        self._closed = False
        threading.Thread(target=self._connect, daemon=True).start()
        atexit.register(self.close)

    def _connect(self):
        if False:
            print('Hello World!')
        while True:
            try:
                self._client = multiprocessing.connection.Client(('localhost', self.port))
                for stuff in self._buffer:
                    self._client.send(stuff)
                self._buffer.clear()
                return
            except Exception:
                time.sleep(1)

    def send(self, stuff):
        if False:
            return 10
        if self._client:
            try:
                self._client.send(stuff)
            except Exception:
                pass
        else:
            self._buffer.append(stuff)

    def close(self):
        if False:
            for i in range(10):
                print('nop')
        if self._closed:
            return
        try:
            while not self._client:
                time.sleep(0.5)
            for stuff in self._buffer:
                self._client.send(stuff)
            self._client.send(_DISCONNECT_MESSAGE)
            self._client.close()
            self._client = None
            self._closed = True
        except Exception as e:
            pass