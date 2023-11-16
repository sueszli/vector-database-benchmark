"""Use pika with the Asyncio EventLoop"""
import asyncio
import logging
import sys
from pika.adapters import base_connection
from pika.adapters.utils import nbio_interface, io_services_utils
LOGGER = logging.getLogger(__name__)
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

class AsyncioConnection(base_connection.BaseConnection):
    """ The AsyncioConnection runs on the Asyncio EventLoop.

    """

    def __init__(self, parameters=None, on_open_callback=None, on_open_error_callback=None, on_close_callback=None, custom_ioloop=None, internal_connection_workflow=True):
        if False:
            while True:
                i = 10
        " Create a new instance of the AsyncioConnection class, connecting\n        to RabbitMQ automatically\n\n        :param pika.connection.Parameters parameters: Connection parameters\n        :param callable on_open_callback: The method to call when the connection\n            is open\n        :param None | method on_open_error_callback: Called if the connection\n            can't be established or connection establishment is interrupted by\n            `Connection.close()`: on_open_error_callback(Connection, exception).\n        :param None | method on_close_callback: Called when a previously fully\n            open connection is closed:\n            `on_close_callback(Connection, exception)`, where `exception` is\n            either an instance of `exceptions.ConnectionClosed` if closed by\n            user or broker or exception of another type that describes the cause\n            of connection failure.\n        :param None | asyncio.AbstractEventLoop |\n            nbio_interface.AbstractIOServices custom_ioloop:\n                Defaults to asyncio.get_event_loop().\n        :param bool internal_connection_workflow: True for autonomous connection\n            establishment which is default; False for externally-managed\n            connection workflow via the `create_connection()` factory.\n\n        "
        if isinstance(custom_ioloop, nbio_interface.AbstractIOServices):
            nbio = custom_ioloop
        else:
            nbio = _AsyncioIOServicesAdapter(custom_ioloop)
        super().__init__(parameters, on_open_callback, on_open_error_callback, on_close_callback, nbio, internal_connection_workflow=internal_connection_workflow)

    @classmethod
    def create_connection(cls, connection_configs, on_done, custom_ioloop=None, workflow=None):
        if False:
            for i in range(10):
                print('nop')
        'Implement\n        :py:classmethod::`pika.adapters.BaseConnection.create_connection()`.\n\n        '
        nbio = _AsyncioIOServicesAdapter(custom_ioloop)

        def connection_factory(params):
            if False:
                while True:
                    i = 10
            'Connection factory.'
            if params is None:
                raise ValueError('Expected pika.connection.Parameters instance, but got None in params arg.')
            return cls(parameters=params, custom_ioloop=nbio, internal_connection_workflow=False)
        return cls._start_connection_workflow(connection_configs=connection_configs, connection_factory=connection_factory, nbio=nbio, workflow=workflow, on_done=on_done)

class _AsyncioIOServicesAdapter(io_services_utils.SocketConnectionMixin, io_services_utils.StreamingConnectionMixin, nbio_interface.AbstractIOServices, nbio_interface.AbstractFileDescriptorServices):
    """Implements
    :py:class:`.utils.nbio_interface.AbstractIOServices` interface
    on top of `asyncio`.

    NOTE:
    :py:class:`.utils.nbio_interface.AbstractFileDescriptorServices`
    interface is only required by the mixins.

    """

    def __init__(self, loop=None):
        if False:
            i = 10
            return i + 15
        '\n        :param asyncio.AbstractEventLoop | None loop: If None, gets default\n            event loop from asyncio.\n\n        '
        self._loop = loop or asyncio.get_event_loop()

    def get_native_ioloop(self):
        if False:
            while True:
                i = 10
        'Implement\n        :py:meth:`.utils.nbio_interface.AbstractIOServices.get_native_ioloop()`.\n\n        '
        return self._loop

    def close(self):
        if False:
            return 10
        'Implement\n        :py:meth:`.utils.nbio_interface.AbstractIOServices.close()`.\n\n        '
        self._loop.close()

    def run(self):
        if False:
            while True:
                i = 10
        'Implement :py:meth:`.utils.nbio_interface.AbstractIOServices.run()`.\n\n        '
        self._loop.run_forever()

    def stop(self):
        if False:
            return 10
        'Implement :py:meth:`.utils.nbio_interface.AbstractIOServices.stop()`.\n\n        '
        self._loop.stop()

    def add_callback_threadsafe(self, callback):
        if False:
            print('Hello World!')
        'Implement\n        :py:meth:`.utils.nbio_interface.AbstractIOServices.add_callback_threadsafe()`.\n\n        '
        self._loop.call_soon_threadsafe(callback)

    def call_later(self, delay, callback):
        if False:
            for i in range(10):
                print('nop')
        'Implement\n        :py:meth:`.utils.nbio_interface.AbstractIOServices.call_later()`.\n\n        '
        return _TimerHandle(self._loop.call_later(delay, callback))

    def getaddrinfo(self, host, port, on_done, family=0, socktype=0, proto=0, flags=0):
        if False:
            for i in range(10):
                print('nop')
        'Implement\n        :py:meth:`.utils.nbio_interface.AbstractIOServices.getaddrinfo()`.\n\n        '
        return self._schedule_and_wrap_in_io_ref(self._loop.getaddrinfo(host, port, family=family, type=socktype, proto=proto, flags=flags), on_done)

    def set_reader(self, fd, on_readable):
        if False:
            i = 10
            return i + 15
        'Implement\n        :py:meth:`.utils.nbio_interface.AbstractFileDescriptorServices.set_reader()`.\n\n        '
        self._loop.add_reader(fd, on_readable)
        LOGGER.debug('set_reader(%s, _)', fd)

    def remove_reader(self, fd):
        if False:
            for i in range(10):
                print('nop')
        'Implement\n        :py:meth:`.utils.nbio_interface.AbstractFileDescriptorServices.remove_reader()`.\n\n        '
        LOGGER.debug('remove_reader(%s)', fd)
        return self._loop.remove_reader(fd)

    def set_writer(self, fd, on_writable):
        if False:
            while True:
                i = 10
        'Implement\n        :py:meth:`.utils.nbio_interface.AbstractFileDescriptorServices.set_writer()`.\n\n        '
        self._loop.add_writer(fd, on_writable)
        LOGGER.debug('set_writer(%s, _)', fd)

    def remove_writer(self, fd):
        if False:
            i = 10
            return i + 15
        'Implement\n        :py:meth:`.utils.nbio_interface.AbstractFileDescriptorServices.remove_writer()`.\n\n        '
        LOGGER.debug('remove_writer(%s)', fd)
        return self._loop.remove_writer(fd)

    def _schedule_and_wrap_in_io_ref(self, coro, on_done):
        if False:
            while True:
                i = 10
        'Schedule the coroutine to run and return _AsyncioIOReference\n\n        :param coroutine-obj coro:\n        :param callable on_done: user callback that takes the completion result\n            or exception as its only arg. It will not be called if the operation\n            was cancelled.\n        :rtype: _AsyncioIOReference which is derived from\n            nbio_interface.AbstractIOReference\n\n        '
        if not callable(on_done):
            raise TypeError(f'on_done arg must be callable, but got {on_done!r}')
        return _AsyncioIOReference(asyncio.ensure_future(coro, loop=self._loop), on_done)

class _TimerHandle(nbio_interface.AbstractTimerReference):
    """This module's adaptation of `nbio_interface.AbstractTimerReference`.

    """

    def __init__(self, handle):
        if False:
            while True:
                i = 10
        '\n\n        :param asyncio.Handle handle:\n        '
        self._handle = handle

    def cancel(self):
        if False:
            while True:
                i = 10
        if self._handle is not None:
            self._handle.cancel()
            self._handle = None

class _AsyncioIOReference(nbio_interface.AbstractIOReference):
    """This module's adaptation of `nbio_interface.AbstractIOReference`.

    """

    def __init__(self, future, on_done):
        if False:
            i = 10
            return i + 15
        '\n        :param asyncio.Future future:\n        :param callable on_done: user callback that takes the completion result\n            or exception as its only arg. It will not be called if the operation\n            was cancelled.\n\n        '
        if not callable(on_done):
            raise TypeError(f'on_done arg must be callable, but got {on_done!r}')
        self._future = future

        def on_done_adapter(future):
            if False:
                i = 10
                return i + 15
            'Handle completion callback from the future instance'
            if not future.cancelled():
                on_done(future.exception() or future.result())
        future.add_done_callback(on_done_adapter)

    def cancel(self):
        if False:
            i = 10
            return i + 15
        'Cancel pending operation\n\n        :returns: False if was already done or cancelled; True otherwise\n        :rtype: bool\n\n        '
        return self._future.cancel()