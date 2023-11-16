"""Use pika with the Tornado IOLoop

"""
import logging
from tornado import ioloop
from pika.adapters.utils import nbio_interface, selector_ioloop_adapter
from pika.adapters import base_connection
LOGGER = logging.getLogger(__name__)

class TornadoConnection(base_connection.BaseConnection):
    """The TornadoConnection runs on the Tornado IOLoop.
    """

    def __init__(self, parameters=None, on_open_callback=None, on_open_error_callback=None, on_close_callback=None, custom_ioloop=None, internal_connection_workflow=True):
        if False:
            print('Hello World!')
        "Create a new instance of the TornadoConnection class, connecting\n        to RabbitMQ automatically.\n\n        :param pika.connection.Parameters|None parameters: The connection\n            parameters\n        :param callable|None on_open_callback: The method to call when the\n            connection is open\n        :param callable|None on_open_error_callback: Called if the connection\n            can't be established or connection establishment is interrupted by\n            `Connection.close()`:\n            on_open_error_callback(Connection, exception)\n        :param callable|None on_close_callback: Called when a previously fully\n            open connection is closed:\n            `on_close_callback(Connection, exception)`, where `exception` is\n            either an instance of `exceptions.ConnectionClosed` if closed by\n            user or broker or exception of another type that describes the\n            cause of connection failure\n        :param ioloop.IOLoop|nbio_interface.AbstractIOServices|None custom_ioloop:\n            Override using the global IOLoop in Tornado\n        :param bool internal_connection_workflow: True for autonomous connection\n            establishment which is default; False for externally-managed\n            connection workflow via the `create_connection()` factory\n\n        "
        if isinstance(custom_ioloop, nbio_interface.AbstractIOServices):
            nbio = custom_ioloop
        else:
            nbio = selector_ioloop_adapter.SelectorIOServicesAdapter(custom_ioloop or ioloop.IOLoop.instance())
        super().__init__(parameters, on_open_callback, on_open_error_callback, on_close_callback, nbio, internal_connection_workflow=internal_connection_workflow)

    @classmethod
    def create_connection(cls, connection_configs, on_done, custom_ioloop=None, workflow=None):
        if False:
            i = 10
            return i + 15
        'Implement\n        :py:classmethod::`pika.adapters.BaseConnection.create_connection()`.\n\n        '
        nbio = selector_ioloop_adapter.SelectorIOServicesAdapter(custom_ioloop or ioloop.IOLoop.instance())

        def connection_factory(params):
            if False:
                i = 10
                return i + 15
            'Connection factory.'
            if params is None:
                raise ValueError('Expected pika.connection.Parameters instance, but got None in params arg.')
            return cls(parameters=params, custom_ioloop=nbio, internal_connection_workflow=False)
        return cls._start_connection_workflow(connection_configs=connection_configs, connection_factory=connection_factory, nbio=nbio, workflow=workflow, on_done=on_done)