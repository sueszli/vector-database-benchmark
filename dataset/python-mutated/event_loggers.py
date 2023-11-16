"""Example event logger classes.

.. versionadded:: 3.11

These loggers can be registered using :func:`register` or
:class:`~pymongo.mongo_client.MongoClient`.

``monitoring.register(CommandLogger())``

or

``MongoClient(event_listeners=[CommandLogger()])``
"""
from __future__ import annotations
import logging
from pymongo import monitoring

class CommandLogger(monitoring.CommandListener):
    """A simple listener that logs command events.

    Listens for :class:`~pymongo.monitoring.CommandStartedEvent`,
    :class:`~pymongo.monitoring.CommandSucceededEvent` and
    :class:`~pymongo.monitoring.CommandFailedEvent` events and
    logs them at the `INFO` severity level using :mod:`logging`.
    .. versionadded:: 3.11
    """

    def started(self, event: monitoring.CommandStartedEvent) -> None:
        if False:
            return 10
        logging.info(f'Command {event.command_name} with request id {event.request_id} started on server {event.connection_id}')

    def succeeded(self, event: monitoring.CommandSucceededEvent) -> None:
        if False:
            for i in range(10):
                print('nop')
        logging.info(f'Command {event.command_name} with request id {event.request_id} on server {event.connection_id} succeeded in {event.duration_micros} microseconds')

    def failed(self, event: monitoring.CommandFailedEvent) -> None:
        if False:
            i = 10
            return i + 15
        logging.info(f'Command {event.command_name} with request id {event.request_id} on server {event.connection_id} failed in {event.duration_micros} microseconds')

class ServerLogger(monitoring.ServerListener):
    """A simple listener that logs server discovery events.

    Listens for :class:`~pymongo.monitoring.ServerOpeningEvent`,
    :class:`~pymongo.monitoring.ServerDescriptionChangedEvent`,
    and :class:`~pymongo.monitoring.ServerClosedEvent`
    events and logs them at the `INFO` severity level using :mod:`logging`.

    .. versionadded:: 3.11
    """

    def opened(self, event: monitoring.ServerOpeningEvent) -> None:
        if False:
            i = 10
            return i + 15
        logging.info(f'Server {event.server_address} added to topology {event.topology_id}')

    def description_changed(self, event: monitoring.ServerDescriptionChangedEvent) -> None:
        if False:
            return 10
        previous_server_type = event.previous_description.server_type
        new_server_type = event.new_description.server_type
        if new_server_type != previous_server_type:
            logging.info(f'Server {event.server_address} changed type from {event.previous_description.server_type_name} to {event.new_description.server_type_name}')

    def closed(self, event: monitoring.ServerClosedEvent) -> None:
        if False:
            return 10
        logging.warning(f'Server {event.server_address} removed from topology {event.topology_id}')

class HeartbeatLogger(monitoring.ServerHeartbeatListener):
    """A simple listener that logs server heartbeat events.

    Listens for :class:`~pymongo.monitoring.ServerHeartbeatStartedEvent`,
    :class:`~pymongo.monitoring.ServerHeartbeatSucceededEvent`,
    and :class:`~pymongo.monitoring.ServerHeartbeatFailedEvent`
    events and logs them at the `INFO` severity level using :mod:`logging`.

    .. versionadded:: 3.11
    """

    def started(self, event: monitoring.ServerHeartbeatStartedEvent) -> None:
        if False:
            return 10
        logging.info(f'Heartbeat sent to server {event.connection_id}')

    def succeeded(self, event: monitoring.ServerHeartbeatSucceededEvent) -> None:
        if False:
            for i in range(10):
                print('nop')
        logging.info(f'Heartbeat to server {event.connection_id} succeeded with reply {event.reply.document}')

    def failed(self, event: monitoring.ServerHeartbeatFailedEvent) -> None:
        if False:
            for i in range(10):
                print('nop')
        logging.warning(f'Heartbeat to server {event.connection_id} failed with error {event.reply}')

class TopologyLogger(monitoring.TopologyListener):
    """A simple listener that logs server topology events.

    Listens for :class:`~pymongo.monitoring.TopologyOpenedEvent`,
    :class:`~pymongo.monitoring.TopologyDescriptionChangedEvent`,
    and :class:`~pymongo.monitoring.TopologyClosedEvent`
    events and logs them at the `INFO` severity level using :mod:`logging`.

    .. versionadded:: 3.11
    """

    def opened(self, event: monitoring.TopologyOpenedEvent) -> None:
        if False:
            while True:
                i = 10
        logging.info(f'Topology with id {event.topology_id} opened')

    def description_changed(self, event: monitoring.TopologyDescriptionChangedEvent) -> None:
        if False:
            print('Hello World!')
        logging.info(f'Topology description updated for topology id {event.topology_id}')
        previous_topology_type = event.previous_description.topology_type
        new_topology_type = event.new_description.topology_type
        if new_topology_type != previous_topology_type:
            logging.info(f'Topology {event.topology_id} changed type from {event.previous_description.topology_type_name} to {event.new_description.topology_type_name}')
        if not event.new_description.has_writable_server():
            logging.warning('No writable servers available.')
        if not event.new_description.has_readable_server():
            logging.warning('No readable servers available.')

    def closed(self, event: monitoring.TopologyClosedEvent) -> None:
        if False:
            print('Hello World!')
        logging.info(f'Topology with id {event.topology_id} closed')

class ConnectionPoolLogger(monitoring.ConnectionPoolListener):
    """A simple listener that logs server connection pool events.

    Listens for :class:`~pymongo.monitoring.PoolCreatedEvent`,
    :class:`~pymongo.monitoring.PoolClearedEvent`,
    :class:`~pymongo.monitoring.PoolClosedEvent`,
    :~pymongo.monitoring.class:`ConnectionCreatedEvent`,
    :class:`~pymongo.monitoring.ConnectionReadyEvent`,
    :class:`~pymongo.monitoring.ConnectionClosedEvent`,
    :class:`~pymongo.monitoring.ConnectionCheckOutStartedEvent`,
    :class:`~pymongo.monitoring.ConnectionCheckOutFailedEvent`,
    :class:`~pymongo.monitoring.ConnectionCheckedOutEvent`,
    and :class:`~pymongo.monitoring.ConnectionCheckedInEvent`
    events and logs them at the `INFO` severity level using :mod:`logging`.

    .. versionadded:: 3.11
    """

    def pool_created(self, event: monitoring.PoolCreatedEvent) -> None:
        if False:
            for i in range(10):
                print('nop')
        logging.info(f'[pool {event.address}] pool created')

    def pool_ready(self, event: monitoring.PoolReadyEvent) -> None:
        if False:
            return 10
        logging.info(f'[pool {event.address}] pool ready')

    def pool_cleared(self, event: monitoring.PoolClearedEvent) -> None:
        if False:
            while True:
                i = 10
        logging.info(f'[pool {event.address}] pool cleared')

    def pool_closed(self, event: monitoring.PoolClosedEvent) -> None:
        if False:
            i = 10
            return i + 15
        logging.info(f'[pool {event.address}] pool closed')

    def connection_created(self, event: monitoring.ConnectionCreatedEvent) -> None:
        if False:
            return 10
        logging.info(f'[pool {event.address}][conn #{event.connection_id}] connection created')

    def connection_ready(self, event: monitoring.ConnectionReadyEvent) -> None:
        if False:
            print('Hello World!')
        logging.info(f'[pool {event.address}][conn #{event.connection_id}] connection setup succeeded')

    def connection_closed(self, event: monitoring.ConnectionClosedEvent) -> None:
        if False:
            i = 10
            return i + 15
        logging.info(f'[pool {event.address}][conn #{event.connection_id}] connection closed, reason: "{event.reason}"')

    def connection_check_out_started(self, event: monitoring.ConnectionCheckOutStartedEvent) -> None:
        if False:
            return 10
        logging.info(f'[pool {event.address}] connection check out started')

    def connection_check_out_failed(self, event: monitoring.ConnectionCheckOutFailedEvent) -> None:
        if False:
            i = 10
            return i + 15
        logging.info(f'[pool {event.address}] connection check out failed, reason: {event.reason}')

    def connection_checked_out(self, event: monitoring.ConnectionCheckedOutEvent) -> None:
        if False:
            print('Hello World!')
        logging.info(f'[pool {event.address}][conn #{event.connection_id}] connection checked out of pool')

    def connection_checked_in(self, event: monitoring.ConnectionCheckedInEvent) -> None:
        if False:
            i = 10
            return i + 15
        logging.info(f'[pool {event.address}][conn #{event.connection_id}] connection checked into pool')