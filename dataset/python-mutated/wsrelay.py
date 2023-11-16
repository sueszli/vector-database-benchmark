import json
import logging
import asyncio
from typing import Dict
import ipaddress
import aiohttp
from aiohttp import client_exceptions
import aioredis
from channels.layers import get_channel_layer
from django.conf import settings
from django.apps import apps
import psycopg
from awx.main.analytics.broadcast_websocket import RelayWebsocketStats, RelayWebsocketStatsManager
import awx.main.analytics.subsystem_metrics as s_metrics
logger = logging.getLogger('awx.main.wsrelay')

def wrap_broadcast_msg(group, message: str):
    if False:
        print('Hello World!')
    return dict(group=group, message=message)

def get_local_host():
    if False:
        while True:
            i = 10
    Instance = apps.get_model('main', 'Instance')
    return Instance.objects.my_hostname()

class WebsocketRelayConnection:

    def __init__(self, name, stats: RelayWebsocketStats, remote_host: str, remote_port: int=settings.BROADCAST_WEBSOCKET_PORT, protocol: str=settings.BROADCAST_WEBSOCKET_PROTOCOL, verify_ssl: bool=settings.BROADCAST_WEBSOCKET_VERIFY_CERT):
        if False:
            print('Hello World!')
        self.name = name
        self.event_loop = asyncio.get_event_loop()
        self.stats = stats
        self.remote_host = remote_host
        self.remote_port = remote_port
        self.protocol = protocol
        self.verify_ssl = verify_ssl
        self.channel_layer = None
        self.subsystem_metrics = s_metrics.Metrics(instance_name=name)
        self.producers = dict()
        self.connected = False

    async def run_loop(self, websocket: aiohttp.ClientWebSocketResponse):
        raise RuntimeError('Implement me')

    async def connect(self):
        from awx.main.consumers import WebsocketSecretAuthHelper
        logger.debug(f'Connection attempt from {self.name} to {self.remote_host}')
        '\n        Can not put get_channel_layer() in the init code because it is in the init\n        path of channel layers i.e. RedisChannelLayer() calls our init code.\n        '
        if not self.channel_layer:
            self.channel_layer = get_channel_layer()
        uri_hostname = self.remote_host
        try:
            if isinstance(ipaddress.ip_address(uri_hostname), ipaddress.IPv6Address):
                uri_hostname = f'[{uri_hostname}]'
        except ValueError:
            pass
        uri = f'{self.protocol}://{uri_hostname}:{self.remote_port}/websocket/relay/'
        timeout = aiohttp.ClientTimeout(total=10)
        secret_val = WebsocketSecretAuthHelper.construct_secret()
        try:
            async with aiohttp.ClientSession(headers={'secret': secret_val}, timeout=timeout) as session:
                async with session.ws_connect(uri, ssl=self.verify_ssl, heartbeat=20) as websocket:
                    logger.info(f'Connection from {self.name} to {self.remote_host} established.')
                    self.stats.record_connection_established()
                    self.connected = True
                    await self.run_connection(websocket)
        except asyncio.CancelledError:
            logger.warning(f'Connection from {self.name} to {self.remote_host} cancelled.')
        except client_exceptions.ClientConnectorError as e:
            logger.warning(f"Connection from {self.name} to {self.remote_host} failed: '{e}'.", exc_info=True)
        except asyncio.TimeoutError:
            logger.warning(f'Connection from {self.name} to {self.remote_host} timed out.')
        except Exception as e:
            logger.warning(f"Connection from {self.name} to {self.remote_host} failed for unknown reason: '{e}'.", exc_info=True)
        else:
            logger.debug(f'Connection from {self.name} to {self.remote_host} lost, but no exception was raised.')
        finally:
            self.connected = False
            self.stats.record_connection_lost()

    def start(self):
        if False:
            return 10
        self.async_task = self.event_loop.create_task(self.connect())
        return self.async_task

    def cancel(self):
        if False:
            for i in range(10):
                print('nop')
        self.async_task.cancel()

    async def run_connection(self, websocket: aiohttp.ClientWebSocketResponse):
        producer = self.event_loop.create_task(self.run_producer('metrics', websocket, 'metrics'))
        self.producers['metrics'] = {'task': producer, 'subscriptions': {'metrics'}}
        async for msg in websocket:
            self.stats.record_message_received()
            if msg.type == aiohttp.WSMsgType.ERROR:
                break
            elif msg.type == aiohttp.WSMsgType.TEXT:
                try:
                    payload = json.loads(msg.data)
                except json.JSONDecodeError:
                    logmsg = 'Failed to decode message from web node'
                    if logger.isEnabledFor(logging.DEBUG):
                        logmsg = '{} {}'.format(logmsg, payload)
                    logger.warning(logmsg)
                    continue
            if payload.get('type') == 'consumer.subscribe':
                for group in payload['groups']:
                    name = f'{self.remote_host}-{group}'
                    origin_channel = payload['origin_channel']
                    if not self.producers.get(name):
                        producer = self.event_loop.create_task(self.run_producer(name, websocket, group))
                        self.producers[name] = {'task': producer, 'subscriptions': {origin_channel}}
                        logger.debug(f'Producer {name} started.')
                    else:
                        self.producers[name]['subscriptions'].add(origin_channel)
                        logger.debug(f'Connection from {self.name} to {self.remote_host} added subscription to {group}.')
            if payload.get('type') == 'consumer.unsubscribe':
                for group in payload['groups']:
                    name = f'{self.remote_host}-{group}'
                    origin_channel = payload['origin_channel']
                    try:
                        self.producers[name]['subscriptions'].remove(origin_channel)
                        logger.debug(f'Unsubscribed {origin_channel} from {name}')
                    except KeyError:
                        logger.warning(f'Producer {name} not found.')

    async def run_producer(self, name, websocket, group):
        try:
            logger.info(f'Starting producer for {name}')
            consumer_channel = await self.channel_layer.new_channel()
            await self.channel_layer.group_add(group, consumer_channel)
            logger.debug(f'Producer {name} added to group {group} and is now awaiting messages.')
            while True:
                try:
                    msg = await asyncio.wait_for(self.channel_layer.receive(consumer_channel), timeout=10)
                    if not msg.get('needs_relay'):
                        continue
                    msg = dict(msg)
                    del msg['needs_relay']
                except asyncio.TimeoutError:
                    current_subscriptions = self.producers[name]['subscriptions']
                    if len(current_subscriptions) == 0:
                        logger.info(f'Producer {name} has no subscribers, shutting down.')
                        return
                    continue
                except aioredis.errors.ConnectionClosedError:
                    logger.info(f'Producer {name} lost connection to Redis, shutting down.')
                    return
                await websocket.send_json(wrap_broadcast_msg(group, msg))
        except ConnectionResetError:
            logger.debug(f'Producer {name} connection reset.')
            pass
        except Exception:
            logger.exception(f'Event relay producer {name} crashed')
        finally:
            await self.channel_layer.group_discard(group, consumer_channel)
            del self.producers[name]

class WebSocketRelayManager(object):

    def __init__(self):
        if False:
            return 10
        self.local_hostname = get_local_host()
        self.relay_connections = dict()
        self.known_hosts: Dict[str, str] = dict()

    async def on_ws_heartbeat(self, conn):
        await conn.execute('LISTEN web_ws_heartbeat')
        async for notif in conn.notifies():
            if notif is None:
                continue
            try:
                if not notif.payload or notif.channel != 'web_ws_heartbeat':
                    return
                try:
                    payload = json.loads(notif.payload)
                except json.JSONDecodeError:
                    logmsg = 'Failed to decode message from pg_notify channel `web_ws_heartbeat`'
                    if logger.isEnabledFor(logging.DEBUG):
                        logmsg = '{} {}'.format(logmsg, payload)
                        logger.warning(logmsg)
                    return
                if payload.get('hostname') == self.local_hostname:
                    return
                action = payload.get('action')
                if action in ('online', 'offline'):
                    hostname = payload.get('hostname')
                    ip = payload.get('ip') or hostname
                    if ip is None:
                        logger.warning(f'Received invalid {action} ws_heartbeat, missing hostname and ip: {payload}')
                        return
                    logger.debug(f'Web host {hostname} ({ip}) {action} heartbeat received.')
                if action == 'online':
                    self.known_hosts[hostname] = ip
                elif action == 'offline':
                    await self.cleanup_offline_host(hostname)
            except Exception as e:
                logger.exception(f'on_ws_heartbeat exception: {e}')

    async def cleanup_offline_host(self, hostname):
        """
        Given a hostname, try to cancel its task/connection and remove it from
        the list of hosts we know about.
        If the host isn't in the list, assume that it was already deleted and
        don't error.
        """
        if hostname in self.relay_connections:
            self.relay_connections[hostname].cancel()
            try:
                await asyncio.wait_for(self.relay_connections[hostname].async_task, timeout=10)
            except asyncio.TimeoutError:
                logger.warning(f'Tried to cancel relay connection for {hostname} but it timed out during cleanup.')
            except asyncio.CancelledError:
                pass
            del self.relay_connections[hostname]
        if hostname in self.known_hosts:
            del self.known_hosts[hostname]
        try:
            self.stats_mgr.delete_remote_host_stats(hostname)
        except KeyError:
            pass

    async def run(self):
        event_loop = asyncio.get_running_loop()
        self.stats_mgr = RelayWebsocketStatsManager(event_loop, self.local_hostname)
        self.stats_mgr.start()
        database_conf = settings.DATABASES['default']
        async_conn = await psycopg.AsyncConnection.connect(dbname=database_conf['NAME'], host=database_conf['HOST'], user=database_conf['USER'], password=database_conf['PASSWORD'], port=database_conf['PORT'], **database_conf.get('OPTIONS', {}))
        await async_conn.set_autocommit(True)
        event_loop.create_task(self.on_ws_heartbeat(async_conn))
        while True:
            future_remote_hosts = self.known_hosts.keys()
            current_remote_hosts = self.relay_connections.keys()
            deleted_remote_hosts = set(current_remote_hosts) - set(future_remote_hosts)
            new_remote_hosts = set(future_remote_hosts) - set(current_remote_hosts)
            for (hostname, address) in self.known_hosts.items():
                if hostname not in self.relay_connections:
                    continue
                if address != self.relay_connections[hostname].remote_host:
                    deleted_remote_hosts.add(hostname)
                    new_remote_hosts.add(hostname)
            for (hostname, relay_conn) in self.relay_connections.items():
                if not relay_conn.connected:
                    deleted_remote_hosts.add(hostname)
            if deleted_remote_hosts:
                logger.info(f'Removing {deleted_remote_hosts} from websocket broadcast list')
                await asyncio.gather((self.cleanup_offline_host(h) for h in deleted_remote_hosts))
            if new_remote_hosts:
                logger.info(f'Adding {new_remote_hosts} to websocket broadcast list')
            for h in new_remote_hosts:
                stats = self.stats_mgr.new_remote_host_stats(h)
                relay_connection = WebsocketRelayConnection(name=self.local_hostname, stats=stats, remote_host=self.known_hosts[h])
                relay_connection.start()
                self.relay_connections[h] = relay_connection
            await asyncio.sleep(settings.BROADCAST_WEBSOCKET_NEW_INSTANCE_POLL_RATE_SECONDS)