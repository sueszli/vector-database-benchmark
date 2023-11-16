import logging
import asyncio
import datetime
import re
import redis
import time
from datetime import datetime as dt
from django.core.management.base import BaseCommand
from django.db import connection
from django.db.migrations.executor import MigrationExecutor
from awx.main.analytics.broadcast_websocket import RelayWebsocketStatsManager, safe_name
from awx.main.wsrelay import WebSocketRelayManager
logger = logging.getLogger('awx.main.wsrelay')

class Command(BaseCommand):
    help = 'Launch the websocket broadcaster'

    def add_arguments(self, parser):
        if False:
            i = 10
            return i + 15
        parser.add_argument('--status', dest='status', action='store_true', help='print the internal state of any running broadcast websocket')

    @classmethod
    def display_len(cls, s):
        if False:
            i = 10
            return i + 15
        return len(re.sub('\x1b.*?m', '', s))

    @classmethod
    def _format_lines(cls, host_stats, padding=5):
        if False:
            while True:
                i = 10
        widths = [0 for i in host_stats[0]]
        for entry in host_stats:
            for (i, e) in enumerate(entry):
                if Command.display_len(e) > widths[i]:
                    widths[i] = Command.display_len(e)
        paddings = [padding for i in widths]
        lines = []
        for entry in host_stats:
            line = ''
            for (pad, width, value) in zip(paddings, widths, entry):
                if len(value) > Command.display_len(value):
                    width += len(value) - Command.display_len(value)
                total_width = width + pad
                line += f'{value:{total_width}}'
            lines.append(line)
        return lines

    @classmethod
    def get_connection_status(cls, hostnames, data):
        if False:
            return 10
        host_stats = [('hostname', 'state', 'start time', 'duration (sec)')]
        for h in hostnames:
            connection_color = '91'
            h_safe = safe_name(h)
            prefix = f'awx_{h_safe}'
            connection_state = data.get(f'{prefix}_connection', 'N/A')
            connection_started = 'N/A'
            connection_duration = 'N/A'
            if connection_state is None:
                connection_state = 'unknown'
            if connection_state == 'connected':
                connection_color = '92'
                connection_started = data.get(f'{prefix}_connection_start', 'Error')
                if connection_started != 'Error':
                    connection_started = datetime.datetime.fromtimestamp(connection_started)
                    connection_duration = int((dt.now() - connection_started).total_seconds())
            connection_state = f'\x1b[{connection_color}m{connection_state}\x1b[0m'
            host_stats.append((h, connection_state, str(connection_started), str(connection_duration)))
        return host_stats

    @classmethod
    def get_connection_stats(cls, hostnames, data):
        if False:
            i = 10
            return i + 15
        host_stats = [('hostname', 'total', 'per minute')]
        for h in hostnames:
            h_safe = safe_name(h)
            prefix = f'awx_{h_safe}'
            messages_total = data.get(f'{prefix}_messages_received', '0')
            messages_per_minute = data.get(f'{prefix}_messages_received_per_minute', '0')
            host_stats.append((h, str(int(messages_total)), str(int(messages_per_minute))))
        return host_stats

    def handle(self, *arg, **options):
        if False:
            for i in range(10):
                print('nop')
        from awx.main.models.ha import Instance
        try:
            executor = MigrationExecutor(connection)
            migrating = bool(executor.migration_plan(executor.loader.graph.leaf_nodes()))
            connection.close()
        except Exception as exc:
            logger.warning(f'Error on startup of run_wsrelay (error: {exc}), retry in 10s...')
            time.sleep(10)
            return
        if migrating:
            logger.info('AWX is currently migrating, retry in 10s...')
            time.sleep(10)
            return
        try:
            my_hostname = Instance.objects.my_hostname()
            logger.info('Active instance with hostname {} is registered.'.format(my_hostname))
        except RuntimeError as e:
            logger.info('Unable to return currently active instance: {}, retry in 5s...'.format(e))
            time.sleep(5)
            return
        if options.get('status'):
            try:
                stats_all = RelayWebsocketStatsManager.get_stats_sync()
            except redis.exceptions.ConnectionError as e:
                print(f'Unable to get Relay Websocket Status. Failed to connect to redis {e}')
                return
            data = {}
            for family in stats_all:
                if family.type == 'gauge' and len(family.samples) > 1:
                    for sample in family.samples:
                        if sample.value >= 1:
                            data[family.name] = sample.labels[family.name]
                            break
                else:
                    data[family.name] = family.samples[0].value
            my_hostname = Instance.objects.my_hostname()
            hostnames = [i.hostname for i in Instance.objects.exclude(hostname=my_hostname)]
            host_stats = Command.get_connection_status(hostnames, data)
            lines = Command._format_lines(host_stats)
            print(f'Relay websocket connection status from "{my_hostname}" to:')
            print('\n'.join(lines))
            host_stats = Command.get_connection_stats(hostnames, data)
            lines = Command._format_lines(host_stats)
            print(f'\nRelay websocket connection stats from "{my_hostname}" to:')
            print('\n'.join(lines))
            return
        try:
            websocket_relay_manager = WebSocketRelayManager()
            asyncio.run(websocket_relay_manager.run())
        except KeyboardInterrupt:
            logger.info('Terminating Websocket Relayer')