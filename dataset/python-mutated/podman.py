"""Podman Extension unit for Glances' Containers plugin."""
from datetime import datetime
from glances.globals import iterkeys, itervalues, nativestr, pretty_date, string_value_to_float
from glances.logger import logger
from glances.plugins.containers.stats_streamer import StatsStreamer
try:
    from podman import PodmanClient
except Exception as e:
    import_podman_error_tag = True
    logger.warning('Error loading Podman deps Lib. Podman feature in the Containers plugin is disabled ({})'.format(e))
else:
    import_podman_error_tag = False

class PodmanContainerStatsFetcher:
    MANDATORY_FIELDS = ['CPU', 'MemUsage', 'MemLimit', 'NetInput', 'NetOutput', 'BlockInput', 'BlockOutput']

    def __init__(self, container):
        if False:
            for i in range(10):
                print('nop')
        self._container = container
        stats_iterable = container.stats(decode=True)
        self._streamer = StatsStreamer(stats_iterable, initial_stream_value={})

    def _log_debug(self, msg, exception=None):
        if False:
            return 10
        logger.debug('containers (Podman) ID: {} - {} ({})'.format(self._container.id, msg, exception))
        logger.debug(self._streamer.stats)

    def stop(self):
        if False:
            while True:
                i = 10
        self._streamer.stop()

    @property
    def stats(self):
        if False:
            return 10
        stats = self._streamer.stats
        if stats['Error']:
            self._log_debug('Stats fetching failed', stats['Error'])
        return stats['Stats'][0]

    @property
    def activity_stats(self):
        if False:
            i = 10
            return i + 15
        result_stats = {'cpu': {}, 'memory': {}, 'io': {}, 'network': {}}
        api_stats = self.stats
        if any((field not in api_stats for field in self.MANDATORY_FIELDS)):
            self._log_debug('Missing mandatory fields')
            return result_stats
        try:
            cpu_usage = float(api_stats.get('CPU', 0))
            mem_usage = float(api_stats['MemUsage'])
            mem_limit = float(api_stats['MemLimit'])
            rx = float(api_stats['NetInput'])
            tx = float(api_stats['NetOutput'])
            ior = float(api_stats['BlockInput'])
            iow = float(api_stats['BlockOutput'])
            result_stats = {'cpu': {'total': cpu_usage}, 'memory': {'usage': mem_usage, 'limit': mem_limit}, 'io': {'ior': ior, 'iow': iow, 'time_since_update': 1}, 'network': {'rx': rx, 'tx': tx, 'time_since_update': 1}}
        except ValueError as e:
            self._log_debug('Non float stats values found', e)
        return result_stats

class PodmanPodStatsFetcher:

    def __init__(self, pod_manager):
        if False:
            print('Hello World!')
        self._pod_manager = pod_manager
        stats_iterable = (pod_manager.stats(decode=True) for _ in iter(int, 1))
        self._streamer = StatsStreamer(stats_iterable, initial_stream_value={}, sleep_duration=2)

    def _log_debug(self, msg, exception=None):
        if False:
            for i in range(10):
                print('nop')
        logger.debug('containers (Podman): Pod Manager - {} ({})'.format(msg, exception))
        logger.debug(self._streamer.stats)

    def stop(self):
        if False:
            print('Hello World!')
        self._streamer.stop()

    @property
    def activity_stats(self):
        if False:
            while True:
                i = 10
        result_stats = {}
        container_stats = self._streamer.stats
        for stat in container_stats:
            io_stats = self._get_io_stats(stat)
            cpu_stats = self._get_cpu_stats(stat)
            memory_stats = self._get_memory_stats(stat)
            network_stats = self._get_network_stats(stat)
            computed_stats = {'name': stat['Name'], 'cid': stat['CID'], 'pod_id': stat['Pod'], 'io': io_stats or {}, 'memory': memory_stats or {}, 'network': network_stats or {}, 'cpu': cpu_stats or {'total': 0.0}}
            result_stats[stat['CID']] = computed_stats
        return result_stats

    def _get_cpu_stats(self, stats):
        if False:
            i = 10
            return i + 15
        "Return the container CPU usage.\n\n        Output: a dict {'total': 1.49}\n        "
        if 'CPU' not in stats:
            self._log_debug('Missing CPU usage fields')
            return None
        cpu_usage = string_value_to_float(stats['CPU'].rstrip('%'))
        return {'total': cpu_usage}

    def _get_memory_stats(self, stats):
        if False:
            while True:
                i = 10
        "Return the container MEMORY.\n\n        Output: a dict {'rss': 1015808, 'cache': 356352,  'usage': ..., 'max_usage': ...}\n        "
        if 'MemUsage' not in stats or '/' not in stats['MemUsage']:
            self._log_debug('Missing MEM usage fields')
            return None
        memory_usage_str = stats['MemUsage']
        (usage_str, limit_str) = memory_usage_str.split('/')
        try:
            usage = string_value_to_float(usage_str)
            limit = string_value_to_float(limit_str)
        except ValueError as e:
            self._log_debug('Compute MEM usage failed', e)
            return None
        return {'usage': usage, 'limit': limit}

    def _get_network_stats(self, stats):
        if False:
            print('Hello World!')
        "Return the container network usage using the Docker API (v1.0 or higher).\n\n        Output: a dict {'time_since_update': 3000, 'rx': 10, 'tx': 65}.\n        with:\n            time_since_update: number of seconds elapsed between the latest grab\n            rx: Number of bytes received\n            tx: Number of bytes transmitted\n        "
        if 'NetIO' not in stats or '/' not in stats['NetIO']:
            self._log_debug('Compute MEM usage failed')
            return None
        net_io_str = stats['NetIO']
        (rx_str, tx_str) = net_io_str.split('/')
        try:
            rx = string_value_to_float(rx_str)
            tx = string_value_to_float(tx_str)
        except ValueError as e:
            self._log_debug('Compute MEM usage failed', e)
            return None
        return {'rx': rx, 'tx': tx, 'time_since_update': 1}

    def _get_io_stats(self, stats):
        if False:
            i = 10
            return i + 15
        "Return the container IO usage using the Docker API (v1.0 or higher).\n\n        Output: a dict {'time_since_update': 3000, 'ior': 10, 'iow': 65}.\n        with:\n            time_since_update: number of seconds elapsed between the latest grab\n            ior: Number of bytes read\n            iow: Number of bytes written\n        "
        if 'BlockIO' not in stats or '/' not in stats['BlockIO']:
            self._log_debug('Missing BlockIO usage fields')
            return None
        block_io_str = stats['BlockIO']
        (ior_str, iow_str) = block_io_str.split('/')
        try:
            ior = string_value_to_float(ior_str)
            iow = string_value_to_float(iow_str)
        except ValueError as e:
            self._log_debug('Compute BlockIO usage failed', e)
            return None
        return {'ior': ior, 'iow': iow, 'time_since_update': 1}

class PodmanContainersExtension:
    """Glances' Containers Plugin's Docker Extension unit"""
    CONTAINER_ACTIVE_STATUS = ['running', 'paused']

    def __init__(self, podman_sock):
        if False:
            while True:
                i = 10
        if import_podman_error_tag:
            raise Exception('Missing libs required to run Podman Extension (Containers)')
        self.client = None
        self.ext_name = 'containers (Podman)'
        self.podman_sock = podman_sock
        self.pods_stats_fetcher = None
        self.container_stats_fetchers = {}
        self.connect()

    def connect(self):
        if False:
            for i in range(10):
                print('nop')
        'Connect to Podman.'
        try:
            self.client = PodmanClient(base_url=self.podman_sock)
            self.client.ping()
        except Exception as e:
            logger.error("{} plugin - Can't connect to Podman ({})".format(self.ext_name, e))
            self.client = None

    def update_version(self):
        if False:
            for i in range(10):
                print('nop')
        return {}

    def stop(self):
        if False:
            return 10
        for t in itervalues(self.container_stats_fetchers):
            t.stop()
        if self.pods_stats_fetcher:
            self.pods_stats_fetcher.stop()

    def update(self, all_tag):
        if False:
            while True:
                i = 10
        'Update Podman stats using the input method.'
        if not self.client:
            return ({}, [])
        version_stats = self.update_version()
        try:
            containers = self.client.containers.list(all=all_tag)
            if not self.pods_stats_fetcher:
                self.pods_stats_fetcher = PodmanPodStatsFetcher(self.client.pods)
        except Exception as e:
            logger.error("{} plugin - Can't get containers list ({})".format(self.ext_name, e))
            return (version_stats, [])
        for container in containers:
            if container.id not in self.container_stats_fetchers:
                logger.debug('{} plugin - Create thread for container {}'.format(self.ext_name, container.id[:12]))
                self.container_stats_fetchers[container.id] = PodmanContainerStatsFetcher(container)
        absent_containers = set(iterkeys(self.container_stats_fetchers)) - set((c.id for c in containers))
        for container_id in absent_containers:
            logger.debug('{} plugin - Stop thread for old container {}'.format(self.ext_name, container_id[:12]))
            self.container_stats_fetchers[container_id].stop()
            del self.container_stats_fetchers[container_id]
        container_stats = [self.generate_stats(container) for container in containers]
        pod_stats = self.pods_stats_fetcher.activity_stats
        for stats in container_stats:
            if stats['Id'][:12] in pod_stats:
                stats['pod_name'] = pod_stats[stats['Id'][:12]]['name']
                stats['pod_id'] = pod_stats[stats['Id'][:12]]['pod_id']
        return (version_stats, container_stats)

    @property
    def key(self):
        if False:
            i = 10
            return i + 15
        'Return the key of the list.'
        return 'name'

    def generate_stats(self, container):
        if False:
            i = 10
            return i + 15
        stats = {'key': self.key, 'name': nativestr(container.name), 'Id': container.id, 'Image': str(container.image.tags), 'Status': container.attrs['State'], 'Created': container.attrs['Created'], 'Command': container.attrs.get('Command') or []}
        if stats['Status'] in self.CONTAINER_ACTIVE_STATUS:
            started_at = datetime.fromtimestamp(container.attrs['StartedAt'])
            stats_fetcher = self.container_stats_fetchers[container.id]
            activity_stats = stats_fetcher.activity_stats
            stats.update(activity_stats)
            stats['cpu_percent'] = stats['cpu']['total']
            stats['memory_usage'] = stats['memory'].get('usage')
            if stats['memory'].get('cache') is not None:
                stats['memory_usage'] -= stats['memory']['cache']
            stats['io_r'] = stats['io'].get('ior')
            stats['io_w'] = stats['io'].get('iow')
            stats['network_rx'] = stats['network'].get('rx')
            stats['network_tx'] = stats['network'].get('tx')
            stats['Uptime'] = pretty_date(started_at)
        else:
            stats['io'] = {}
            stats['cpu'] = {}
            stats['memory'] = {}
            stats['network'] = {}
            stats['io_r'] = None
            stats['io_w'] = None
            stats['cpu_percent'] = None
            stats['memory_percent'] = None
            stats['network_rx'] = None
            stats['network_tx'] = None
            stats['Uptime'] = None
        return stats