import asyncio
import datetime
import json
import logging
import os
import socket
import sys
import traceback
import warnings
import psutil
from typing import List, Optional, Tuple
from collections import defaultdict
import ray
import ray._private.services
import ray._private.utils
from ray.dashboard.consts import GCS_RPC_TIMEOUT_SECONDS, COMPONENT_METRICS_TAG_KEYS
from ray.dashboard.modules.reporter.profile_manager import CpuProfilingManager
import ray.dashboard.modules.reporter.reporter_consts as reporter_consts
import ray.dashboard.utils as dashboard_utils
from opencensus.stats import stats as stats_module
import ray._private.prometheus_exporter as prometheus_exporter
from prometheus_client.core import REGISTRY
from ray._private.metrics_agent import Gauge, MetricsAgent, Record
from ray._private.ray_constants import DEBUG_AUTOSCALING_STATUS
from ray.core.generated import reporter_pb2, reporter_pb2_grpc
from ray.util.debug import log_once
from ray.dashboard import k8s_utils
from ray._raylet import WorkerID
logger = logging.getLogger(__name__)
enable_gpu_usage_check = True
IN_KUBERNETES_POD = 'KUBERNETES_SERVICE_HOST' in os.environ
ENABLE_K8S_DISK_USAGE = os.environ.get('RAY_DASHBOARD_ENABLE_K8S_DISK_USAGE') == '1'
IN_CONTAINER = os.path.exists('/sys/fs/cgroup')
try:
    import gpustat.core as gpustat
except ModuleNotFoundError:
    gpustat = None
    if log_once('gpustat_import_warning'):
        warnings.warn('`gpustat` package is not installed. GPU monitoring is not available. To have full functionality of the dashboard please install `pip install ray[default]`.)')
except ImportError as e:
    gpustat = None
    if log_once('gpustat_import_warning'):
        warnings.warn('Importing gpustat failed, fix this to have full functionality of the dashboard. The original error was:\n\n' + e.msg)

def recursive_asdict(o):
    if False:
        print('Hello World!')
    if isinstance(o, tuple) and hasattr(o, '_asdict'):
        return recursive_asdict(o._asdict())
    if isinstance(o, (tuple, list)):
        L = []
        for k in o:
            L.append(recursive_asdict(k))
        return L
    if isinstance(o, dict):
        D = {k: recursive_asdict(v) for (k, v) in o.items()}
        return D
    return o

def jsonify_asdict(o) -> str:
    if False:
        while True:
            i = 10
    return json.dumps(dashboard_utils.to_google_style(recursive_asdict(o)))
METRICS_GAUGES = {'node_cpu_utilization': Gauge('node_cpu_utilization', 'Total CPU usage on a ray node', 'percentage', ['ip', 'SessionName']), 'node_cpu_count': Gauge('node_cpu_count', 'Total CPUs available on a ray node', 'cores', ['ip', 'SessionName']), 'node_mem_used': Gauge('node_mem_used', 'Memory usage on a ray node', 'bytes', ['ip', 'SessionName']), 'node_mem_available': Gauge('node_mem_available', 'Memory available on a ray node', 'bytes', ['ip', 'SessionName']), 'node_mem_total': Gauge('node_mem_total', 'Total memory on a ray node', 'bytes', ['ip', 'SessionName']), 'node_mem_shared_bytes': Gauge('node_mem_shared_bytes', 'Total shared memory usage on a ray node', 'bytes', ['ip', 'SessionName']), 'node_gpus_available': Gauge('node_gpus_available', 'Total GPUs available on a ray node', 'percentage', ['ip', 'SessionName', 'GpuDeviceName', 'GpuIndex']), 'node_gpus_utilization': Gauge('node_gpus_utilization', 'Total GPUs usage on a ray node', 'percentage', ['ip', 'SessionName', 'GpuDeviceName', 'GpuIndex']), 'node_gram_used': Gauge('node_gram_used', 'Total GPU RAM usage on a ray node', 'bytes', ['ip', 'SessionName', 'GpuDeviceName', 'GpuIndex']), 'node_gram_available': Gauge('node_gram_available', 'Total GPU RAM available on a ray node', 'bytes', ['ip', 'SessionName', 'GpuDeviceName', 'GpuIndex']), 'node_disk_io_read': Gauge('node_disk_io_read', 'Total read from disk', 'bytes', ['ip', 'SessionName']), 'node_disk_io_write': Gauge('node_disk_io_write', 'Total written to disk', 'bytes', ['ip', 'SessionName']), 'node_disk_io_read_count': Gauge('node_disk_io_read_count', 'Total read ops from disk', 'io', ['ip', 'SessionName']), 'node_disk_io_write_count': Gauge('node_disk_io_write_count', 'Total write ops to disk', 'io', ['ip', 'SessionName']), 'node_disk_io_read_speed': Gauge('node_disk_io_read_speed', 'Disk read speed', 'bytes/sec', ['ip', 'SessionName']), 'node_disk_io_write_speed': Gauge('node_disk_io_write_speed', 'Disk write speed', 'bytes/sec', ['ip', 'SessionName']), 'node_disk_read_iops': Gauge('node_disk_read_iops', 'Disk read iops', 'iops', ['ip', 'SessionName']), 'node_disk_write_iops': Gauge('node_disk_write_iops', 'Disk write iops', 'iops', ['ip', 'SessionName']), 'node_disk_usage': Gauge('node_disk_usage', 'Total disk usage (bytes) on a ray node', 'bytes', ['ip', 'SessionName']), 'node_disk_free': Gauge('node_disk_free', 'Total disk free (bytes) on a ray node', 'bytes', ['ip', 'SessionName']), 'node_disk_utilization_percentage': Gauge('node_disk_utilization_percentage', 'Total disk utilization (percentage) on a ray node', 'percentage', ['ip', 'SessionName']), 'node_network_sent': Gauge('node_network_sent', 'Total network sent', 'bytes', ['ip', 'SessionName']), 'node_network_received': Gauge('node_network_received', 'Total network received', 'bytes', ['ip', 'SessionName']), 'node_network_send_speed': Gauge('node_network_send_speed', 'Network send speed', 'bytes/sec', ['ip', 'SessionName']), 'node_network_receive_speed': Gauge('node_network_receive_speed', 'Network receive speed', 'bytes/sec', ['ip', 'SessionName']), 'component_cpu_percentage': Gauge('component_cpu_percentage', 'Total CPU usage of the components on a node.', 'percentage', COMPONENT_METRICS_TAG_KEYS), 'component_mem_shared_bytes': Gauge('component_mem_shared_bytes', "SHM usage of all components of the node. It is equivalent to the top command's SHR column.", 'bytes', COMPONENT_METRICS_TAG_KEYS), 'component_rss_mb': Gauge('component_rss_mb', 'RSS usage of all components on the node.', 'MB', COMPONENT_METRICS_TAG_KEYS), 'component_uss_mb': Gauge('component_uss_mb', 'USS usage of all components on the node.', 'MB', COMPONENT_METRICS_TAG_KEYS), 'component_num_fds': Gauge('component_num_fds', 'Number of open fds of all components on the node.', 'count', COMPONENT_METRICS_TAG_KEYS), 'cluster_active_nodes': Gauge('cluster_active_nodes', 'Active nodes on the cluster', 'count', ['node_type', 'SessionName']), 'cluster_failed_nodes': Gauge('cluster_failed_nodes', 'Failed nodes on the cluster', 'count', ['node_type', 'SessionName']), 'cluster_pending_nodes': Gauge('cluster_pending_nodes', 'Pending nodes on the cluster', 'count', ['node_type', 'SessionName'])}

class ReporterAgent(dashboard_utils.DashboardAgentModule, reporter_pb2_grpc.ReporterServiceServicer):
    """A monitor process for monitoring Ray nodes.

    Attributes:
        dashboard_agent: The DashboardAgent object contains global config
    """

    def __init__(self, dashboard_agent):
        if False:
            print('Hello World!')
        'Initialize the reporter object.'
        super().__init__(dashboard_agent)
        if IN_KUBERNETES_POD or IN_CONTAINER:
            logical_cpu_count = ray._private.utils.get_num_cpus(override_docker_cpu_warning=True)
            physical_cpu_count = psutil.cpu_count(logical=False)
        else:
            logical_cpu_count = psutil.cpu_count()
            physical_cpu_count = psutil.cpu_count(logical=False)
        self._cpu_counts = (logical_cpu_count, physical_cpu_count)
        self._gcs_aio_client = dashboard_agent.gcs_aio_client
        self._ip = dashboard_agent.ip
        self._log_dir = dashboard_agent.log_dir
        self._is_head_node = self._ip == dashboard_agent.gcs_address.split(':')[0]
        self._hostname = socket.gethostname()
        self._workers = {}
        self._raylet_proc = None
        self._agent_proc = None
        self._latest_worker_proc_names = set()
        self._network_stats_hist = [(0, (0.0, 0.0))]
        self._disk_io_stats_hist = [(0, (0.0, 0.0, 0, 0))]
        self._metrics_collection_disabled = dashboard_agent.metrics_collection_disabled
        self._metrics_agent = None
        self._session_name = dashboard_agent.session_name
        if not self._metrics_collection_disabled:
            try:
                stats_exporter = prometheus_exporter.new_stats_exporter(prometheus_exporter.Options(namespace='ray', port=dashboard_agent.metrics_export_port, address='127.0.0.1' if self._ip == '127.0.0.1' else ''))
            except Exception:
                logger.exception('Failed to start prometheus stats exporter. Agent will stay alive but disable the stats.')
                stats_exporter = None
            self._metrics_agent = MetricsAgent(stats_module.stats.view_manager, stats_module.stats.stats_recorder, stats_exporter)
            if self._metrics_agent.proxy_exporter_collector:
                REGISTRY.register(self._metrics_agent.proxy_exporter_collector)
        self._key = f'{reporter_consts.REPORTER_PREFIX}{self._dashboard_agent.node_id}'

    async def GetTraceback(self, request, context):
        pid = request.pid
        native = request.native
        p = CpuProfilingManager(self._log_dir)
        (success, output) = await p.trace_dump(pid, native=native)
        return reporter_pb2.GetTracebackReply(output=output, success=success)

    async def CpuProfiling(self, request, context):
        pid = request.pid
        duration = request.duration
        format = request.format
        native = request.native
        p = CpuProfilingManager(self._log_dir)
        (success, output) = await p.cpu_profile(pid, format=format, duration=duration, native=native)
        return reporter_pb2.CpuProfilingReply(output=output, success=success)

    async def ReportOCMetrics(self, request, context):
        if self._metrics_collection_disabled:
            return reporter_pb2.ReportOCMetricsReply()
        try:
            worker_id = WorkerID(request.worker_id)
            worker_id = None if worker_id.is_nil() else worker_id.hex()
            self._metrics_agent.proxy_export_metrics(request.metrics, worker_id)
        except Exception:
            logger.error(traceback.format_exc())
        return reporter_pb2.ReportOCMetricsReply()

    @staticmethod
    def _get_cpu_percent(in_k8s: bool):
        if False:
            while True:
                i = 10
        if in_k8s:
            return k8s_utils.cpu_percent()
        else:
            return psutil.cpu_percent()

    @staticmethod
    def _get_gpu_usage():
        if False:
            for i in range(10):
                print('nop')
        global enable_gpu_usage_check
        if gpustat is None or not enable_gpu_usage_check:
            return []
        gpu_utilizations = []
        gpus = []
        try:
            gpus = gpustat.new_query().gpus
        except Exception as e:
            logger.debug(f'gpustat failed to retrieve GPU information: {e}')
            if type(e).__name__ == 'NVMLError_DriverNotLoaded':
                enable_gpu_usage_check = False
        for gpu in gpus:
            gpu_data = {'_'.join(key.split('.')): val for (key, val) in gpu.entry.items()}
            gpu_utilizations.append(gpu_data)
        return gpu_utilizations

    @staticmethod
    def _get_boot_time():
        if False:
            while True:
                i = 10
        if IN_KUBERNETES_POD:
            return psutil.Process(pid=1).create_time()
        else:
            return psutil.boot_time()

    @staticmethod
    def _get_network_stats():
        if False:
            while True:
                i = 10
        ifaces = [v for (k, v) in psutil.net_io_counters(pernic=True).items() if k[0] == 'e']
        sent = sum((iface.bytes_sent for iface in ifaces))
        recv = sum((iface.bytes_recv for iface in ifaces))
        return (sent, recv)

    @staticmethod
    def _get_mem_usage():
        if False:
            for i in range(10):
                print('nop')
        total = ray._private.utils.get_system_memory()
        used = ray._private.utils.get_used_memory()
        available = total - used
        percent = round(used / total, 3) * 100
        return (total, available, percent, used)

    @staticmethod
    def _get_disk_usage():
        if False:
            for i in range(10):
                print('nop')
        if IN_KUBERNETES_POD and (not ENABLE_K8S_DISK_USAGE):
            return {'/': psutil._common.sdiskusage(total=1, used=0, free=1, percent=0.0)}
        if sys.platform == 'win32':
            root = psutil.disk_partitions()[0].mountpoint
        else:
            root = os.sep
        tmp = ray._private.utils.get_user_temp_dir()
        return {'/': psutil.disk_usage(root), tmp: psutil.disk_usage(tmp)}

    @staticmethod
    def _get_disk_io_stats():
        if False:
            for i in range(10):
                print('nop')
        stats = psutil.disk_io_counters()
        if not stats:
            return (0, 0, 0, 0)
        else:
            return (stats.read_bytes, stats.write_bytes, stats.read_count, stats.write_count)

    def _get_agent_proc(self) -> psutil.Process:
        if False:
            for i in range(10):
                print('nop')
        return psutil.Process()

    def _generate_worker_key(self, proc: psutil.Process) -> Tuple[int, float]:
        if False:
            print('Hello World!')
        return (proc.pid, proc.create_time())

    def _get_workers(self):
        if False:
            for i in range(10):
                print('nop')
        raylet_proc = self._get_raylet_proc()
        if raylet_proc is None:
            return []
        else:
            workers = {self._generate_worker_key(proc): proc for proc in raylet_proc.children()}
            keys_to_pop = []
            for (key, worker) in workers.items():
                if key not in self._workers:
                    self._workers[key] = worker
            for key in self._workers:
                if key not in workers:
                    keys_to_pop.append(key)
            for k in keys_to_pop:
                self._workers.pop(k)
            self._workers.pop(self._generate_worker_key(self._get_agent_proc()))
            result = []
            for w in self._workers.values():
                try:
                    if w.status() == psutil.STATUS_ZOMBIE:
                        continue
                except psutil.NoSuchProcess:
                    continue
                result.append(w.as_dict(attrs=['pid', 'create_time', 'cpu_percent', 'cpu_times', 'cmdline', 'memory_info', 'memory_full_info', 'num_fds']))
            return result

    def _get_raylet_proc(self):
        if False:
            print('Hello World!')
        try:
            if not self._raylet_proc:
                curr_proc = psutil.Process()
                self._raylet_proc = curr_proc.parent()
            if self._raylet_proc is not None:
                if self._raylet_proc.pid == 1:
                    return None
                if self._raylet_proc.status() == psutil.STATUS_ZOMBIE:
                    return None
            return self._raylet_proc
        except (psutil.AccessDenied, ProcessLookupError):
            pass
        return None

    def _get_raylet(self):
        if False:
            while True:
                i = 10
        raylet_proc = self._get_raylet_proc()
        if raylet_proc is None:
            return {}
        else:
            return raylet_proc.as_dict(attrs=['pid', 'create_time', 'cpu_percent', 'cpu_times', 'cmdline', 'memory_info', 'memory_full_info', 'num_fds'])

    def _get_agent(self):
        if False:
            while True:
                i = 10
        if not self._agent_proc:
            self._agent_proc = psutil.Process()
        return self._agent_proc.as_dict(attrs=['pid', 'create_time', 'cpu_percent', 'cpu_times', 'cmdline', 'memory_info', 'memory_full_info', 'num_fds'])

    def _get_load_avg(self):
        if False:
            for i in range(10):
                print('nop')
        if sys.platform == 'win32':
            cpu_percent = psutil.cpu_percent()
            load = (cpu_percent, cpu_percent, cpu_percent)
        else:
            load = os.getloadavg()
        if self._cpu_counts[0] > 0:
            per_cpu_load = tuple((round(x / self._cpu_counts[0], 2) for x in load))
        else:
            per_cpu_load = None
        return (load, per_cpu_load)

    @staticmethod
    def _compute_speed_from_hist(hist):
        if False:
            print('Hello World!')
        while len(hist) > 7:
            hist.pop(0)
        (then, prev_stats) = hist[0]
        (now, now_stats) = hist[-1]
        time_delta = now - then
        return tuple(((y - x) / time_delta for (x, y) in zip(prev_stats, now_stats)))

    def _get_shm_usage(self):
        if False:
            i = 10
            return i + 15
        "Return the shm usage.\n\n        If shm doesn't exist (e.g., MacOS), it returns None.\n        "
        mem = psutil.virtual_memory()
        if not hasattr(mem, 'shared'):
            return None
        return mem.shared

    def _get_all_stats(self):
        if False:
            while True:
                i = 10
        now = dashboard_utils.to_posix_time(datetime.datetime.utcnow())
        network_stats = self._get_network_stats()
        self._network_stats_hist.append((now, network_stats))
        network_speed_stats = self._compute_speed_from_hist(self._network_stats_hist)
        disk_stats = self._get_disk_io_stats()
        self._disk_io_stats_hist.append((now, disk_stats))
        disk_speed_stats = self._compute_speed_from_hist(self._disk_io_stats_hist)
        return {'now': now, 'hostname': self._hostname, 'ip': self._ip, 'cpu': self._get_cpu_percent(IN_KUBERNETES_POD), 'cpus': self._cpu_counts, 'mem': self._get_mem_usage(), 'shm': self._get_shm_usage(), 'workers': self._get_workers(), 'raylet': self._get_raylet(), 'agent': self._get_agent(), 'bootTime': self._get_boot_time(), 'loadAvg': self._get_load_avg(), 'disk': self._get_disk_usage(), 'disk_io': disk_stats, 'disk_io_speed': disk_speed_stats, 'gpus': self._get_gpu_usage(), 'network': network_stats, 'network_speed': network_speed_stats, 'cmdline': self._get_raylet().get('cmdline', [])}

    def _generate_reseted_stats_record(self, component_name: str) -> List[Record]:
        if False:
            while True:
                i = 10
        'Return a list of Record that will reset\n        the system metrics of a given component name.\n\n        Args:\n            component_name: a component name for a given stats.\n\n        Returns:\n            a list of Record instances of all values 0.\n        '
        tags = {'ip': self._ip, 'Component': component_name}
        records = []
        records.append(Record(gauge=METRICS_GAUGES['component_cpu_percentage'], value=0.0, tags=tags))
        records.append(Record(gauge=METRICS_GAUGES['component_mem_shared_bytes'], value=0.0, tags=tags))
        records.append(Record(gauge=METRICS_GAUGES['component_rss_mb'], value=0.0, tags=tags))
        records.append(Record(gauge=METRICS_GAUGES['component_uss_mb'], value=0.0, tags=tags))
        records.append(Record(gauge=METRICS_GAUGES['component_num_fds'], value=0, tags=tags))
        return records

    def _generate_system_stats_record(self, stats: List[dict], component_name: str, pid: Optional[str]=None) -> List[Record]:
        if False:
            for i in range(10):
                print('nop')
        'Generate a list of Record class from a given component names.\n\n        Args:\n            stats: a list of stats dict generated by `psutil.as_dict`.\n                If empty, it will create the metrics of a given "component_name"\n                which has all 0 values.\n            component_name: a component name for a given stats.\n            pid: optionally provided pids.\n\n        Returns:\n            a list of Record class that will be exposed to Prometheus.\n        '
        total_cpu_percentage = 0.0
        total_rss = 0.0
        total_uss = 0.0
        total_shm = 0.0
        total_num_fds = 0
        for stat in stats:
            total_cpu_percentage += float(stat.get('cpu_percent', 0.0))
            memory_info = stat.get('memory_info')
            if memory_info:
                mem = stat['memory_info']
                total_rss += float(mem.rss) / 1000000.0
                if hasattr(mem, 'shared'):
                    total_shm += float(mem.shared)
            mem_full_info = stat.get('memory_full_info')
            if mem_full_info is not None:
                total_uss += float(mem_full_info.uss) / 1000000.0
            total_num_fds += int(stat.get('num_fds', 0))
        tags = {'ip': self._ip, 'Component': component_name}
        if pid:
            tags['pid'] = pid
        records = []
        records.append(Record(gauge=METRICS_GAUGES['component_cpu_percentage'], value=total_cpu_percentage, tags=tags))
        records.append(Record(gauge=METRICS_GAUGES['component_mem_shared_bytes'], value=total_shm, tags=tags))
        records.append(Record(gauge=METRICS_GAUGES['component_rss_mb'], value=total_rss, tags=tags))
        if total_uss > 0.0:
            records.append(Record(gauge=METRICS_GAUGES['component_uss_mb'], value=total_uss, tags=tags))
        records.append(Record(gauge=METRICS_GAUGES['component_num_fds'], value=total_num_fds, tags=tags))
        return records

    def generate_worker_stats_record(self, worker_stats: List[dict]) -> List[Record]:
        if False:
            while True:
                i = 10
        'Generate a list of Record class for worker proceses.\n\n        This API automatically sets the component_name of record as\n        the name of worker processes. I.e., ray::* so that we can report\n        per task/actor (grouped by a func/class name) resource usages.\n\n        Args:\n            stats: a list of stats dict generated by `psutil.as_dict`\n                for worker processes.\n        '
        proc_name_to_stats = defaultdict(list)
        for stat in worker_stats:
            cmdline = stat.get('cmdline')
            if cmdline and len(cmdline) > 0 and cmdline[0].startswith('ray::'):
                proc_name = cmdline[0]
                proc_name_to_stats[proc_name].append(stat)
        records = []
        for (proc_name, stats) in proc_name_to_stats.items():
            records.extend(self._generate_system_stats_record(stats, proc_name))
        new_proc_names = set(proc_name_to_stats.keys())
        stale_procs = self._latest_worker_proc_names - new_proc_names
        self._latest_worker_proc_names = new_proc_names
        for stale_proc_name in stale_procs:
            records.extend(self._generate_reseted_stats_record(stale_proc_name))
        return records

    def _record_stats(self, stats, cluster_stats):
        if False:
            while True:
                i = 10
        records_reported = []
        ip = stats['ip']
        if 'autoscaler_report' in cluster_stats and self._is_head_node:
            active_nodes = cluster_stats['autoscaler_report']['active_nodes']
            for (node_type, active_node_count) in active_nodes.items():
                records_reported.append(Record(gauge=METRICS_GAUGES['cluster_active_nodes'], value=active_node_count, tags={'node_type': node_type}))
            failed_nodes = cluster_stats['autoscaler_report']['failed_nodes']
            failed_nodes_dict = {}
            for (node_ip, node_type) in failed_nodes:
                if node_type in failed_nodes_dict:
                    failed_nodes_dict[node_type] += 1
                else:
                    failed_nodes_dict[node_type] = 1
            for (node_type, failed_node_count) in failed_nodes_dict.items():
                records_reported.append(Record(gauge=METRICS_GAUGES['cluster_failed_nodes'], value=failed_node_count, tags={'node_type': node_type}))
            pending_nodes = cluster_stats['autoscaler_report']['pending_nodes']
            pending_nodes_dict = {}
            for (node_ip, node_type, status_message) in pending_nodes:
                if node_type in pending_nodes_dict:
                    pending_nodes_dict[node_type] += 1
                else:
                    pending_nodes_dict[node_type] = 1
            for (node_type, pending_node_count) in pending_nodes_dict.items():
                records_reported.append(Record(gauge=METRICS_GAUGES['cluster_pending_nodes'], value=pending_node_count, tags={'node_type': node_type}))
        cpu_usage = float(stats['cpu'])
        cpu_record = Record(gauge=METRICS_GAUGES['node_cpu_utilization'], value=cpu_usage, tags={'ip': ip})
        (cpu_count, _) = stats['cpus']
        cpu_count_record = Record(gauge=METRICS_GAUGES['node_cpu_count'], value=cpu_count, tags={'ip': ip})
        (mem_total, mem_available, _, mem_used) = stats['mem']
        mem_used_record = Record(gauge=METRICS_GAUGES['node_mem_used'], value=mem_used, tags={'ip': ip})
        mem_available_record = Record(gauge=METRICS_GAUGES['node_mem_available'], value=mem_available, tags={'ip': ip})
        mem_total_record = Record(gauge=METRICS_GAUGES['node_mem_total'], value=mem_total, tags={'ip': ip})
        shm_used = stats['shm']
        if shm_used:
            node_mem_shared = Record(gauge=METRICS_GAUGES['node_mem_shared_bytes'], value=shm_used, tags={'ip': ip})
            records_reported.append(node_mem_shared)
        "\n        {'index': 0,\n        'uuid': 'GPU-36e1567d-37ed-051e-f8ff-df807517b396',\n        'name': 'NVIDIA A10G',\n        'temperature_gpu': 20,\n        'fan_speed': 0,\n        'utilization_gpu': 1,\n        'utilization_enc': 0,\n        'utilization_dec': 0,\n        'power_draw': 51,\n        'enforced_power_limit': 300,\n        'memory_used': 0,\n        'memory_total': 22731,\n        'processes': []}\n        "
        gpus = stats['gpus']
        gpus_available = len(gpus)
        if gpus_available:
            gpu_tags = {'ip': ip}
            for gpu in gpus:
                (gpus_utilization, gram_used, gram_total) = (0, 0, 0)
                if gpu['utilization_gpu'] is not None:
                    gpus_utilization += gpu['utilization_gpu']
                gram_used += gpu['memory_used']
                gram_total += gpu['memory_total']
                gpu_index = gpu.get('index')
                gpu_name = gpu.get('name')
                gram_available = gram_total - gram_used
                if gpu_index is not None:
                    gpu_tags = {'ip': ip, 'GpuIndex': str(gpu_index)}
                    if gpu_name:
                        gpu_tags['GpuDeviceName'] = gpu_name
                    gpus_available_record = Record(gauge=METRICS_GAUGES['node_gpus_available'], value=1, tags=gpu_tags)
                    gpus_utilization_record = Record(gauge=METRICS_GAUGES['node_gpus_utilization'], value=gpus_utilization, tags=gpu_tags)
                    gram_used_record = Record(gauge=METRICS_GAUGES['node_gram_used'], value=gram_used, tags=gpu_tags)
                    gram_available_record = Record(gauge=METRICS_GAUGES['node_gram_available'], value=gram_available, tags=gpu_tags)
                    records_reported.extend([gpus_available_record, gpus_utilization_record, gram_used_record, gram_available_record])
        disk_io_stats = stats['disk_io']
        disk_read_record = Record(gauge=METRICS_GAUGES['node_disk_io_read'], value=disk_io_stats[0], tags={'ip': ip})
        disk_write_record = Record(gauge=METRICS_GAUGES['node_disk_io_write'], value=disk_io_stats[1], tags={'ip': ip})
        disk_read_count_record = Record(gauge=METRICS_GAUGES['node_disk_io_read_count'], value=disk_io_stats[2], tags={'ip': ip})
        disk_write_count_record = Record(gauge=METRICS_GAUGES['node_disk_io_write_count'], value=disk_io_stats[3], tags={'ip': ip})
        disk_io_speed_stats = stats['disk_io_speed']
        disk_read_speed_record = Record(gauge=METRICS_GAUGES['node_disk_io_read_speed'], value=disk_io_speed_stats[0], tags={'ip': ip})
        disk_write_speed_record = Record(gauge=METRICS_GAUGES['node_disk_io_write_speed'], value=disk_io_speed_stats[1], tags={'ip': ip})
        disk_read_iops_record = Record(gauge=METRICS_GAUGES['node_disk_read_iops'], value=disk_io_speed_stats[2], tags={'ip': ip})
        disk_write_iops_record = Record(gauge=METRICS_GAUGES['node_disk_write_iops'], value=disk_io_speed_stats[3], tags={'ip': ip})
        used = stats['disk']['/'].used
        free = stats['disk']['/'].free
        disk_utilization = float(used / (used + free)) * 100
        disk_usage_record = Record(gauge=METRICS_GAUGES['node_disk_usage'], value=used, tags={'ip': ip})
        disk_free_record = Record(gauge=METRICS_GAUGES['node_disk_free'], value=free, tags={'ip': ip})
        disk_utilization_percentage_record = Record(gauge=METRICS_GAUGES['node_disk_utilization_percentage'], value=disk_utilization, tags={'ip': ip})
        network_stats = stats['network']
        network_sent_record = Record(gauge=METRICS_GAUGES['node_network_sent'], value=network_stats[0], tags={'ip': ip})
        network_received_record = Record(gauge=METRICS_GAUGES['node_network_received'], value=network_stats[1], tags={'ip': ip})
        network_speed_stats = stats['network_speed']
        network_send_speed_record = Record(gauge=METRICS_GAUGES['node_network_send_speed'], value=network_speed_stats[0], tags={'ip': ip})
        network_receive_speed_record = Record(gauge=METRICS_GAUGES['node_network_receive_speed'], value=network_speed_stats[1], tags={'ip': ip})
        '\n        Record system stats.\n        '
        raylet_stats = stats['raylet']
        if raylet_stats:
            raylet_pid = str(raylet_stats['pid'])
            records_reported.extend(self._generate_system_stats_record([raylet_stats], 'raylet', pid=raylet_pid))
        workers_stats = stats['workers']
        records_reported.extend(self.generate_worker_stats_record(workers_stats))
        agent_stats = stats['agent']
        if agent_stats:
            agent_pid = str(agent_stats['pid'])
            records_reported.extend(self._generate_system_stats_record([agent_stats], 'agent', pid=agent_pid))
        records_reported.extend([cpu_record, cpu_count_record, mem_used_record, mem_available_record, mem_total_record, disk_read_record, disk_write_record, disk_read_count_record, disk_write_count_record, disk_read_speed_record, disk_write_speed_record, disk_read_iops_record, disk_write_iops_record, disk_usage_record, disk_free_record, disk_utilization_percentage_record, network_sent_record, network_received_record, network_send_speed_record, network_receive_speed_record])
        return records_reported

    async def _perform_iteration(self, publisher):
        """Get any changes to the log files and push updates to kv."""
        while True:
            try:
                formatted_status_string = await self._gcs_aio_client.internal_kv_get(DEBUG_AUTOSCALING_STATUS.encode(), None, timeout=GCS_RPC_TIMEOUT_SECONDS)
                stats = self._get_all_stats()
                if not self._metrics_collection_disabled:
                    cluster_stats = json.loads(formatted_status_string.decode()) if formatted_status_string else {}
                    records_reported = self._record_stats(stats, cluster_stats)
                    self._metrics_agent.record_and_export(records_reported, global_tags={'SessionName': self._session_name})
                    self._metrics_agent.clean_all_dead_worker_metrics()
                await publisher.publish_resource_usage(self._key, jsonify_asdict(stats))
            except Exception:
                logger.exception('Error publishing node physical stats.')
            await asyncio.sleep(reporter_consts.REPORTER_UPDATE_INTERVAL_MS / 1000)

    async def run(self, server):
        if server:
            reporter_pb2_grpc.add_ReporterServiceServicer_to_server(self, server)
        await self._perform_iteration(self._dashboard_agent.publisher)

    @staticmethod
    def is_minimal_module():
        if False:
            for i in range(10):
                print('nop')
        return False