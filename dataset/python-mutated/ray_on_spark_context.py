import os
import re
import subprocess
import time
import uuid
import random
import warnings
import tempfile
import filelock
import multiprocessing
from packaging import version
from bigdl.orca.ray.process import session_execute, ProcessMonitor
from bigdl.orca.ray.utils import is_local
from bigdl.orca.ray.utils import resource_to_bytes
from bigdl.orca.ray.utils import get_parent_pid
from bigdl.dllib.utils.log4Error import invalidInputError
from typing import TYPE_CHECKING
from typing import Any, Dict, Optional, List
if TYPE_CHECKING:
    from pyspark.context import SparkContext

def kill_redundant_log_monitors(redis_address: str) -> None:
    if False:
        while True:
            i = 10
    '\n    Killing redundant log_monitor.py processes.\n    If multiple ray nodes are started on the same machine,\n    there will be multiple ray log_monitor.py processes\n    monitoring the same log dir. As a result, the logs\n    will be replicated multiple times and forwarded to driver.\n    See issue https://github.com/ray-project/ray/issues/10392\n    '
    import psutil
    import subprocess
    log_monitor_processes = []
    for proc in psutil.process_iter(['name', 'cmdline']):
        try:
            if proc.name() is None or proc.name() == 'lwsslauncher':
                continue
            cmdline = subprocess.list2cmdline(proc.cmdline())
            is_log_monitor = 'log_monitor.py' in cmdline
            is_same_redis = '--redis-address={}'.format(redis_address)
            if is_log_monitor and is_same_redis in cmdline:
                log_monitor_processes.append(proc)
        except (psutil.AccessDenied, psutil.ZombieProcess, psutil.ProcessLookupError):
            if psutil.MACOS:
                continue
            else:
                invalidInputError(False, 'List process with list2cmdline failed!')
    if len(log_monitor_processes) > 1:
        for proc in log_monitor_processes[1:]:
            proc.kill()

class RayServiceFuncGenerator(object):
    """
    This should be a pickable class.
    """

    def _prepare_env(self):
        if False:
            print('Hello World!')
        modified_env = os.environ.copy()
        if self.python_loc == 'python_env/bin/python':
            executor_python_path = '{}/{}'.format(os.getcwd(), '/'.join(self.python_loc.split('/')[:-1]))
        else:
            executor_python_path = '/'.join(self.python_loc.split('/')[:-1])
        if 'PATH' in os.environ:
            modified_env['PATH'] = '{}:{}'.format(executor_python_path, os.environ['PATH'])
        else:
            modified_env['PATH'] = executor_python_path
        modified_env.pop('MALLOC_ARENA_MAX', None)
        modified_env.pop('RAY_BACKEND_LOG_LEVEL', None)
        modified_env.pop('intra_op_parallelism_threads', None)
        modified_env.pop('inter_op_parallelism_threads', None)
        modified_env.pop('OMP_NUM_THREADS', None)
        modified_env.pop('KMP_BLOCKTIME', None)
        modified_env.pop('KMP_AFFINITY', None)
        modified_env.pop('KMP_SETTINGS', None)
        modified_env.pop('PYTHONHOME', None)
        if self.env:
            modified_env.update(self.env)
        if self.verbose:
            print('Executing with these environment settings:')
            for pair in modified_env.items():
                print(pair)
            print('The $PATH is: {}'.format(modified_env['PATH']))
        return modified_env

    def __init__(self, python_loc, redis_port, redis_password, ray_node_cpu_cores, object_store_memory, verbose=False, env=None, include_webui=False, extra_params=None, system_config=None):
        if False:
            for i in range(10):
                print('nop')
        'object_store_memory: integer in bytes'
        self.env = env
        self.python_loc = python_loc
        self.redis_port = redis_port
        self.redis_password = redis_password
        self.ray_node_cpu_cores = ray_node_cpu_cores
        self.ray_exec = self._get_ray_exec()
        self.object_store_memory = object_store_memory
        self.extra_params = extra_params
        self.system_config = system_config
        self.include_webui = include_webui
        self.verbose = verbose
        self.labels = '--resources \'{"_mxnet_worker": %s, "_mxnet_server": %s, "_reserved": %s}\'' % (1, 1, 2)
        tag = uuid.uuid4().hex
        self.ray_master_flag = 'ray_master_{}'.format(tag)
        self.ray_master_lock = 'ray_master_start_{}.lock'.format(tag)
        self.raylet_lock = 'raylet_start_{}.lock'.format(tag)

    def gen_stop(self):
        if False:
            i = 10
            return i + 15

        def _stop(iter):
            if False:
                print('Hello World!')
            command = '{} stop'.format(self.ray_exec)
            print('Start to end the ray services: {}'.format(command))
            session_execute(command=command, fail_fast=True)
            return iter
        return _stop

    @staticmethod
    def _enrich_command(command, object_store_memory, extra_params):
        if False:
            i = 10
            return i + 15
        if object_store_memory:
            command = command + ' --object-store-memory {}'.format(str(object_store_memory))
        if extra_params:
            for (k, v) in extra_params.items():
                kw = k.replace('_', '-')
                command = command + ' --{} {}'.format(kw, v)
        return command

    def _gen_master_command(self):
        if False:
            print('Hello World!')
        webui = 'true' if self.include_webui else 'false'
        command = '{} start --head --include-dashboard {} --dashboard-host 0.0.0.0 --port {} --num-cpus {}'.format(self.ray_exec, webui, self.redis_port, self.ray_node_cpu_cores)
        if self.redis_password:
            command = command + ' --redis-password {}'.format(self.redis_password)
        if self.labels:
            command = command + ' ' + self.labels
        if self.system_config:
            import json
            command = command + ' ' + "--system-config='" + json.dumps(self.system_config) + "'"
        return RayServiceFuncGenerator._enrich_command(command=command, object_store_memory=self.object_store_memory, extra_params=self.extra_params)

    @staticmethod
    def _get_raylet_command(redis_address, ray_exec, redis_password, ray_node_cpu_cores, labels='', object_store_memory=None, extra_params=None):
        if False:
            while True:
                i = 10
        command = '{} start --address {} --num-cpus {}'.format(ray_exec, redis_address, ray_node_cpu_cores)
        if redis_password:
            command = command + ' --redis-password {}'.format(redis_password)
        if labels:
            command = command + ' ' + labels
        return RayServiceFuncGenerator._enrich_command(command=command, object_store_memory=object_store_memory, extra_params=extra_params)

    @staticmethod
    def _get_spark_executor_pid():
        if False:
            i = 10
            return i + 15
        this_pid = os.getpid()
        pyspark_daemon_pid = get_parent_pid(this_pid)
        spark_executor_pid = get_parent_pid(pyspark_daemon_pid)
        return spark_executor_pid

    @staticmethod
    def start_ray_daemon(python_loc, pid_to_watch, pgid_to_kill):
        if False:
            i = 10
            return i + 15
        daemon_path = os.path.join(os.path.dirname(__file__), 'ray_daemon.py')
        invalidInputError(pid_to_watch > 0, 'pid_to_watch should be a positive integer.')
        invalidInputError(pgid_to_kill > 0, 'pgid_to_kill should be a positive integer.')
        start_daemon_command = ['nohup', python_loc, daemon_path, str(pid_to_watch), str(pgid_to_kill)]
        subprocess.Popen(start_daemon_command, preexec_fn=os.setpgrp)
        time.sleep(1)

    def _start_ray_node(self, command, tag):
        if False:
            print('Hello World!')
        modified_env = self._prepare_env()
        print('Starting {} by running: {}'.format(tag, command))
        process_info = session_execute(command=command, env=modified_env, tag=tag)
        spark_executor_pid = RayServiceFuncGenerator._get_spark_executor_pid()
        RayServiceFuncGenerator.start_ray_daemon(self.python_loc, pid_to_watch=spark_executor_pid, pgid_to_kill=process_info.pgid)
        import ray._private.services as rservices
        process_info.node_ip = rservices.get_node_ip_address()
        return process_info

    def _get_ray_exec(self):
        if False:
            return 10
        if 'envs' in self.python_loc:
            python_bin_dir = '/'.join(self.python_loc.split('/')[:-1])
            return '{}/python {}/ray'.format(python_bin_dir, python_bin_dir)
        elif self.python_loc == 'python_env/bin/python':
            return 'python_env/bin/python python_env/bin/ray'
        else:
            return 'ray'

    def gen_ray_master_start(self):
        if False:
            for i in range(10):
                print('nop')

        def _start_ray_master(index, iter):
            if False:
                i = 10
                return i + 15
            from bigdl.dllib.utils.utils import get_node_ip
            process_info = None
            if index == 0:
                print('partition id is : {}'.format(index))
                current_ip = get_node_ip()
                print('master address {}'.format(current_ip))
                redis_address = '{}:{}'.format(current_ip, self.redis_port)
                process_info = self._start_ray_node(command=self._gen_master_command(), tag='ray-master')
                process_info.master_addr = redis_address
            yield process_info
        return _start_ray_master

    def gen_raylet_start(self, redis_address):
        if False:
            return 10

        def _start_raylets(iter):
            if False:
                i = 10
                return i + 15
            from bigdl.dllib.utils.utils import get_node_ip
            current_ip = get_node_ip()
            master_ip = redis_address.split(':')[0]
            do_start = True
            process_info = None
            base_path = tempfile.gettempdir()
            ray_master_flag_path = os.path.join(base_path, self.ray_master_flag)
            if current_ip == master_ip:
                ray_master_lock_path = os.path.join(base_path, self.ray_master_lock)
                with filelock.FileLock(ray_master_lock_path):
                    if not os.path.exists(ray_master_flag_path):
                        os.mknod(ray_master_flag_path)
                        do_start = False
            if do_start:
                raylet_lock_path = os.path.join(base_path, self.raylet_lock)
                with filelock.FileLock(raylet_lock_path):
                    process_info = self._start_ray_node(command=RayServiceFuncGenerator._get_raylet_command(redis_address=redis_address, ray_exec=self.ray_exec, redis_password=self.redis_password, ray_node_cpu_cores=self.ray_node_cpu_cores, labels=self.labels, object_store_memory=self.object_store_memory, extra_params=self.extra_params), tag='raylet')
                    kill_redundant_log_monitors(redis_address=redis_address)
            yield process_info
        return _start_raylets

    def gen_ray_start(self, master_ip):
        if False:
            i = 10
            return i + 15

        def _start_ray_services(iter):
            if False:
                i = 10
                return i + 15
            from pyspark import BarrierTaskContext
            from bigdl.dllib.utils.utils import get_node_ip
            tc = BarrierTaskContext.get()
            current_ip = get_node_ip()
            print('current address {}'.format(current_ip))
            print('master address {}'.format(master_ip))
            redis_address = '{}:{}'.format(master_ip, self.redis_port)
            process_info = None
            base_path = tempfile.gettempdir()
            ray_master_flag_path = os.path.join(base_path, self.ray_master_flag)
            if current_ip == master_ip:
                ray_master_lock_path = os.path.join(base_path, self.ray_master_lock)
                with filelock.FileLock(ray_master_lock_path):
                    if not os.path.exists(ray_master_flag_path):
                        print('partition id is : {}'.format(tc.partitionId()))
                        process_info = self._start_ray_node(command=self._gen_master_command(), tag='ray-master')
                        process_info.master_addr = redis_address
                        os.mknod(ray_master_flag_path)
            tc.barrier()
            if not process_info:
                raylet_lock_path = os.path.join(base_path, self.raylet_lock)
                with filelock.FileLock(raylet_lock_path):
                    print('partition id is : {}'.format(tc.partitionId()))
                    process_info = self._start_ray_node(command=RayServiceFuncGenerator._get_raylet_command(redis_address=redis_address, ray_exec=self.ray_exec, redis_password=self.redis_password, ray_node_cpu_cores=self.ray_node_cpu_cores, labels=self.labels, object_store_memory=self.object_store_memory, extra_params=self.extra_params), tag='raylet')
                    kill_redundant_log_monitors(redis_address=redis_address)
            if os.path.exists(ray_master_flag_path):
                os.remove(ray_master_flag_path)
            yield process_info
        return _start_ray_services

class RayOnSparkContext(object):
    _active_ray_context = None

    def __init__(self, sc: 'SparkContext', redis_port: Optional[int]=None, redis_password: Optional[str]=None, object_store_memory: Optional[str]=None, verbose: bool=False, env: Optional[Dict[str, str]]=None, extra_params: Optional[Dict[str, Any]]=None, include_webui: bool=True, num_ray_nodes: Optional[int]=None, ray_node_cpu_cores: Optional[int]=None, system_config: Optional[Dict[str, str]]=None):
        if False:
            i = 10
            return i + 15
        '\n        The RayOnSparkContext would initiate a ray cluster on top of the configuration of\n        SparkContext.\n        After creating RayOnSparkContext, call the init method to set up the cluster.\n        - For Spark local mode: The total available cores for Ray is equal to the number of\n        Spark local cores.\n        - For Spark cluster mode: The number of raylets to be created is equal to the number of\n        Spark executors. The number of cores allocated for each raylet is equal to the number of\n        cores for each Spark executor.\n        You are allowed to specify num_ray_nodes and ray_node_cpu_cores for configurations\n        to start raylets.\n        :param sc: An instance of SparkContext.\n        :param redis_port: The redis port for the ray head node. Default is None.\n        The value would be randomly picked if not specified.\n        :param redis_password: The password for redis. Default to be None if not specified.\n        :param object_store_memory: The memory size for ray object_store in string.\n        This can be specified in bytes(b), kilobytes(k), megabytes(m) or gigabytes(g).\n        For example, "50b", "100k", "250m", "30g".\n        :param verbose: True for more logs when starting ray. Default is False.\n        :param env: The environment variable dict for running ray processes. Default is None.\n        :param extra_params: The key value dict for extra options to launch ray.\n        For example, extra_params={"dashboard-port": "11281", "temp-dir": "/tmp/ray/"}.\n        :param include_webui: Default is True for including web ui when starting ray.\n        :param num_ray_nodes: The number of ray processes to start across the cluster.\n        For Spark local mode, you don\'t need to specify this value.\n        For Spark cluster mode, it is default to be the number of Spark executors. If\n        spark.executor.instances can\'t be detected in your SparkContext, you need to explicitly\n        specify this. It is recommended that num_ray_nodes is not larger than the number of\n        Spark executors to make sure there are enough resources in your cluster.\n        :param ray_node_cpu_cores: The number of available cores for each ray process.\n        For Spark local mode, it is default to be the number of Spark local cores.\n        For Spark cluster mode, it is default to be the number of cores for each Spark executor. If\n        spark.executor.cores or spark.cores.max can\'t be detected in your SparkContext, you need to\n        explicitly specify this. It is recommended that ray_node_cpu_cores is not larger than the\n        number of cores for each Spark executor to make sure there are enough resources in your\n        cluster.\n        :param system_config: The key value dict for overriding RayConfig defaults. Mainly for\n        testing purposes. An example for system_config could be:\n        {"object_spilling_config":"{"type":"filesystem",\n                                   "params":{"directory_path":"/tmp/spill"}}"}\n        '
        invalidInputError(sc is not None, 'sc cannot be None, please create a SparkContext first')
        self.sc = sc
        self.initialized = False
        self.is_local = is_local(sc)
        self.verbose = verbose
        self.redis_password = redis_password
        self.object_store_memory = resource_to_bytes(object_store_memory)
        self.ray_processesMonitor = None
        self.env = env
        self.extra_params = extra_params
        self.system_config = system_config
        if extra_params:
            invalidInputError(isinstance(extra_params, dict), 'extra_params should be a dict for extra options to launch ray')
            if self.system_config:
                self.extra_params.pop('system_config', None)
                self.extra_params.pop('_system_config', None)
            elif 'system_config' in self.extra_params:
                self.system_config = self.extra_params.pop('system_config')
            elif '_system_config' in self.extra_params:
                self.system_config = self.extra_params.pop('_system_config')
        self.include_webui = include_webui
        self._address_info = None
        self.redis_port = random.randint(20000, 65535) if not redis_port else int(redis_port)
        self.ray_node_cpu_cores = ray_node_cpu_cores
        self.num_ray_nodes = num_ray_nodes
        RayOnSparkContext._active_ray_context = self

    def setup(self) -> None:
        if False:
            i = 10
            return i + 15
        if self.is_local:
            self.num_ray_nodes = 1
            spark_cores = self._get_spark_local_cores()
            if self.ray_node_cpu_cores:
                ray_node_cpu_cores = int(self.ray_node_cpu_cores)
                if ray_node_cpu_cores > spark_cores:
                    warnings.warn('ray_node_cpu_cores is larger than available Spark cores, make sure there are enough resources on your machine')
                self.ray_node_cpu_cores = ray_node_cpu_cores
            else:
                self.ray_node_cpu_cores = spark_cores
        else:
            if self.sc.getConf().contains('spark.executor.cores'):
                executor_cores = int(self.sc.getConf().get('spark.executor.cores'))
            else:
                executor_cores = None
            if self.ray_node_cpu_cores:
                ray_node_cpu_cores = int(self.ray_node_cpu_cores)
                if executor_cores and ray_node_cpu_cores > executor_cores:
                    warnings.warn('ray_node_cpu_cores is larger than Spark executor cores, make sure there are enough resources on your cluster')
                self.ray_node_cpu_cores = ray_node_cpu_cores
            elif executor_cores:
                self.ray_node_cpu_cores = executor_cores
            else:
                invalidInputError(False, 'spark.executor.cores not detected in the SparkContext, you need to manually specify num_ray_nodes and ray_node_cpu_cores for RayOnSparkContext to start ray services')
            if self.sc.getConf().contains('spark.executor.instances'):
                num_executors = int(self.sc.getConf().get('spark.executor.instances'))
            elif self.sc.getConf().contains('spark.cores.max'):
                import math
                num_executors = math.floor(int(self.sc.getConf().get('spark.cores.max')) / self.ray_node_cpu_cores)
            else:
                num_executors = None
            if self.num_ray_nodes:
                num_ray_nodes = int(self.num_ray_nodes)
                if num_executors and num_ray_nodes > num_executors:
                    warnings.warn('num_ray_nodes is larger than the number of Spark executors, make sure there are enough resources on your cluster')
                self.num_ray_nodes = num_ray_nodes
            elif num_executors:
                self.num_ray_nodes = num_executors
            else:
                invalidInputError(False, 'spark.executor.cores not detected in the SparkContext, you need to manually specify num_ray_nodes and ray_node_cpu_cores for RayOnSparkContext to start ray services')
            from bigdl.dllib.utils.utils import detect_python_location
            self.python_loc = os.environ.get('PYSPARK_PYTHON', detect_python_location())
            self.ray_service = RayServiceFuncGenerator(python_loc=self.python_loc, redis_port=self.redis_port, redis_password=self.redis_password, ray_node_cpu_cores=self.ray_node_cpu_cores, object_store_memory=self.object_store_memory, verbose=self.verbose, env=self.env, include_webui=self.include_webui, extra_params=self.extra_params, system_config=self.system_config)
        self.total_cores = self.num_ray_nodes * self.ray_node_cpu_cores

    @classmethod
    def get(cls, initialize: bool=True) -> Optional['RayOnSparkContext']:
        if False:
            print('Hello World!')
        if RayOnSparkContext._active_ray_context:
            ray_ctx = RayOnSparkContext._active_ray_context
            if initialize and (not ray_ctx.initialized):
                ray_ctx.init()
            return ray_ctx
        else:
            invalidInputError(False, 'No active RayOnSparkContext. Please create a RayOnSparkContext and init it first')
        return None

    def _gather_cluster_ips(self) -> List[str]:
        if False:
            print('Hello World!')
        '\n        Get the ips of all Spark executors in the cluster. The first ip returned would be the\n        ray master.\n        '

        def info_fn(iter):
            if False:
                i = 10
                return i + 15
            from bigdl.dllib.utils.utils import get_node_ip
            yield get_node_ip()
        ips = self.sc.range(0, self.total_cores, numSlices=self.total_cores).mapPartitions(info_fn).collect()
        ips = list(set(ips))
        return ips

    def stop(self) -> None:
        if False:
            print('Hello World!')
        if not self.initialized:
            print('The Ray cluster has not been launched.')
            return
        import ray
        ray.shutdown()
        self.initialized = False

    def purge(self) -> None:
        if False:
            while True:
                i = 10
        '\n        Invoke ray stop to clean ray processes.\n        '
        if not self.initialized:
            print('The Ray cluster has not been launched.')
            return
        if self.is_local:
            import ray
            ray.shutdown()
        else:
            self.sc.range(0, self.total_cores, numSlices=self.total_cores).mapPartitions(self.ray_service.gen_stop()).collect()
        self.initialized = False

    def _get_spark_local_cores(self) -> int:
        if False:
            while True:
                i = 10
        local_symbol = re.match('local\\[(.*)\\]', self.sc.master).group(1)
        if local_symbol == '*':
            return multiprocessing.cpu_count()
        else:
            return int(local_symbol)

    def _update_extra_params(self, extra_params: Optional[Dict[str, str]]) -> Dict[str, str]:
        if False:
            return 10
        kwargs = {}
        if extra_params is not None:
            for (k, v) in extra_params.items():
                kw = k.replace('-', '_')
                kwargs[kw] = v
        return kwargs

    def init(self, driver_cores: int=0):
        if False:
            while True:
                i = 10
        "\n        Initiate the ray cluster.\n        :param driver_cores: The number of cores for the raylet on driver for Spark cluster mode.\n        Default is 0 and in this case the local driver wouldn't have any ray workload.\n        :return The dictionary of address information about the ray cluster.\n        Information contains node_ip_address, redis_address, object_store_address,\n        raylet_socket_name, webui_url and session_dir.\n        "
        if self.initialized:
            print('The Ray cluster has been launched.')
        else:
            self.setup()
            if self.is_local:
                if self.env:
                    os.environ.update(self.env)
                import ray
                kwargs = self._update_extra_params(self.extra_params)
                init_params = dict(num_cpus=self.ray_node_cpu_cores, object_store_memory=self.object_store_memory, include_dashboard=self.include_webui, dashboard_host='0.0.0.0', _system_config=self.system_config, namespace='bigdl')
                if self.redis_password:
                    init_params['_redis_password'] = self.redis_password
                init_params.update(kwargs)
                self._address_info = ray.init(**init_params)
            else:
                self.cluster_ips = self._gather_cluster_ips()
                redis_address = self._start_cluster()
                self._address_info = self._start_driver(num_cores=driver_cores, redis_address=redis_address)
            print(self._address_info)
            kill_redundant_log_monitors(self._address_info['redis_address'])
            self.initialized = True
        return self._address_info

    @property
    def address_info(self):
        if False:
            for i in range(10):
                print('nop')
        if self._address_info:
            return self._address_info
        else:
            invalidInputError(False, 'The Ray cluster has not been launched yet. Please call init first')

    @property
    def redis_address(self) -> str:
        if False:
            i = 10
            return i + 15
        return self.address_info['redis_address']

    def _start_cluster(self) -> str:
        if False:
            while True:
                i = 10
        ray_rdd = self.sc.range(0, self.num_ray_nodes, numSlices=self.num_ray_nodes)
        from bigdl.dllib.nncontext import ZooContext
        if ZooContext.barrier_mode:
            print('Launching Ray on cluster with Spark barrier mode')
            process_infos = ray_rdd.barrier().mapPartitions(self.ray_service.gen_ray_start(self.cluster_ips[0])).collect()
        else:
            print('Launching Ray on cluster without Spark barrier mode')
            master_process_infos = ray_rdd.mapPartitionsWithIndex(self.ray_service.gen_ray_master_start()).collect()
            master_process_infos = [process for process in master_process_infos if process]
            invalidInputError(len(master_process_infos) == 1, 'There should be only one ray master launched, but got {}'.format(len(master_process_infos)))
            master_process_info = master_process_infos[0]
            redis_address = master_process_info.master_addr
            raylet_process_infos = ray_rdd.mapPartitions(self.ray_service.gen_raylet_start(redis_address)).collect()
            raylet_process_infos = [process for process in raylet_process_infos if process]
            invalidInputError(len(raylet_process_infos) == self.num_ray_nodes - 1, 'There should be {} raylets launched across the cluster, but got {}'.format(self.num_ray_nodes - 1, len(raylet_process_infos)))
            process_infos = master_process_infos + raylet_process_infos
        self.ray_processesMonitor = ProcessMonitor(process_infos, self.sc, ray_rdd, self, verbose=self.verbose)
        return self.ray_processesMonitor.master.master_addr

    def _start_restricted_worker(self, num_cores: int, node_ip_address: str, redis_address: str) -> None:
        if False:
            return 10
        extra_param = {'node-ip-address': node_ip_address}
        if self.extra_params is not None:
            extra_param.update(self.extra_params)
        command = RayServiceFuncGenerator._get_raylet_command(redis_address=redis_address, ray_exec='ray', redis_password=self.redis_password, ray_node_cpu_cores=num_cores, object_store_memory=self.object_store_memory, extra_params=extra_param)
        modified_env = self.ray_service._prepare_env()
        print('Executing command: {}'.format(command))
        process_info = session_execute(command=command, env=modified_env, tag='raylet', fail_fast=True)
        RayServiceFuncGenerator.start_ray_daemon('python', pid_to_watch=os.getpid(), pgid_to_kill=process_info.pgid)

    def _start_driver(self, num_cores: int, redis_address: str):
        if False:
            return 10
        print('Start to launch ray driver')
        import ray._private.services
        node_ip = ray._private.services.get_node_ip_address(redis_address)
        self._start_restricted_worker(num_cores=num_cores, node_ip_address=node_ip, redis_address=redis_address)
        ray.shutdown()
        init_params = dict(address=redis_address, _node_ip_address=node_ip, namespace='bigdl')
        if self.redis_password:
            init_params['_redis_password'] = self.redis_password
        return ray.init(**init_params)