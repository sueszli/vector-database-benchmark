import logging
import os
import subprocess
import time
import ray
from ray._private import ray_constants
from ray._private.ray_logging import setup_component_logger
from ray._private.services import get_node_ip_address
from ray._private.utils import try_to_create_directory
from ray.autoscaler._private.kuberay.autoscaling_config import AutoscalingConfigProducer
from ray.autoscaler._private.monitor import Monitor
logger = logging.getLogger(__name__)
BACKOFF_S = 5

def run_kuberay_autoscaler(cluster_name: str, cluster_namespace: str):
    if False:
        for i in range(10):
            print('nop')
    'Wait until the Ray head container is ready. Then start the autoscaler.'
    head_ip = get_node_ip_address()
    ray_address = f'{head_ip}:6379'
    while True:
        try:
            subprocess.check_call(['ray', 'health-check', '--address', ray_address, '--skip-version-check'])
            print('The Ray head is ready. Starting the autoscaler.')
            break
        except subprocess.CalledProcessError:
            print('The Ray head is not yet ready.')
            print(f'Will check again in {BACKOFF_S} seconds.')
            time.sleep(BACKOFF_S)
    _setup_logging()
    autoscaling_config_producer = AutoscalingConfigProducer(cluster_name, cluster_namespace)
    Monitor(address=ray_address, autoscaling_config=autoscaling_config_producer, monitor_ip=head_ip, retry_on_failure=False).run()

def _setup_logging() -> None:
    if False:
        for i in range(10):
            print('nop')
    'Log to autoscaler log file\n    (typically, /tmp/ray/session_latest/logs/monitor.*)\n\n    Also log to pod stdout (logs viewable with `kubectl logs <head-pod> -c autoscaler`).\n    '
    log_dir = os.path.join(ray._private.utils.get_ray_temp_dir(), ray._private.ray_constants.SESSION_LATEST, 'logs')
    try_to_create_directory(log_dir)
    setup_component_logger(logging_level=ray_constants.LOGGER_LEVEL, logging_format=ray_constants.LOGGER_FORMAT, log_dir=log_dir, filename=ray_constants.MONITOR_LOG_FILE_NAME, max_bytes=ray_constants.LOGGING_ROTATE_BYTES, backup_count=ray_constants.LOGGING_ROTATE_BACKUP_COUNT)
    level = logging.getLevelName(ray_constants.LOGGER_LEVEL.upper())
    stderr_handler = logging._StderrHandler()
    stderr_handler.setFormatter(logging.Formatter(ray_constants.LOGGER_FORMAT))
    stderr_handler.setLevel(level)
    logging.root.setLevel(level)
    logging.root.addHandler(stderr_handler)