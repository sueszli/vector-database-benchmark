import asyncio
import io
import logging
import sys
import os
import ray
from ray.dashboard.consts import _PARENT_DEATH_THREASHOLD
import ray.dashboard.consts as dashboard_consts
import ray._private.ray_constants as ray_constants
from ray._private.utils import run_background_task
import psutil
logger = logging.getLogger(__name__)
_RAYLET_LOG_MAX_PUBLISH_LINES = 20
_RAYLET_LOG_MAX_TAIL_SIZE = 1 * 1024 ** 2
try:
    create_task = asyncio.create_task
except AttributeError:
    create_task = asyncio.ensure_future

def get_raylet_pid():
    if False:
        i = 10
        return i + 15
    if sys.platform in ['win32', 'cygwin']:
        return None
    raylet_pid = int(os.environ['RAY_RAYLET_PID'])
    assert raylet_pid > 0
    logger.info('raylet pid is %s', raylet_pid)
    return raylet_pid

def create_check_raylet_task(log_dir, gcs_address, parent_dead_callback, loop):
    if False:
        return 10
    '\n    Creates an asyncio task to periodically check if the raylet process is still\n    running. If raylet is dead for _PARENT_DEATH_THREASHOLD (5) times, prepare to exit\n    as follows:\n\n    - Write logs about whether the raylet exit is graceful, by looking into the raylet\n    log and search for term "SIGTERM",\n    - Flush the logs via GcsPublisher,\n    - Exit.\n    '
    if sys.platform in ['win32', 'cygwin']:
        raise RuntimeError("can't check raylet process in Windows.")
    raylet_pid = get_raylet_pid()
    return run_background_task(_check_parent(raylet_pid, log_dir, gcs_address, parent_dead_callback))

async def _check_parent(raylet_pid, log_dir, gcs_address, parent_dead_callback):
    """Check if raylet is dead and fate-share if it is."""
    try:
        curr_proc = psutil.Process()
        parent_death_cnt = 0
        while True:
            parent = curr_proc.parent()
            parent_gone = parent is None
            init_assigned_for_parent = False
            parent_changed = False
            if parent:
                init_assigned_for_parent = parent.pid == 1
                parent_changed = raylet_pid != parent.pid
            if parent_gone or init_assigned_for_parent or parent_changed:
                parent_death_cnt += 1
                logger.warning(f'Raylet is considered dead {parent_death_cnt} X. If it reaches to {_PARENT_DEATH_THREASHOLD}, the agent will kill itself. Parent: {parent}, parent_gone: {parent_gone}, init_assigned_for_parent: {init_assigned_for_parent}, parent_changed: {parent_changed}.')
                if parent_death_cnt < _PARENT_DEATH_THREASHOLD:
                    await asyncio.sleep(dashboard_consts.DASHBOARD_AGENT_CHECK_PARENT_INTERVAL_S)
                    continue
                log_path = os.path.join(log_dir, 'raylet.out')
                error = False
                parent_dead_callback()
                msg = 'Raylet is terminated. '
                try:
                    with open(log_path, 'r', encoding='utf-8') as f:
                        f.seek(0, io.SEEK_END)
                        pos = max(0, f.tell() - _RAYLET_LOG_MAX_TAIL_SIZE)
                        f.seek(pos, io.SEEK_SET)
                        raylet_logs = f.readlines()
                        if any(('Raylet received SIGTERM' in line for line in raylet_logs)):
                            msg += 'Termination is graceful.'
                            logger.info(msg)
                        else:
                            msg += f'Termination is unexpected. Possible reasons include: (1) SIGKILL by the user or system OOM killer, (2) Invalid memory access from Raylet causing SIGSEGV or SIGBUS, (3) Other termination signals. Last {_RAYLET_LOG_MAX_PUBLISH_LINES} lines of the Raylet logs:\n'
                            msg += '    ' + '    '.join(raylet_logs[-_RAYLET_LOG_MAX_PUBLISH_LINES:])
                            error = True
                except Exception as e:
                    msg += f'Failed to read Raylet logs at {log_path}: {e}!'
                    logger.exception(msg)
                    error = True
                if error:
                    logger.error(msg)
                    ray._private.utils.publish_error_to_driver(ray_constants.RAYLET_DIED_ERROR, msg, gcs_publisher=ray._raylet.GcsPublisher(address=gcs_address))
                else:
                    logger.info(msg)
                sys.exit(0)
            else:
                parent_death_cnt = 0
            await asyncio.sleep(dashboard_consts.DASHBOARD_AGENT_CHECK_PARENT_INTERVAL_S)
    except Exception:
        logger.exception('Failed to check parent PID, exiting.')
        sys.exit(1)