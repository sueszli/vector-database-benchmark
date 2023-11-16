import os.path
import subprocess
import sys
import time
import shutil
import fcntl
import signal
import socket
import logging
import threading
from ray.util.spark.cluster_init import RAY_ON_SPARK_COLLECT_LOG_TO_PATH, RAY_ON_SPARK_START_RAY_PARENT_PID
from ray._private.ray_process_reaper import SIGTERM_GRACE_PERIOD_SECONDS
_logger = logging.getLogger(__name__)
if __name__ == '__main__':
    arg_list = sys.argv[1:]
    collect_log_to_path = os.environ[RAY_ON_SPARK_COLLECT_LOG_TO_PATH]
    temp_dir_arg_prefix = '--temp-dir='
    temp_dir = None
    for arg in arg_list:
        if arg.startswith(temp_dir_arg_prefix):
            temp_dir = arg[len(temp_dir_arg_prefix):]
    if temp_dir is None:
        raise ValueError('Please explicitly set --temp-dir option.')
    temp_dir = os.path.normpath(temp_dir)
    ray_cli_cmd = 'ray'
    lock_file = temp_dir + '.lock'
    lock_fd = os.open(lock_file, os.O_RDWR | os.O_CREAT | os.O_TRUNC)
    fcntl.flock(lock_fd, fcntl.LOCK_SH)
    process = subprocess.Popen([ray_cli_cmd, 'start', *arg_list], text=True)

    def try_clean_temp_dir_at_exit():
        if False:
            i = 10
            return i + 15
        try:
            time.sleep(SIGTERM_GRACE_PERIOD_SECONDS + 0.5)
            if process.poll() is None:
                process.kill()
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                lock_acquired = True
            except BlockingIOError:
                lock_acquired = False
            if lock_acquired:
                if collect_log_to_path:
                    try:
                        base_dir = os.path.join(collect_log_to_path, os.path.basename(temp_dir) + '-logs')
                        os.makedirs(base_dir, exist_ok=True)
                        copy_log_dest_path = os.path.join(base_dir, socket.gethostname())
                        ray_session_dir = os.readlink(os.path.join(temp_dir, 'session_latest'))
                        shutil.copytree(os.path.join(ray_session_dir, 'logs'), copy_log_dest_path)
                    except Exception as e:
                        _logger.warning(f'Collect logs to destination directory failed, error: {repr(e)}.')
                shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception:
            pass
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            os.close(lock_fd)

    def check_parent_alive() -> None:
        if False:
            return 10
        orig_parent_pid = int(os.environ[RAY_ON_SPARK_START_RAY_PARENT_PID])
        while True:
            time.sleep(0.5)
            if os.getppid() != orig_parent_pid:
                process.terminate()
                try_clean_temp_dir_at_exit()
                os._exit(143)
    threading.Thread(target=check_parent_alive, daemon=True).start()
    try:

        def sighup_handler(*args):
            if False:
                print('Hello World!')
            pass
        signal.signal(signal.SIGHUP, sighup_handler)

        def sigterm_handler(*args):
            if False:
                return 10
            process.terminate()
            try_clean_temp_dir_at_exit()
            os._exit(143)
        signal.signal(signal.SIGTERM, sigterm_handler)
        ret_code = process.wait()
        try_clean_temp_dir_at_exit()
        sys.exit(ret_code)
    except Exception:
        try_clean_temp_dir_at_exit()
        raise