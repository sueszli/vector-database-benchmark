from __future__ import annotations
import signal
from argparse import Namespace
from typing import Callable
from daemon import daemon
from daemon.pidfile import TimeoutPIDLockFile
from airflow import settings
from airflow.utils.cli import setup_locations, setup_logging, sigint_handler, sigquit_handler
from airflow.utils.process_utils import check_if_pidfile_process_is_running

def run_command_with_daemon_option(*, args: Namespace, process_name: str, callback: Callable, should_setup_logging: bool=False, umask: str=settings.DAEMON_UMASK, pid_file: str | None=None):
    if False:
        print('Hello World!')
    'Run the command in a daemon process if daemon mode enabled or within this process if not.\n\n    :param args: the set of arguments passed to the original CLI command\n    :param process_name: process name used in naming log and PID files for the daemon\n    :param callback: the actual command to run with or without daemon context\n    :param should_setup_logging: if true, then a log file handler for the daemon process will be created\n    :param umask: file access creation mask ("umask") to set for the process on daemon start\n    :param pid_file: if specified, this file path us used to store daemon process PID.\n        If not specified, a file path is generated with the default pattern.\n    '
    if args.daemon:
        (pid, stdout, stderr, log_file) = setup_locations(process=process_name, stdout=args.stdout, stderr=args.stderr, log=args.log_file)
        if pid_file:
            pid = pid_file
        check_if_pidfile_process_is_running(pid_file=pid, process_name=process_name)
        if should_setup_logging:
            files_preserve = [setup_logging(log_file)]
        else:
            files_preserve = None
        with open(stdout, 'a') as stdout_handle, open(stderr, 'a') as stderr_handle:
            stdout_handle.truncate(0)
            stderr_handle.truncate(0)
            ctx = daemon.DaemonContext(pidfile=TimeoutPIDLockFile(pid, -1), files_preserve=files_preserve, stdout=stdout_handle, stderr=stderr_handle, umask=int(umask, 8))
            with ctx:
                from airflow.stats import Stats
                Stats.instance = None
                callback()
    else:
        signal.signal(signal.SIGINT, sigint_handler)
        signal.signal(signal.SIGTERM, sigint_handler)
        signal.signal(signal.SIGQUIT, sigquit_handler)
        callback()