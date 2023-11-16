"""Celery command."""
from __future__ import annotations
import logging
import sys
from contextlib import contextmanager
from multiprocessing import Process
import psutil
import sqlalchemy.exc
from celery import maybe_patch_concurrency
from celery.app.defaults import DEFAULT_TASK_LOG_FMT
from celery.signals import after_setup_logger
from lockfile.pidlockfile import read_pid_from_pidfile, remove_existing_pidfile
from airflow import settings
from airflow.cli.commands.daemon_utils import run_command_with_daemon_option
from airflow.configuration import conf
from airflow.utils import cli as cli_utils
from airflow.utils.cli import setup_locations
from airflow.utils.providers_configuration_loader import providers_configuration_loaded
from airflow.utils.serve_logs import serve_logs
WORKER_PROCESS_NAME = 'worker'

@cli_utils.action_cli
@providers_configuration_loaded
def flower(args):
    if False:
        i = 10
        return i + 15
    'Start Flower, Celery monitoring tool.'
    from airflow.providers.celery.executors.celery_executor import app as celery_app
    options = ['flower', conf.get('celery', 'BROKER_URL'), f'--address={args.hostname}', f'--port={args.port}']
    if args.broker_api:
        options.append(f'--broker-api={args.broker_api}')
    if args.url_prefix:
        options.append(f'--url-prefix={args.url_prefix}')
    if args.basic_auth:
        options.append(f'--basic-auth={args.basic_auth}')
    if args.flower_conf:
        options.append(f'--conf={args.flower_conf}')
    run_command_with_daemon_option(args=args, process_name='flower', callback=lambda : celery_app.start(options))

@contextmanager
def _serve_logs(skip_serve_logs: bool=False):
    if False:
        return 10
    'Start serve_logs sub-process.'
    sub_proc = None
    if skip_serve_logs is False:
        sub_proc = Process(target=serve_logs)
        sub_proc.start()
    yield
    if sub_proc:
        sub_proc.terminate()

@after_setup_logger.connect()
@providers_configuration_loaded
def logger_setup_handler(logger, **kwargs):
    if False:
        print('Hello World!')
    '\n    Reconfigure the logger.\n\n    * remove any previously configured handlers\n    * logs of severity error, and above goes to stderr,\n    * logs of severity lower than error goes to stdout.\n    '
    if conf.getboolean('logging', 'celery_stdout_stderr_separation', fallback=False):
        celery_formatter = logging.Formatter(DEFAULT_TASK_LOG_FMT)

        class NoErrorOrAboveFilter(logging.Filter):
            """Allow only logs with level *lower* than ERROR to be reported."""

            def filter(self, record):
                if False:
                    i = 10
                    return i + 15
                return record.levelno < logging.ERROR
        below_error_handler = logging.StreamHandler(sys.stdout)
        below_error_handler.addFilter(NoErrorOrAboveFilter())
        below_error_handler.setFormatter(celery_formatter)
        from_error_handler = logging.StreamHandler(sys.stderr)
        from_error_handler.setLevel(logging.ERROR)
        from_error_handler.setFormatter(celery_formatter)
        logger.handlers[:] = [below_error_handler, from_error_handler]

@cli_utils.action_cli
@providers_configuration_loaded
def worker(args):
    if False:
        i = 10
        return i + 15
    'Start Airflow Celery worker.'
    from airflow.providers.celery.executors.celery_executor import app as celery_app
    settings.reconfigure_orm(disable_connection_pool=True)
    if not settings.validate_session():
        raise SystemExit('Worker exiting, database connection precheck failed.')
    autoscale = args.autoscale
    skip_serve_logs = args.skip_serve_logs
    if autoscale is None and conf.has_option('celery', 'worker_autoscale'):
        autoscale = conf.get('celery', 'worker_autoscale')
    if hasattr(celery_app.backend, 'ResultSession'):
        try:
            session = celery_app.backend.ResultSession()
            session.close()
        except sqlalchemy.exc.IntegrityError:
            pass
    celery_log_level = conf.get('logging', 'CELERY_LOGGING_LEVEL')
    if not celery_log_level:
        celery_log_level = conf.get('logging', 'LOGGING_LEVEL')
    (worker_pid_file_path, _, _, _) = setup_locations(process=WORKER_PROCESS_NAME, pid=args.pid)
    options = ['worker', '-O', 'fair', '--queues', args.queues, '--concurrency', args.concurrency, '--hostname', args.celery_hostname, '--loglevel', celery_log_level, '--pidfile', worker_pid_file_path]
    if autoscale:
        options.extend(['--autoscale', autoscale])
    if args.without_mingle:
        options.append('--without-mingle')
    if args.without_gossip:
        options.append('--without-gossip')
    if conf.has_option('celery', 'pool'):
        pool = conf.get('celery', 'pool')
        options.extend(['--pool', pool])
        maybe_patch_concurrency(['-P', pool])
    (_, stdout, stderr, log_file) = setup_locations(process=WORKER_PROCESS_NAME, stdout=args.stdout, stderr=args.stderr, log=args.log_file)

    def run_celery_worker():
        if False:
            for i in range(10):
                print('nop')
        with _serve_logs(skip_serve_logs):
            celery_app.worker_main(options)
    if args.umask:
        umask = args.umask
    else:
        umask = conf.get('celery', 'worker_umask', fallback=settings.DAEMON_UMASK)
    run_command_with_daemon_option(args=args, process_name=WORKER_PROCESS_NAME, callback=run_celery_worker, should_setup_logging=True, umask=umask, pid_file=worker_pid_file_path)

@cli_utils.action_cli
@providers_configuration_loaded
def stop_worker(args):
    if False:
        for i in range(10):
            print('nop')
    'Send SIGTERM to Celery worker.'
    if args.pid:
        pid_file_path = args.pid
    else:
        (pid_file_path, _, _, _) = setup_locations(process=WORKER_PROCESS_NAME)
    pid = read_pid_from_pidfile(pid_file_path)
    if pid:
        worker_process = psutil.Process(pid)
        worker_process.terminate()
    remove_existing_pidfile(pid_file_path)