"""Scheduler command."""
from __future__ import annotations
import logging
from argparse import Namespace
from contextlib import contextmanager
from multiprocessing import Process
from airflow import settings
from airflow.api_internal.internal_api_call import InternalApiConfig
from airflow.cli.commands.daemon_utils import run_command_with_daemon_option
from airflow.configuration import conf
from airflow.executors.executor_loader import ExecutorLoader
from airflow.jobs.job import Job, run_job
from airflow.jobs.scheduler_job_runner import SchedulerJobRunner
from airflow.utils import cli as cli_utils
from airflow.utils.cli import process_subdir
from airflow.utils.providers_configuration_loader import providers_configuration_loaded
from airflow.utils.scheduler_health import serve_health_check
log = logging.getLogger(__name__)

def _run_scheduler_job(args) -> None:
    if False:
        i = 10
        return i + 15
    job_runner = SchedulerJobRunner(job=Job(), subdir=process_subdir(args.subdir), num_runs=args.num_runs, do_pickle=args.do_pickle)
    ExecutorLoader.validate_database_executor_compatibility(job_runner.job.executor)
    InternalApiConfig.force_database_direct_access()
    enable_health_check = conf.getboolean('scheduler', 'ENABLE_HEALTH_CHECK')
    with _serve_logs(args.skip_serve_logs), _serve_health_check(enable_health_check):
        try:
            run_job(job=job_runner.job, execute_callable=job_runner._execute)
        except Exception:
            log.exception('Exception when running scheduler job')

@cli_utils.action_cli
@providers_configuration_loaded
def scheduler(args: Namespace):
    if False:
        return 10
    'Start Airflow Scheduler.'
    print(settings.HEADER)
    run_command_with_daemon_option(args=args, process_name='scheduler', callback=lambda : _run_scheduler_job(args), should_setup_logging=True)

@contextmanager
def _serve_logs(skip_serve_logs: bool=False):
    if False:
        for i in range(10):
            print('nop')
    'Start serve_logs sub-process.'
    from airflow.utils.serve_logs import serve_logs
    sub_proc = None
    (executor_class, _) = ExecutorLoader.import_default_executor_cls()
    if executor_class.serve_logs:
        if skip_serve_logs is False:
            sub_proc = Process(target=serve_logs)
            sub_proc.start()
    yield
    if sub_proc:
        sub_proc.terminate()

@contextmanager
def _serve_health_check(enable_health_check: bool=False):
    if False:
        while True:
            i = 10
    'Start serve_health_check sub-process.'
    sub_proc = None
    if enable_health_check:
        sub_proc = Process(target=serve_health_check)
        sub_proc.start()
    yield
    if sub_proc:
        sub_proc.terminate()