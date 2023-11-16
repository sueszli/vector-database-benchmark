"""Triggerer command."""
from __future__ import annotations
from contextlib import contextmanager
from functools import partial
from multiprocessing import Process
from typing import Generator
from airflow import settings
from airflow.cli.commands.daemon_utils import run_command_with_daemon_option
from airflow.configuration import conf
from airflow.jobs.job import Job, run_job
from airflow.jobs.triggerer_job_runner import TriggererJobRunner
from airflow.utils import cli as cli_utils
from airflow.utils.providers_configuration_loader import providers_configuration_loaded
from airflow.utils.serve_logs import serve_logs

@contextmanager
def _serve_logs(skip_serve_logs: bool=False) -> Generator[None, None, None]:
    if False:
        while True:
            i = 10
    'Start serve_logs sub-process.'
    sub_proc = None
    if skip_serve_logs is False:
        port = conf.getint('logging', 'trigger_log_server_port', fallback=8794)
        sub_proc = Process(target=partial(serve_logs, port=port))
        sub_proc.start()
    try:
        yield
    finally:
        if sub_proc:
            sub_proc.terminate()

def triggerer_run(skip_serve_logs: bool, capacity: int, triggerer_heartrate: float):
    if False:
        while True:
            i = 10
    with _serve_logs(skip_serve_logs):
        triggerer_job_runner = TriggererJobRunner(job=Job(heartrate=triggerer_heartrate), capacity=capacity)
        run_job(job=triggerer_job_runner.job, execute_callable=triggerer_job_runner._execute)

@cli_utils.action_cli
@providers_configuration_loaded
def triggerer(args):
    if False:
        while True:
            i = 10
    'Start Airflow Triggerer.'
    settings.MASK_SECRETS_IN_LOGS = True
    print(settings.HEADER)
    triggerer_heartrate = conf.getfloat('triggerer', 'JOB_HEARTBEAT_SEC')
    run_command_with_daemon_option(args=args, process_name='triggerer', callback=lambda : triggerer_run(args.skip_serve_logs, args.capacity, triggerer_heartrate), should_setup_logging=True)