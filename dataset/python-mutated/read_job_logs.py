from __future__ import annotations
from typing import NoReturn
from google.cloud import batch_v1
from google.cloud import logging

def print_job_logs(project_id: str, job: batch_v1.Job) -> NoReturn:
    if False:
        return 10
    '\n    Prints the log messages created by given job.\n\n    Args:\n        project_id: name of the project hosting the job.\n        job: the job which logs you want to print.\n    '
    log_client = logging.Client(project=project_id)
    logger = log_client.logger('batch_task_logs')
    for log_entry in logger.list_entries(filter_=f'labels.job_uid={job.uid}'):
        print(log_entry.payload)