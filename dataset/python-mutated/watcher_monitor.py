"""A module of widgets for job monitoring"""
import sys
import time
import threading

def _job_monitor(job, status, watcher):
    if False:
        i = 10
        return i + 15
    'Monitor the status of a IBMQJob instance.\n\n    Args:\n        job (BaseJob): Job to monitor.\n        status (Enum): Job status.\n        watcher (JobWatcher): Job watcher instance\n    '
    thread = threading.Thread(target=_job_checker, args=(job, status, watcher))
    thread.start()

def _job_checker(job, status, watcher):
    if False:
        return 10
    'A simple job status checker\n\n    Args:\n        job (BaseJob): The job to check.\n        status (Enum): Job status.\n        watcher (JobWatcher): Job watcher instance\n\n    '
    prev_status_name = None
    prev_queue_pos = None
    interval = 2
    exception_count = 0
    while status.name not in ['DONE', 'CANCELLED', 'ERROR']:
        time.sleep(interval)
        try:
            status = job.status()
            exception_count = 0
            if status.name == 'QUEUED':
                queue_pos = job.queue_position()
                if queue_pos != prev_queue_pos:
                    update_info = (job.job_id(), status.name, queue_pos, status.value)
                    watcher.update_single_job(update_info)
                    interval = max(queue_pos, 2)
                    prev_queue_pos = queue_pos
            elif status.name != prev_status_name:
                update_info = (job.job_id(), status.name, 0, status.value)
                watcher.update_single_job(update_info)
                interval = 2
                prev_status_name = status.name
        except Exception:
            exception_count += 1
            if exception_count == 5:
                update_info = (job.job_id(), 'NA', 0, 'Could not query job.')
                watcher.update_single_job(update_info)
                sys.exit()