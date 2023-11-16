"""
Concurrent programming with an event loop is a relatively new concept in
Python 3.x. This module aims to highlight how it could be used in the
context of a scheduler which runs a fire-and-forget operation for starting
jobs. In the real world, it takes time for a scheduler to start a job (i.e.
hit an API endpoint, ask the operating system for resources) so we assume
that starting a job has some intrinsic delay.
"""
import asyncio
from dataclasses import dataclass
from datetime import datetime
from uuid import uuid4
_DELAY_SMALL = 0.001
_DELAY_LARGE = 3600

@dataclass
class JobRecord:
    """Job record with useful metadata."""
    guid: str
    queued_at: datetime
    started_at: datetime

def _is_valid_record(record):
    if False:
        return 10
    'Check whether job record is valid or not.'
    return record.queued_at < record.started_at

def _current_time():
    if False:
        return 10
    'Return current time that is timezone-naive.'
    return datetime.now()

async def start_job(job_id, delay):
    """Start job ID after a certain amount of delay."""
    queue_time = _current_time()
    await asyncio.sleep(delay)
    start_time = _current_time()
    return JobRecord(job_id, queue_time, start_time)

async def schedule_jobs():
    """Schedule jobs concurrently."""
    single_job = start_job(uuid4().hex, _DELAY_SMALL)
    assert asyncio.iscoroutine(single_job)
    single_record = await single_job
    assert _is_valid_record(single_record)
    single_task = asyncio.create_task(start_job(uuid4().hex, _DELAY_LARGE))
    assert asyncio.isfuture(single_task)
    single_task.cancel()
    task_failed = False
    try:
        await single_task
    except asyncio.exceptions.CancelledError:
        assert single_task.cancelled()
        task_failed = True
    assert task_failed is True
    batch_jobs = [start_job(uuid4().hex, _DELAY_SMALL) for _ in range(10)]
    batch_records = await asyncio.gather(*batch_jobs)
    assert len(batch_records) == len(batch_jobs)
    assert all((_is_valid_record(record) for record in batch_records))

def main():
    if False:
        while True:
            i = 10
    asyncio.run(schedule_jobs())
if __name__ == '__main__':
    main()