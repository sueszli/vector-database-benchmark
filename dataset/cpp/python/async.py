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

# Module-level constants
_DELAY_SMALL = .001
_DELAY_LARGE = 3600


@dataclass
class JobRecord:
    """Job record with useful metadata."""

    guid: str
    queued_at: datetime
    started_at: datetime


def _is_valid_record(record):
    """Check whether job record is valid or not."""
    return record.queued_at < record.started_at


def _current_time():
    """Return current time that is timezone-naive."""
    return datetime.now()


async def start_job(job_id, delay):
    """Start job ID after a certain amount of delay."""
    queue_time = _current_time()
    await asyncio.sleep(delay)
    start_time = _current_time()
    return JobRecord(job_id, queue_time, start_time)


async def schedule_jobs():
    """Schedule jobs concurrently."""
    # Start a job which also represents a coroutine
    single_job = start_job(uuid4().hex, _DELAY_SMALL)
    assert asyncio.iscoroutine(single_job)

    # Grab a job record from the coroutine
    single_record = await single_job
    assert _is_valid_record(single_record)

    # Task is a wrapped coroutine which also represents a future
    single_task = asyncio.create_task(start_job(uuid4().hex, _DELAY_LARGE))
    assert asyncio.isfuture(single_task)

    # Futures are different from other coroutines since they can be cancelled
    single_task.cancel()
    task_failed = False
    try:
        await single_task
    except asyncio.exceptions.CancelledError:
        assert single_task.cancelled()
        task_failed = True
    assert task_failed is True

    # Gather coroutines for batch start
    batch_jobs = [start_job(uuid4().hex, _DELAY_SMALL) for _ in range(10)]
    batch_records = await asyncio.gather(*batch_jobs)

    # We get the same amount of records as we have coroutines
    assert len(batch_records) == len(batch_jobs)
    assert all(_is_valid_record(record) for record in batch_records)


def main():
    asyncio.run(schedule_jobs())


if __name__ == "__main__":
    main()
