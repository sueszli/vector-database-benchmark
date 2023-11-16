from __future__ import annotations
import math
from typing import Any, Type
import sentry_sdk
from celery import Task
from django.conf import settings
from django.db.models import Max, Min
from sentry.models.outbox import ControlOutboxBase, OutboxBase, OutboxFlushError, RegionOutboxBase
from sentry.silo.base import SiloMode
from sentry.tasks.backfill_outboxes import backfill_outboxes_for
from sentry.tasks.base import instrumented_task
from sentry.utils import metrics
from sentry.utils.env import in_test_environment

@instrumented_task(name='sentry.tasks.enqueue_outbox_jobs_control', queue='outbox.control', silo_mode=SiloMode.CONTROL)
def enqueue_outbox_jobs_control(concurrency: int | None=None, process_outbox_backfills=True, **kwargs):
    if False:
        i = 10
        return i + 15
    schedule_batch(silo_mode=SiloMode.CONTROL, drain_task=drain_outbox_shards_control, concurrency=concurrency, process_outbox_backfills=process_outbox_backfills)

@instrumented_task(name='sentry.tasks.enqueue_outbox_jobs', silo_mode=SiloMode.REGION)
def enqueue_outbox_jobs(concurrency: int | None=None, process_outbox_backfills=True, **kwargs):
    if False:
        return 10
    schedule_batch(silo_mode=SiloMode.REGION, drain_task=drain_outbox_shards, concurrency=concurrency, process_outbox_backfills=process_outbox_backfills)
CONCURRENCY = 5

def schedule_batch(silo_mode: SiloMode, drain_task: Task, concurrency: int | None=None, process_outbox_backfills=True):
    if False:
        i = 10
        return i + 15
    scheduled_count = 0
    if not concurrency:
        concurrency = CONCURRENCY
    try:
        for outbox_name in settings.SENTRY_OUTBOX_MODELS[silo_mode.name]:
            outbox_model: Type[OutboxBase] = OutboxBase.from_outbox_name(outbox_name)
            lo = outbox_model.objects.all().aggregate(Min('id'))['id__min'] or 0
            hi = outbox_model.objects.all().aggregate(Max('id'))['id__max'] or -1
            if hi < lo:
                continue
            scheduled_count += hi - lo + 1
            batch_size = math.ceil((hi - lo + 1) / concurrency)
            metrics.gauge('deliver_from_outbox.queued_batch_size', value=batch_size, tags=dict(silo_mode=silo_mode.name), sample_rate=1.0)
            for i in range(concurrency):
                drain_task.delay(outbox_name=outbox_name, outbox_identifier_low=lo + i * batch_size, outbox_identifier_hi=lo + (i + 1) * batch_size)
        if process_outbox_backfills:
            backfill_outboxes_for(silo_mode, scheduled_count)
    except Exception:
        sentry_sdk.capture_exception()
        raise

@instrumented_task(name='sentry.tasks.drain_outbox_shard_control', queue='outbox.control', silo_mode=SiloMode.CONTROL)
def drain_outbox_shard_control(**kwargs: Any):
    if False:
        print('Hello World!')
    return

@instrumented_task(name='sentry.tasks.drain_outbox_shard', silo_mode=SiloMode.REGION)
def drain_outbox_shard(**kwds: Any):
    if False:
        while True:
            i = 10
    return

@instrumented_task(name='sentry.tasks.drain_outbox_shards', silo_mode=SiloMode.REGION)
def drain_outbox_shards(outbox_identifier_low: int=0, outbox_identifier_hi: int=0, outbox_name: str | None=None):
    if False:
        print('Hello World!')
    try:
        if outbox_name is None:
            outbox_name = settings.SENTRY_OUTBOX_MODELS['REGION'][0]
        assert outbox_name, 'Could not determine outbox name'
        outbox_model: Type[RegionOutboxBase] = RegionOutboxBase.from_outbox_name(outbox_name)
        process_outbox_batch(outbox_identifier_hi, outbox_identifier_low, outbox_model)
    except Exception:
        sentry_sdk.capture_exception()
        raise

@instrumented_task(name='sentry.tasks.drain_outbox_shards_control', silo_mode=SiloMode.CONTROL)
def drain_outbox_shards_control(outbox_identifier_low: int=0, outbox_identifier_hi: int=0, outbox_name: str | None=None):
    if False:
        i = 10
        return i + 15
    try:
        if outbox_name is None:
            outbox_name = settings.SENTRY_OUTBOX_MODELS['CONTROL'][0]
        assert outbox_name, 'Could not determine outbox name'
        outbox_model: Type[ControlOutboxBase] = ControlOutboxBase.from_outbox_name(outbox_name)
        process_outbox_batch(outbox_identifier_hi, outbox_identifier_low, outbox_model)
    except Exception:
        sentry_sdk.capture_exception()
        raise

def process_outbox_batch(outbox_identifier_hi: int, outbox_identifier_low: int, outbox_model: Type[OutboxBase]) -> int:
    if False:
        return 10
    processed_count: int = 0
    for shard_attributes in outbox_model.find_scheduled_shards(outbox_identifier_low, outbox_identifier_hi):
        shard_outbox: ControlOutboxBase | None = outbox_model.prepare_next_from_shard(shard_attributes)
        if not shard_outbox:
            continue
        try:
            processed_count += 1
            shard_outbox.drain_shard(flush_all=True)
        except Exception as e:
            with sentry_sdk.push_scope() as scope:
                if isinstance(e, OutboxFlushError):
                    scope.set_tag('outbox.category', e.outbox.category)
                    scope.set_tag('outbox.shard_scope', e.outbox.shard_scope)
                    scope.set_context('outbox', {'shard_identifier': e.outbox.shard_identifier, 'object_identifier': e.outbox.object_identifier, 'payload': e.outbox.payload})
                sentry_sdk.capture_exception(e)
                if in_test_environment():
                    raise
    return processed_count