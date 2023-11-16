from __future__ import annotations
import asyncio
import datetime
import pendulum
import pytest
from airflow.triggers.base import TriggerEvent
from airflow.triggers.temporal import DateTimeTrigger, TimeDeltaTrigger
from airflow.utils import timezone

def test_input_validation():
    if False:
        print('Hello World!')
    '\n    Tests that the DateTimeTrigger validates input to moment arg, it should only accept datetime.\n    '
    with pytest.raises(TypeError, match="Expected datetime.datetime type for moment. Got <class 'str'>"):
        DateTimeTrigger('2012-01-01T03:03:03+00:00')

def test_datetime_trigger_serialization():
    if False:
        print('Hello World!')
    '\n    Tests that the DateTimeTrigger correctly serializes its arguments\n    and classpath.\n    '
    moment = pendulum.instance(datetime.datetime(2020, 4, 1, 13, 0), pendulum.UTC)
    trigger = DateTimeTrigger(moment)
    (classpath, kwargs) = trigger.serialize()
    assert classpath == 'airflow.triggers.temporal.DateTimeTrigger'
    assert kwargs == {'moment': moment}

def test_timedelta_trigger_serialization():
    if False:
        i = 10
        return i + 15
    '\n    Tests that the TimeDeltaTrigger correctly serializes its arguments\n    and classpath (it turns into a DateTimeTrigger).\n    '
    trigger = TimeDeltaTrigger(datetime.timedelta(seconds=10))
    expected_moment = timezone.utcnow() + datetime.timedelta(seconds=10)
    (classpath, kwargs) = trigger.serialize()
    assert classpath == 'airflow.triggers.temporal.DateTimeTrigger'
    assert -2 < (kwargs['moment'] - expected_moment).total_seconds() < 2

@pytest.mark.parametrize('tz', [pendulum.tz.timezone('UTC'), pendulum.tz.timezone('Europe/Paris'), pendulum.tz.timezone('America/Toronto')])
@pytest.mark.asyncio
async def test_datetime_trigger_timing(tz):
    """
    Tests that the DateTimeTrigger only goes off on or after the appropriate
    time.
    """
    past_moment = pendulum.instance((timezone.utcnow() - datetime.timedelta(seconds=60)).astimezone(tz))
    future_moment = pendulum.instance((timezone.utcnow() + datetime.timedelta(seconds=60)).astimezone(tz))
    trigger = DateTimeTrigger(future_moment)
    trigger_task = asyncio.create_task(trigger.run().__anext__())
    await asyncio.sleep(0.5)
    assert trigger_task.done() is False
    trigger_task.cancel()
    trigger = DateTimeTrigger(past_moment)
    trigger_task = asyncio.create_task(trigger.run().__anext__())
    await asyncio.sleep(0.5)
    assert trigger_task.done() is True
    result = trigger_task.result()
    assert isinstance(result, TriggerEvent)
    assert result.payload == past_moment