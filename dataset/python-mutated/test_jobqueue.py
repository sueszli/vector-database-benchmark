import asyncio
import calendar
import datetime as dtm
import logging
import platform
import time
import pytest
from telegram.ext import ApplicationBuilder, CallbackContext, ContextTypes, Defaults, Job, JobQueue
from telegram.warnings import PTBUserWarning
from tests.auxil.envvars import GITHUB_ACTION, TEST_WITH_OPT_DEPS
from tests.auxil.pytest_classes import make_bot
from tests.auxil.slots import mro_slots
if TEST_WITH_OPT_DEPS:
    import pytz
    UTC = pytz.utc
else:
    import datetime
    UTC = datetime.timezone.utc

class CustomContext(CallbackContext):
    pass

@pytest.fixture()
async def job_queue(app):
    jq = JobQueue()
    jq.set_application(app)
    await jq.start()
    yield jq
    await jq.stop()

@pytest.mark.skipif(TEST_WITH_OPT_DEPS, reason='Only relevant if the optional dependency is not installed')
class TestNoJobQueue:

    def test_init_job_queue(self):
        if False:
            return 10
        with pytest.raises(RuntimeError, match='python-telegram-bot\\[job-queue\\]'):
            JobQueue()

    def test_init_job(self):
        if False:
            return 10
        with pytest.raises(RuntimeError, match='python-telegram-bot\\[job-queue\\]'):
            Job(None)

@pytest.mark.skipif(not TEST_WITH_OPT_DEPS, reason='Only relevant if the optional dependency is installed')
@pytest.mark.skipif(bool(GITHUB_ACTION and platform.system() in ['Windows', 'Darwin']), reason='On Windows & MacOS precise timings are not accurate.')
@pytest.mark.flaky(10, 1)
class TestJobQueue:
    result = 0
    job_time = 0
    received_error = None
    expected_warning = "Prior to v20.0 the `days` parameter was not aligned to that of cron's weekday scheme. We recommend double checking if the passed value is correct."

    async def test_repr(self, app):
        jq = JobQueue()
        jq.set_application(app)
        assert repr(jq) == f'JobQueue[application={app!r}]'
        when = dtm.datetime.utcnow() + dtm.timedelta(days=1)
        callback = self.job_run_once
        job = jq.run_once(callback, when, name='name2')
        assert repr(job) == f"Job[id={job.job.id}, name={job.name}, callback=job_run_once, trigger=date[{when.strftime('%Y-%m-%d %H:%M:%S UTC')}]]"

    @pytest.fixture(autouse=True)
    def _reset(self):
        if False:
            for i in range(10):
                print('nop')
        self.result = 0
        self.job_time = 0
        self.received_error = None

    def test_scheduler_configuration(self, job_queue, timezone, bot):
        if False:
            i = 10
            return i + 15
        assert job_queue.scheduler_configuration['timezone'] is UTC
        tz_app = ApplicationBuilder().defaults(Defaults(tzinfo=timezone)).token(bot.token).build()
        assert tz_app.job_queue.scheduler_configuration['timezone'] is timezone

    async def job_run_once(self, context):
        if isinstance(context, CallbackContext) and isinstance(context.job, Job) and isinstance(context.update_queue, asyncio.Queue) and (context.job.data is None) and (context.chat_data is None) and (context.user_data is None) and isinstance(context.bot_data, dict):
            self.result += 1

    async def job_with_exception(self, context):
        raise Exception('Test Error')

    async def job_remove_self(self, context):
        self.result += 1
        context.job.schedule_removal()

    async def job_run_once_with_data(self, context):
        self.result += context.job.data

    async def job_datetime_tests(self, context):
        self.job_time = time.time()

    async def error_handler_context(self, update, context):
        self.received_error = (str(context.error), context.job, context.user_data, context.chat_data)

    async def error_handler_raise_error(self, *args):
        raise Exception('Failing bigly')

    def test_slot_behaviour(self, job_queue):
        if False:
            i = 10
            return i + 15
        for attr in job_queue.__slots__:
            assert getattr(job_queue, attr, 'err') != 'err', f"got extra slot '{attr}'"
        assert len(mro_slots(job_queue)) == len(set(mro_slots(job_queue))), 'duplicate slot'

    def test_application_weakref(self, bot):
        if False:
            print('Hello World!')
        jq = JobQueue()
        application = ApplicationBuilder().token(bot.token).job_queue(None).build()
        with pytest.raises(RuntimeError, match='No application was set'):
            jq.application
        jq.set_application(application)
        assert jq.application is application
        del application
        with pytest.raises(RuntimeError, match='no longer alive'):
            jq.application

    async def test_run_once(self, job_queue):
        job_queue.run_once(self.job_run_once, 0.1)
        await asyncio.sleep(0.2)
        assert self.result == 1

    async def test_run_once_timezone(self, job_queue, timezone):
        """Test the correct handling of aware datetimes"""
        when = dtm.datetime.now(timezone)
        job_queue.run_once(self.job_run_once, when)
        await asyncio.sleep(0.1)
        assert self.result == 1

    async def test_job_with_data(self, job_queue):
        job_queue.run_once(self.job_run_once_with_data, 0.1, data=5)
        await asyncio.sleep(0.2)
        assert self.result == 5

    async def test_run_repeating(self, job_queue):
        job_queue.run_repeating(self.job_run_once, 0.1)
        await asyncio.sleep(0.25)
        assert self.result == 2

    async def test_run_repeating_first(self, job_queue):
        job_queue.run_repeating(self.job_run_once, 0.5, first=0.2)
        await asyncio.sleep(0.15)
        assert self.result == 0
        await asyncio.sleep(0.1)
        assert self.result == 1

    async def test_run_repeating_first_timezone(self, job_queue, timezone):
        """Test correct scheduling of job when passing a timezone-aware datetime as ``first``"""
        job_queue.run_repeating(self.job_run_once, 0.5, first=dtm.datetime.now(timezone) + dtm.timedelta(seconds=0.2))
        await asyncio.sleep(0.05)
        assert self.result == 0
        await asyncio.sleep(0.25)
        assert self.result == 1

    async def test_run_repeating_last(self, job_queue):
        job_queue.run_repeating(self.job_run_once, 0.25, last=0.4)
        await asyncio.sleep(0.3)
        assert self.result == 1
        await asyncio.sleep(0.4)
        assert self.result == 1

    async def test_run_repeating_last_timezone(self, job_queue, timezone):
        """Test correct scheduling of job when passing a timezone-aware datetime as ``last``"""
        job_queue.run_repeating(self.job_run_once, 0.25, last=dtm.datetime.now(timezone) + dtm.timedelta(seconds=0.4))
        await asyncio.sleep(0.3)
        assert self.result == 1
        await asyncio.sleep(0.4)
        assert self.result == 1

    async def test_run_repeating_last_before_first(self, job_queue):
        with pytest.raises(ValueError, match="'last' must not be before 'first'!"):
            job_queue.run_repeating(self.job_run_once, 0.5, first=1, last=0.5)

    async def test_run_repeating_timedelta(self, job_queue):
        job_queue.run_repeating(self.job_run_once, dtm.timedelta(seconds=0.1))
        await asyncio.sleep(0.25)
        assert self.result == 2

    async def test_run_custom(self, job_queue):
        job_queue.run_custom(self.job_run_once, {'trigger': 'interval', 'seconds': 0.2})
        await asyncio.sleep(0.5)
        assert self.result == 2

    async def test_multiple(self, job_queue):
        job_queue.run_once(self.job_run_once, 0.1)
        job_queue.run_once(self.job_run_once, 0.2)
        job_queue.run_repeating(self.job_run_once, 0.2)
        await asyncio.sleep(0.55)
        assert self.result == 4

    async def test_disabled(self, job_queue):
        j1 = job_queue.run_once(self.job_run_once, 0.1)
        j2 = job_queue.run_repeating(self.job_run_once, 0.5)
        j1.enabled = False
        j2.enabled = False
        await asyncio.sleep(0.6)
        assert self.result == 0
        j1.enabled = True
        await asyncio.sleep(0.6)
        assert self.result == 1

    async def test_schedule_removal(self, job_queue):
        j1 = job_queue.run_once(self.job_run_once, 0.3)
        j2 = job_queue.run_repeating(self.job_run_once, 0.2)
        await asyncio.sleep(0.25)
        assert self.result == 1
        j1.schedule_removal()
        j2.schedule_removal()
        await asyncio.sleep(0.4)
        assert self.result == 1

    async def test_schedule_removal_from_within(self, job_queue):
        job_queue.run_repeating(self.job_remove_self, 0.1)
        await asyncio.sleep(0.5)
        assert self.result == 1

    async def test_longer_first(self, job_queue):
        job_queue.run_once(self.job_run_once, 0.2)
        job_queue.run_once(self.job_run_once, 0.1)
        await asyncio.sleep(0.15)
        assert self.result == 1

    async def test_error(self, job_queue):
        job_queue.run_repeating(self.job_with_exception, 0.1)
        job_queue.run_repeating(self.job_run_once, 0.2)
        await asyncio.sleep(0.3)
        assert self.result == 1

    async def test_in_application(self, bot_info):
        app = ApplicationBuilder().bot(make_bot(bot_info)).build()
        async with app:
            assert not app.job_queue.scheduler.running
            await app.start()
            assert app.job_queue.scheduler.running
            app.job_queue.run_repeating(self.job_run_once, 0.2)
            await asyncio.sleep(0.3)
            assert self.result == 1
            await app.stop()
            assert not app.job_queue.scheduler.running
            await asyncio.sleep(1)
            assert self.result == 1

    async def test_time_unit_int(self, job_queue):
        delta = 0.5
        expected_time = time.time() + delta
        job_queue.run_once(self.job_datetime_tests, delta)
        await asyncio.sleep(0.6)
        assert pytest.approx(self.job_time) == expected_time

    async def test_time_unit_dt_timedelta(self, job_queue):
        interval = dtm.timedelta(seconds=0.5)
        expected_time = time.time() + interval.total_seconds()
        job_queue.run_once(self.job_datetime_tests, interval)
        await asyncio.sleep(0.6)
        assert pytest.approx(self.job_time) == expected_time

    async def test_time_unit_dt_datetime(self, job_queue):
        (delta, now) = (dtm.timedelta(seconds=0.5), dtm.datetime.now(UTC))
        when = now + delta
        expected_time = when.timestamp()
        job_queue.run_once(self.job_datetime_tests, when)
        await asyncio.sleep(0.6)
        assert self.job_time == pytest.approx(expected_time)

    async def test_time_unit_dt_time_today(self, job_queue):
        (delta, now) = (0.5, dtm.datetime.now(UTC))
        expected_time = now + dtm.timedelta(seconds=delta)
        when = expected_time.time()
        expected_time = expected_time.timestamp()
        job_queue.run_once(self.job_datetime_tests, when)
        await asyncio.sleep(0.6)
        assert self.job_time == pytest.approx(expected_time)

    async def test_time_unit_dt_time_tomorrow(self, job_queue):
        (delta, now) = (-2, dtm.datetime.now(UTC))
        when = (now + dtm.timedelta(seconds=delta)).time()
        expected_time = (now + dtm.timedelta(seconds=delta, days=1)).timestamp()
        job_queue.run_once(self.job_datetime_tests, when)
        scheduled_time = job_queue.jobs()[0].next_t.timestamp()
        assert scheduled_time == pytest.approx(expected_time)

    async def test_run_daily(self, job_queue):
        (delta, now) = (1, dtm.datetime.now(UTC))
        time_of_day = (now + dtm.timedelta(seconds=delta)).time()
        expected_reschedule_time = (now + dtm.timedelta(seconds=delta, days=1)).timestamp()
        job_queue.run_daily(self.job_run_once, time_of_day)
        await asyncio.sleep(delta + 0.1)
        assert self.result == 1
        scheduled_time = job_queue.jobs()[0].next_t.timestamp()
        assert scheduled_time == pytest.approx(expected_reschedule_time)

    async def test_run_daily_warning(self, job_queue, recwarn):
        (delta, now) = (1, dtm.datetime.now(UTC))
        time_of_day = (now + dtm.timedelta(seconds=delta)).time()
        job_queue.run_daily(self.job_run_once, time_of_day)
        assert len(recwarn) == 0
        job_queue.run_daily(self.job_run_once, time_of_day, days=(0, 1, 2, 3))
        assert len(recwarn) == 1
        assert str(recwarn[0].message) == self.expected_warning
        assert recwarn[0].category is PTBUserWarning
        assert recwarn[0].filename == __file__, 'wrong stacklevel'

    @pytest.mark.parametrize('weekday', [0, 1, 2, 3, 4, 5, 6])
    async def test_run_daily_days_of_week(self, job_queue, recwarn, weekday):
        (delta, now) = (1, dtm.datetime.now(UTC))
        time_of_day = (now + dtm.timedelta(seconds=delta)).time()
        offset = (weekday + 6 - now.weekday()) % 7
        offset = offset if offset > 0 else 7
        expected_reschedule_time = (now + dtm.timedelta(seconds=delta, days=offset)).timestamp()
        job_queue.run_daily(self.job_run_once, time_of_day, days=[weekday])
        await asyncio.sleep(delta + 0.1)
        scheduled_time = job_queue.jobs()[0].next_t.timestamp()
        assert scheduled_time == pytest.approx(expected_reschedule_time)
        assert len(recwarn) == 1
        assert str(recwarn[0].message) == self.expected_warning
        assert recwarn[0].category is PTBUserWarning
        assert recwarn[0].filename == __file__, 'wrong stacklevel'

    async def test_run_monthly(self, job_queue, timezone):
        (delta, now) = (1, dtm.datetime.now(timezone))
        expected_reschedule_time = now + dtm.timedelta(seconds=delta)
        time_of_day = expected_reschedule_time.time().replace(tzinfo=timezone)
        day = now.day
        this_months_days = calendar.monthrange(now.year, now.month)[1]
        if now.month == 12:
            next_months_days = calendar.monthrange(now.year + 1, 1)[1]
        else:
            next_months_days = calendar.monthrange(now.year, now.month + 1)[1]
        expected_reschedule_time += dtm.timedelta(this_months_days)
        if day > next_months_days:
            expected_reschedule_time += dtm.timedelta(next_months_days)
        expected_reschedule_time = timezone.normalize(expected_reschedule_time)
        expected_reschedule_time += dtm.timedelta(hours=time_of_day.hour - expected_reschedule_time.hour)
        expected_reschedule_time = expected_reschedule_time.timestamp()
        job_queue.run_monthly(self.job_run_once, time_of_day, day)
        await asyncio.sleep(delta + 0.1)
        assert self.result == 1
        scheduled_time = job_queue.jobs()[0].next_t.timestamp()
        assert scheduled_time == pytest.approx(expected_reschedule_time, rel=0.001)

    async def test_run_monthly_non_strict_day(self, job_queue, timezone):
        (delta, now) = (1, dtm.datetime.now(timezone))
        expected_reschedule_time = now + dtm.timedelta(seconds=delta)
        time_of_day = expected_reschedule_time.time().replace(tzinfo=timezone)
        expected_reschedule_time += dtm.timedelta(calendar.monthrange(now.year, now.month)[1]) - dtm.timedelta(days=now.day)
        expected_reschedule_time = timezone.normalize(expected_reschedule_time)
        expected_reschedule_time += dtm.timedelta(hours=time_of_day.hour - expected_reschedule_time.hour)
        expected_reschedule_time = expected_reschedule_time.timestamp()
        job_queue.run_monthly(self.job_run_once, time_of_day, -1)
        scheduled_time = job_queue.jobs()[0].next_t.timestamp()
        assert scheduled_time == pytest.approx(expected_reschedule_time)

    async def test_default_tzinfo(self, tz_bot):
        app = ApplicationBuilder().bot(tz_bot).build()
        jq = app.job_queue
        await jq.start()
        when = dtm.datetime.now(tz_bot.defaults.tzinfo) + dtm.timedelta(seconds=0.1)
        jq.run_once(self.job_run_once, when.time())
        await asyncio.sleep(0.15)
        assert self.result == 1
        await jq.stop()

    async def test_get_jobs(self, job_queue):
        callback = self.job_run_once
        job1 = job_queue.run_once(callback, 10, name='name1')
        await asyncio.sleep(0.03)
        job2 = job_queue.run_once(callback, 10, name='name1')
        await asyncio.sleep(0.03)
        job3 = job_queue.run_once(callback, 10, name='name2')
        await asyncio.sleep(0.03)
        assert job_queue.jobs() == (job1, job2, job3)
        assert job_queue.get_jobs_by_name('name1') == (job1, job2)
        assert job_queue.get_jobs_by_name('name2') == (job3,)

    async def test_job_run(self, app):
        job = app.job_queue.run_repeating(self.job_run_once, 0.02)
        await asyncio.sleep(0.05)
        assert self.result == 0
        await job.run(app)
        assert self.result == 1

    async def test_enable_disable_job(self, job_queue):
        job = job_queue.run_repeating(self.job_run_once, 0.2)
        await asyncio.sleep(0.5)
        assert self.result == 2
        job.enabled = False
        assert not job.enabled
        await asyncio.sleep(0.5)
        assert self.result == 2
        job.enabled = True
        assert job.enabled
        await asyncio.sleep(0.5)
        assert self.result == 4

    async def test_remove_job(self, job_queue):
        job = job_queue.run_repeating(self.job_run_once, 0.2)
        await asyncio.sleep(0.5)
        assert self.result == 2
        assert not job.removed
        job.schedule_removal()
        assert job.removed
        await asyncio.sleep(0.5)
        assert self.result == 2

    async def test_equality(self, job_queue):
        job = job_queue.run_repeating(self.job_run_once, 0.2)
        job_2 = job_queue.run_repeating(self.job_run_once, 0.2)
        job_3 = Job(self.job_run_once, 0.2)
        job_3._job = job.job
        assert job == job
        assert job != job_queue
        assert job != job_2
        assert job == job_3
        assert hash(job) == hash(job)
        assert hash(job) != hash(job_queue)
        assert hash(job) != hash(job_2)
        assert hash(job) == hash(job_3)

    async def test_process_error_context(self, job_queue, app):
        app.add_error_handler(self.error_handler_context)
        job = job_queue.run_once(self.job_with_exception, 0.1, chat_id=42, user_id=43)
        await asyncio.sleep(0.15)
        assert self.received_error[0] == 'Test Error'
        assert self.received_error[1] is job
        self.received_error = None
        await job.run(app)
        assert self.received_error[0] == 'Test Error'
        assert self.received_error[1] is job
        assert self.received_error[2] is app.user_data[43]
        assert self.received_error[3] is app.chat_data[42]
        app.remove_error_handler(self.error_handler_context)
        self.received_error = None
        job = job_queue.run_once(self.job_with_exception, 0.1)
        await asyncio.sleep(0.15)
        assert self.received_error is None
        await job.run(app)
        assert self.received_error is None

    async def test_process_error_that_raises_errors(self, job_queue, app, caplog):
        app.add_error_handler(self.error_handler_raise_error)
        with caplog.at_level(logging.ERROR):
            job = job_queue.run_once(self.job_with_exception, 0.1)
        await asyncio.sleep(0.15)
        assert len(caplog.records) == 1
        rec = caplog.records[-1]
        assert 'An error was raised and an uncaught' in rec.getMessage()
        caplog.clear()
        with caplog.at_level(logging.ERROR):
            await job.run(app)
        assert len(caplog.records) == 1
        rec = caplog.records[-1]
        assert 'uncaught error was raised while handling' in rec.getMessage()
        caplog.clear()
        app.remove_error_handler(self.error_handler_raise_error)
        self.received_error = None
        with caplog.at_level(logging.ERROR):
            job = job_queue.run_once(self.job_with_exception, 0.1)
        await asyncio.sleep(0.15)
        assert len(caplog.records) == 1
        rec = caplog.records[-1]
        assert 'No error handlers are registered' in rec.getMessage()
        caplog.clear()
        with caplog.at_level(logging.ERROR):
            await job.run(app)
        assert len(caplog.records) == 1
        rec = caplog.records[-1]
        assert rec.name == 'telegram.ext.Application'
        assert 'No error handlers are registered' in rec.getMessage()

    async def test_custom_context(self, bot):
        application = ApplicationBuilder().token(bot.token).context_types(ContextTypes(context=CustomContext, bot_data=int, user_data=float, chat_data=complex)).build()
        job_queue = JobQueue()
        job_queue.set_application(application)

        async def callback(context):
            self.result = (type(context), context.user_data, context.chat_data, type(context.bot_data))
        await job_queue.start()
        job_queue.run_once(callback, 0.1)
        await asyncio.sleep(0.15)
        assert self.result == (CustomContext, None, None, int)
        await job_queue.stop()

    async def test_attribute_error(self):
        job = Job(self.job_run_once)
        with pytest.raises(AttributeError, match="nor 'apscheduler.job.Job' has attribute 'error'"):
            job.error

    @pytest.mark.parametrize('wait', [True, False])
    async def test_wait_on_shut_down(self, job_queue, wait):
        ready_event = asyncio.Event()

        async def callback(_):
            await ready_event.wait()
        await job_queue.start()
        job_queue.run_once(callback, when=0.1)
        await asyncio.sleep(0.15)
        task = asyncio.create_task(job_queue.stop(wait=wait))
        if wait:
            assert not task.done()
            ready_event.set()
            await asyncio.sleep(0.1)
            assert task.done()
        else:
            await asyncio.sleep(0.1)
            assert task.done()

    async def test_from_aps_job(self, job_queue):
        job = job_queue.run_once(self.job_run_once, 0.1, name='test_job')
        aps_job = job_queue.scheduler.get_job(job.id)
        tg_job = Job.from_aps_job(aps_job)
        assert tg_job is job
        assert tg_job.job is aps_job

    async def test_from_aps_job_missing_reference(self, job_queue):
        """We manually create a ext.Job and an aps job such that the former has no reference to the
        latter. Then we test that Job.from_aps_job() still sets the reference correctly.
        """
        job = Job(self.job_run_once)
        aps_job = job_queue.scheduler.add_job(func=job_queue.job_callback, args=(job_queue, job), trigger='interval', seconds=2, id='test_id')
        assert job.job is None
        tg_job = Job.from_aps_job(aps_job)
        assert tg_job is job
        assert tg_job.job is aps_job