import datetime
import croniter
from twisted.internet import defer
from twisted.internet import reactor
from twisted.python import log
from zope.interface import implementer
from buildbot import config
from buildbot import util
from buildbot.changes.filter import ChangeFilter
from buildbot.interfaces import ITriggerableScheduler
from buildbot.process import buildstep
from buildbot.process import properties
from buildbot.schedulers import base
from buildbot.util.codebase import AbsoluteSourceStampsMixin

class Timed(AbsoluteSourceStampsMixin, base.BaseScheduler):
    """
    Parent class for timed schedulers.  This takes care of the (surprisingly
    subtle) mechanics of ensuring that each timed actuation runs to completion
    before the service stops.
    """
    compare_attrs = ('reason', 'createAbsoluteSourceStamps', 'onlyIfChanged', 'branch', 'fileIsImportant', 'change_filter', 'onlyImportant')
    reason = ''

    class NoBranch:
        pass

    def __init__(self, name, builderNames, reason='', createAbsoluteSourceStamps=False, onlyIfChanged=False, branch=NoBranch, change_filter=None, fileIsImportant=None, onlyImportant=False, **kwargs):
        if False:
            print('Hello World!')
        super().__init__(name, builderNames, **kwargs)
        self.lastActuated = None
        self.actuationLock = defer.DeferredLock()
        self.actuateOk = False
        self.actuateAt = None
        self.actuateAtTimer = None
        self.reason = util.bytes2unicode(reason % {'name': name})
        self.branch = branch
        self.change_filter = ChangeFilter.fromSchedulerConstructorArgs(change_filter=change_filter)
        self.createAbsoluteSourceStamps = createAbsoluteSourceStamps
        self.onlyIfChanged = onlyIfChanged
        if fileIsImportant and (not callable(fileIsImportant)):
            config.error('fileIsImportant must be a callable')
        self.fileIsImportant = fileIsImportant
        self.onlyImportant = onlyImportant
        self._reactor = reactor
        self.is_first_build = None

    @defer.inlineCallbacks
    def activate(self):
        if False:
            print('Hello World!')
        yield super().activate()
        if not self.enabled:
            return None
        self.actuateOk = True
        self.lastActuated = (yield self.getState('last_build', None))
        if self.lastActuated is None:
            self.is_first_build = True
        else:
            self.is_first_build = False
        yield self.scheduleNextBuild()
        if self.onlyIfChanged or self.createAbsoluteSourceStamps:
            yield self.startConsumingChanges(fileIsImportant=self.fileIsImportant, change_filter=self.change_filter, onlyImportant=self.onlyImportant)
        else:
            yield self.master.db.schedulers.flushChangeClassifications(self.serviceid)
        return None

    @defer.inlineCallbacks
    def deactivate(self):
        if False:
            while True:
                i = 10
        yield super().deactivate()
        if not self.enabled:
            return None

        def stop_actuating():
            if False:
                return 10
            self.actuateOk = False
            self.actuateAt = None
            if self.actuateAtTimer:
                self.actuateAtTimer.cancel()
            self.actuateAtTimer = None
        yield self.actuationLock.run(stop_actuating)
        return None

    def gotChange(self, change, important):
        if False:
            i = 10
            return i + 15
        if self.branch is not Timed.NoBranch and change.branch != self.branch:
            return defer.succeed(None)
        d = self.master.db.schedulers.classifyChanges(self.serviceid, {change.number: important})
        if self.createAbsoluteSourceStamps:
            d.addCallback(lambda _: self.recordChange(change))
        return d

    @defer.inlineCallbacks
    def startBuild(self):
        if False:
            i = 10
            return i + 15
        if not self.enabled:
            log.msg(format='ignoring build from %(name)s because scheduler has been disabled by the user', name=self.name)
            return
        scheds = self.master.db.schedulers
        classifications = (yield scheds.getChangeClassifications(self.serviceid))
        last_only_if_changed = (yield self.getState('last_only_if_changed', True))
        if last_only_if_changed and self.onlyIfChanged and (not any(classifications.values())) and (not self.is_first_build) and (not self.maybe_force_build_on_unimportant_changes(self.lastActuated)):
            log.msg(('{} scheduler <{}>: skipping build ' + '- No important changes').format(self.__class__.__name__, self.name))
            self.is_first_build = False
            return
        if last_only_if_changed != self.onlyIfChanged:
            yield self.setState('last_only_if_changed', self.onlyIfChanged)
        changeids = sorted(classifications.keys())
        if changeids:
            max_changeid = changeids[-1]
            yield self.addBuildsetForChanges(reason=self.reason, changeids=changeids, priority=self.priority)
            yield scheds.flushChangeClassifications(self.serviceid, less_than=max_changeid + 1)
        else:
            sourcestamps = [{'codebase': cb} for cb in self.codebases]
            yield self.addBuildsetForSourceStampsWithDefaults(reason=self.reason, sourcestamps=sourcestamps, priority=self.priority)
        self.is_first_build = False

    def getCodebaseDict(self, codebase):
        if False:
            for i in range(10):
                print('nop')
        if self.createAbsoluteSourceStamps:
            return super().getCodebaseDict(codebase)
        return self.codebases[codebase]

    def getNextBuildTime(self, lastActuation):
        if False:
            return 10
        '\n        Called by to calculate the next time to actuate a BuildSet.  Override\n        in subclasses.  To trigger a fresh call to this method, use\n        L{rescheduleNextBuild}.\n\n        @param lastActuation: the time of the last actuation, or None for never\n\n        @returns: a Deferred firing with the next time a build should occur (in\n        the future), or None for never.\n        '
        raise NotImplementedError

    def scheduleNextBuild(self):
        if False:
            return 10
        '\n        Schedule the next build, re-invoking L{getNextBuildTime}.  This can be\n        called at any time, and it will avoid contention with builds being\n        started concurrently.\n\n        @returns: Deferred\n        '
        return self.actuationLock.run(self._scheduleNextBuild_locked)

    def maybe_force_build_on_unimportant_changes(self, current_actuation_time):
        if False:
            i = 10
            return i + 15
        '\n        Allows forcing a build in cases when there are no important changes and onlyIfChanged is\n        enabled.\n        '
        return False

    def now(self):
        if False:
            return 10
        'Similar to util.now, but patchable by tests'
        return util.now(self._reactor)

    def current_utc_offset(self, tm):
        if False:
            i = 10
            return i + 15
        return (datetime.datetime.fromtimestamp(tm) - datetime.datetime.utcfromtimestamp(tm)).total_seconds()

    @defer.inlineCallbacks
    def _scheduleNextBuild_locked(self):
        if False:
            for i in range(10):
                print('nop')
        if self.actuateAtTimer:
            self.actuateAtTimer.cancel()
        self.actuateAtTimer = None
        actuateAt = (yield self.getNextBuildTime(self.lastActuated))
        if actuateAt is None:
            self.actuateAt = None
        else:
            now = self.now()
            self.actuateAt = max(actuateAt, now)
            untilNext = self.actuateAt - now
            if untilNext == 0:
                log.msg(f'{self.__class__.__name__} scheduler <{self.name}>: missed scheduled build time - building immediately')
            self.actuateAtTimer = self._reactor.callLater(untilNext, self._actuate)

    @defer.inlineCallbacks
    def _actuate(self):
        if False:
            i = 10
            return i + 15
        self.actuateAtTimer = None
        self.lastActuated = self.actuateAt

        @defer.inlineCallbacks
        def set_state_and_start():
            if False:
                i = 10
                return i + 15
            if not self.actuateOk:
                return
            self.actuateAt = None
            yield self.setState('last_build', self.lastActuated)
            try:
                yield self.startBuild()
            except Exception as e:
                log.err(e, 'while actuating')
            finally:
                yield self._scheduleNextBuild_locked()
        yield self.actuationLock.run(set_state_and_start)

class Periodic(Timed):
    compare_attrs = ('periodicBuildTimer',)

    def __init__(self, name, builderNames, periodicBuildTimer, reason="The Periodic scheduler named '%(name)s' triggered this build", **kwargs):
        if False:
            return 10
        super().__init__(name, builderNames, reason=reason, **kwargs)
        if periodicBuildTimer <= 0:
            config.error('periodicBuildTimer must be positive')
        self.periodicBuildTimer = periodicBuildTimer

    def getNextBuildTime(self, lastActuated):
        if False:
            print('Hello World!')
        if lastActuated is None:
            return defer.succeed(self.now())
        return defer.succeed(lastActuated + self.periodicBuildTimer)

class NightlyBase(Timed):
    compare_attrs = ('minute', 'hour', 'dayOfMonth', 'month', 'dayOfWeek', 'force_at_minute', 'force_at_hour', 'force_at_day_of_month', 'force_at_month', 'force_at_day_of_week')

    def __init__(self, name, builderNames, minute=0, hour='*', dayOfMonth='*', month='*', dayOfWeek='*', force_at_minute=None, force_at_hour=None, force_at_day_of_month=None, force_at_month=None, force_at_day_of_week=None, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__(name, builderNames, **kwargs)
        self.minute = minute
        self.hour = hour
        self.dayOfMonth = dayOfMonth
        self.month = month
        self.dayOfWeek = dayOfWeek
        self.force_at_enabled = force_at_minute is not None or force_at_hour is not None or force_at_day_of_month is not None or (force_at_month is not None) or (force_at_day_of_week is not None)

        def default_if_none(value, default):
            if False:
                for i in range(10):
                    print('nop')
            if value is None:
                return default
            return value
        self.force_at_minute = default_if_none(force_at_minute, 0)
        self.force_at_hour = default_if_none(force_at_hour, '*')
        self.force_at_day_of_month = default_if_none(force_at_day_of_month, '*')
        self.force_at_month = default_if_none(force_at_month, '*')
        self.force_at_day_of_week = default_if_none(force_at_day_of_week, '*')

    def _timeToCron(self, time, isDayOfWeek=False):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(time, int):
            if isDayOfWeek:
                time = (time + 1) % 7
            return time
        if isinstance(time, str):
            if isDayOfWeek:
                time_array = str(time).split(',')
                for (i, time_val) in enumerate(time_array):
                    try:
                        time_array[i] = (int(time_val) + 1) % 7
                    except ValueError:
                        pass
                return ','.join([str(s) for s in time_array])
            return time
        if isDayOfWeek:
            time = [(t + 1) % 7 for t in time]
        return ','.join([str(s) for s in time])

    def _times_to_cron_line(self, minute, hour, day_of_month, month, day_of_week):
        if False:
            for i in range(10):
                print('nop')
        return ' '.join([str(self._timeToCron(minute)), str(self._timeToCron(hour)), str(self._timeToCron(day_of_month)), str(self._timeToCron(month)), str(self._timeToCron(day_of_week, True))])

    def _time_to_croniter_tz_time(self, ts):
        if False:
            print('Hello World!')
        tz = datetime.timezone(datetime.timedelta(seconds=self.current_utc_offset(ts)))
        return datetime.datetime.fromtimestamp(ts, tz)

    def getNextBuildTime(self, lastActuated):
        if False:
            return 10
        ts = lastActuated or self.now()
        sched = self._times_to_cron_line(self.minute, self.hour, self.dayOfMonth, self.month, self.dayOfWeek)
        cron = croniter.croniter(sched, self._time_to_croniter_tz_time(ts))
        nextdate = cron.get_next(float)
        return defer.succeed(nextdate)

    def maybe_force_build_on_unimportant_changes(self, current_actuation_time):
        if False:
            while True:
                i = 10
        if not self.force_at_enabled:
            return False
        cron_string = self._times_to_cron_line(self.force_at_minute, self.force_at_hour, self.force_at_day_of_month, self.force_at_month, self.force_at_day_of_week)
        return croniter.croniter.match(cron_string, self._time_to_croniter_tz_time(current_actuation_time))

class Nightly(NightlyBase):

    def __init__(self, name, builderNames, minute=0, hour='*', dayOfMonth='*', month='*', dayOfWeek='*', reason="The Nightly scheduler named '%(name)s' triggered this build", force_at_minute=None, force_at_hour=None, force_at_day_of_month=None, force_at_month=None, force_at_day_of_week=None, **kwargs):
        if False:
            while True:
                i = 10
        super().__init__(name=name, builderNames=builderNames, minute=minute, hour=hour, dayOfMonth=dayOfMonth, month=month, dayOfWeek=dayOfWeek, reason=reason, force_at_minute=force_at_minute, force_at_hour=force_at_hour, force_at_day_of_month=force_at_day_of_month, force_at_month=force_at_month, force_at_day_of_week=force_at_day_of_week, **kwargs)

@implementer(ITriggerableScheduler)
class NightlyTriggerable(NightlyBase):

    def __init__(self, name, builderNames, minute=0, hour='*', dayOfMonth='*', month='*', dayOfWeek='*', reason="The NightlyTriggerable scheduler named '%(name)s' triggered this build", force_at_minute=None, force_at_hour=None, force_at_day_of_month=None, force_at_month=None, force_at_day_of_week=None, **kwargs):
        if False:
            while True:
                i = 10
        super().__init__(name=name, builderNames=builderNames, minute=minute, hour=hour, dayOfMonth=dayOfMonth, month=month, dayOfWeek=dayOfWeek, reason=reason, force_at_minute=force_at_minute, force_at_hour=force_at_hour, force_at_day_of_month=force_at_day_of_month, force_at_month=force_at_month, force_at_day_of_week=force_at_day_of_week, **kwargs)
        self._lastTrigger = None

    @defer.inlineCallbacks
    def activate(self):
        if False:
            while True:
                i = 10
        yield super().activate()
        if not self.enabled:
            return
        lastTrigger = (yield self.getState('lastTrigger', None))
        self._lastTrigger = None
        if lastTrigger:
            try:
                if isinstance(lastTrigger[0], list):
                    self._lastTrigger = (lastTrigger[0], properties.Properties.fromDict(lastTrigger[1]), lastTrigger[2], lastTrigger[3])
                elif isinstance(lastTrigger[0], dict):
                    self._lastTrigger = (list(lastTrigger[0].values()), properties.Properties.fromDict(lastTrigger[1]), None, None)
            except Exception:
                pass
            if not self._lastTrigger:
                log.msg(format='NightlyTriggerable Scheduler <%(scheduler)s>: could not load previous state; starting fresh', scheduler=self.name)

    def trigger(self, waited_for, sourcestamps=None, set_props=None, parent_buildid=None, parent_relationship=None):
        if False:
            for i in range(10):
                print('nop')
        'Trigger this scheduler with the given sourcestamp ID. Returns a\n        deferred that will fire when the buildset is finished.'
        assert isinstance(sourcestamps, list), 'trigger requires a list of sourcestamps'
        self._lastTrigger = (sourcestamps, set_props, parent_buildid, parent_relationship)
        if set_props:
            propsDict = set_props.asDict()
        else:
            propsDict = {}
        d = self.setState('lastTrigger', (sourcestamps, propsDict, parent_buildid, parent_relationship))
        return (defer.succeed((None, {})), d.addCallback(lambda _: buildstep.SUCCESS))

    @defer.inlineCallbacks
    def startBuild(self):
        if False:
            print('Hello World!')
        if not self.enabled:
            log.msg(format='ignoring build from %(name)s because scheduler has been disabled by the user', name=self.name)
            return
        if self._lastTrigger is None:
            return
        (sourcestamps, set_props, parent_buildid, parent_relationship) = self._lastTrigger
        self._lastTrigger = None
        yield self.setState('lastTrigger', None)
        props = properties.Properties()
        props.updateFromProperties(self.properties)
        if set_props:
            props.updateFromProperties(set_props)
        yield self.addBuildsetForSourceStampsWithDefaults(reason=self.reason, sourcestamps=sourcestamps, properties=props, parent_buildid=parent_buildid, parent_relationship=parent_relationship, priority=self.priority)