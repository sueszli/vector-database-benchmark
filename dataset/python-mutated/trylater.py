"""A delayed execution system.

The ``trylater`` module provides tools for performing an action at a set time
in the future.  To use it, you must do two things.

First, make a scheduling call::

    from datetime import timedelta

    from r2.models.trylater import TryLater

    def make_breakfast(spam):
        breakfast = cook(spam)
        later = timedelta(minutes=45)
        # The storage layer only likes strings.
        data = json.dumps(breakfast)
        TryLater.schedule('wash_dishes', data, later)

Then, write the delayed code and decorate it with a hook, using the same
identifier as you used when you scheduled it::

    from r2.lib import hooks
    trylater_hooks = hooks.HookRegistrar()

    @trylater_hooks.on('trylater.wash_dishes')
    def on_dish_washing(data):
        # data is an ordered dictionary of timeuuid -> data pairs.
        for datum in data.values():
            meal = json.loads(datum)
            for dish in meal.dishes:
                dish.wash()

Note: once you've scheduled a ``TryLater`` task, there's no stopping it!  If
you might need to cancel your jobs later, use ``TryLaterBySubject``, which uses
almost the exact same semantics, but has a useful ``unschedule`` method.
"""
from collections import OrderedDict
from datetime import datetime, timedelta
import uuid
from pycassa.system_manager import TIME_UUID_TYPE, UTF8_TYPE
from pycassa.util import convert_uuid_to_time
from pylons import app_globals as g
from r2.lib.db import tdb_cassandra
from r2.lib.utils import tup

class TryLater(tdb_cassandra.View):
    _use_db = True
    _read_consistency_level = tdb_cassandra.CL.QUORUM
    _write_consistency_level = tdb_cassandra.CL.QUORUM
    _compare_with = TIME_UUID_TYPE

    @classmethod
    def process_ready_items(cls, rowkey, ready_fn):
        if False:
            i = 10
            return i + 15
        cutoff = datetime.now(g.tz)
        columns = cls._cf.xget(rowkey, include_timestamp=True)
        ready_items = OrderedDict()
        ready_timestamps = []
        unripe_timestamps = []
        for (ready_time_uuid, (data, timestamp)) in columns:
            ready_time = convert_uuid_to_time(ready_time_uuid)
            ready_datetime = datetime.fromtimestamp(ready_time, tz=g.tz)
            if ready_datetime <= cutoff:
                ready_items[ready_time_uuid] = data
                ready_timestamps.append(timestamp)
            else:
                unripe_timestamps.append(timestamp)
        g.stats.simple_event('trylater.{system}.ready'.format(system=rowkey), delta=len(ready_items))
        g.stats.simple_event('trylater.{system}.pending'.format(system=rowkey), delta=len(unripe_timestamps))
        if not ready_items:
            return
        try:
            ready_fn(ready_items)
        except:
            g.stats.simple_event('trylater.{system}.failed'.format(system=rowkey))
        cls.cleanup(rowkey, ready_items, ready_timestamps, unripe_timestamps)

    @classmethod
    def cleanup(cls, rowkey, ready_items, ready_timestamps, unripe_timestamps):
        if False:
            print('Hello World!')
        'Remove ALL ready items from the C* row'
        if not unripe_timestamps or min(unripe_timestamps) > max(ready_timestamps):
            cls._cf.remove(rowkey, timestamp=max(ready_timestamps))
            g.stats.simple_event('trylater.{system}.row_delete'.format(system=rowkey), delta=len(ready_items))
        else:
            cls._cf.remove(rowkey, ready_items.keys())
            g.stats.simple_event('trylater.{system}.column_delete'.format(system=rowkey), delta=len(ready_items))

    @classmethod
    def run(cls):
        if False:
            i = 10
            return i + 15
        'Run all ready items through their processing hook.'
        from r2.lib import amqp
        from r2.lib.hooks import all_hooks
        for (hook_name, hook) in all_hooks().items():
            if hook_name.startswith('trylater.'):
                rowkey = hook_name[len('trylater.'):]

                def ready_fn(ready_items):
                    if False:
                        return 10
                    return hook.call(data=ready_items)
                g.log.info('Trying %s', rowkey)
                cls.process_ready_items(rowkey, ready_fn)
        amqp.worker.join()
        g.stats.flush()

    @classmethod
    def search(cls, rowkey, when):
        if False:
            i = 10
            return i + 15
        if isinstance(when, uuid.UUID):
            when = convert_uuid_to_time(when)
        try:
            return cls._cf.get(rowkey, column_start=when, column_finish=when)
        except tdb_cassandra.NotFoundException:
            return {}

    @classmethod
    def schedule(cls, system, data, delay=None):
        if False:
            for i in range(10):
                print('nop')
        'Schedule code for later execution.\n\n        system:  an string identifying the hook to be executed\n        data:    passed to the hook as an argument\n        delay:   (optional) a datetime.timedelta indicating the desired\n                 execution time\n        '
        if delay is None:
            delay = timedelta(minutes=60)
        key = datetime.now(g.tz) + delay
        scheduled = {key: data}
        cls._set_values(system, scheduled)
        return scheduled

    @classmethod
    def unschedule(cls, rowkey, column_keys):
        if False:
            print('Hello World!')
        column_keys = tup(column_keys)
        return cls._cf.remove(rowkey, column_keys)

class TryLaterBySubject(tdb_cassandra.View):
    _use_db = True
    _read_consistency_level = tdb_cassandra.CL.QUORUM
    _write_consistency_level = tdb_cassandra.CL.QUORUM
    _compare_with = UTF8_TYPE
    _extra_schema_creation_args = {'key_validation_class': UTF8_TYPE, 'default_validation_class': TIME_UUID_TYPE}
    _value_type = 'date'

    @classmethod
    def schedule(cls, system, subject, data, delay, trylater_rowkey=None):
        if False:
            for i in range(10):
                print('nop')
        if trylater_rowkey is None:
            trylater_rowkey = system
        scheduled = TryLater.schedule(trylater_rowkey, data, delay)
        when = scheduled.keys()[0]
        ttl = (delay + timedelta(minutes=10)).total_seconds()
        coldict = {subject: when}
        cls._set_values(system, coldict, ttl=ttl)
        return scheduled

    @classmethod
    def search(cls, rowkey, subjects=None):
        if False:
            i = 10
            return i + 15
        try:
            if subjects:
                subjects = tup(subjects)
                return cls._cf.get(rowkey, subjects)
            else:
                return cls._cf.get(rowkey)
        except tdb_cassandra.NotFoundException:
            return {}

    @classmethod
    def unschedule(cls, rowkey, colkey, schedule_rowkey):
        if False:
            i = 10
            return i + 15
        colkey = tup(colkey)
        victims = cls.search(rowkey, colkey)
        for uu in victims.itervalues():
            keys = TryLater.search(schedule_rowkey, uu).keys()
            TryLater.unschedule(schedule_rowkey, keys)
        cls._cf.remove(rowkey, colkey)