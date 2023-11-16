import datetime as dt
import pytest
try:
    import dateutil
except ImportError:
    dateutil = None
from mongoengine import *
from mongoengine import connection
from tests.utils import MongoDBTestCase, get_as_pymongo

class TestDateTimeField(MongoDBTestCase):

    def test_datetime_from_empty_string(self):
        if False:
            return 10
        '\n        Ensure an exception is raised when trying to\n        cast an empty string to datetime.\n        '

        class MyDoc(Document):
            dt = DateTimeField()
        md = MyDoc(dt='')
        with pytest.raises(ValidationError):
            md.save()

    def test_datetime_from_whitespace_string(self):
        if False:
            print('Hello World!')
        '\n        Ensure an exception is raised when trying to\n        cast a whitespace-only string to datetime.\n        '

        class MyDoc(Document):
            dt = DateTimeField()
        md = MyDoc(dt='   ')
        with pytest.raises(ValidationError):
            md.save()

    def test_default_value_utcnow(self):
        if False:
            i = 10
            return i + 15
        'Ensure that default field values are used when creating\n        a document.\n        '

        class Person(Document):
            created = DateTimeField(default=dt.datetime.utcnow)
        utcnow = dt.datetime.utcnow()
        person = Person()
        person.validate()
        person_created_t0 = person.created
        assert person.created - utcnow < dt.timedelta(seconds=1)
        assert person_created_t0 == person.created
        assert person._data['created'] == person.created

    def test_set_using_callable(self):
        if False:
            i = 10
            return i + 15

        class Person(Document):
            created = DateTimeField()
        Person.drop_collection()
        person = Person()
        frozen_dt = dt.datetime(2020, 7, 25, 9, 56, 1)
        person.created = lambda : frozen_dt
        person.save()
        assert callable(person.created)
        assert get_as_pymongo(person) == {'_id': person.id, 'created': frozen_dt}

    def test_handling_microseconds(self):
        if False:
            return 10
        'Tests showing pymongo datetime fields handling of microseconds.\n        Microseconds are rounded to the nearest millisecond and pre UTC\n        handling is wonky.\n\n        See: http://api.mongodb.org/python/current/api/bson/son.html#dt\n        '

        class LogEntry(Document):
            date = DateTimeField()
        LogEntry.drop_collection()
        log = LogEntry()
        log.date = dt.date.today()
        log.save()
        log.reload()
        assert log.date.date() == dt.date.today()
        d1 = dt.datetime(1970, 1, 1, 0, 0, 1, 999)
        d2 = dt.datetime(1970, 1, 1, 0, 0, 1)
        log = LogEntry()
        log.date = d1
        log.save()
        log.reload()
        assert log.date != d1
        assert log.date == d2
        d1 = dt.datetime(1970, 1, 1, 0, 0, 1, 9999)
        d2 = dt.datetime(1970, 1, 1, 0, 0, 1, 9000)
        log.date = d1
        log.save()
        log.reload()
        assert log.date != d1
        assert log.date == d2

    def test_regular_usage(self):
        if False:
            for i in range(10):
                print('nop')
        'Tests for regular datetime fields'

        class LogEntry(Document):
            date = DateTimeField()
        LogEntry.drop_collection()
        d1 = dt.datetime(1970, 1, 1, 0, 0, 1)
        log = LogEntry()
        log.date = d1
        log.validate()
        log.save()
        for query in (d1, d1.isoformat(' ')):
            log1 = LogEntry.objects.get(date=query)
            assert log == log1
        if dateutil:
            log1 = LogEntry.objects.get(date=d1.isoformat('T'))
            assert log == log1
        for i in range(1971, 1990):
            d = dt.datetime(i, 1, 1, 0, 0, 1)
            LogEntry(date=d).save()
        assert LogEntry.objects.count() == 20
        logs = LogEntry.objects.order_by('date')
        i = 0
        while i < 19:
            assert logs[i].date <= logs[i + 1].date
            i += 1
        logs = LogEntry.objects.order_by('-date')
        i = 0
        while i < 19:
            assert logs[i].date >= logs[i + 1].date
            i += 1
        logs = LogEntry.objects.filter(date__gte=dt.datetime(1980, 1, 1))
        assert logs.count() == 10
        logs = LogEntry.objects.filter(date__lte=dt.datetime(1980, 1, 1))
        assert logs.count() == 10
        logs = LogEntry.objects.filter(date__lte=dt.datetime(1980, 1, 1), date__gte=dt.datetime(1975, 1, 1))
        assert logs.count() == 5

    def test_datetime_validation(self):
        if False:
            for i in range(10):
                print('nop')
        'Ensure that invalid values cannot be assigned to datetime\n        fields.\n        '

        class LogEntry(Document):
            time = DateTimeField()
        log = LogEntry()
        log.time = dt.datetime.now()
        log.validate()
        log.time = dt.date.today()
        log.validate()
        log.time = dt.datetime.now().isoformat(' ')
        log.validate()
        log.time = '2019-05-16 21:42:57.897847'
        log.validate()
        if dateutil:
            log.time = dt.datetime.now().isoformat('T')
            log.validate()
        log.time = -1
        with pytest.raises(ValidationError):
            log.validate()
        log.time = 'ABC'
        with pytest.raises(ValidationError):
            log.validate()
        log.time = '2019-05-16 21:GARBAGE:12'
        with pytest.raises(ValidationError):
            log.validate()
        log.time = '2019-05-16 21:42:57.GARBAGE'
        with pytest.raises(ValidationError):
            log.validate()
        log.time = '2019-05-16 21:42:57.123.456'
        with pytest.raises(ValidationError):
            log.validate()

    def test_parse_datetime_as_str(self):
        if False:
            while True:
                i = 10

        class DTDoc(Document):
            date = DateTimeField()
        date_str = '2019-03-02 22:26:01'
        dtd = DTDoc()
        dtd.date = date_str
        assert isinstance(dtd.date, str)
        dtd.save()
        dtd.reload()
        assert isinstance(dtd.date, dt.datetime)
        assert str(dtd.date) == date_str
        dtd.date = 'January 1st, 9999999999'
        with pytest.raises(ValidationError):
            dtd.validate()

class TestDateTimeTzAware(MongoDBTestCase):

    def test_datetime_tz_aware_mark_as_changed(self):
        if False:
            i = 10
            return i + 15
        connection._connection_settings = {}
        connection._connections = {}
        connection._dbs = {}
        connect(db='mongoenginetest', tz_aware=True)

        class LogEntry(Document):
            time = DateTimeField()
        LogEntry.drop_collection()
        LogEntry(time=dt.datetime(2013, 1, 1, 0, 0, 0)).save()
        log = LogEntry.objects.first()
        log.time = dt.datetime(2013, 1, 1, 0, 0, 0)
        assert ['time'] == log._changed_fields