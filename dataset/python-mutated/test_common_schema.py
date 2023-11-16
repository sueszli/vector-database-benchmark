from __future__ import annotations
import datetime
import pytest
from dateutil import relativedelta
from airflow.api_connexion.schemas.common_schema import CronExpression, CronExpressionSchema, RelativeDeltaSchema, ScheduleIntervalSchema, TimeDeltaSchema

class TestTimeDeltaSchema:

    def test_should_serialize(self):
        if False:
            print('Hello World!')
        instance = datetime.timedelta(days=12)
        schema_instance = TimeDeltaSchema()
        result = schema_instance.dump(instance)
        assert {'__type': 'TimeDelta', 'days': 12, 'seconds': 0, 'microseconds': 0} == result

    def test_should_deserialize(self):
        if False:
            print('Hello World!')
        instance = {'__type': 'TimeDelta', 'days': 12, 'seconds': 0, 'microseconds': 0}
        schema_instance = TimeDeltaSchema()
        result = schema_instance.load(instance)
        expected_instance = datetime.timedelta(days=12)
        assert expected_instance == result

class TestRelativeDeltaSchema:

    def test_should_serialize(self):
        if False:
            print('Hello World!')
        instance = relativedelta.relativedelta(days=+12)
        schema_instance = RelativeDeltaSchema()
        result = schema_instance.dump(instance)
        assert {'__type': 'RelativeDelta', 'day': None, 'days': 12, 'hour': None, 'hours': 0, 'leapdays': 0, 'microsecond': None, 'microseconds': 0, 'minute': None, 'minutes': 0, 'month': None, 'months': 0, 'second': None, 'seconds': 0, 'year': None, 'years': 0} == result

    def test_should_deserialize(self):
        if False:
            i = 10
            return i + 15
        instance = {'__type': 'RelativeDelta', 'days': 12, 'seconds': 0}
        schema_instance = RelativeDeltaSchema()
        result = schema_instance.load(instance)
        expected_instance = relativedelta.relativedelta(days=+12)
        assert expected_instance == result

class TestCronExpressionSchema:

    def test_should_deserialize(self):
        if False:
            print('Hello World!')
        instance = {'__type': 'CronExpression', 'value': '5 4 * * *'}
        schema_instance = CronExpressionSchema()
        result = schema_instance.load(instance)
        expected_instance = CronExpression('5 4 * * *')
        assert expected_instance == result

class TestScheduleIntervalSchema:

    def test_should_serialize_timedelta(self):
        if False:
            i = 10
            return i + 15
        instance = datetime.timedelta(days=12)
        schema_instance = ScheduleIntervalSchema()
        result = schema_instance.dump(instance)
        assert {'__type': 'TimeDelta', 'days': 12, 'seconds': 0, 'microseconds': 0} == result

    def test_should_deserialize_timedelta(self):
        if False:
            i = 10
            return i + 15
        instance = {'__type': 'TimeDelta', 'days': 12, 'seconds': 0, 'microseconds': 0}
        schema_instance = ScheduleIntervalSchema()
        result = schema_instance.load(instance)
        expected_instance = datetime.timedelta(days=12)
        assert expected_instance == result

    def test_should_serialize_relative_delta(self):
        if False:
            while True:
                i = 10
        instance = relativedelta.relativedelta(days=+12)
        schema_instance = ScheduleIntervalSchema()
        result = schema_instance.dump(instance)
        assert {'__type': 'RelativeDelta', 'day': None, 'days': 12, 'hour': None, 'hours': 0, 'leapdays': 0, 'microsecond': None, 'microseconds': 0, 'minute': None, 'minutes': 0, 'month': None, 'months': 0, 'second': None, 'seconds': 0, 'year': None, 'years': 0} == result

    def test_should_deserialize_relative_delta(self):
        if False:
            print('Hello World!')
        instance = {'__type': 'RelativeDelta', 'days': 12, 'seconds': 0}
        schema_instance = ScheduleIntervalSchema()
        result = schema_instance.load(instance)
        expected_instance = relativedelta.relativedelta(days=+12)
        assert expected_instance == result

    def test_should_serialize_cron_expression(self):
        if False:
            for i in range(10):
                print('nop')
        instance = '5 4 * * *'
        schema_instance = ScheduleIntervalSchema()
        result = schema_instance.dump(instance)
        expected_instance = {'__type': 'CronExpression', 'value': '5 4 * * *'}
        assert expected_instance == result

    def test_should_error_unknown_obj_type(self):
        if False:
            return 10
        instance = 342
        schema_instance = ScheduleIntervalSchema()
        with pytest.raises(Exception, match='Unknown object type: int'):
            schema_instance.dump(instance)