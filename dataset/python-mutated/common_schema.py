from __future__ import annotations
import datetime
import inspect
import json
import typing
import marshmallow
from dateutil import relativedelta
from marshmallow import Schema, fields, validate
from marshmallow_oneofschema import OneOfSchema
from airflow.models.mappedoperator import MappedOperator
from airflow.serialization.serialized_objects import SerializedBaseOperator
from airflow.utils.weight_rule import WeightRule

class CronExpression(typing.NamedTuple):
    """Cron expression schema."""
    value: str

class TimeDeltaSchema(Schema):
    """Time delta schema."""
    objectType = fields.Constant('TimeDelta', data_key='__type')
    days = fields.Integer()
    seconds = fields.Integer()
    microseconds = fields.Integer()

    @marshmallow.post_load
    def make_time_delta(self, data, **kwargs):
        if False:
            while True:
                i = 10
        'Create time delta based on data.'
        data.pop('objectType', None)
        return datetime.timedelta(**data)

class RelativeDeltaSchema(Schema):
    """Relative delta schema."""
    objectType = fields.Constant('RelativeDelta', data_key='__type')
    years = fields.Integer()
    months = fields.Integer()
    days = fields.Integer()
    leapdays = fields.Integer()
    hours = fields.Integer()
    minutes = fields.Integer()
    seconds = fields.Integer()
    microseconds = fields.Integer()
    year = fields.Integer()
    month = fields.Integer()
    day = fields.Integer()
    hour = fields.Integer()
    minute = fields.Integer()
    second = fields.Integer()
    microsecond = fields.Integer()

    @marshmallow.post_load
    def make_relative_delta(self, data, **kwargs):
        if False:
            while True:
                i = 10
        'Create relative delta based on data.'
        data.pop('objectType', None)
        return relativedelta.relativedelta(**data)

class CronExpressionSchema(Schema):
    """Cron expression schema."""
    objectType = fields.Constant('CronExpression', data_key='__type')
    value = fields.String(required=True)

    @marshmallow.post_load
    def make_cron_expression(self, data, **kwargs):
        if False:
            while True:
                i = 10
        'Create cron expression based on data.'
        return CronExpression(data['value'])

class ScheduleIntervalSchema(OneOfSchema):
    """
    Schedule interval.

    It supports the following types:

    * TimeDelta
    * RelativeDelta
    * CronExpression
    """
    type_field = '__type'
    type_schemas = {'TimeDelta': TimeDeltaSchema, 'RelativeDelta': RelativeDeltaSchema, 'CronExpression': CronExpressionSchema}

    def _dump(self, obj, update_fields=True, **kwargs):
        if False:
            while True:
                i = 10
        if isinstance(obj, str):
            obj = CronExpression(obj)
        return super()._dump(obj, update_fields=update_fields, **kwargs)

    def get_obj_type(self, obj):
        if False:
            return 10
        'Select schema based on object type.'
        if isinstance(obj, datetime.timedelta):
            return 'TimeDelta'
        elif isinstance(obj, relativedelta.relativedelta):
            return 'RelativeDelta'
        elif isinstance(obj, CronExpression):
            return 'CronExpression'
        else:
            raise Exception(f'Unknown object type: {obj.__class__.__name__}')

class ColorField(fields.String):
    """Schema for color property."""

    def __init__(self, **metadata):
        if False:
            return 10
        super().__init__(**metadata)
        self.validators = [validate.Regexp('^#[a-fA-F0-9]{3,6}$'), *self.validators]

class WeightRuleField(fields.String):
    """Schema for WeightRule."""

    def __init__(self, **metadata):
        if False:
            return 10
        super().__init__(**metadata)
        self.validators = [validate.OneOf(WeightRule.all_weight_rules()), *self.validators]

class TimezoneField(fields.String):
    """Schema for timezone."""

class ClassReferenceSchema(Schema):
    """Class reference schema."""
    module_path = fields.Method('_get_module', required=True)
    class_name = fields.Method('_get_class_name', required=True)

    def _get_module(self, obj):
        if False:
            print('Hello World!')
        if isinstance(obj, (MappedOperator, SerializedBaseOperator)):
            return obj._task_module
        return inspect.getmodule(obj).__name__

    def _get_class_name(self, obj):
        if False:
            print('Hello World!')
        if isinstance(obj, (MappedOperator, SerializedBaseOperator)):
            return obj._task_type
        if isinstance(obj, type):
            return obj.__name__
        return type(obj).__name__

class JsonObjectField(fields.Field):
    """JSON object field."""

    def _serialize(self, value, attr, obj, **kwargs):
        if False:
            while True:
                i = 10
        if not value:
            return {}
        return json.loads(value) if isinstance(value, str) else value

    def _deserialize(self, value, attr, data, **kwargs):
        if False:
            print('Hello World!')
        if isinstance(value, str):
            return json.loads(value)
        return value