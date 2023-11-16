"""Migrate num_period_compare and period_ratio_type

Revision ID: 3dda56f1c4c6
Revises: bddc498dd179
Create Date: 2018-07-05 15:19:14.609299

"""
import datetime
import json
import isodate
from alembic import op
from sqlalchemy import Column, Integer, String, Text
from sqlalchemy.ext.declarative import declarative_base
from superset import db
from superset.utils.date_parser import parse_human_timedelta
revision = '3dda56f1c4c6'
down_revision = 'bddc498dd179'
Base = declarative_base()

class Slice(Base):
    __tablename__ = 'slices'
    id = Column(Integer, primary_key=True)
    datasource_type = Column(String(200))
    params = Column(Text)
comparison_type_map = {'factor': 'ratio', 'growth': 'percentage', 'value': 'absolute'}
db_engine_specs_map = {'second': 'PT1S', 'minute': 'PT1M', '5 minute': 'PT5M', '10 minute': 'PT10M', 'half hour': 'PT0.5H', 'hour': 'PT1H', 'day': 'P1D', 'week': 'P1W', 'week_ending_saturday': 'P1W', 'week_start_sunday': 'P1W', 'week_start_monday': 'P1W', 'week_starting_sunday': 'P1W', 'P1W/1970-01-03T00:00:00Z': 'P1W', '1969-12-28T00:00:00Z/P1W': 'P1W', 'month': 'P1M', 'quarter': 'P0.25Y', 'year': 'P1Y'}

def isodate_duration_to_string(obj):
    if False:
        print('Hello World!')
    if obj.tdelta:
        if not obj.months and (not obj.years):
            return format_seconds(obj.tdelta.total_seconds())
        raise Exception(f'Unable to convert: {obj}')
    if obj.months % 12 != 0:
        months = obj.months + 12 * obj.years
        return f'{months} months'
    return f'{obj.years + obj.months // 12} years'

def timedelta_to_string(obj):
    if False:
        return 10
    if obj.microseconds:
        raise Exception(f'Unable to convert: {obj}')
    elif obj.seconds:
        return format_seconds(obj.total_seconds())
    elif obj.days % 7 == 0:
        return f'{obj.days // 7} weeks'
    else:
        return f'{obj.days} days'

def format_seconds(value):
    if False:
        print('Hello World!')
    periods = [('minute', 60), ('hour', 3600), ('day', 86400), ('week', 604800)]
    for (period, multiple) in periods:
        if value % multiple == 0:
            value //= multiple
            break
    else:
        period = 'second'
    return f"{value} {period}{('s' if value > 1 else '')}"

def compute_time_compare(granularity, periods):
    if False:
        return 10
    if not granularity:
        return None
    if granularity in db_engine_specs_map:
        granularity = db_engine_specs_map[granularity]
    try:
        obj = isodate.parse_duration(granularity) * periods
    except isodate.isoerror.ISO8601Error:
        delta = f"{periods} {granularity}{('s' if periods > 1 else '')}"
        obj = parse_human_timedelta(delta)
        if obj:
            return delta
        raise Exception(f'Unable to parse: {granularity}')
    if isinstance(obj, isodate.duration.Duration):
        return isodate_duration_to_string(obj)
    elif isinstance(obj, datetime.timedelta):
        return timedelta_to_string(obj)

def upgrade():
    if False:
        i = 10
        return i + 15
    bind = op.get_bind()
    session = db.Session(bind=bind)
    for chart in session.query(Slice):
        params = json.loads(chart.params or '{}')
        if not params.get('num_period_compare'):
            continue
        num_period_compare = int(params.get('num_period_compare'))
        granularity = params.get('granularity') if chart.datasource_type == 'druid' else params.get('time_grain_sqla')
        time_compare = compute_time_compare(granularity, num_period_compare)
        period_ratio_type = params.get('period_ratio_type') or 'growth'
        comparison_type = comparison_type_map[period_ratio_type.lower()]
        params['time_compare'] = [time_compare]
        params['comparison_type'] = comparison_type
        chart.params = json.dumps(params, sort_keys=True)
    session.commit()
    session.close()

def downgrade():
    if False:
        return 10
    bind = op.get_bind()
    session = db.Session(bind=bind)
    for chart in session.query(Slice):
        params = json.loads(chart.params or '{}')
        if 'time_compare' in params or 'comparison_type' in params:
            params.pop('time_compare', None)
            params.pop('comparison_type', None)
            chart.params = json.dumps(params, sort_keys=True)
    session.commit()
    session.close()