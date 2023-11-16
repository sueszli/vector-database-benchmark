from __future__ import annotations
import pendulum
from airflow.utils.log.json_formatter import JSONFormatter

class ElasticsearchJSONFormatter(JSONFormatter):
    """Convert a log record to JSON with ISO 8601 date and time format."""
    default_time_format = '%Y-%m-%dT%H:%M:%S'
    default_msec_format = '%s.%03d'
    default_tz_format = '%z'

    def formatTime(self, record, datefmt=None):
        if False:
            print('Hello World!')
        'Return the creation time of the LogRecord in ISO 8601 date/time format in the local time zone.'
        dt = pendulum.from_timestamp(record.created, tz=pendulum.local_timezone())
        s = dt.strftime(datefmt or self.default_time_format)
        if self.default_msec_format:
            s = self.default_msec_format % (s, record.msecs)
        if self.default_tz_format:
            s += dt.strftime(self.default_tz_format)
        return s