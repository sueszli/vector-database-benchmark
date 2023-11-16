from __future__ import annotations
import logging
import pendulum

class TimezoneAware(logging.Formatter):
    """Override time-formatting methods to include UTC offset.

    Since Airflow parses the logs to perform time conversion, UTC offset is
    critical information. This formatter ensures ``%(asctime)s`` is formatted
    containing the offset in ISO 8601, e.g. ``2022-06-12T13:00:00.123+0000``.
    """
    default_time_format = '%Y-%m-%dT%H:%M:%S'
    default_msec_format = '%s.%03d'
    default_tz_format = '%z'

    def formatTime(self, record, datefmt=None):
        if False:
            return 10
        'Format time in record.\n\n        This returns the creation time of the specified LogRecord in ISO 8601\n        date and time format in the local time zone.\n        '
        dt = pendulum.from_timestamp(record.created, tz=pendulum.local_timezone())
        s = dt.strftime(datefmt or self.default_time_format)
        if self.default_msec_format:
            s = self.default_msec_format % (s, record.msecs)
        if self.default_tz_format:
            s += dt.strftime(self.default_tz_format)
        return s