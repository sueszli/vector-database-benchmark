"""json_formatter module stores all related to ElasticSearch specific logger classes."""
from __future__ import annotations
import json
import logging
from airflow.utils.helpers import merge_dicts

class JSONFormatter(logging.Formatter):
    """JSONFormatter instances are used to convert a log record to json."""

    def __init__(self, fmt=None, datefmt=None, style='%', json_fields=None, extras=None):
        if False:
            i = 10
            return i + 15
        super().__init__(fmt, datefmt, style)
        if extras is None:
            extras = {}
        if json_fields is None:
            json_fields = []
        self.json_fields = json_fields
        self.extras = extras

    def usesTime(self):
        if False:
            return 10
        return 'asctime' in self.json_fields

    def format(self, record):
        if False:
            print('Hello World!')
        super().format(record)
        record_dict = {label: getattr(record, label, None) for label in self.json_fields}
        if 'message' in self.json_fields:
            msg = record_dict['message']
            if record.exc_text:
                if msg[-1:] != '\n':
                    msg = msg + '\n'
                msg = msg + record.exc_text
            if record.stack_info:
                if msg[-1:] != '\n':
                    msg = msg + '\n'
                msg = msg + self.formatStack(record.stack_info)
            record_dict['message'] = msg
        merged_record = merge_dicts(record_dict, self.extras)
        return json.dumps(merged_record)