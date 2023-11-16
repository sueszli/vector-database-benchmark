import re
from datetime import datetime, timedelta
from functools import partial
from typing import Any, Dict, SupportsInt
from superset.jinja_context import PrestoTemplateProcessor

def DATE(ts: datetime, day_offset: SupportsInt=0, hour_offset: SupportsInt=0) -> str:
    if False:
        while True:
            i = 10
    'Current day as a string'
    (day_offset, hour_offset) = (int(day_offset), int(hour_offset))
    offset_day = (ts + timedelta(days=day_offset, hours=hour_offset)).date()
    return str(offset_day)

class CustomPrestoTemplateProcessor(PrestoTemplateProcessor):
    """A custom presto template processor for test."""
    engine = 'db_for_macros_testing'

    def process_template(self, sql: str, **kwargs) -> str:
        if False:
            i = 10
            return i + 15
        'Processes a sql template with $ style macro using regex.'
        macros = {'DATE': partial(DATE, datetime.utcnow())}
        macros.update(self._context)
        macros.update(kwargs)

        def replacer(match):
            if False:
                for i in range(10):
                    print('nop')
            'Expands $ style macros with corresponding function calls.'
            (macro_name, args_str) = match.groups()
            args = [a.strip() for a in args_str.split(',')]
            if args == ['']:
                args = []
            f = macros[macro_name[1:]]
            return f(*args)
        macro_names = ['$' + name for name in macros.keys()]
        pattern = '(%s)\\s*\\(([^()]*)\\)' % '|'.join(map(re.escape, macro_names))
        return re.sub(pattern, replacer, sql)