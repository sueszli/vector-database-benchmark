"""Support for http server log files"""
from __future__ import annotations
import datetime
import re
from typing import Any, Dict
from ivre.parser import Parser
from ivre.utils import LOGGER
LINE_RE = re.compile('^(?P<addr>[^ ]*) (?P<identity>[^ ]*) (?P<username>[^ ]*) \\[(?P<datetime>[^]]*)\\] "(?P<request>[^"]*)" (?P<status>[^ ]*) (?P<size>[^ ]*) "(?P<referer>[^"]*)" "(?P<useragent>[^"]*)"(?: "(?P<x_forwarded_for>[^"]*)")?\\r?$')

class WeblogFile(Parser):
    """Http server log generator"""

    def parse_line(self, line: bytes) -> Dict[str, Any]:
        if False:
            return 10
        m = LINE_RE.match(line.decode())
        if not m:
            LOGGER.warning('Cannot parse line [%r]', line)
            return {}
        try:
            timestamp = datetime.datetime.strptime(m.group('datetime').split()[0], '%d/%b/%Y:%H:%M:%S')
        except ValueError:
            LOGGER.warning('Cannot parse timestamp from line [%r]', line)
            return {}
        return {'host': m.group('addr'), 'ts': timestamp, 'useragent': m.group('useragent')}