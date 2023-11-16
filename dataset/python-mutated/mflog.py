import heapq
import re
import time
import uuid
from datetime import datetime
from collections import namedtuple
from metaflow.util import to_bytes, to_fileobj, to_unicode
VERSION = b'0'
RE = b'(\\[!)?\\[MFLOG\\|(0)\\|(.+?)Z\\|(.+?)\\|(.+?)\\](.*)'
MFLogline = namedtuple('MFLogline', ['should_persist', 'version', 'utc_tstamp_str', 'logsource', 'id', 'msg', 'utc_tstamp'])
LINE_PARSER = re.compile(RE)
ISOFORMAT = '%Y-%m-%dT%H:%M:%S.%f'
MISSING_TIMESTAMP = datetime(3000, 1, 1)
MISSING_TIMESTAMP_STR = MISSING_TIMESTAMP.strftime(ISOFORMAT)
if time.timezone == 0:
    utc_to_local = lambda x: x
else:
    try:
        from datetime import timezone

        def utc_to_local(utc_dt):
            if False:
                for i in range(10):
                    print('nop')
            return utc_dt.replace(tzinfo=timezone.utc).astimezone(tz=None)
    except ImportError:
        import calendar

        def utc_to_local(utc_dt):
            if False:
                i = 10
                return i + 15
            timestamp = calendar.timegm(utc_dt.timetuple())
            local_dt = datetime.fromtimestamp(timestamp)
            return local_dt.replace(microsecond=utc_dt.microsecond)

def decorate(source, line, version=VERSION, now=None, lineid=None):
    if False:
        while True:
            i = 10
    if now is None:
        now = datetime.utcnow()
    tstamp = to_bytes(now.strftime(ISOFORMAT))
    if not lineid:
        lineid = to_bytes(str(uuid.uuid4()))
    line = to_bytes(line)
    source = to_bytes(source)
    return b''.join((b'[MFLOG|', version, b'|', tstamp, b'Z|', source, b'|', lineid, b']', line))

def is_structured(line):
    if False:
        i = 10
        return i + 15
    line = to_bytes(line)
    return line.startswith(b'[MFLOG|') or line.startswith(b'[![MFLOG|')

def parse(line):
    if False:
        for i in range(10):
            print('nop')
    line = to_bytes(line)
    m = LINE_PARSER.match(to_bytes(line))
    if m:
        try:
            fields = list(m.groups())
            fields.append(datetime.strptime(to_unicode(fields[2]), ISOFORMAT))
            return MFLogline(*fields)
        except:
            pass

def set_should_persist(line):
    if False:
        i = 10
        return i + 15
    line = to_bytes(line)
    if is_structured(line) and (not line.startswith(b'[![')):
        return b'[!' + line
    else:
        return line

def unset_should_persist(line):
    if False:
        for i in range(10):
            print('nop')
    line = to_bytes(line)
    if is_structured(line) and line.startswith(b'[!['):
        return line[2:]
    else:
        return line

def refine(line, prefix=None, suffix=None):
    if False:
        print('Hello World!')
    line = to_bytes(line)
    prefix = to_bytes(prefix) if prefix else b''
    suffix = to_bytes(suffix) if suffix else b''
    parts = line.split(b']', 1)
    if len(parts) == 2:
        (header, body) = parts
        return b''.join((header, b']', prefix, body, suffix))
    else:
        return line

def merge_logs(logs):
    if False:
        return 10

    def line_iter(logblob):
        if False:
            return 10
        missing = []
        for line in to_fileobj(logblob):
            res = parse(line)
            if res:
                yield (res.utc_tstamp_str, res)
            else:
                missing.append(line)
        for line in missing:
            res = MFLogline(False, None, MISSING_TIMESTAMP_STR.encode('utf-8'), None, None, line, MISSING_TIMESTAMP)
            yield (res.utc_tstamp_str, res)
    for (_, line) in heapq.merge(*[sorted(line_iter(blob)) for blob in logs]):
        yield line