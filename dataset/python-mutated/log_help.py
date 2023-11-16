"""
A module to assist with using the Python logging module

"""
import datetime
import errno
import logging
import os
from logging import handlers
from os import path
HANDLERS = []

class IndentFormatter(logging.Formatter):

    def format(self, record, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Format a message in the log\n\n        Act like the normal format, but indent anything that is a\n        newline within the message.\n\n        '
        return logging.Formatter.format(self, record, *args, **kwargs).replace('\n', '\n' + ' ' * 8)

def configure(*args, **kwargs):
    if False:
        return 10
    '\n    Configure logging.\n\n    Borrowed from logging.basicConfig\n\n    Uses the IndentFormatter instead of the regular Formatter\n\n    Also, opts the caller into Syslog output, unless syslog could not\n    be opened for some reason or another, in which case a warning will\n    be printed to the other log handlers.\n\n    '
    assert len(HANDLERS) == 0
    log_destinations = get_log_destinations()
    if 'stderr' in log_destinations:
        HANDLERS.append(logging.StreamHandler())

    def terrible_log_output(s):
        if False:
            i = 10
            return i + 15
        import sys
        print(s, file=sys.stderr)
    places = ['/dev/log', '/var/run/log', '/var/run/syslog']
    default_syslog_address = places[0]
    for p in places:
        if path.exists(p):
            default_syslog_address = p
            break
    syslog_address = kwargs.setdefault('syslog_address', default_syslog_address)
    valid_facility = False
    if 'syslog' in log_destinations:
        (facility, valid_facility) = get_syslog_facility()
        if not valid_facility:
            terrible_log_output('invalid syslog facility level specified')
        try:
            HANDLERS.append(handlers.SysLogHandler(syslog_address, facility=facility))
        except EnvironmentError as e:
            if e.errno in [errno.EACCES, errno.ECONNREFUSED]:
                message = 'wal-e: Could not set up syslog, continuing anyway.  Reason: {0}'.format(errno.errorcode[e.errno])
                terrible_log_output(message)
    fs = kwargs.get('format', logging.BASIC_FORMAT)
    dfs = kwargs.get('datefmt', None)
    fmt = IndentFormatter(fs, dfs)
    for handler in HANDLERS:
        handler.setFormatter(fmt)
        logging.root.addHandler(handler)
    set_level(kwargs.get('level', logging.INFO))

def get_log_destinations():
    if False:
        print('Hello World!')
    'Parse env string'
    env = os.getenv('WALE_LOG_DESTINATION', 'stderr,syslog')
    return env.split(',')

def get_syslog_facility():
    if False:
        print('Hello World!')
    'Get syslog facility from ENV var'
    facil = os.getenv('WALE_SYSLOG_FACILITY', 'user')
    valid_facility = True
    try:
        facility = handlers.SysLogHandler.facility_names[facil.lower()]
    except KeyError:
        valid_facility = False
        facility = handlers.SysLogHandler.LOG_USER
    return (facility, valid_facility)

def set_level(level):
    if False:
        i = 10
        return i + 15
    'Adjust the logging level of WAL-E'
    for handler in HANDLERS:
        handler.setLevel(level)
    logging.root.setLevel(level)

class WalELogger(object):

    def __init__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        self._logger = logging.getLogger(*args, **kwargs)

    @staticmethod
    def _fmt_structured(d):
        if False:
            return 10
        "Formats '{k1:v1, k2:v2}' => 'time=... pid=... k1=v1 k2=v2'\n\n        Output is lexically sorted, *except* the time and pid always\n        come first, to assist with human scanning of the data.\n        "
        timeEntry = datetime.datetime.utcnow().strftime('time=%Y-%m-%dT%H:%M:%S.%f-00')
        pidEntry = 'pid=' + str(os.getpid())
        rest = sorted(('='.join([str(k), str(v)]) for (k, v) in list(d.items())))
        return ' '.join([timeEntry, pidEntry] + rest)

    @staticmethod
    def fmt_logline(msg, detail=None, hint=None, structured=None):
        if False:
            i = 10
            return i + 15
        msg_parts = ['MSG: ' + msg]
        if detail is not None:
            msg_parts.append('DETAIL: ' + detail)
        if hint is not None:
            msg_parts.append('HINT: ' + hint)
        if structured is None:
            structured = {}
        msg_parts.append('STRUCTURED: ' + WalELogger._fmt_structured(structured))
        return '\n'.join(msg_parts)

    def log(self, level, msg, *args, **kwargs):
        if False:
            print('Hello World!')
        detail = kwargs.pop('detail', None)
        hint = kwargs.pop('hint', None)
        structured = kwargs.pop('structured', None)
        self._logger.log(level, self.fmt_logline(msg, detail, hint, structured), *args, **kwargs)

    def debug(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        self.log(logging.DEBUG, *args, **kwargs)

    def info(self, *args, **kwargs):
        if False:
            print('Hello World!')
        self.log(logging.INFO, *args, **kwargs)

    def warning(self, *args, **kwargs):
        if False:
            return 10
        self.log(logging.WARNING, *args, **kwargs)

    def error(self, *args, **kwargs):
        if False:
            return 10
        self.log(logging.ERROR, *args, **kwargs)

    def critical(self, *args, **kwargs):
        if False:
            return 10
        self.log(logging.CRITICAL, *args, **kwargs)