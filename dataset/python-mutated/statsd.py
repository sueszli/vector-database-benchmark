"""Bare-bones implementation of statsD's protocol, client-side"""
import logging
import socket
from re import sub
from gunicorn.glogging import Logger
METRIC_VAR = 'metric'
VALUE_VAR = 'value'
MTYPE_VAR = 'mtype'
GAUGE_TYPE = 'gauge'
COUNTER_TYPE = 'counter'
HISTOGRAM_TYPE = 'histogram'

class Statsd(Logger):
    """statsD-based instrumentation, that passes as a logger
    """

    def __init__(self, cfg):
        if False:
            for i in range(10):
                print('nop')
        Logger.__init__(self, cfg)
        self.prefix = sub('^(.+[^.]+)\\.*$', '\\g<1>.', cfg.statsd_prefix)
        if isinstance(cfg.statsd_host, str):
            address_family = socket.AF_UNIX
        else:
            address_family = socket.AF_INET
        try:
            self.sock = socket.socket(address_family, socket.SOCK_DGRAM)
            self.sock.connect(cfg.statsd_host)
        except Exception:
            self.sock = None
        self.dogstatsd_tags = cfg.dogstatsd_tags

    def critical(self, msg, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        Logger.critical(self, msg, *args, **kwargs)
        self.increment('gunicorn.log.critical', 1)

    def error(self, msg, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        Logger.error(self, msg, *args, **kwargs)
        self.increment('gunicorn.log.error', 1)

    def warning(self, msg, *args, **kwargs):
        if False:
            while True:
                i = 10
        Logger.warning(self, msg, *args, **kwargs)
        self.increment('gunicorn.log.warning', 1)

    def exception(self, msg, *args, **kwargs):
        if False:
            while True:
                i = 10
        Logger.exception(self, msg, *args, **kwargs)
        self.increment('gunicorn.log.exception', 1)

    def info(self, msg, *args, **kwargs):
        if False:
            print('Hello World!')
        self.log(logging.INFO, msg, *args, **kwargs)

    def debug(self, msg, *args, **kwargs):
        if False:
            print('Hello World!')
        self.log(logging.DEBUG, msg, *args, **kwargs)

    def log(self, lvl, msg, *args, **kwargs):
        if False:
            return 10
        'Log a given statistic if metric, value and type are present\n        '
        try:
            extra = kwargs.get('extra', None)
            if extra is not None:
                metric = extra.get(METRIC_VAR, None)
                value = extra.get(VALUE_VAR, None)
                typ = extra.get(MTYPE_VAR, None)
                if metric and value and typ:
                    if typ == GAUGE_TYPE:
                        self.gauge(metric, value)
                    elif typ == COUNTER_TYPE:
                        self.increment(metric, value)
                    elif typ == HISTOGRAM_TYPE:
                        self.histogram(metric, value)
                    else:
                        pass
            if msg:
                Logger.log(self, lvl, msg, *args, **kwargs)
        except Exception:
            Logger.warning(self, 'Failed to log to statsd', exc_info=True)

    def access(self, resp, req, environ, request_time):
        if False:
            i = 10
            return i + 15
        'Measure request duration\n        request_time is a datetime.timedelta\n        '
        Logger.access(self, resp, req, environ, request_time)
        duration_in_ms = request_time.seconds * 1000 + float(request_time.microseconds) / 10 ** 3
        status = resp.status
        if isinstance(status, str):
            status = int(status.split(None, 1)[0])
        self.histogram('gunicorn.request.duration', duration_in_ms)
        self.increment('gunicorn.requests', 1)
        self.increment('gunicorn.request.status.%d' % status, 1)

    def gauge(self, name, value):
        if False:
            for i in range(10):
                print('nop')
        self._sock_send('{0}{1}:{2}|g'.format(self.prefix, name, value))

    def increment(self, name, value, sampling_rate=1.0):
        if False:
            i = 10
            return i + 15
        self._sock_send('{0}{1}:{2}|c|@{3}'.format(self.prefix, name, value, sampling_rate))

    def decrement(self, name, value, sampling_rate=1.0):
        if False:
            return 10
        self._sock_send('{0}{1}:-{2}|c|@{3}'.format(self.prefix, name, value, sampling_rate))

    def histogram(self, name, value):
        if False:
            return 10
        self._sock_send('{0}{1}:{2}|ms'.format(self.prefix, name, value))

    def _sock_send(self, msg):
        if False:
            for i in range(10):
                print('nop')
        try:
            if isinstance(msg, str):
                msg = msg.encode('ascii')
            if self.dogstatsd_tags:
                msg = msg + b'|#' + self.dogstatsd_tags.encode('ascii')
            if self.sock:
                self.sock.send(msg)
        except Exception:
            Logger.warning(self, 'Error sending message to statsd', exc_info=True)