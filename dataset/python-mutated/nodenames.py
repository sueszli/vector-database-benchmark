"""Worker name utilities."""
import os
import socket
from functools import partial
from kombu.entity import Exchange, Queue
from .functional import memoize
from .text import simple_format
WORKER_DIRECT_EXCHANGE = Exchange('C.dq2')
WORKER_DIRECT_QUEUE_FORMAT = '{hostname}.dq2'
NODENAME_SEP = '@'
NODENAME_DEFAULT = 'celery'
gethostname = memoize(1, Cache=dict)(socket.gethostname)
__all__ = ('worker_direct', 'gethostname', 'nodename', 'anon_nodename', 'nodesplit', 'default_nodename', 'node_format', 'host_format')

def worker_direct(hostname):
    if False:
        i = 10
        return i + 15
    'Return the :class:`kombu.Queue` being a direct route to a worker.\n\n    Arguments:\n        hostname (str, ~kombu.Queue): The fully qualified node name of\n            a worker (e.g., ``w1@example.com``).  If passed a\n            :class:`kombu.Queue` instance it will simply return\n            that instead.\n    '
    if isinstance(hostname, Queue):
        return hostname
    return Queue(WORKER_DIRECT_QUEUE_FORMAT.format(hostname=hostname), WORKER_DIRECT_EXCHANGE, hostname)

def nodename(name, hostname):
    if False:
        i = 10
        return i + 15
    'Create node name from name/hostname pair.'
    return NODENAME_SEP.join((name, hostname))

def anon_nodename(hostname=None, prefix='gen'):
    if False:
        while True:
            i = 10
    'Return the nodename for this process (not a worker).\n\n    This is used for e.g. the origin task message field.\n    '
    return nodename(''.join([prefix, str(os.getpid())]), hostname or gethostname())

def nodesplit(name):
    if False:
        i = 10
        return i + 15
    'Split node name into tuple of name/hostname.'
    parts = name.split(NODENAME_SEP, 1)
    if len(parts) == 1:
        return (None, parts[0])
    return parts

def default_nodename(hostname):
    if False:
        for i in range(10):
            print('nop')
    'Return the default nodename for this process.'
    (name, host) = nodesplit(hostname or '')
    return nodename(name or NODENAME_DEFAULT, host or gethostname())

def node_format(s, name, **extra):
    if False:
        return 10
    'Format worker node name (name@host.com).'
    (shortname, host) = nodesplit(name)
    return host_format(s, host, shortname or NODENAME_DEFAULT, p=name, **extra)

def _fmt_process_index(prefix='', default='0'):
    if False:
        while True:
            i = 10
    from .log import current_process_index
    index = current_process_index()
    return f'{prefix}{index}' if index else default
_fmt_process_index_with_prefix = partial(_fmt_process_index, '-', '')

def host_format(s, host=None, name=None, **extra):
    if False:
        i = 10
        return i + 15
    'Format host %x abbreviations.'
    host = host or gethostname()
    (hname, _, domain) = host.partition('.')
    name = name or hname
    keys = dict({'h': host, 'n': name, 'd': domain, 'i': _fmt_process_index, 'I': _fmt_process_index_with_prefix}, **extra)
    return simple_format(s, keys)