from __future__ import absolute_import
from __future__ import division
from __future__ import nested_scopes
from __future__ import print_function
from multiprocessing.managers import BaseManager
from multiprocessing import JoinableQueue

class TFManager(BaseManager):
    """Python multiprocessing.Manager for distributed, multi-process communication."""
    pass
mgr = None
qdict = {}
kdict = {}

def _get(key):
    if False:
        return 10
    return kdict[key]

def _set(key, value):
    if False:
        return 10
    kdict[key] = value

def _get_queue(qname):
    if False:
        while True:
            i = 10
    try:
        return qdict[qname]
    except KeyError:
        return None

def start(authkey, queues, mode='local'):
    if False:
        while True:
            i = 10
    "Create a new multiprocess.Manager (or return existing one).\n\n  Args:\n    :authkey: string authorization key\n    :queues: *INTERNAL_USE*\n    :mode: 'local' indicates that the manager will only be accessible from the same host, otherwise remotely accessible.\n\n  Returns:\n    A TFManager instance, which is also cached in local memory of the Python worker process.\n  "
    global mgr, qdict, kdict
    qdict.clear()
    kdict.clear()
    for q in queues:
        qdict[q] = JoinableQueue()
    TFManager.register('get_queue', callable=lambda qname: _get_queue(qname))
    TFManager.register('get', callable=lambda key: _get(key))
    TFManager.register('set', callable=lambda key, value: _set(key, value))
    if mode == 'remote':
        mgr = TFManager(address=('', 0), authkey=authkey)
    else:
        mgr = TFManager(authkey=authkey)
    mgr.start()
    return mgr

def connect(address, authkey):
    if False:
        print('Hello World!')
    "Connect to a multiprocess.Manager.\n\n  Args:\n    :address: unique address to the TFManager, either a unique connection string for 'local', or a (host, port) tuple for remote.\n    :authkey: string authorization key\n\n  Returns:\n    A TFManager instance referencing the remote TFManager at the supplied address.\n  "
    TFManager.register('get_queue')
    TFManager.register('get')
    TFManager.register('set')
    m = TFManager(address, authkey=authkey)
    m.connect()
    return m