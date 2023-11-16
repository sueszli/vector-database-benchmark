try:
    from urllib.parse import urlparse, urlunparse
except ImportError as e:
    raise ImportError('urllib cannot be found, urlparse from python2 is no longer supported.') from e
import numbers
import os
import sys
from datetime import timedelta
from typing import Dict, Optional
from torch.distributed import FileStore, PrefixStore, Store, TCPStore
from .constants import default_pg_timeout
_rendezvous_handlers = {}

def register_rendezvous_handler(scheme, handler):
    if False:
        print('Hello World!')
    '\n    Register a new rendezvous handler.\n\n    Before we can run collective algorithms, participating processes\n    need to find each other and exchange information to be able to\n    communicate. We call this process rendezvous.\n\n    The outcome of the rendezvous process is a triplet containing a\n    shared key/value store, the rank of the process, and the total\n    number of participating processes.\n\n    If none of the bundled rendezvous methods apply to your execution\n    environment you can opt to register your own rendezvous handler.\n    Pick a unique name and use the URL scheme to identify it when\n    calling the `rendezvous()` function.\n\n    Args:\n        scheme (str): URL scheme to identify your rendezvous handler.\n        handler (function): Handler that is invoked when the\n            `rendezvous()` function is called with a URL that uses\n            the corresponding scheme. It must be a generator function\n            that yields the triplet.\n    '
    global _rendezvous_handlers
    if scheme in _rendezvous_handlers:
        raise RuntimeError(f'Rendezvous handler for {scheme}:// already registered')
    _rendezvous_handlers[scheme] = handler

def _query_to_dict(query: str) -> Dict[str, str]:
    if False:
        print('Hello World!')
    return {pair[0]: pair[1] for pair in (pair.split('=') for pair in filter(None, query.split('&')))}

def _rendezvous_helper(url: str, rank: int, world_size_opt: Optional[int], **kwargs):
    if False:
        i = 10
        return i + 15
    result = urlparse(url)
    if world_size_opt is None:
        world_size = -1
        if result.scheme == 'env':
            rank = int(os.environ.get('RANK', rank))
            world_size = int(os.environ.get('WORLD_SIZE', world_size))
    else:
        world_size = world_size_opt
    if rank != -1 or world_size != -1 or world_size_opt is None:
        query_dict = _query_to_dict(result.query)
        assert 'rank' not in query_dict and 'world_size' not in query_dict, f'The url: {url} has node-specific arguments(rank, world_size) already.'
        if rank != -1:
            query_dict['rank'] = str(rank)
        if world_size != -1 or world_size_opt is None:
            query_dict['world_size'] = str(world_size)
        result = result._replace(query=f"{'&'.join([f'{k}={v}' for (k, v) in query_dict.items()])}")
        url = urlunparse(result)
    if result.scheme not in _rendezvous_handlers:
        raise RuntimeError(f'No rendezvous handler for {result.scheme}://')
    return _rendezvous_handlers[result.scheme](url, **kwargs)

def rendezvous(url: str, rank: int=-1, world_size: int=-1, **kwargs):
    if False:
        while True:
            i = 10
    if not isinstance(url, (str, bytes)):
        raise RuntimeError(f'`url` must be a string. {type(url)}: {url}')
    if not isinstance(rank, numbers.Integral):
        raise RuntimeError(f'`rank` must be an integer. {rank}')
    if not isinstance(world_size, numbers.Integral):
        raise RuntimeError(f'`world_size` must be an integer. {world_size}')
    return _rendezvous_helper(url, rank, world_size, **kwargs)

def _create_store_from_options(backend_options, rank):
    if False:
        i = 10
        return i + 15
    (store, _, _) = next(_rendezvous_helper(backend_options.init_method, rank, None))
    return store

def _rendezvous_error(msg):
    if False:
        for i in range(10):
            print('nop')
    return ValueError('Error initializing torch.distributed using ' + msg)

def _file_rendezvous_handler(url: str, **kwargs):
    if False:
        return 10

    def _error(msg):
        if False:
            print('Hello World!')
        return _rendezvous_error('file:// rendezvous: ' + msg)
    result = urlparse(url)
    path = result.path
    if sys.platform == 'win32':
        import urllib.request
        full_path = result.netloc + result.path
        path = urllib.request.url2pathname(full_path)
        if path:
            path = os.path.normpath(path)
    if not path:
        raise _error('path missing')
    query_dict = _query_to_dict(result.query)
    if 'rank' not in query_dict:
        raise _error('rank parameter missing')
    if 'world_size' not in query_dict:
        raise _error('world size parameter missing')
    rank = int(query_dict['rank'])
    world_size = int(query_dict['world_size'])
    store = FileStore(path, world_size)
    yield (store, rank, world_size)
    raise RuntimeError('Unable to perform rerendezvous using file:// method')

def _torchelastic_use_agent_store() -> bool:
    if False:
        while True:
            i = 10
    return os.environ.get('TORCHELASTIC_USE_AGENT_STORE', None) == str(True)

def _create_c10d_store(hostname, port, rank, world_size, timeout, use_libuv=False) -> Store:
    if False:
        i = 10
        return i + 15
    "\n    Smartly creates a c10d Store object on ``rank`` based on whether we need to re-use agent store.\n\n    The TCPStore server is assumed to be hosted\n    on ``hostname:port``.\n\n    If ``torchelastic_use_agent_store()`` is ``True``, then it is assumed that\n    the agent leader (node rank 0) hosts the TCPStore server (for which the\n    endpoint is specified by the given ``hostname:port``). Hence\n    ALL ranks will create and return a TCPStore client (e.g. ``start_daemon=False``).\n\n    If ``torchelastic_use_agent_store()`` is ``False``, then rank 0 will host\n    the TCPStore (with multi-tenancy) and it is assumed that rank 0's hostname\n    and port are correctly passed via ``hostname`` and ``port``. All\n    non-zero ranks will create and return a TCPStore client.\n    "
    if not 0 <= port < 2 ** 16:
        raise ValueError(f'port must have value from 0 to 65535 but was {port}.')
    if _torchelastic_use_agent_store():
        attempt = os.environ['TORCHELASTIC_RESTART_COUNT']
        tcp_store = TCPStore(hostname, port, world_size, False, timeout)
        return PrefixStore(f'/worker/attempt_{attempt}', tcp_store)
    else:
        start_daemon = rank == 0
        return TCPStore(hostname, port, world_size, start_daemon, timeout, multi_tenant=True, use_libuv=use_libuv)

def _tcp_rendezvous_handler(url: str, timeout: timedelta=default_pg_timeout, **kwargs):
    if False:
        print('Hello World!')

    def _error(msg):
        if False:
            for i in range(10):
                print('nop')
        return _rendezvous_error('tcp:// rendezvous: ' + msg)
    result = urlparse(url)
    if not result.port:
        raise _error('port number missing')
    query_dict = _query_to_dict(result.query)
    if 'rank' not in query_dict:
        raise _error('rank parameter missing')
    if 'world_size' not in query_dict:
        raise _error('world size parameter missing')
    rank = int(query_dict['rank'])
    world_size = int(query_dict['world_size'])
    use_libuv = query_dict.get('use_libuv', '0') == '1'
    assert result.hostname is not None
    store = _create_c10d_store(result.hostname, result.port, rank, world_size, timeout, use_libuv)
    yield (store, rank, world_size)
    raise RuntimeError('Unable to perform re-rendezvous using tcp:// method')

def _env_rendezvous_handler(url: str, timeout: timedelta=default_pg_timeout, **kwargs):
    if False:
        print('Hello World!')

    def _error(msg):
        if False:
            for i in range(10):
                print('nop')
        return _rendezvous_error('env:// rendezvous: ' + msg)

    def _env_error(var):
        if False:
            while True:
                i = 10
        return _error(f'environment variable {var} expected, but not set')

    def _get_env_or_raise(env_var: str) -> str:
        if False:
            i = 10
            return i + 15
        env_val = os.environ.get(env_var, None)
        if not env_val:
            raise _env_error(env_var)
        else:
            return env_val
    result = urlparse(url)
    query_dict = _query_to_dict(result.query)
    rank: int
    world_size: int
    master_port: int
    master_addr: str
    if 'rank' in query_dict:
        rank = int(query_dict['rank'])
    else:
        rank = int(_get_env_or_raise('RANK'))
    if 'world_size' in query_dict:
        world_size = int(query_dict['world_size'])
    else:
        world_size = int(_get_env_or_raise('WORLD_SIZE'))
    master_addr = _get_env_or_raise('MASTER_ADDR')
    master_port = int(_get_env_or_raise('MASTER_PORT'))
    use_libuv = query_dict.get('use_libuv', os.environ.get('USE_LIBUV', '0')) == '1'
    store = _create_c10d_store(master_addr, master_port, rank, world_size, timeout, use_libuv)
    yield (store, rank, world_size)
    raise RuntimeError('Unable to perform re-rendezvous using env:// method')
register_rendezvous_handler('tcp', _tcp_rendezvous_handler)
register_rendezvous_handler('env', _env_rendezvous_handler)
register_rendezvous_handler('file', _file_rendezvous_handler)