from typing import Dict
import dns.exception
from dns._asyncbackend import Backend, DatagramSocket, Socket, StreamSocket
_default_backend = None
_backends: Dict[str, Backend] = {}
_no_sniffio = False

class AsyncLibraryNotFoundError(dns.exception.DNSException):
    pass

def get_backend(name: str) -> Backend:
    if False:
        i = 10
        return i + 15
    'Get the specified asynchronous backend.\n\n    *name*, a ``str``, the name of the backend.  Currently the "trio"\n    and "asyncio" backends are available.\n\n    Raises NotImplementError if an unknown backend name is specified.\n    '
    backend = _backends.get(name)
    if backend:
        return backend
    if name == 'trio':
        import dns._trio_backend
        backend = dns._trio_backend.Backend()
    elif name == 'asyncio':
        import dns._asyncio_backend
        backend = dns._asyncio_backend.Backend()
    else:
        raise NotImplementedError(f'unimplemented async backend {name}')
    _backends[name] = backend
    return backend

def sniff() -> str:
    if False:
        print('Hello World!')
    'Attempt to determine the in-use asynchronous I/O library by using\n    the ``sniffio`` module if it is available.\n\n    Returns the name of the library, or raises AsyncLibraryNotFoundError\n    if the library cannot be determined.\n    '
    try:
        if _no_sniffio:
            raise ImportError
        import sniffio
        try:
            return sniffio.current_async_library()
        except sniffio.AsyncLibraryNotFoundError:
            raise AsyncLibraryNotFoundError('sniffio cannot determine async library')
    except ImportError:
        import asyncio
        try:
            asyncio.get_running_loop()
            return 'asyncio'
        except RuntimeError:
            raise AsyncLibraryNotFoundError('no async library detected')

def get_default_backend() -> Backend:
    if False:
        print('Hello World!')
    'Get the default backend, initializing it if necessary.'
    if _default_backend:
        return _default_backend
    return set_default_backend(sniff())

def set_default_backend(name: str) -> Backend:
    if False:
        i = 10
        return i + 15
    "Set the default backend.\n\n    It's not normally necessary to call this method, as\n    ``get_default_backend()`` will initialize the backend\n    appropriately in many cases.  If ``sniffio`` is not installed, or\n    in testing situations, this function allows the backend to be set\n    explicitly.\n    "
    global _default_backend
    _default_backend = get_backend(name)
    return _default_backend