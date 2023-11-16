from __future__ import annotations
import asyncio
from logging import getLogger
from sys import exc_info
from typing import Any, NoReturn
from reactpy.backend.types import BackendType
from reactpy.backend.utils import SUPPORTED_BACKENDS, all_implementations
from reactpy.types import RootComponentConstructor
logger = getLogger(__name__)
_DEFAULT_IMPLEMENTATION: BackendType[Any] | None = None

class Options:
    """Configuration options that can be provided to the backend.
    This definition should not be used/instantiated. It exists only for
    type hinting purposes."""

    def __init__(self, *args: Any, **kwds: Any) -> NoReturn:
        if False:
            for i in range(10):
                print('nop')
        msg = 'Default implementation has no options.'
        raise ValueError(msg)

def configure(app: Any, component: RootComponentConstructor, options: None=None) -> None:
    if False:
        i = 10
        return i + 15
    'Configure the given app instance to display the given component'
    if options is not None:
        msg = 'Default implementation cannot be configured with options'
        raise ValueError(msg)
    return _default_implementation().configure(app, component)

def create_development_app() -> Any:
    if False:
        return 10
    'Create an application instance for development purposes'
    return _default_implementation().create_development_app()

async def serve_development_app(app: Any, host: str, port: int, started: asyncio.Event | None=None) -> None:
    """Run an application using a development server"""
    return await _default_implementation().serve_development_app(app, host, port, started)

def _default_implementation() -> BackendType[Any]:
    if False:
        i = 10
        return i + 15
    'Get the first available server implementation'
    global _DEFAULT_IMPLEMENTATION
    if _DEFAULT_IMPLEMENTATION is not None:
        return _DEFAULT_IMPLEMENTATION
    try:
        implementation = next(all_implementations())
    except StopIteration:
        logger.debug('Backend implementation import failed', exc_info=exc_info())
        supported_backends = ', '.join(SUPPORTED_BACKENDS)
        msg = f"""It seems you haven't installed a backend. To resolve this issue, you can install a backend by running:\n\n\x1b[1mpip install "reactpy[starlette]"\x1b[0m\n\nOther supported backends include: {supported_backends}."""
        raise RuntimeError(msg) from None
    else:
        _DEFAULT_IMPLEMENTATION = implementation
        return implementation