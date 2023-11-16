"""Faust environment variables."""
import os
from typing import Any, Sequence
from yarl import URL
__all__ = ['CONSOLE_PORT', 'DATADIR', 'DEBUG', 'STRICT', 'WEB_PORT', 'WEB_BIND', 'WEB_TRANSPORT', 'WORKDIR']
PREFICES: Sequence[str] = ['FAUST_', 'F_']

def _getenv(name: str, *default: Any, prefices: Sequence[str]=PREFICES) -> Any:
    if False:
        for i in range(10):
            print('nop')
    for prefix in prefices:
        try:
            return os.environ[prefix + name]
        except KeyError:
            pass
    if default:
        return default[0]
    raise KeyError(prefices[0] + name)
DEBUG: bool = bool(_getenv('DEBUG', False))
WORKDIR: str = _getenv('WORKDIR', None)
DATADIR: str = _getenv('DATADIR', '{conf.name}-data')
BLOCKING_TIMEOUT: float = float(_getenv('BLOCKING_TIMEOUT', '10.0'))
FORCE_BLOCKING_TIMEOUT: bool = bool(_getenv('FORCE_BLOCKING_TIMEOUT', ''))
CONSOLE_PORT: int = int(_getenv('CONSOLE_PORT', 50101))
STRICT: bool = bool(_getenv('STRICT', False))
WEB_PORT: int = int(_getenv('WEB_PORT', '6066'))
WEB_BIND: str = _getenv('F_WEB_BIND', '0.0.0.0')
WEB_TRANSPORT: URL = URL(_getenv('WEB_TRANSPORT', 'tcp://'))