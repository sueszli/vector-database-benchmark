"""This is a non compliant process that does not listens to signals"""
from __future__ import annotations
import os
import signal
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from types import FrameType
out = sys.stdout

def handler(signum: int, _: FrameType | None) -> None:
    if False:
        while True:
            i = 10
    _p(f'how about no signal {signum!r}')

def _p(m: str) -> None:
    if False:
        return 10
    out.write(f'{m}{os.linesep}')
    out.flush()
_p(f'start {__name__} with {sys.argv!r}')
signal.signal(signal.SIGINT, handler)
signal.signal(signal.SIGTERM, handler)
try:
    start_file = Path(sys.argv[1])
    _p(f'create {start_file}')
    start_file.write_text('')
    _p(f'created {start_file}')
    while True:
        time.sleep(0.01)
finally:
    _p(f'done {__name__}')