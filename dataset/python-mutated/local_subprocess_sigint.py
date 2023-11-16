from __future__ import annotations
import logging
import os
import signal
import sys
from io import TextIOWrapper
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock
from tox.execute.local_sub_process import LocalSubProcessExecutor
from tox.execute.request import ExecuteRequest, StdinSource
from tox.report import NamedBytesIO
if TYPE_CHECKING:
    from types import FrameType
    from tox.execute import Outcome
logging.basicConfig(level=logging.DEBUG, format='%(relativeCreated)d\t%(levelname).1s\t%(message)s')
bad_process = Path(__file__).parent / 'bad_process.py'
executor = LocalSubProcessExecutor(colored=False)
request = ExecuteRequest(cmd=[sys.executable, bad_process, sys.argv[1]], cwd=Path().absolute(), env=os.environ.copy(), stdin=StdinSource.API, run_id='')
out_err = (TextIOWrapper(NamedBytesIO('out')), TextIOWrapper(NamedBytesIO('err')))

def show_outcome(outcome: Outcome | None) -> None:
    if False:
        for i in range(10):
            print('nop')
    if outcome is not None:
        print(outcome.exit_code)
        print(repr(outcome.out))
        print(repr(outcome.err))
        print(outcome.elapsed, end='')
        print('done show outcome', file=sys.stderr)

def handler(s: int, f: FrameType | None) -> None:
    if False:
        for i in range(10):
            print('nop')
    logging.info('signal %s at %s', s, f)
    global interrupt_done
    if interrupt_done is False:
        interrupt_done = True
        logging.info('interrupt via %s', status)
        status.interrupt()
        logging.info('interrupt finished via %s', status)
interrupt_done = False
signal.signal(signal.SIGINT, handler)
logging.info('PID %d start %r', os.getpid(), request)
tox_env = MagicMock(conf={'suicide_timeout': 0.01, 'interrupt_timeout': 0.05, 'terminate_timeout': 0.07})
try:
    with executor.call(request, show=False, out_err=out_err, env=tox_env) as status:
        logging.info('wait on %r', status)
        while status.exit_code is None:
            status.wait(timeout=0.01)
        logging.info('wait over on %r', status)
    show_outcome(status.outcome)
except Exception as exception:
    logging.exception(exception)
finally:
    logging.info('done')
    logging.shutdown()