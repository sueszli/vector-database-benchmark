import signal
from types import FrameType
from typing import Any, Callable, Dict, Optional, Union
SignalHandlerT = Union[Callable[[int, Optional[FrameType]], Any], int, signal.Handlers, None]
signal_names: Dict[int, str] = {}
for signame in dir(signal):
    if signame.startswith('SIG') and (not signame.startswith('SIG_')):
        signum = getattr(signal, signame)
        if isinstance(signum, int):
            signal_names[signum] = signame

def install_shutdown_handlers(function: SignalHandlerT, override_sigint: bool=True) -> None:
    if False:
        return 10
    "Install the given function as a signal handler for all common shutdown\n    signals (such as SIGINT, SIGTERM, etc). If ``override_sigint`` is ``False`` the\n    SIGINT handler won't be installed if there is already a handler in place\n    (e.g. Pdb)\n    "
    signal.signal(signal.SIGTERM, function)
    if signal.getsignal(signal.SIGINT) == signal.default_int_handler or override_sigint:
        signal.signal(signal.SIGINT, function)
    if hasattr(signal, 'SIGBREAK'):
        signal.signal(signal.SIGBREAK, function)