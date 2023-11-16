import argparse
import importlib
import itertools
import multiprocessing
import os
import signal
import sys
from types import FrameType
from typing import Any, Callable, Dict, List, Optional
from twisted.internet.main import installReactor
_original_signal_handlers: Dict[int, Any] = {}

class ProxiedReactor:
    """
    Twisted tracks the 'installed' reactor as a global variable.
    (Actually, it does some module trickery, but the effect is similar.)

    The default EpollReactor is buggy if it's created before a process is
    forked, then used in the child.
    See https://twistedmatrix.com/trac/ticket/4759#comment:17.

    However, importing certain Twisted modules will automatically create and
    install a reactor if one hasn't already been installed.
    It's not normally possible to re-install a reactor.

    Given the goal of launching workers with fork() to only import the code once,
    this presents a conflict.
    Our work around is to 'install' this ProxiedReactor which prevents Twisted
    from creating and installing one, but which lets us replace the actual reactor
    in use later on.
    """

    def __init__(self) -> None:
        if False:
            return 10
        self.___reactor_target: Any = None

    def _install_real_reactor(self, new_reactor: Any) -> None:
        if False:
            while True:
                i = 10
        '\n        Install a real reactor for this ProxiedReactor to forward lookups onto.\n\n        This method is specific to our ProxiedReactor and should not clash with\n        any names used on an actual Twisted reactor.\n        '
        self.___reactor_target = new_reactor

    def __getattr__(self, attr_name: str) -> Any:
        if False:
            print('Hello World!')
        return getattr(self.___reactor_target, attr_name)

def _worker_entrypoint(func: Callable[[], None], proxy_reactor: ProxiedReactor, args: List[str]) -> None:
    if False:
        i = 10
        return i + 15
    "\n    Entrypoint for a forked worker process.\n\n    We just need to set up the command-line arguments, create our real reactor\n    and then kick off the worker's main() function.\n    "
    from synapse.util.stringutils import strtobool
    sys.argv = args
    for (sig, handler) in _original_signal_handlers.items():
        signal.signal(sig, handler)
    if strtobool(os.environ.get('SYNAPSE_COMPLEMENT_FORKING_LAUNCHER_ASYNC_IO_REACTOR', '0')):
        import asyncio
        from twisted.internet.asyncioreactor import AsyncioSelectorReactor
        reactor = AsyncioSelectorReactor(asyncio.get_event_loop())
        proxy_reactor._install_real_reactor(reactor)
    else:
        from twisted.internet.epollreactor import EPollReactor
        proxy_reactor._install_real_reactor(EPollReactor())
    func()

def main() -> None:
    if False:
        print('Hello World!')
    '\n    Entrypoint for the forking launcher.\n    '
    parser = argparse.ArgumentParser()
    parser.add_argument('db_config', help='Path to database config file')
    parser.add_argument('args', nargs='...', help='Argument groups separated by `--`. The first argument of each group is a Synapse app name. Subsequent arguments are passed through.')
    ns = parser.parse_args()
    args_by_worker: List[List[str]] = [list(args) for (cond, args) in itertools.groupby(ns.args, lambda ele: ele != '--') if cond and args]
    proxy_reactor = ProxiedReactor()
    installReactor(proxy_reactor)
    worker_functions = []
    for worker_args in args_by_worker:
        worker_module = importlib.import_module(worker_args[0])
        worker_functions.append(worker_module.main)
    from synapse._scripts import update_synapse_database
    update_proc = multiprocessing.Process(target=_worker_entrypoint, args=(update_synapse_database.main, proxy_reactor, ['update_synapse_database', '--database-config', ns.db_config, '--run-background-updates']))
    print('===== PREPARING DATABASE =====', file=sys.stderr)
    update_proc.start()
    update_proc.join()
    print('===== PREPARED DATABASE =====', file=sys.stderr)
    processes: List[multiprocessing.Process] = []

    def handle_signal(signum: int, frame: Optional[FrameType]) -> None:
        if False:
            return 10
        print(f'complement_fork_starter: Caught signal {signum}. Stopping children.', file=sys.stderr)
        for p in processes:
            if p.pid:
                os.kill(p.pid, signum)
    for sig in (signal.SIGINT, signal.SIGTERM):
        _original_signal_handlers[sig] = signal.signal(sig, handle_signal)
    for (func, worker_args) in zip(worker_functions, args_by_worker):
        process = multiprocessing.Process(target=_worker_entrypoint, args=(func, proxy_reactor, worker_args))
        process.start()
        processes.append(process)
    for process in processes:
        process.join()
if __name__ == '__main__':
    main()