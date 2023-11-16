"""
Some light wrappers around Python's multiprocessing, to deal with cleanly
starting child processes.
"""
import multiprocessing
import os
import sys
from multiprocessing.context import SpawnProcess
from socket import socket
from typing import Callable, List, Optional
from uvicorn.config import Config
multiprocessing.allow_connection_pickling()
spawn = multiprocessing.get_context('spawn')

def get_subprocess(config: Config, target: Callable[..., None], sockets: List[socket]) -> SpawnProcess:
    if False:
        return 10
    '\n    Called in the parent process, to instantiate a new child process instance.\n    The child is not yet started at this point.\n\n    * config - The Uvicorn configuration instance.\n    * target - A callable that accepts a list of sockets. In practice this will\n               be the `Server.run()` method.\n    * sockets - A list of sockets to pass to the server. Sockets are bound once\n                by the parent process, and then passed to the child processes.\n    '
    stdin_fileno: Optional[int]
    try:
        stdin_fileno = sys.stdin.fileno()
    except OSError:
        stdin_fileno = None
    kwargs = {'config': config, 'target': target, 'sockets': sockets, 'stdin_fileno': stdin_fileno}
    return spawn.Process(target=subprocess_started, kwargs=kwargs)

def subprocess_started(config: Config, target: Callable[..., None], sockets: List[socket], stdin_fileno: Optional[int]) -> None:
    if False:
        i = 10
        return i + 15
    '\n    Called when the child process starts.\n\n    * config - The Uvicorn configuration instance.\n    * target - A callable that accepts a list of sockets. In practice this will\n               be the `Server.run()` method.\n    * sockets - A list of sockets to pass to the server. Sockets are bound once\n                by the parent process, and then passed to the child processes.\n    * stdin_fileno - The file number of sys.stdin, so that it can be reattached\n                     to the child process.\n    '
    if stdin_fileno is not None:
        sys.stdin = os.fdopen(stdin_fileno)
    config.configure_logging()
    target(sockets=sockets)