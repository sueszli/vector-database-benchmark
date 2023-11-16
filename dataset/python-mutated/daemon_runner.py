import argparse
from contextlib import redirect_stderr, redirect_stdout
from typing import List
from httpie.context import Environment
from httpie.internal.update_warnings import _fetch_updates, _get_suppress_context
from httpie.status import ExitStatus
STATUS_FILE = '.httpie-test-daemon-status'

def _check_status(env):
    if False:
        for i in range(10):
            print('nop')
    import tempfile
    from pathlib import Path
    status_file = Path(tempfile.gettempdir()) / STATUS_FILE
    status_file.touch()
DAEMONIZED_TASKS = {'check_status': _check_status, 'fetch_updates': _fetch_updates}

def _parse_options(args: List[str]) -> argparse.Namespace:
    if False:
        while True:
            i = 10
    parser = argparse.ArgumentParser()
    parser.add_argument('task_id')
    parser.add_argument('--daemon', action='store_true')
    return parser.parse_known_args(args)[0]

def is_daemon_mode(args: List[str]) -> bool:
    if False:
        i = 10
        return i + 15
    return '--daemon' in args

def run_daemon_task(env: Environment, args: List[str]) -> ExitStatus:
    if False:
        i = 10
        return i + 15
    options = _parse_options(args)
    assert options.daemon
    assert options.task_id in DAEMONIZED_TASKS
    with redirect_stdout(env.devnull), redirect_stderr(env.devnull):
        with _get_suppress_context(env):
            DAEMONIZED_TASKS[options.task_id](env)
    return ExitStatus.SUCCESS