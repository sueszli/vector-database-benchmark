import json
from contextlib import nullcontext, suppress
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional, Callable
import requests
import httpie
from httpie.context import Environment, LogLevel
from httpie.internal.__build_channel__ import BUILD_CHANNEL
from httpie.internal.daemons import spawn_daemon
from httpie.utils import is_version_greater, open_with_lockfile
PACKAGE_INDEX_LINK = 'https://packages.httpie.io/latest.json'
FETCH_INTERVAL = timedelta(weeks=2)
WARN_INTERVAL = timedelta(weeks=1)
UPDATE_MESSAGE_FORMAT = 'A new HTTPie release ({last_released_version}) is available.\nTo see how you can update, please visit https://httpie.io/docs/cli/{installation_method}\n'
ALREADY_UP_TO_DATE_MESSAGE = 'You are already up-to-date.\n'

def _read_data_error_free(file: Path) -> Any:
    if False:
        i = 10
        return i + 15
    try:
        with open(file) as stream:
            return json.load(stream)
    except (ValueError, OSError):
        return {}

def _fetch_updates(env: Environment) -> str:
    if False:
        print('Hello World!')
    file = env.config.version_info_file
    data = _read_data_error_free(file)
    response = requests.get(PACKAGE_INDEX_LINK, verify=False)
    response.raise_for_status()
    data.setdefault('last_warned_date', None)
    data['last_fetched_date'] = datetime.now().isoformat()
    data['last_released_versions'] = response.json()
    with open_with_lockfile(file, 'w') as stream:
        json.dump(data, stream)

def fetch_updates(env: Environment, lazy: bool=True):
    if False:
        return 10
    if lazy:
        spawn_daemon('fetch_updates')
    else:
        _fetch_updates(env)

def maybe_fetch_updates(env: Environment) -> None:
    if False:
        return 10
    if env.config.get('disable_update_warnings'):
        return None
    data = _read_data_error_free(env.config.version_info_file)
    if data:
        current_date = datetime.now()
        last_fetched_date = datetime.fromisoformat(data['last_fetched_date'])
        earliest_fetch_date = last_fetched_date + FETCH_INTERVAL
        if current_date < earliest_fetch_date:
            return None
    fetch_updates(env)

def _get_suppress_context(env: Environment) -> Any:
    if False:
        while True:
            i = 10
    'Return a context manager that suppress\n    all possible errors.\n\n    Note: if you have set the developer_mode=True in\n    your config, then it will show all errors for easier\n    debugging.'
    if env.config.developer_mode:
        return nullcontext()
    else:
        return suppress(BaseException)

def _update_checker(func: Callable[[Environment], None]) -> Callable[[Environment], None]:
    if False:
        while True:
            i = 10
    'Control the execution of the update checker (suppress errors, trigger\n    auto updates etc.)'

    def wrapper(env: Environment) -> None:
        if False:
            for i in range(10):
                print('nop')
        with _get_suppress_context(env):
            func(env)
        with _get_suppress_context(env):
            maybe_fetch_updates(env)
    return wrapper

def _get_update_status(env: Environment) -> Optional[str]:
    if False:
        return 10
    'If there is a new update available, return the warning text.\n    Otherwise just return None.'
    file = env.config.version_info_file
    if not file.exists():
        return None
    with _get_suppress_context(env):
        with open_with_lockfile(file) as stream:
            version_info = json.load(stream)
        available_channels = version_info['last_released_versions']
        if BUILD_CHANNEL not in available_channels:
            return None
        current_version = httpie.__version__
        last_released_version = available_channels[BUILD_CHANNEL]
        if not is_version_greater(last_released_version, current_version):
            return None
        text = UPDATE_MESSAGE_FORMAT.format(last_released_version=last_released_version, installation_method=BUILD_CHANNEL)
        return text

def get_update_status(env: Environment) -> str:
    if False:
        return 10
    return _get_update_status(env) or ALREADY_UP_TO_DATE_MESSAGE

@_update_checker
def check_updates(env: Environment) -> None:
    if False:
        for i in range(10):
            print('nop')
    if env.config.get('disable_update_warnings'):
        return None
    file = env.config.version_info_file
    update_status = _get_update_status(env)
    if not update_status:
        return None
    with open_with_lockfile(file) as stream:
        version_info = json.load(stream)
    current_date = datetime.now()
    last_warned_date = version_info['last_warned_date']
    if last_warned_date is not None:
        earliest_warn_date = datetime.fromisoformat(last_warned_date) + WARN_INTERVAL
        if current_date < earliest_warn_date:
            return None
    env.log_error(update_status, level=LogLevel.INFO)
    version_info['last_warned_date'] = current_date.isoformat()
    with open_with_lockfile(file, 'w') as stream:
        json.dump(version_info, stream)