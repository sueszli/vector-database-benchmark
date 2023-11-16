"""
Contains information about newer version checker for SAM CLI
"""
import logging
from datetime import datetime, timedelta
from functools import wraps
import click
from requests import get
from samcli import __version__ as installed_version
from samcli.cli.global_config import GlobalConfig
LOG = logging.getLogger(__name__)
AWS_SAM_CLI_PYPI_ENDPOINT = 'https://pypi.org/pypi/aws-sam-cli/json'
AWS_SAM_CLI_INSTALL_DOCS = 'https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/serverless-sam-cli-install.html'
PYPI_CALL_TIMEOUT_IN_SECONDS = 5
DELTA_DAYS = 7

def check_newer_version(func):
    if False:
        return 10
    '\n    This function returns a wrapped function definition, which checks if there are newer version of SAM CLI available\n\n    Parameters\n    ----------\n    func: function reference\n        Actual function (command) which will be executed\n\n    Returns\n    -------\n    function reference:\n        A wrapped function reference which executes original function and checks newer version of SAM CLI\n    '

    @wraps(func)
    def wrapped(*args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        actual_result = func(*args, **kwargs)
        _inform_newer_version()
        return actual_result
    return wrapped

def _inform_newer_version(force_check=False) -> None:
    if False:
        i = 10
        return i + 15
    "\n    Compares installed SAM CLI version with the up to date version from PyPi,\n    and print information if up to date version is different then what is installed now\n\n    It will store last version check time into GlobalConfig, so that it won't be running all the time\n    Currently, it will be checking weekly\n\n    Parameters\n    ----------\n    force_check: bool\n        When it is True, it will trigger checking new version of SAM CLI. Default value is False\n\n    "
    global_config = None
    need_to_update_last_check_time = True
    try:
        global_config = GlobalConfig()
        last_version_check = global_config.last_version_check
        if force_check or is_version_check_overdue(last_version_check):
            fetch_and_compare_versions()
        else:
            need_to_update_last_check_time = False
    except Exception as e:
        LOG.debug('New version check failed', exc_info=e)
    finally:
        if need_to_update_last_check_time:
            update_last_check_time()

def fetch_and_compare_versions() -> None:
    if False:
        print('Hello World!')
    '\n    Compare current up to date version with the installed one, and inform if a newer version available\n    '
    response = get(AWS_SAM_CLI_PYPI_ENDPOINT, timeout=PYPI_CALL_TIMEOUT_IN_SECONDS)
    result = response.json()
    latest_version = result.get('info', {}).get('version', None)
    LOG.debug('Installed version %s, current version %s', installed_version, latest_version)
    if latest_version and installed_version != latest_version:
        click.secho(f'\nSAM CLI update available ({latest_version}); ({installed_version} installed)', fg='green', err=True)
        click.echo(f'To download: {AWS_SAM_CLI_INSTALL_DOCS}', err=True)

def update_last_check_time() -> None:
    if False:
        for i in range(10):
            print('nop')
    '\n    Update last_check_time in GlobalConfig\n    '
    try:
        gc = GlobalConfig()
        gc.last_version_check = datetime.utcnow().timestamp()
    except Exception as e:
        LOG.debug('Updating last version check time was failed', exc_info=e)

def is_version_check_overdue(last_version_check) -> bool:
    if False:
        while True:
            i = 10
    '\n    Check if last version check have been made longer then a week ago\n\n    Parameters\n    ----------\n    last_version_check: epoch time\n        last_version_check epoch time read from GlobalConfig\n\n    Returns\n    -------\n    bool:\n        True if last_version_check is None or older then a week, False otherwise\n    '
    if last_version_check is None or type(last_version_check) not in [int, float]:
        return True
    epoch_week_ago = datetime.utcnow() - timedelta(days=DELTA_DAYS)
    return datetime.utcfromtimestamp(last_version_check) < epoch_week_ago