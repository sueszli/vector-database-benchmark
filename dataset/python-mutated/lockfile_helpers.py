import datetime
import os
import stat
import warnings
import xmlrpc.client as xmlrpc
from contextlib import suppress
from urllib.parse import urlparse
from astropy import log
from astropy.config.paths import _find_home
from astropy.utils.data import get_readable_fileobj
from .errors import SAMPHubError, SAMPWarning

def read_lockfile(lockfilename):
    if False:
        i = 10
        return i + 15
    '\n    Read in the lockfile given by ``lockfilename`` into a dictionary.\n    '
    lockfiledict = {}
    with get_readable_fileobj(lockfilename) as f:
        for line in f:
            if not line.startswith('#'):
                (kw, val) = line.split('=')
                lockfiledict[kw.strip()] = val.strip()
    return lockfiledict

def write_lockfile(lockfilename, lockfiledict):
    if False:
        while True:
            i = 10
    lockfile = open(lockfilename, 'w')
    lockfile.close()
    os.chmod(lockfilename, stat.S_IREAD + stat.S_IWRITE)
    lockfile = open(lockfilename, 'w')
    now_iso = datetime.datetime.now().isoformat()
    lockfile.write(f'# SAMP lockfile written on {now_iso}\n')
    lockfile.write('# Standard Profile required keys\n')
    for (key, value) in lockfiledict.items():
        lockfile.write(f'{key}={value}\n')
    lockfile.close()

def create_lock_file(lockfilename=None, mode=None, hub_id=None, hub_params=None):
    if False:
        i = 10
        return i + 15
    remove_garbage_lock_files()
    lockfiledir = ''
    if 'SAMP_HUB' in os.environ:
        if os.environ['SAMP_HUB'].startswith('std-lockurl:'):
            lockfilename = os.environ['SAMP_HUB'][len('std-lockurl:'):]
            lockfile_parsed = urlparse(lockfilename)
            if lockfile_parsed[0] != 'file':
                warnings.warn(f'Unable to start a Hub with lockfile {lockfilename}. Start-up process aborted.', SAMPWarning)
                return False
            else:
                lockfilename = lockfile_parsed[2]
    elif lockfilename is None:
        log.debug('Running mode: ' + mode)
        if mode == 'single':
            lockfilename = os.path.join(_find_home(), '.samp')
        else:
            lockfiledir = os.path.join(_find_home(), '.samp-1')
            try:
                os.mkdir(lockfiledir)
            except OSError:
                pass
            finally:
                os.chmod(lockfiledir, stat.S_IREAD + stat.S_IWRITE + stat.S_IEXEC)
            lockfilename = os.path.join(lockfiledir, f'samp-hub-{hub_id}')
    else:
        log.debug('Running mode: multiple')
    (hub_is_running, lockfiledict) = check_running_hub(lockfilename)
    if hub_is_running:
        warnings.warn('Another SAMP Hub is already running. Start-up process aborted.', SAMPWarning)
        return False
    log.debug('Lock-file: ' + lockfilename)
    write_lockfile(lockfilename, hub_params)
    return lockfilename

def get_main_running_hub():
    if False:
        while True:
            i = 10
    '\n    Get either the hub given by the environment variable SAMP_HUB, or the one\n    given by the lockfile .samp in the user home directory.\n    '
    hubs = get_running_hubs()
    if not hubs:
        raise SAMPHubError('Unable to find a running SAMP Hub.')
    if 'SAMP_HUB' in os.environ:
        if os.environ['SAMP_HUB'].startswith('std-lockurl:'):
            lockfilename = os.environ['SAMP_HUB'][len('std-lockurl:'):]
        else:
            raise SAMPHubError('SAMP Hub profile not supported.')
    else:
        lockfilename = os.path.join(_find_home(), '.samp')
    return hubs[lockfilename]

def get_running_hubs():
    if False:
        while True:
            i = 10
    '\n    Return a dictionary containing the lock-file contents of all the currently\n    running hubs (single and/or multiple mode).\n\n    The dictionary format is:\n\n    ``{<lock-file>: {<token-name>: <token-string>, ...}, ...}``\n\n    where ``{<lock-file>}`` is the lock-file name, ``{<token-name>}`` and\n    ``{<token-string>}`` are the lock-file tokens (name and content).\n\n    Returns\n    -------\n    running_hubs : dict\n        Lock-file contents of all the currently running hubs.\n    '
    hubs = {}
    lockfilename = ''
    if 'SAMP_HUB' in os.environ:
        if os.environ['SAMP_HUB'].startswith('std-lockurl:'):
            lockfilename = os.environ['SAMP_HUB'][len('std-lockurl:'):]
    else:
        lockfilename = os.path.join(_find_home(), '.samp')
    (hub_is_running, lockfiledict) = check_running_hub(lockfilename)
    if hub_is_running:
        hubs[lockfilename] = lockfiledict
    lockfiledir = ''
    lockfiledir = os.path.join(_find_home(), '.samp-1')
    if os.path.isdir(lockfiledir):
        for filename in os.listdir(lockfiledir):
            if filename.startswith('samp-hub'):
                lockfilename = os.path.join(lockfiledir, filename)
                (hub_is_running, lockfiledict) = check_running_hub(lockfilename)
                if hub_is_running:
                    hubs[lockfilename] = lockfiledict
    return hubs

def check_running_hub(lockfilename):
    if False:
        i = 10
        return i + 15
    '\n    Test whether a hub identified by ``lockfilename`` is running or not.\n\n    Parameters\n    ----------\n    lockfilename : str\n        Lock-file name (path + file name) of the Hub to be tested.\n\n    Returns\n    -------\n    is_running : bool\n        Whether the hub is running\n    hub_params : dict\n        If the hub is running this contains the parameters from the lockfile\n    '
    is_running = False
    lockfiledict = {}
    try:
        lockfiledict = read_lockfile(lockfilename)
    except OSError:
        return (is_running, lockfiledict)
    if 'samp.hub.xmlrpc.url' in lockfiledict:
        try:
            proxy = xmlrpc.ServerProxy(lockfiledict['samp.hub.xmlrpc.url'].replace('\\', ''), allow_none=1)
            proxy.samp.hub.ping()
            is_running = True
        except xmlrpc.ProtocolError:
            is_running = True
        except OSError:
            pass
    return (is_running, lockfiledict)

def remove_garbage_lock_files():
    if False:
        i = 10
        return i + 15
    lockfilename = ''
    lockfilename = os.path.join(_find_home(), '.samp')
    (hub_is_running, lockfiledict) = check_running_hub(lockfilename)
    if not hub_is_running:
        if os.path.isfile(lockfilename):
            with suppress(OSError):
                os.remove(lockfilename)
    lockfiledir = os.path.join(_find_home(), '.samp-1')
    if os.path.isdir(lockfiledir):
        for filename in os.listdir(lockfiledir):
            if filename.startswith('samp-hub'):
                lockfilename = os.path.join(lockfiledir, filename)
                (hub_is_running, lockfiledict) = check_running_hub(lockfilename)
                if not hub_is_running:
                    if os.path.isfile(lockfilename):
                        with suppress(OSError):
                            os.remove(lockfilename)