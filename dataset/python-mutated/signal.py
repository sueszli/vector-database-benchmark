import tempfile
import time
import os
import os.path
import mycroft
from .file_utils import ensure_directory_exists, create_file

def get_ipc_directory(domain=None):
    if False:
        print('Hello World!')
    'Get the directory used for Inter Process Communication\n\n    Files in this folder can be accessed by different processes on the\n    machine.  Useful for communication.  This is often a small RAM disk.\n\n    Args:\n        domain (str): The IPC domain.  Basically a subdirectory to prevent\n            overlapping signal filenames.\n\n    Returns:\n        str: a path to the IPC directory\n    '
    config = mycroft.configuration.Configuration.get()
    dir = config.get('ipc_path')
    if not dir:
        dir = os.path.join(tempfile.gettempdir(), 'mycroft', 'ipc')
    return ensure_directory_exists(dir, domain)

def create_signal(signal_name):
    if False:
        i = 10
        return i + 15
    "Create a named signal\n\n    Args:\n        signal_name (str): The signal's name.  Must only contain characters\n            valid in filenames.\n    "
    try:
        path = os.path.join(get_ipc_directory(), 'signal', signal_name)
        create_file(path)
        return os.path.isfile(path)
    except IOError:
        return False

def check_for_signal(signal_name, sec_lifetime=0):
    if False:
        return 10
    "See if a named signal exists\n\n    Args:\n        signal_name (str): The signal's name.  Must only contain characters\n            valid in filenames.\n        sec_lifetime (int, optional): How many seconds the signal should\n            remain valid.  If 0 or not specified, it is a single-use signal.\n            If -1, it never expires.\n\n    Returns:\n        bool: True if the signal is defined, False otherwise\n    "
    path = os.path.join(get_ipc_directory(), 'signal', signal_name)
    if os.path.isfile(path):
        if sec_lifetime == 0:
            os.remove(path)
        elif sec_lifetime == -1:
            return True
        elif int(os.path.getctime(path) + sec_lifetime) < int(time.time()):
            os.remove(path)
            return False
        return True
    return False