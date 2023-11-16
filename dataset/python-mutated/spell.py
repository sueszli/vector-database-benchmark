"""Installing and configuring spell-checking for QtWebEngine."""
import glob
import os
import os.path
import re
from qutebrowser.utils import log, message, standarddir
_DICT_VERSION_RE = re.compile('.+-(?P<version>[0-9]+-[0-9]+?)\\.bdic')

def version(filename):
    if False:
        print('Hello World!')
    'Extract the version number from the dictionary file name.'
    match = _DICT_VERSION_RE.fullmatch(filename)
    if match is None:
        message.warning('Found a dictionary with a malformed name: {}'.format(filename))
        return None
    return tuple((int(n) for n in match.group('version').split('-')))

def dictionary_dir():
    if False:
        while True:
            i = 10
    "Return the path (str) to the QtWebEngine's dictionaries directory."
    return os.path.join(standarddir.data(), 'qtwebengine_dictionaries')

def local_files(code):
    if False:
        for i in range(10):
            print('nop')
    'Return all installed dictionaries for the given code.\n\n    The returned dictionaries are sorted by version, therefore the latest will\n    be the first element. The list will be empty if no dictionaries are found.\n    '
    pathname = os.path.join(dictionary_dir(), '{}*.bdic'.format(code))
    matching_dicts = glob.glob(pathname)
    versioned_dicts = []
    for matching_dict in matching_dicts:
        parsed_version = version(matching_dict)
        if parsed_version is not None:
            filename = os.path.basename(matching_dict)
            log.config.debug('Found file for dict {}: {}'.format(code, filename))
            versioned_dicts.append((parsed_version, filename))
    return [filename for (version, filename) in sorted(versioned_dicts, reverse=True)]

def local_filename(code):
    if False:
        i = 10
        return i + 15
    'Return the newest installed dictionary for the given code.\n\n    Return the filename of the installed dictionary with the highest version\n    number or None if the dictionary is not installed.\n    '
    all_installed = local_files(code)
    return all_installed[0] if all_installed else None

def init():
    if False:
        return 10
    'Initialize the dictionary path.'
    dict_dir = dictionary_dir()
    os.environ['QTWEBENGINE_DICTIONARIES_PATH'] = dict_dir