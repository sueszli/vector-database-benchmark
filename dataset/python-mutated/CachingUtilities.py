import hashlib
import logging
import os
import pickle
import appdirs
from coalib.misc import Constants
from coalib import VERSION
USER_DATA_DIR = appdirs.user_data_dir('coala', version=VERSION)
Constants.USER_DATA_DIR = USER_DATA_DIR

def get_data_path(log_printer, identifier):
    if False:
        return 10
    "\n    Get the full path of ``identifier`` present in the user's data directory.\n\n    :param log_printer: A LogPrinter object to use for logging.\n    :param identifier:  The file whose path needs to be expanded.\n    :return:            Full path of the file, assuming it's present in the\n                        user's config directory.\n                        Returns ``None`` if there is a ``PermissionError``\n                        in creating the directory.\n    "
    try:
        os.makedirs(Constants.USER_DATA_DIR, exist_ok=True)
        return os.path.join(Constants.USER_DATA_DIR, hash_id(identifier))
    except PermissionError:
        logging.error(f"Unable to create user data directory '{Constants.USER_DATA_DIR}'. Continuing without caching.")
    return None

def delete_files(log_printer, identifiers):
    if False:
        for i in range(10):
            print('nop')
    "\n    Delete the given identifiers from the user's coala data directory.\n\n    :param log_printer: A LogPrinter object to use for logging.\n    :param identifiers: The list of files to be deleted.\n    :return:            True if all the given files were successfully deleted.\n                        False otherwise.\n    "
    error_files = []
    result = True
    for identifier in identifiers:
        try:
            file_path = get_data_path(None, identifier)
            if os.path.isfile(file_path):
                os.remove(file_path)
            else:
                result = False
        except (OSError, TypeError):
            error_files.append(hash_id(identifier))
    if len(error_files) > 0:
        error_files = ', '.join(error_files)
        logging.warning(f"There was a problem deleting the following files: {error_files}. Please delete them manually from '{Constants.USER_DATA_DIR}'.")
        result = False
    return result

def pickle_load(log_printer, identifier, fallback=None):
    if False:
        while True:
            i = 10
    "\n    Unpickle the data stored in ``identifier`` file and return it.\n\n    Example usage:\n\n    >>> test_data = {'answer': 42}\n    >>> pickle_dump(None, 'test_project', test_data)\n    True\n    >>> pickle_load(None, 'test_project')\n    {'answer': 42}\n    >>> pickle_load(None, 'nonexistent_project')\n    >>> pickle_load(None, 'nonexistent_project', fallback=42)\n    42\n\n    :param log_printer: A LogPrinter object to use for logging.\n    :param identifier:  The name of the file present in the user config\n                        directory.\n    :param fallback:    Return value to fallback to in case the file doesn't\n                        exist.\n    :return:            Data that is present in the file, if the file exists.\n                        Otherwise the ``default`` value is returned.\n    "
    file_path = get_data_path(None, identifier)
    if file_path is None or not os.path.isfile(file_path):
        return fallback
    with open(file_path, 'rb') as f:
        try:
            return pickle.load(f)
        except (pickle.UnpicklingError, EOFError):
            logging.warning('The given file is corrupted and will be removed.')
            delete_files(None, [identifier])
            return fallback

def pickle_dump(log_printer, identifier, data):
    if False:
        return 10
    '\n    Pickle the ``data`` and write into the ``identifier`` file.\n\n    :param log_printer: A LogPrinter object to use for logging.\n    :param identifier:  The name of the file present in the user config\n                        directory.\n    :param data:        Data to be serialized and written to the file using\n                        pickle.\n    :return:            True if the write was successful.\n                        False if there was a permission error in writing.\n    '
    file_path = get_data_path(None, identifier)
    if file_path is None:
        return False
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)
    return True

def hash_id(text):
    if False:
        for i in range(10):
            print('nop')
    '\n    Hashes the given text.\n\n    :param text: String to to be hashed\n    :return:     A MD5 hash of the given string\n    '
    return hashlib.md5(text.encode('utf-8')).hexdigest()

def get_settings_hash(sections, targets=[], ignore_settings: list=['disable_caching']):
    if False:
        return 10
    '\n    Compute and return a unique hash for the settings.\n\n    :param sections:        A dict containing the settings for each section.\n    :param targets:         The list of sections that are enabled.\n    :param ignore_settings: Setting keys to remove from sections before\n                            hashing.\n    :return:                A MD5 hash that is unique to the settings used.\n    '
    settings = []
    for section in sections:
        if section in targets or targets == []:
            section_copy = sections[section].copy()
            for setting in ignore_settings:
                try:
                    section_copy.__getitem__(setting, ignore_defaults=True)
                    section_copy.delete_setting(setting)
                except IndexError:
                    continue
            settings.append(str(section_copy))
    return hash_id(str(settings))

def settings_changed(log_printer, settings_hash):
    if False:
        i = 10
        return i + 15
    '\n    Determine if the settings have changed since the last run with caching.\n\n    :param log_printer:   A LogPrinter object to use for logging.\n    :param settings_hash: A MD5 hash that is unique to the settings used.\n    :return:              Return True if the settings hash has changed\n                          Return False otherwise.\n    '
    project_hash = hash_id(os.getcwd())
    settings_hash_db = pickle_load(None, 'settings_hash_db', {})
    if project_hash not in settings_hash_db:
        return False
    result = settings_hash_db[project_hash] != settings_hash
    if result:
        del settings_hash_db[project_hash]
        logging.debug('Since the configuration settings have changed since the last run, the cache will be flushed and rebuilt.')
    return result

def update_settings_db(log_printer, settings_hash):
    if False:
        while True:
            i = 10
    '\n    Update the config file last modification date.\n\n    :param log_printer:   A LogPrinter object to use for logging.\n    :param settings_hash: A MD5 hash that is unique to the settings used.\n    '
    project_hash = hash_id(os.getcwd())
    settings_hash_db = pickle_load(None, 'settings_hash_db', {})
    settings_hash_db[project_hash] = settings_hash
    pickle_dump(None, 'settings_hash_db', settings_hash_db)