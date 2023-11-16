import os
from streamlit import config, file_util, util
DEFAULT_FOLDER_BLACKLIST = ['**/.*', '**/anaconda', '**/anaconda2', '**/anaconda3', '**/dist-packages', '**/miniconda', '**/miniconda2', '**/miniconda3', '**/node_modules', '**/pyenv', '**/site-packages', '**/venv', '**/virtualenv']

class FolderBlackList(object):
    """Implement a black list object with globbing.

    Note
    ----
    Blacklist any path that matches a glob in `DEFAULT_FOLDER_BLACKLIST`.

    """

    def __init__(self, folder_blacklist):
        if False:
            for i in range(10):
                print('nop')
        'Constructor.\n\n        Parameters\n        ----------\n        folder_blacklist : list of str\n            list of folder names with globbing to blacklist.\n\n        '
        self._folder_blacklist = list(folder_blacklist)
        self._folder_blacklist.extend(DEFAULT_FOLDER_BLACKLIST)
        if config.get_option('global.developmentMode'):
            self._folder_blacklist.append(os.path.dirname(__file__))

    def __repr__(self) -> str:
        if False:
            print('Hello World!')
        return util.repr_(self)

    def is_blacklisted(self, filepath):
        if False:
            for i in range(10):
                print('nop')
        'Test if filepath is in the blacklist.\n\n        Parameters\n        ----------\n        filepath : str\n            File path that we intend to test.\n\n        '
        return any((file_util.file_is_in_folder_glob(filepath, blacklisted_folder) for blacklisted_folder in self._folder_blacklist))