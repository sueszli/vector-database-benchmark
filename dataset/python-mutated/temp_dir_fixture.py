import logging
import os
import tempfile
import unittest
import ethereum.keys
import shutil
from time import sleep
from pathlib import Path
from golem.core.common import is_windows, is_osx
from golem.core.simpleenv import get_local_datadir
logger = logging.getLogger(__name__)

class TempDirFixture(unittest.TestCase):
    root_dir = None

    @classmethod
    def setUpClass(cls):
        if False:
            print('Hello World!')
        logging.basicConfig(level=logging.DEBUG)
        if cls.root_dir is None:
            if is_osx():
                cls.root_dir = os.path.join(get_local_datadir('tests'))
                os.makedirs(cls.root_dir, exist_ok=True)
            else:
                cls.root_dir = tempfile.mkdtemp(prefix='golem-tests-')
                if is_windows():
                    import win32api
                    cls.root_dir = win32api.GetLongPathName(cls.root_dir)

    def setUp(self):
        if False:
            while True:
                i = 10
        ethereum.keys.PBKDF2_CONSTANTS['c'] = 1
        prefix = self.id().rsplit('.', 1)[1]
        self.tempdir = tempfile.mkdtemp(prefix=prefix, dir=self.root_dir)
        self.path = self.tempdir
        if not is_windows():
            os.chmod(self.tempdir, 504)
        self.new_path = Path(self.path)

    def tearDown(self):
        if False:
            return 10
        try:
            self.__remove_files()
        except OSError as e:
            logger.debug('%r', e, exc_info=True)
            tree = ''
            for (path, dirs, files) in os.walk(self.path):
                tree += path + '\n'
                for f in files:
                    tree += f + '\n'
            logger.error('Failed to remove files %r', tree)
            import gc
            gc.collect()
            sleep(3)
            self.__remove_files()

    def temp_file_name(self, name: str) -> str:
        if False:
            while True:
                i = 10
        return os.path.join(self.tempdir, name)

    def additional_dir_content(self, file_num_list, dir_=None, results=None, sub_dir=None):
        if False:
            return 10
        '\n        Create recursively additional temporary files in directories in given\n        directory.\n        For example file_num_list in format [5, [2], [4, []]] will create\n        5 files in self.tempdir directory, and 2 subdirectories - first one will\n        contain 2 tempfiles, second will contain 4 tempfiles and an empty\n        subdirectory.\n        :param file_num_list: list containing number of new files that should\n            be created in this directory or list describing file_num_list for\n            new inner directories\n        :param dir_: directory in which files should be created\n        :param results: list of created temporary files\n        :return:\n        '
        if dir_ is None:
            dir_ = self.tempdir
        if sub_dir:
            dir_ = os.path.join(dir_, sub_dir)
            if not os.path.exists(dir_):
                os.makedirs(dir_)
        if results is None:
            results = []
        for el in file_num_list:
            if isinstance(el, int):
                for i in range(el):
                    t = tempfile.NamedTemporaryFile(dir=dir_, delete=False)
                    results.append(t.name)
            else:
                new_dir = tempfile.mkdtemp(dir=dir_)
                self.additional_dir_content(el, new_dir, results)
        return results

    def __remove_files(self):
        if False:
            print('Hello World!')
        if os.path.isdir(self.tempdir):
            shutil.rmtree(self.tempdir)