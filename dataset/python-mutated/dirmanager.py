import logging
import os
import shutil
import time
from typing import Iterator
logger = logging.getLogger(__name__)

def symlink_or_copy(source, target):
    if False:
        while True:
            i = 10
    try:
        os.symlink(source, target)
    except OSError:
        if os.path.isfile(source):
            if os.path.exists(target):
                os.remove(target)
            shutil.copy(source, target)
        else:
            from distutils import dir_util
            dir_util.copy_tree(source, target, update=1)

def rmlink_or_rmtree(target):
    if False:
        print('Hello World!')
    try:
        os.unlink(target)
    except OSError:
        if os.path.isfile(target):
            os.remove(target)
        else:
            shutil.rmtree(target)

def split_path(path):
    if False:
        return 10
    ' Split given path into a list of directories\n    :param str path: path that should be split\n    :return list: list of directories on a given path\n    '
    (head, tail) = os.path.split(path)
    if not tail:
        return []
    if not head:
        return [tail]
    return split_path(head) + [tail]

def list_dir_recursive(dir: str) -> Iterator[str]:
    if False:
        while True:
            i = 10
    for (dirpath, dirnames, filenames) in os.walk(dir, followlinks=True):
        for name in filenames:
            yield os.path.join(dirpath, name)

class DirManager(object):
    """ Manage working directories for application. Return paths, create them if it's needed """

    def __init__(self, root_path, tmp='tmp', res='resources', output='output', global_resource='golemres', reference_data_dir='reference_data', test='test'):
        if False:
            for i in range(10):
                print('nop')
        ' Creates new dir manager instance\n        :param str root_path: path to the main directory where all other working directories are placed\n        :param str tmp: temporary directory name\n        :param res: resource directory name\n        :param output: output directory name\n        :param global_resource: global resources directory name\n        '
        self.root_path = root_path
        self.tmp = tmp
        self.res = res
        self.output = output
        self.global_resource = global_resource
        self.ref = reference_data_dir
        self.test = test

    def get_file_extension(self, fullpath):
        if False:
            return 10
        (filename, file_extension) = os.path.splitext(fullpath)
        return file_extension

    def clear_dir(self, d, older_than_seconds: int=0):
        if False:
            return 10
        ' Remove everything from given directory\n        :param str d: directory that should be cleared\n        :param older_than_seconds: delete contents, that are older than given\n                                   amount of seconds.\n        '
        if not os.path.isdir(d):
            return
        current_time_seconds = time.time()
        min_allowed_mtime = current_time_seconds - older_than_seconds
        for i in os.listdir(d):
            path = os.path.join(d, i)
            if older_than_seconds > 0:
                mtime = os.path.getmtime(path)
                if mtime > min_allowed_mtime:
                    continue
            if os.path.isfile(path):
                os.remove(path)
            if os.path.isdir(path):
                self.clear_dir(path)
                if not os.listdir(path):
                    shutil.rmtree(path, ignore_errors=True)

    def create_dir(self, full_path):
        if False:
            return 10
        ' Create new directory, remove old directory if it exists.\n        :param str full_path: path to directory that should be created\n        '
        if os.path.exists(full_path):
            os.remove(full_path)
        os.makedirs(full_path)

    def get_dir(self, full_path, create, err_msg):
        if False:
            print('Hello World!')
        " Return path to a give directory if it exists. If it doesn't exist and option create is set to False\n        than return nothing and write given error message to a log. If it's set to True, create a directory and return\n        it's path\n        :param str full_path: path to directory should be checked or created\n        :param bool create: if directory doesn't exist, should it be created?\n        :param str err_msg: what should be written to a log if directory doesn't exists and create is set to False?\n        :return:\n        "
        if os.path.isdir(full_path):
            return full_path
        elif create:
            self.create_dir(full_path)
            return full_path
        else:
            logger.error(err_msg)
            return ''

    def get_node_dir(self, create=True):
        if False:
            return 10
        " Get node's directory\n        :param bool create: *Default: True* should directory be created if it doesn't exist\n        :return str: path to directory\n        "
        full_path = self.__get_node_path()
        return self.get_dir(full_path, create, 'resource dir does not exist')

    def get_resource_dir(self, create=True):
        if False:
            return 10
        " Get global resource directory\n        :param bool create: *Default: True* should directory be created if it doesn't exist\n        :return str: path to directory\n        "
        full_path = self.__get_global_resource_path()
        return self.get_dir(full_path, create, 'resource dir does not exist')

    def get_task_temporary_dir(self, task_id, create=True):
        if False:
            for i in range(10):
                print('nop')
        " Get temporary directory\n        :param task_id:\n        :param bool create: *Default: True* should directory be created if it doesn't exist\n        :return str: path to directory\n        "
        full_path = self.__get_tmp_path(task_id)
        return self.get_dir(full_path, create, 'temporary dir does not exist')

    def get_task_resource_dir(self, task_id, create=True):
        if False:
            print('Hello World!')
        " Get task resource directory\n        :param task_id:\n        :param bool create: *Default: True* should directory be created if it doesn't exist\n        :return str: path to directory\n        "
        full_path = self.__get_res_path(task_id)
        return self.get_dir(full_path, create, 'resource dir does not exist')

    def get_task_output_dir(self, task_id, create=True):
        if False:
            print('Hello World!')
        " Get task output directory\n        :param task_id:\n        :param bool create: *Default: True* should directory be created if it doesn't exist\n        :return str: path to directory\n        "
        full_path = self.__get_out_path(task_id)
        return self.get_dir(full_path, create, 'output dir does not exist')

    def get_ref_data_dir(self, task_id, create=True, counter=None):
        if False:
            print('Hello World!')
        " Get directory for storing reference data created by the requestor for validation of providers results\n        :param task_id:\n        :param bool create: *Default: True* should directory be created if it doesn't exist\n        :return str: path to directory\n        "
        full_path = self.__get_ref_path(task_id, counter)
        return self.get_dir(full_path, create, 'reference dir does not exist')

    def get_task_test_dir(self, task_id, create=True):
        if False:
            while True:
                i = 10
        " Get task test directory\n        :param task_id:\n        :param bool create: *Default: True* should directory be created if it doesn't exist\n        :return str: path to directory\n        "
        full_path = self.__get_test_path(task_id)
        return self.get_dir(full_path, create, 'test dir does not exist')

    @staticmethod
    def list_dir_names(task_dir):
        if False:
            for i in range(10):
                print('nop')
        ' Get the names of subdirectories as task ids\n        :param task_dir: Task temporary / resource / output directory\n        :return list: list of task ids\n        '
        if os.path.isdir(task_dir):
            return next(os.walk(task_dir))[1]
        return []

    def clear_temporary(self, task_id):
        if False:
            i = 10
            return i + 15
        ' Remove everything from temporary directory for given task\n        :param task_id: temporary directory of a task with that id should be cleared\n        '
        self.clear_dir(self.__get_tmp_path(task_id))

    def clear_resource(self, task_id):
        if False:
            while True:
                i = 10
        ' Remove everything from resource directory for given task\n        :param task_id: resource directory of a task with that id should be cleared\n        '
        self.clear_dir(self.__get_res_path(task_id))

    def clear_output(self, task_id):
        if False:
            return 10
        ' Remove everything from output directory for given task\n        :param task_id: output directory of a task with that id should be cleared\n        '
        self.clear_dir(self.__get_out_path(task_id))

    def __get_tmp_path(self, task_id):
        if False:
            for i in range(10):
                print('nop')
        return os.path.join(self.root_path, task_id, self.tmp)

    def __get_res_path(self, task_id):
        if False:
            print('Hello World!')
        return os.path.join(self.root_path, task_id, self.res)

    def __get_out_path(self, task_id):
        if False:
            for i in range(10):
                print('nop')
        return os.path.join(self.root_path, task_id, self.output)

    def __get_node_path(self):
        if False:
            for i in range(10):
                print('nop')
        return os.path.join(self.root_path)

    def __get_global_resource_path(self):
        if False:
            print('Hello World!')
        return os.path.join(self.root_path, self.global_resource)

    def __get_ref_path(self, task_id, counter):
        if False:
            i = 10
            return i + 15
        return os.path.join(self.root_path, task_id, self.ref, ''.join(['runNumber', str(counter)]))

    def __get_test_path(self, task_id):
        if False:
            while True:
                i = 10
        return os.path.join(self.root_path, task_id, self.test)

class DirectoryType(object):
    DISTRIBUTED = 1
    RECEIVED = 2