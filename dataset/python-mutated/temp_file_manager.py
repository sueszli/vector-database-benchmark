import contextlib
import os
import shutil
import tempfile
from dagster._core.definitions.resource_definition import dagster_maintained_resource, resource

class TempfileManager:

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.paths = []
        self.files = []
        self.dirs = []

    def tempfile(self):
        if False:
            while True:
                i = 10
        temporary_file = tempfile.NamedTemporaryFile('w+b', delete=False)
        self.files.append(temporary_file)
        self.paths.append(temporary_file.name)
        return temporary_file

    def tempdir(self):
        if False:
            return 10
        temporary_directory = tempfile.mkdtemp()
        self.dirs.append(temporary_directory)
        return temporary_directory

    def close(self):
        if False:
            return 10
        for fobj in self.files:
            fobj.close()
        for path in self.paths:
            if os.path.exists(path):
                os.remove(path)
        for dir_ in self.dirs:
            shutil.rmtree(dir_)

@contextlib.contextmanager
def _tempfile_manager():
    if False:
        for i in range(10):
            print('nop')
    manager = TempfileManager()
    try:
        yield manager
    finally:
        manager.close()

@dagster_maintained_resource
@resource
def tempfile_resource(_init_context):
    if False:
        print('Hello World!')
    with _tempfile_manager() as manager:
        yield manager