import itertools
import os
import shutil
import tempfile
from contextlib import contextmanager
import dagster._check as check
from dagster._core.storage.file_manager import LocalFileHandle

def _unlink_swallow_errors(path):
    if False:
        for i in range(10):
            print('nop')
    check.str_param(path, 'path')
    try:
        os.unlink(path)
    except Exception:
        pass

@contextmanager
def get_temp_file_handle_with_data(data):
    if False:
        i = 10
        return i + 15
    with get_temp_file_name_with_data(data) as temp_file:
        yield LocalFileHandle(temp_file)

@contextmanager
def get_temp_file_name_with_data(data):
    if False:
        print('Hello World!')
    with get_temp_file_name() as temp_file:
        with open(temp_file, 'wb') as ff:
            ff.write(data)
        yield temp_file

@contextmanager
def get_temp_file_handle():
    if False:
        return 10
    with get_temp_file_name() as temp_file:
        yield LocalFileHandle(temp_file)

@contextmanager
def get_temp_file_name():
    if False:
        for i in range(10):
            print('nop')
    (handle, temp_file_name) = tempfile.mkstemp()
    os.close(handle)
    try:
        yield temp_file_name
    finally:
        _unlink_swallow_errors(temp_file_name)

@contextmanager
def get_temp_file_names(number):
    if False:
        i = 10
        return i + 15
    check.int_param(number, 'number')
    temp_file_names = list()
    for _ in itertools.repeat(None, number):
        (handle, temp_file_name) = tempfile.mkstemp()
        os.close(handle)
        temp_file_names.append(temp_file_name)
    try:
        yield tuple(temp_file_names)
    finally:
        for temp_file_name in temp_file_names:
            _unlink_swallow_errors(temp_file_name)

@contextmanager
def get_temp_dir(in_directory=None):
    if False:
        while True:
            i = 10
    temp_dir = None
    try:
        temp_dir = tempfile.mkdtemp(dir=in_directory)
        yield temp_dir
    finally:
        if temp_dir:
            shutil.rmtree(temp_dir)