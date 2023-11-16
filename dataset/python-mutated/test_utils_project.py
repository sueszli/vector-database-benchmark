import contextlib
import os
import shutil
import tempfile
import unittest
import warnings
from pathlib import Path
from scrapy.utils.project import data_path, get_project_settings

@contextlib.contextmanager
def inside_a_project():
    if False:
        i = 10
        return i + 15
    prev_dir = os.getcwd()
    project_dir = tempfile.mkdtemp()
    try:
        os.chdir(project_dir)
        Path('scrapy.cfg').touch()
        yield project_dir
    finally:
        os.chdir(prev_dir)
        shutil.rmtree(project_dir)

class ProjectUtilsTest(unittest.TestCase):

    def test_data_path_outside_project(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(str(Path('.scrapy', 'somepath')), data_path('somepath'))
        abspath = str(Path(os.path.sep, 'absolute', 'path'))
        self.assertEqual(abspath, data_path(abspath))

    def test_data_path_inside_project(self):
        if False:
            while True:
                i = 10
        with inside_a_project() as proj_path:
            expected = Path(proj_path, '.scrapy', 'somepath')
            self.assertEqual(expected.resolve(), Path(data_path('somepath')).resolve())
            abspath = str(Path(os.path.sep, 'absolute', 'path').resolve())
            self.assertEqual(abspath, data_path(abspath))

@contextlib.contextmanager
def set_env(**update):
    if False:
        for i in range(10):
            print('nop')
    modified = set(update.keys()) & set(os.environ.keys())
    update_after = {k: os.environ[k] for k in modified}
    remove_after = frozenset((k for k in update if k not in os.environ))
    try:
        os.environ.update(update)
        yield
    finally:
        os.environ.update(update_after)
        for k in remove_after:
            os.environ.pop(k)

class GetProjectSettingsTestCase(unittest.TestCase):

    def test_valid_envvar(self):
        if False:
            while True:
                i = 10
        value = 'tests.test_cmdline.settings'
        envvars = {'SCRAPY_SETTINGS_MODULE': value}
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            with set_env(**envvars):
                settings = get_project_settings()
        assert settings.get('SETTINGS_MODULE') == value

    def test_invalid_envvar(self):
        if False:
            for i in range(10):
                print('nop')
        envvars = {'SCRAPY_FOO': 'bar'}
        with set_env(**envvars):
            settings = get_project_settings()
        assert settings.get('SCRAPY_FOO') is None

    def test_valid_and_invalid_envvars(self):
        if False:
            return 10
        value = 'tests.test_cmdline.settings'
        envvars = {'SCRAPY_FOO': 'bar', 'SCRAPY_SETTINGS_MODULE': value}
        with set_env(**envvars):
            settings = get_project_settings()
        assert settings.get('SETTINGS_MODULE') == value
        assert settings.get('SCRAPY_FOO') is None