import pytest
import os
import subprocess
import sys
import shutil
if sys.platform != 'win32':
    pytestmark = pytest.mark.skip('PyInstaller is currently only tested on Windows')
else:
    try:
        import PyInstaller
    except ImportError:
        pytestmark = pytest.mark.skip('PyInstaller is not available')

@pytest.mark.incremental
class PyinstallerBase(object):
    pinstall_path = ''
    env = None

    @classmethod
    def setup_class(cls):
        if False:
            while True:
                i = 10
        cls.env = cls.get_env()

    @classmethod
    def get_env(cls):
        if False:
            print('Hello World!')
        env = os.environ.copy()
        env['__KIVY_PYINSTALLER_DIR'] = cls.pinstall_path
        if 'PYTHONPATH' not in env:
            env['PYTHONPATH'] = cls.pinstall_path
        else:
            env['PYTHONPATH'] = cls.pinstall_path + os.sep + env['PYTHONPATH']
        return env

    @classmethod
    def get_run_env(cls):
        if False:
            for i in range(10):
                print('nop')
        return os.environ.copy()

    def test_project(self):
        if False:
            for i in range(10):
                print('nop')
        try:
            subprocess.check_output([sys.executable or 'python', os.path.join(self.pinstall_path, 'main.py')], stderr=subprocess.STDOUT, env=self.env)
        except subprocess.CalledProcessError as e:
            print(e.output.decode('utf8'))
            raise

    def test_packaging(self):
        if False:
            print('Hello World!')
        dist = os.path.join(self.pinstall_path, 'dist')
        build = os.path.join(self.pinstall_path, 'build')
        try:
            subprocess.check_output([sys.executable or 'python', '-m', 'PyInstaller', os.path.join(self.pinstall_path, 'main.spec'), '--distpath', dist, '--workpath', build], stderr=subprocess.STDOUT, env=self.env)
        except subprocess.CalledProcessError as e:
            print(e.output.decode('utf8'))
            raise

    def test_packaged_project(self):
        if False:
            return 10
        try:
            subprocess.check_output(os.path.join(self.pinstall_path, 'dist', 'main', 'main'), stderr=subprocess.STDOUT, env=self.get_run_env())
        except subprocess.CalledProcessError as e:
            print(e.output.decode('utf8'))
            raise

    @classmethod
    def teardown_class(cls):
        if False:
            return 10
        shutil.rmtree(os.path.join(cls.pinstall_path, '__pycache__'), ignore_errors=True)
        shutil.rmtree(os.path.join(cls.pinstall_path, 'build'), ignore_errors=True)
        shutil.rmtree(os.path.join(cls.pinstall_path, 'dist'), ignore_errors=True)
        shutil.rmtree(os.path.join(cls.pinstall_path, 'project', '__pycache__'), ignore_errors=True)

class TestSimpleWidget(PyinstallerBase):
    pinstall_path = os.path.join(os.path.dirname(__file__), 'simple_widget')

class TestVideoWidget(PyinstallerBase):
    pinstall_path = os.path.join(os.path.dirname(__file__), 'video_widget')

    @classmethod
    def get_env(cls):
        if False:
            for i in range(10):
                print('nop')
        env = super(TestVideoWidget, cls).get_env()
        import kivy
        env['__KIVY_VIDEO_TEST_FNAME'] = os.path.abspath(os.path.join(kivy.kivy_examples_dir, 'widgets', 'cityCC0.mpg'))
        return env

    @classmethod
    def get_run_env(cls):
        if False:
            i = 10
            return i + 15
        env = super(TestVideoWidget, cls).get_run_env()
        import kivy
        env['__KIVY_VIDEO_TEST_FNAME'] = os.path.abspath(os.path.join(kivy.kivy_examples_dir, 'widgets', 'cityCC0.mpg'))
        return env