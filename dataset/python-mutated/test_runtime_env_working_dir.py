import os
import shutil
import sys
import tempfile
import time
from importlib import import_module
from pathlib import Path
from unittest import mock
import pytest
import ray
from ray._private import gcs_utils
from ray._private.runtime_env.context import RuntimeEnvContext
from ray._private.runtime_env.packaging import get_uri_for_directory, upload_package_if_needed
from ray._private.runtime_env.working_dir import WorkingDirPlugin, set_pythonpath_in_context
from ray._private.utils import get_directory_size_bytes
HTTPS_PACKAGE_URI = 'https://github.com/shrekris-anyscale/test_module/archive/HEAD.zip'
S3_PACKAGE_URI = 's3://runtime-env-test/test_runtime_env.zip'
GS_PACKAGE_URI = 'gs://public-runtime-env-test/test_module.zip'
TEST_IMPORT_DIR = 'test_import_dir'

@pytest.fixture(scope='module')
def insert_test_dir_in_pythonpath():
    if False:
        while True:
            i = 10
    with mock.patch.dict(os.environ, {'PYTHONPATH': TEST_IMPORT_DIR + os.pathsep + os.environ.get('PYTHONPATH', '')}):
        yield

@pytest.mark.asyncio
async def test_create_delete_size_equal(tmpdir, ray_start_regular):
    """Tests that `create` and `delete_uri` return the same size for a URI."""
    gcs_aio_client = gcs_utils.GcsAioClient(address=ray.worker.global_worker.gcs_client.address)
    path = Path(tmpdir)
    dir_to_upload = path / 'dir_to_upload'
    dir_to_upload.mkdir(parents=True)
    filepath = dir_to_upload / 'file'
    with filepath.open('w') as file:
        file.write('F' * 100)
    uri = get_uri_for_directory(dir_to_upload)
    assert get_directory_size_bytes(dir_to_upload) > 0
    uploaded = upload_package_if_needed(uri, tmpdir, dir_to_upload)
    assert uploaded
    manager = WorkingDirPlugin(tmpdir, gcs_aio_client)
    created_size_bytes = await manager.create(uri, {}, RuntimeEnvContext())
    deleted_size_bytes = manager.delete_uri(uri)
    assert created_size_bytes == deleted_size_bytes

def test_inherit_cluster_env_pythonpath(monkeypatch):
    if False:
        for i in range(10):
            print('nop')
    monkeypatch.setenv('PYTHONPATH', 'last' + os.pathsep + os.environ.get('PYTHONPATH', ''))
    context = RuntimeEnvContext(env_vars={'PYTHONPATH': 'middle'})
    set_pythonpath_in_context('first', context)
    assert context.env_vars['PYTHONPATH'].startswith(os.pathsep.join(['first', 'middle', 'last']))

@pytest.mark.parametrize('option', ['failure', 'working_dir', 'working_dir_zip', 'py_modules', 'working_dir_and_py_modules'])
def test_lazy_reads(insert_test_dir_in_pythonpath, start_cluster, tmp_working_dir, option: str):
    if False:
        for i in range(10):
            print('nop')
    'Tests the case where we lazily read files or import inside a task/actor.\n\n    This tests both that this fails *without* the working_dir and that it\n    passes with it.  Also tests that the existing PYTHONPATH is preserved,\n    so packages preinstalled on the cluster are still importable when using\n    py_modules or working_dir.\n    '
    (cluster, address) = start_cluster

    def call_ray_init():
        if False:
            i = 10
            return i + 15
        if option == 'failure':
            ray.init(address)
        elif option == 'working_dir':
            ray.init(address, runtime_env={'working_dir': tmp_working_dir})
        elif option == 'working_dir_zip':
            with tempfile.TemporaryDirectory() as tmp_dir:
                zip_dir = Path(tmp_working_dir)
                package = shutil.make_archive(os.path.join(tmp_dir, 'test'), 'zip', zip_dir)
                ray.init(address, runtime_env={'working_dir': package})
        elif option == 'py_modules':
            ray.init(address, runtime_env={'py_modules': [str(Path(tmp_working_dir) / 'test_module'), Path(os.path.dirname(__file__)) / 'pip_install_test-0.5-py3-none-any.whl']})
        elif option == 'working_dir_and_py_modules':
            ray.init(address, runtime_env={'working_dir': tmp_working_dir, 'py_modules': [str(Path(tmp_working_dir) / 'test_module'), Path(os.path.dirname(__file__)) / 'pip_install_test-0.5-py3-none-any.whl']})
        else:
            raise ValueError(f'unexpected pytest parameter {option}')
    call_ray_init()

    def reinit():
        if False:
            for i in range(10):
                print('nop')
        ray.shutdown()
        time.sleep(5)
        call_ray_init()

    @ray.remote
    def test_import():
        if False:
            while True:
                i = 10
        import test_module
        assert TEST_IMPORT_DIR in os.environ.get('PYTHONPATH', '')
        return test_module.one()
    if option == 'failure':
        with pytest.raises(ImportError):
            ray.get(test_import.remote())
    else:
        assert ray.get(test_import.remote()) == 1
    if option in {'py_modules', 'working_dir_and_py_modules'}:

        @ray.remote
        def test_py_modules_whl():
            if False:
                while True:
                    i = 10
            import pip_install_test
            return True
        assert ray.get(test_py_modules_whl.remote())
    if option in {'py_modules', 'working_dir_zip'}:
        return
    reinit()

    @ray.remote
    def test_read():
        if False:
            return 10
        return open('hello').read()
    if option == 'failure':
        with pytest.raises(FileNotFoundError):
            ray.get(test_read.remote())
    elif option in {'working_dir_and_py_modules', 'working_dir'}:
        assert ray.get(test_read.remote()) == 'world'
    reinit()

    @ray.remote
    class Actor:

        def test_import(self):
            if False:
                print('Hello World!')
            import test_module
            assert TEST_IMPORT_DIR in os.environ.get('PYTHONPATH', '')
            return test_module.one()

        def test_read(self):
            if False:
                while True:
                    i = 10
            assert TEST_IMPORT_DIR in os.environ.get('PYTHONPATH', '')
            return open('hello').read()
    a = Actor.remote()
    if option == 'failure':
        with pytest.raises(ImportError):
            assert ray.get(a.test_import.remote()) == 1
        with pytest.raises(FileNotFoundError):
            assert ray.get(a.test_read.remote()) == 'world'
    elif option in {'working_dir_and_py_modules', 'working_dir'}:
        assert ray.get(a.test_import.remote()) == 1
        assert ray.get(a.test_read.remote()) == 'world'

@pytest.mark.parametrize('option', ['failure', 'working_dir', 'py_modules'])
def test_captured_import(start_cluster, tmp_working_dir, option: str):
    if False:
        i = 10
        return i + 15
    'Tests importing a module in the driver and capturing it in a task/actor.\n\n    This tests both that this fails *without* the working_dir and that it\n    passes with it.\n    '
    (cluster, address) = start_cluster

    def call_ray_init():
        if False:
            print('Hello World!')
        if option == 'failure':
            ray.init(address)
        elif option == 'working_dir':
            ray.init(address, runtime_env={'working_dir': tmp_working_dir})
        elif option == 'py_modules':
            ray.init(address, runtime_env={'py_modules': [os.path.join(tmp_working_dir, 'test_module')]})
    call_ray_init()

    def reinit():
        if False:
            while True:
                i = 10
        ray.shutdown()
        time.sleep(5)
        call_ray_init()
    sys.path.insert(0, tmp_working_dir)
    import test_module

    @ray.remote
    def test_import():
        if False:
            for i in range(10):
                print('nop')
        return test_module.one()
    if option == 'failure':
        with pytest.raises(Exception):
            ray.get(test_import.remote())
    else:
        assert ray.get(test_import.remote()) == 1
    reinit()

    @ray.remote
    class Actor:

        def test_import(self):
            if False:
                return 10
            return test_module.one()
    if option == 'failure':
        with pytest.raises(Exception):
            a = Actor.remote()
            assert ray.get(a.test_import.remote()) == 1
    else:
        a = Actor.remote()
        assert ray.get(a.test_import.remote()) == 1

def test_empty_working_dir(start_cluster):
    if False:
        return 10
    'Tests the case where we pass an empty directory as the working_dir.'
    (cluster, address) = start_cluster
    with tempfile.TemporaryDirectory() as working_dir:
        ray.init(address, runtime_env={'working_dir': working_dir})

        @ray.remote
        def listdir():
            if False:
                while True:
                    i = 10
            return os.listdir()
        assert len(ray.get(listdir.remote())) == 0

        @ray.remote
        class A:

            def listdir(self):
                if False:
                    while True:
                        i = 10
                return os.listdir()
                pass
        a = A.remote()
        assert len(ray.get(a.listdir.remote())) == 0
        ray.shutdown()
        ray.init(address, runtime_env={'working_dir': working_dir})

@pytest.mark.parametrize('option', ['working_dir', 'py_modules'])
def test_input_validation(start_cluster, option: str):
    if False:
        i = 10
        return i + 15
    'Tests input validation for working_dir and py_modules.'
    (cluster, address) = start_cluster
    with pytest.raises(TypeError):
        if option == 'working_dir':
            ray.init(address, runtime_env={'working_dir': 10})
        else:
            ray.init(address, runtime_env={'py_modules': [10]})
    ray.shutdown()
    with pytest.raises(ValueError):
        if option == 'working_dir':
            ray.init(address, runtime_env={'working_dir': '/does/not/exist'})
        else:
            ray.init(address, runtime_env={'py_modules': ['/does/not/exist']})
    ray.shutdown()
    with pytest.raises(ValueError):
        if option == 'working_dir':
            ray.init(address, runtime_env={'working_dir': 'does_not_exist'})
        else:
            ray.init(address, runtime_env={'py_modules': ['does_not_exist']})
    ray.shutdown()
    for uri in ['https://no_dot_zip', 's3://no_dot_zip', 'gs://no_dot_zip']:
        with pytest.raises(ValueError):
            if option == 'working_dir':
                ray.init(address, runtime_env={'working_dir': uri})
            else:
                ray.init(address, runtime_env={'py_modules': [uri]})
        ray.shutdown()
    if option == 'py_modules':
        with pytest.raises(TypeError):
            ray.init(address, runtime_env={'py_modules': '.'})

@pytest.mark.parametrize('option', ['working_dir', 'py_modules'])
def test_exclusion(start_cluster, tmp_working_dir, option):
    if False:
        return 10
    "Tests various forms of the 'excludes' parameter."
    (cluster, address) = start_cluster

    def create_file(p, empty=False):
        if False:
            return 10
        if not p.parent.exists():
            p.parent.mkdir(parents=True)
        with p.open('w') as f:
            if not empty:
                f.write('Test')
    working_path = Path(tmp_working_dir)
    create_file(working_path / '__init__.py', empty=True)
    create_file(working_path / 'test1')
    create_file(working_path / 'test2')
    create_file(working_path / 'test3')
    create_file(working_path / 'tmp_dir' / 'test_1')
    create_file(working_path / 'tmp_dir' / 'test_2')
    create_file(working_path / 'tmp_dir' / 'test_3')
    create_file(working_path / 'tmp_dir' / 'sub_dir' / 'test_1')
    create_file(working_path / 'tmp_dir' / 'sub_dir' / 'test_2')
    create_file(working_path / 'cache' / 'test_1')
    create_file(working_path / 'tmp_dir' / 'cache' / 'test_1')
    create_file(working_path / 'another_dir' / 'cache' / 'test_1')
    module_name = Path(tmp_working_dir).name
    if option == 'working_dir':
        ray.init(address, runtime_env={'working_dir': tmp_working_dir})
    else:
        ray.init(address, runtime_env={'py_modules': [tmp_working_dir]})

    @ray.remote
    def check_file(name):
        if False:
            while True:
                i = 10
        if option == 'py_modules':
            try:
                module = import_module(module_name)
            except ImportError:
                return 'FAILED'
            name = os.path.join(module.__path__[0], name)
        try:
            with open(name) as f:
                return f.read()
        except Exception:
            return 'FAILED'

    def get_all():
        if False:
            print('Hello World!')
        return ray.get([check_file.remote('test1'), check_file.remote('test2'), check_file.remote('test3'), check_file.remote(os.path.join('tmp_dir', 'test_1')), check_file.remote(os.path.join('tmp_dir', 'test_2')), check_file.remote(os.path.join('tmp_dir', 'test_3')), check_file.remote(os.path.join('tmp_dir', 'sub_dir', 'test_1')), check_file.remote(os.path.join('tmp_dir', 'sub_dir', 'test_2')), check_file.remote(os.path.join('cache', 'test_1')), check_file.remote(os.path.join('tmp_dir', 'cache', 'test_1')), check_file.remote(os.path.join('another_dir', 'cache', 'test_1'))])
    assert get_all() == ['Test', 'Test', 'Test', 'Test', 'Test', 'Test', 'Test', 'Test', 'Test', 'Test', 'Test']
    ray.shutdown()
    excludes = ['test2', str((Path('tmp_dir') / 'sub_dir').as_posix()), str((Path('tmp_dir') / 'test_1').as_posix()), str((Path('tmp_dir') / 'test_2').as_posix())]
    if option == 'working_dir':
        ray.init(address, runtime_env={'working_dir': tmp_working_dir, 'excludes': excludes})
    else:
        ray.init(address, runtime_env={'py_modules': [tmp_working_dir], 'excludes': excludes})
    assert get_all() == ['Test', 'FAILED', 'Test', 'FAILED', 'FAILED', 'Test', 'FAILED', 'FAILED', 'Test', 'Test', 'Test']
    ray.shutdown()
    excludes = ['*']
    if option == 'working_dir':
        ray.init(address, runtime_env={'working_dir': tmp_working_dir, 'excludes': excludes})
    else:
        module_name = Path(tmp_working_dir).name
        ray.init(address, runtime_env={'py_modules': [tmp_working_dir], 'excludes': excludes})
    assert get_all() == ['FAILED', 'FAILED', 'FAILED', 'FAILED', 'FAILED', 'FAILED', 'FAILED', 'FAILED', 'FAILED', 'FAILED', 'FAILED']
    ray.shutdown()
    with open(f'{tmp_working_dir}/.gitignore', 'w') as f:
        f.write('\n# Comment\ntest_[12]\n/test1\n!/tmp_dir/sub_dir/test_1\ncache/\n')
    if option == 'working_dir':
        ray.init(address, runtime_env={'working_dir': tmp_working_dir})
    else:
        module_name = Path(tmp_working_dir).name
        ray.init(address, runtime_env={'py_modules': [tmp_working_dir]})
    assert get_all() == ['FAILED', 'Test', 'Test', 'FAILED', 'FAILED', 'Test', 'Test', 'FAILED', 'FAILED', 'FAILED', 'FAILED']

def test_override_failure(shutdown_only):
    if False:
        i = 10
        return i + 15
    'Tests invalid override behaviors.'
    ray.init()
    with pytest.raises(ValueError):

        @ray.remote(runtime_env={'working_dir': '.'})
        def f():
            if False:
                print('Hello World!')
            pass

    @ray.remote
    def g():
        if False:
            while True:
                i = 10
        pass
    with pytest.raises(ValueError):
        g.options(runtime_env={'working_dir': '.'})
    with pytest.raises(ValueError):

        @ray.remote(runtime_env={'working_dir': '.'})
        class A:
            pass

    @ray.remote
    class B:
        pass
    with pytest.raises(ValueError):
        B.options(runtime_env={'working_dir': '.'})
if __name__ == '__main__':
    if os.environ.get('PARALLEL_CI'):
        sys.exit(pytest.main(['-n', 'auto', '--boxed', '-vs', __file__]))
    else:
        sys.exit(pytest.main(['-sv', __file__]))