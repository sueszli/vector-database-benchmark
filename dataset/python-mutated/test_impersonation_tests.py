from __future__ import annotations
import contextlib
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path
import pytest
from airflow.configuration import conf
from airflow.jobs.backfill_job_runner import BackfillJobRunner
from airflow.jobs.job import Job, run_job
from airflow.models import DagBag, DagRun, TaskInstance
from airflow.utils.db import add_default_pool_if_not_exists
from airflow.utils.state import State
from airflow.utils.timezone import datetime
from airflow.utils.types import DagRunType
from tests.test_utils import db
pytestmark = pytest.mark.db_test
DEV_NULL = '/dev/null'
TEST_ROOT_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
TEST_DAG_FOLDER = os.path.join(TEST_ROOT_FOLDER, 'dags')
TEST_DAG_CORRUPTED_FOLDER = os.path.join(TEST_ROOT_FOLDER, 'dags_corrupted')
TEST_UTILS_FOLDER = os.path.join(TEST_ROOT_FOLDER, 'test_utils')
DEFAULT_DATE = datetime(2015, 1, 1)
TEST_USER = 'airflow_test_user'
logger = logging.getLogger(__name__)

@contextlib.contextmanager
def set_permissions(settings: dict[Path | str, int]):
    if False:
        print('Hello World!')
    'Helper for recursively set permissions only for specific path and revert it back.'
    orig_permissions = []
    try:
        print(' Change file/directory permissions '.center(72, '+'))
        for (path, mode) in settings.items():
            if isinstance(path, str):
                path = Path(path)
            if len(path.parts) <= 1:
                raise SystemError(f'Unable to change permission for the root directory: {path}.')
            st_mode = os.stat(path).st_mode
            new_st_mode = st_mode | mode
            if new_st_mode > st_mode:
                print(f'Path={path}, mode={oct(st_mode)}, new_mode={oct(new_st_mode)}')
                orig_permissions.append((path, st_mode))
                os.chmod(path, new_st_mode)
            parent_path = path.parent
            while len(parent_path.parts) > 1:
                st_mode = os.stat(parent_path).st_mode
                new_st_mode = st_mode | 493
                if new_st_mode > st_mode:
                    print(f'Path={parent_path}, mode={oct(st_mode)}, new_mode={oct(new_st_mode)}')
                    orig_permissions.append((parent_path, st_mode))
                    os.chmod(parent_path, new_st_mode)
                parent_path = parent_path.parent
        print(''.center(72, '+'))
        yield
    finally:
        for (path, mode) in orig_permissions:
            os.chmod(path, mode)

@pytest.fixture
def check_original_docker_image():
    if False:
        return 10
    if not os.path.isfile('/.dockerenv') or os.environ.get('PYTHON_BASE_IMAGE') is None:
        raise pytest.skip('Adding/removing a user as part of a test is very bad for host os (especially if the user already existed to begin with on the OS), therefore we check if we run inside a the official docker container and only allow to run the test there. This is done by checking /.dockerenv file (always present inside container) and checking for PYTHON_BASE_IMAGE variable.')
    yield

@pytest.fixture
def create_user(check_original_docker_image):
    if False:
        return 10
    try:
        subprocess.check_output(['sudo', 'useradd', '-m', TEST_USER, '-g', str(os.getegid())], stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        command = e.cmd[1]
        if e.returncode != 9:
            raise pytest.skip(f"{e} Skipping tests.\nDoes command {command!r} exists and the current user have permission to run {command!r} without a password prompt (check sudoers file)?\n{(e.stdout.decode() if e.stdout else '')}")
    yield TEST_USER
    subprocess.check_call(['sudo', 'userdel', '-r', TEST_USER])

@pytest.fixture
def create_airflow_home(create_user, tmp_path, monkeypatch):
    if False:
        i = 10
        return i + 15
    sql_alchemy_conn = conf.get_mandatory_value('database', 'sql_alchemy_conn')
    username = create_user
    airflow_home = tmp_path / 'airflow-home'
    if not airflow_home.exists():
        airflow_home.mkdir(parents=True, exist_ok=True)
    permissions = {airflow_home: 511, tempfile.gettempdir(): 511}
    if sql_alchemy_conn.lower().startswith('sqlite'):
        sqlite_file = Path(sql_alchemy_conn.replace('sqlite:///', ''))
        permissions[sqlite_file] = 502
        permissions[sqlite_file.parent] = 511
    monkeypatch.setenv('AIRFLOW_HOME', str(airflow_home))
    with set_permissions(permissions):
        subprocess.check_call(['sudo', 'chown', f'{username}:root', str(airflow_home), '-R'], close_fds=True)
        yield airflow_home

class BaseImpersonationTest:
    dagbag: DagBag

    @pytest.fixture(autouse=True)
    def setup_impersonation_tests(self, create_airflow_home):
        if False:
            while True:
                i = 10
        'Setup test cases for all impersonation tests.'
        db.clear_db_runs()
        db.clear_db_jobs()
        add_default_pool_if_not_exists()
        yield
        db.clear_db_runs()
        db.clear_db_jobs()

    @staticmethod
    def get_dagbag(dag_folder):
        if False:
            i = 10
            return i + 15
        'Get DagBag and print statistic into the log.'
        dagbag = DagBag(dag_folder=dag_folder, include_examples=False)
        logger.info('Loaded DAGs:')
        logger.info(dagbag.dagbag_report())
        return dagbag

    def run_backfill(self, dag_id, task_id):
        if False:
            for i in range(10):
                print('nop')
        dag = self.dagbag.get_dag(dag_id)
        dag.clear()
        job = Job()
        job_runner = BackfillJobRunner(job=job, dag=dag, start_date=DEFAULT_DATE, end_date=DEFAULT_DATE)
        run_job(job=job, execute_callable=job_runner._execute)
        run_id = DagRun.generate_run_id(DagRunType.BACKFILL_JOB, execution_date=DEFAULT_DATE)
        ti = TaskInstance(task=dag.get_task(task_id), run_id=run_id)
        ti.refresh_from_db()
        assert ti.state == State.SUCCESS

class TestImpersonation(BaseImpersonationTest):

    @classmethod
    def setup_class(cls):
        if False:
            for i in range(10):
                print('nop')
        cls.dagbag = cls.get_dagbag(TEST_DAG_FOLDER)

    def test_impersonation(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Tests that impersonating a unix user works\n        '
        self.run_backfill('test_impersonation', 'test_impersonated_user')

    def test_no_impersonation(self):
        if False:
            i = 10
            return i + 15
        '\n        If default_impersonation=None, tests that the job is run\n        as the current user (which will be a sudoer)\n        '
        self.run_backfill('test_no_impersonation', 'test_superuser')

    def test_default_impersonation(self, monkeypatch):
        if False:
            print('Hello World!')
        "\n        If default_impersonation=TEST_USER, tests that the job defaults\n        to running as TEST_USER for a test without 'run_as_user' set.\n        "
        monkeypatch.setenv('AIRFLOW__CORE__DEFAULT_IMPERSONATION', TEST_USER)
        self.run_backfill('test_default_impersonation', 'test_deelevated_user')

    @pytest.mark.execution_timeout(150)
    def test_impersonation_subdag(self):
        if False:
            return 10
        'Tests that impersonation using a subdag correctly passes the right configuration.'
        self.run_backfill('impersonation_subdag', 'test_subdag_operation')

class TestImpersonationWithCustomPythonPath(BaseImpersonationTest):

    @pytest.fixture(autouse=True)
    def setup_dagbag(self, monkeypatch):
        if False:
            i = 10
            return i + 15
        monkeypatch.syspath_prepend(TEST_UTILS_FOLDER)
        self.dagbag = self.get_dagbag(TEST_DAG_CORRUPTED_FOLDER)
        monkeypatch.undo()
        yield

    def test_impersonation_custom(self, monkeypatch):
        if False:
            print('Hello World!')
        '\n        Tests that impersonation using a unix user works with custom packages in PYTHONPATH.\n        '
        monkeypatch.setenv('PYTHONPATH', TEST_UTILS_FOLDER)
        assert TEST_UTILS_FOLDER not in sys.path
        self.run_backfill('impersonation_with_custom_pkg', 'exec_python_fn')