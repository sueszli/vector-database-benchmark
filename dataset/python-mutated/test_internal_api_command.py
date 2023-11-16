from __future__ import annotations
import os
import subprocess
import sys
import time
from unittest import mock
import psutil
import pytest
from rich.console import Console
from airflow import settings
from airflow.cli import cli_parser
from airflow.cli.commands import internal_api_command
from airflow.cli.commands.internal_api_command import GunicornMonitor
from airflow.settings import _ENABLE_AIP_44
from tests.cli.commands._common_cli_classes import _ComonCLIGunicornTestClass
console = Console(width=400, color_system='standard')

class TestCLIGetNumReadyWorkersRunning:

    @classmethod
    def setup_class(cls):
        if False:
            i = 10
            return i + 15
        cls.parser = cli_parser.get_parser()

    def setup_method(self):
        if False:
            while True:
                i = 10
        self.children = mock.MagicMock()
        self.child = mock.MagicMock()
        self.process = mock.MagicMock()
        self.monitor = GunicornMonitor(gunicorn_master_pid=1, num_workers_expected=4, master_timeout=60, worker_refresh_interval=60, worker_refresh_batch_size=2, reload_on_plugin_change=True)

    def test_ready_prefix_on_cmdline(self):
        if False:
            i = 10
            return i + 15
        self.child.cmdline.return_value = [settings.GUNICORN_WORKER_READY_PREFIX]
        self.process.children.return_value = [self.child]
        with mock.patch('psutil.Process', return_value=self.process):
            assert self.monitor._get_num_ready_workers_running() == 1

    def test_ready_prefix_on_cmdline_no_children(self):
        if False:
            return 10
        self.process.children.return_value = []
        with mock.patch('psutil.Process', return_value=self.process):
            assert self.monitor._get_num_ready_workers_running() == 0

    def test_ready_prefix_on_cmdline_zombie(self):
        if False:
            return 10
        self.child.cmdline.return_value = []
        self.process.children.return_value = [self.child]
        with mock.patch('psutil.Process', return_value=self.process):
            assert self.monitor._get_num_ready_workers_running() == 0

    def test_ready_prefix_on_cmdline_dead_process(self):
        if False:
            return 10
        self.child.cmdline.side_effect = psutil.NoSuchProcess(11347)
        self.process.children.return_value = [self.child]
        with mock.patch('psutil.Process', return_value=self.process):
            assert self.monitor._get_num_ready_workers_running() == 0

@pytest.mark.db_test
@pytest.mark.skipif(not _ENABLE_AIP_44, reason='AIP-44 is disabled')
class TestCliInternalAPI(_ComonCLIGunicornTestClass):
    main_process_regexp = 'airflow internal-api'

    @pytest.mark.execution_timeout(210)
    def test_cli_internal_api_background(self, tmp_path):
        if False:
            print('Hello World!')
        parent_path = tmp_path / 'gunicorn'
        parent_path.mkdir()
        pidfile_internal_api = parent_path / 'pidflow-internal-api.pid'
        pidfile_monitor = parent_path / 'pidflow-internal-api-monitor.pid'
        stdout = parent_path / 'airflow-internal-api.out'
        stderr = parent_path / 'airflow-internal-api.err'
        logfile = parent_path / 'airflow-internal-api.log'
        try:
            console.print('[magenta]Starting airflow internal-api --daemon')
            proc = subprocess.Popen(['airflow', 'internal-api', '--daemon', '--pid', os.fspath(pidfile_internal_api), '--stdout', os.fspath(stdout), '--stderr', os.fspath(stderr), '--log-file', os.fspath(logfile)])
            assert proc.poll() is None
            pid_monitor = self._wait_pidfile(pidfile_monitor)
            console.print(f'[blue]Monitor started at {pid_monitor}')
            pid_internal_api = self._wait_pidfile(pidfile_internal_api)
            console.print(f'[blue]Internal API started at {pid_internal_api}')
            console.print('[blue]Running airflow internal-api process:')
            assert self._find_process('airflow internal-api --daemon', print_found_process=True)
            console.print('[blue]Waiting for gunicorn processes:')
            for i in range(30):
                if self._find_process('^gunicorn'):
                    break
                console.print('[blue]Waiting for gunicorn to start ...')
                time.sleep(1)
            console.print('[blue]Running gunicorn processes:')
            assert self._find_all_processes('^gunicorn', print_found_process=True)
            console.print('[magenta]Internal-api process started successfully.')
            console.print('[magenta]Terminating monitor process and expect internal-api and gunicorn processes to terminate as well')
            proc = psutil.Process(pid_monitor)
            proc.terminate()
            assert proc.wait(120) in (0, None)
            self._check_processes(ignore_running=False)
            console.print('[magenta]All internal-api and gunicorn processes are terminated.')
        except Exception:
            console.print('[red]Exception occurred. Dumping all logs.')
            for file in parent_path.glob('*'):
                console.print(f'Dumping {file} (size: {file.stat().st_size})')
                console.print(file.read_text())
            raise

    def test_cli_internal_api_debug(self, app):
        if False:
            print('Hello World!')
        with mock.patch('airflow.cli.commands.internal_api_command.create_app', return_value=app), mock.patch.object(app, 'run') as app_run:
            args = self.parser.parse_args(['internal-api', '--debug'])
            internal_api_command.internal_api(args)
            app_run.assert_called_with(debug=True, use_reloader=False, port=9080, host='0.0.0.0')

    def test_cli_internal_api_args(self):
        if False:
            i = 10
            return i + 15
        with mock.patch('subprocess.Popen') as Popen, mock.patch.object(internal_api_command, 'GunicornMonitor'):
            args = self.parser.parse_args(['internal-api', '--access-logformat', 'custom_log_format', '--pid', '/tmp/x.pid'])
            internal_api_command.internal_api(args)
            Popen.assert_called_with([sys.executable, '-m', 'gunicorn', '--workers', '4', '--worker-class', 'sync', '--timeout', '120', '--bind', '0.0.0.0:9080', '--name', 'airflow-internal-api', '--pid', '/tmp/x.pid', '--access-logfile', '-', '--error-logfile', '-', '--access-logformat', 'custom_log_format', 'airflow.cli.commands.internal_api_command:cached_app()', '--preload'], close_fds=True)