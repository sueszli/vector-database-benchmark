from __future__ import absolute_import
import os
import sys
import signal
import psutil
from oslo_config import cfg
import st2tests.config
from st2common.util import concurrency
from st2common.models.db import db_setup
from st2reactor.container.process_container import PROCESS_EXIT_TIMEOUT
from st2common.util.green.shell import run_command
from st2common.bootstrap.sensorsregistrar import register_sensors
from st2tests.base import IntegrationTestCase
__all__ = ['SensorContainerTestCase']
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ST2_CONFIG_PATH = os.path.join(BASE_DIR, '../../../conf/st2.tests.conf')
ST2_CONFIG_PATH = os.path.abspath(ST2_CONFIG_PATH)
PYTHON_BINARY = sys.executable
BINARY = os.path.join(BASE_DIR, '../../../st2reactor/bin/st2sensorcontainer')
BINARY = os.path.abspath(BINARY)
PACKS_BASE_PATH = os.path.abspath(os.path.join(BASE_DIR, '../../../contrib'))
DEFAULT_CMD = [PYTHON_BINARY, BINARY, '--config-file', ST2_CONFIG_PATH, '--sensor-ref=examples.SamplePollingSensor']

class SensorContainerTestCase(IntegrationTestCase):
    """
    Note: For those tests MongoDB must be running, virtualenv must exist for
    examples pack and sensors from the example pack must be registered.
    """
    print_stdout_stderr_on_teardown = True

    @classmethod
    def setUpClass(cls):
        if False:
            return 10
        super(SensorContainerTestCase, cls).setUpClass()
        st2tests.config.parse_args()
        username = cfg.CONF.database.username if hasattr(cfg.CONF.database, 'username') else None
        password = cfg.CONF.database.password if hasattr(cfg.CONF.database, 'password') else None
        cls.db_connection = db_setup(cfg.CONF.database.db_name, cfg.CONF.database.host, cfg.CONF.database.port, username=username, password=password, ensure_indexes=False)
        cfg.CONF.content.packs_base_paths = PACKS_BASE_PATH
        register_sensors(packs_base_paths=[PACKS_BASE_PATH], use_pack_cache=False)
        virtualenv_path = '/tmp/virtualenvs/examples'
        run_command(cmd=['rm', '-rf', virtualenv_path])
        cmd = ['virtualenv', '--system-site-packages', '--python', PYTHON_BINARY, virtualenv_path]
        run_command(cmd=cmd)

    def test_child_processes_are_killed_on_sigint(self):
        if False:
            return 10
        process = self._start_sensor_container()
        concurrency.sleep(7)
        self.assertProcessIsRunning(process=process)
        pp = psutil.Process(process.pid)
        children_pp = pp.children()
        self.assertEqual(pp.cmdline()[1:], DEFAULT_CMD[1:])
        self.assertEqual(len(children_pp), 1)
        process.send_signal(signal.SIGINT)
        concurrency.sleep(PROCESS_EXIT_TIMEOUT + 1)
        self.assertProcessExited(proc=pp)
        self.assertProcessExited(proc=children_pp[0])
        self.remove_process(process=process)

    def test_child_processes_are_killed_on_sigterm(self):
        if False:
            while True:
                i = 10
        process = self._start_sensor_container()
        concurrency.sleep(5)
        pp = psutil.Process(process.pid)
        children_pp = pp.children()
        self.assertEqual(pp.cmdline()[1:], DEFAULT_CMD[1:])
        self.assertEqual(len(children_pp), 1)
        process.send_signal(signal.SIGTERM)
        concurrency.sleep(PROCESS_EXIT_TIMEOUT + 8)
        self.assertProcessExited(proc=pp)
        self.assertProcessExited(proc=children_pp[0])
        self.remove_process(process=process)

    def test_child_processes_are_killed_on_sigkill(self):
        if False:
            return 10
        process = self._start_sensor_container()
        concurrency.sleep(5)
        pp = psutil.Process(process.pid)
        children_pp = pp.children()
        self.assertEqual(pp.cmdline()[1:], DEFAULT_CMD[1:])
        self.assertEqual(len(children_pp), 1)
        process.send_signal(signal.SIGKILL)
        concurrency.sleep(1)
        self.assertProcessExited(proc=pp)
        self.assertProcessExited(proc=children_pp[0])
        self.remove_process(process=process)

    def test_single_sensor_mode(self):
        if False:
            return 10
        cmd = [PYTHON_BINARY, BINARY, '--config-file', ST2_CONFIG_PATH, '--single-sensor-mode']
        process = self._start_sensor_container(cmd=cmd)
        pp = psutil.Process(process.pid)
        concurrency.sleep(5)
        stdout = process.stdout.read()
        self.assertTrue(b'--sensor-ref argument must be provided when running in single sensor mode' in stdout)
        self.assertProcessExited(proc=pp)
        self.remove_process(process=process)
        cmd = [BINARY, '--config-file', ST2_CONFIG_PATH, '--single-sensor-mode', '--sensor-ref=examples.SampleSensorExit']
        process = self._start_sensor_container(cmd=cmd)
        pp = psutil.Process(process.pid)
        concurrency.sleep(1)
        stdout = process.stdout.read()
        self.assertTrue(b'Process for sensor examples.SampleSensorExit has exited with code 110')
        self.assertTrue(b'Not respawning a sensor since running in single sensor mode')
        self.assertTrue(b'Process container quit with exit_code 110.')
        concurrency.sleep(2)
        self.assertProcessExited(proc=pp)
        self.remove_process(process=process)

    def _start_sensor_container(self, cmd=DEFAULT_CMD):
        if False:
            i = 10
            return i + 15
        subprocess = concurrency.get_subprocess_module()
        print('Using command: %s' % ' '.join(cmd))
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False, preexec_fn=os.setsid)
        self.add_process(process=process)
        return process