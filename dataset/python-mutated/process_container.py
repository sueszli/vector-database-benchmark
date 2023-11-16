from __future__ import absolute_import
import os
import sys
import time
import json
import subprocess
from collections import defaultdict
import six
from oslo_config import cfg
from st2common import log as logging
from st2common.util import concurrency
from st2common.constants.error_messages import PACK_VIRTUALENV_DOESNT_EXIST
from st2common.constants.system import API_URL_ENV_VARIABLE_NAME
from st2common.constants.system import AUTH_TOKEN_ENV_VARIABLE_NAME
from st2common.constants.triggers import SENSOR_SPAWN_TRIGGER, SENSOR_EXIT_TRIGGER
from st2common.constants.exit_codes import SUCCESS_EXIT_CODE
from st2common.constants.exit_codes import FAILURE_EXIT_CODE
from st2common.models.system.common import ResourceReference
from st2common.services.access import create_token
from st2common.transport.reactor import TriggerDispatcher
from st2common.util.api import get_full_public_api_url
from st2common.util.pack import get_pack_common_libs_path_for_pack_ref
from st2common.util.shell import on_parent_exit
from st2common.util.sandboxing import get_sandbox_python_path
from st2common.util.sandboxing import get_sandbox_python_binary_path
from st2common.util.sandboxing import get_sandbox_virtualenv_path
__all__ = ['ProcessSensorContainer']
LOG = logging.getLogger('st2reactor.process_sensor_container')
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WRAPPER_SCRIPT_NAME = 'sensor_wrapper.py'
WRAPPER_SCRIPT_PATH = os.path.join(BASE_DIR, WRAPPER_SCRIPT_NAME)
SENSOR_MAX_RESPAWN_COUNTS = 2
SENSOR_SUCCESSFUL_START_THRESHOLD = 10
SENSOR_RESPAWN_DELAY = 2.5
PROCESS_EXIT_TIMEOUT = 5

class ProcessSensorContainer(object):
    """
    Sensor container which runs sensors in a separate process.
    """

    def __init__(self, sensors, poll_interval=5, single_sensor_mode=False, dispatcher=None, wrapper_script_path=WRAPPER_SCRIPT_PATH, create_token=True):
        if False:
            for i in range(10):
                print('nop')
        '\n        :param sensors: A list of sensor dicts.\n        :type sensors: ``list`` of ``dict``\n\n        :param poll_interval: How long to sleep between each poll for running / dead sensors.\n        :type poll_interval: ``float``\n\n        :param wrapper_script_path: Path to the sensor wrapper script.\n        :type wrapper_script_path: ``str``\n\n        :param create_token: True to create temporary authentication token for the purpose for each\n                             sensor process and add it to that process environment variables.\n        :type create_token: ``bool``\n        '
        self._poll_interval = poll_interval
        self._single_sensor_mode = single_sensor_mode
        self._wrapper_script_path = wrapper_script_path
        self._create_token = create_token
        if self._single_sensor_mode:
            self._poll_interval = 1
        self._sensors = {}
        self._processes = {}
        self._dispatcher = dispatcher or TriggerDispatcher(LOG)
        self._stopped = False
        self._exit_code = None
        sensors = sensors or []
        for sensor_obj in sensors:
            sensor_id = self._get_sensor_id(sensor=sensor_obj)
            self._sensors[sensor_id] = sensor_obj
        self._sensor_start_times = {}
        self._sensor_respawn_counts = defaultdict(int)
        self._internal_sensor_state_variables = [self._processes, self._sensors, self._sensor_start_times]
        self._enable_common_pack_libs = cfg.CONF.packs.enable_common_libs or False

    def run(self):
        if False:
            return 10
        self._run_all_sensors()
        success_exception_cls = concurrency.get_greenlet_exit_exception_class()
        try:
            while not self._stopped:
                sensor_ids = list(self._sensors.keys())
                if len(sensor_ids) >= 1:
                    LOG.debug('%d active sensor(s)' % len(sensor_ids))
                    self._poll_sensors_for_results(sensor_ids)
                else:
                    LOG.debug('No active sensors')
                concurrency.sleep(self._poll_interval)
        except success_exception_cls:
            self._stopped = True
            return SUCCESS_EXIT_CODE
        except:
            LOG.exception('Container failed to run sensors.')
            self._stopped = True
            return FAILURE_EXIT_CODE
        self._stopped = True
        LOG.error('Process container stopped.')
        exit_code = self._exit_code or SUCCESS_EXIT_CODE
        return exit_code

    def _poll_sensors_for_results(self, sensor_ids):
        if False:
            return 10
        '\n        Main loop which polls sensor for results and detects dead sensors.\n        '
        for sensor_id in sensor_ids:
            now = int(time.time())
            process = self._processes[sensor_id]
            status = process.poll()
            if status is not None:
                LOG.info('Process for sensor %s has exited with code %s', sensor_id, status)
                sensor = self._sensors[sensor_id]
                self._delete_sensor(sensor_id)
                self._dispatch_trigger_for_sensor_exit(sensor=sensor, exit_code=status)
                concurrency.spawn(self._respawn_sensor, sensor_id=sensor_id, sensor=sensor, exit_code=status)
            else:
                sensor_start_time = self._sensor_start_times[sensor_id]
                sensor_respawn_count = self._sensor_respawn_counts[sensor_id]
                successfully_started = now - sensor_start_time >= SENSOR_SUCCESSFUL_START_THRESHOLD
                if successfully_started and sensor_respawn_count >= 1:
                    self._sensor_respawn_counts[sensor_id] = 0

    def running(self):
        if False:
            i = 10
            return i + 15
        return len(self._processes)

    def stopped(self):
        if False:
            return 10
        return self._stopped

    def shutdown(self, force=False):
        if False:
            while True:
                i = 10
        LOG.info('Container shutting down. Invoking cleanup on sensors.')
        self._stopped = True
        if force:
            exit_timeout = 0
        else:
            exit_timeout = PROCESS_EXIT_TIMEOUT
        sensor_ids = list(self._sensors.keys())
        for sensor_id in sensor_ids:
            self._stop_sensor_process(sensor_id=sensor_id, exit_timeout=exit_timeout)
        LOG.info('All sensors are shut down.')
        self._sensors = {}
        self._processes = {}

    def add_sensor(self, sensor):
        if False:
            for i in range(10):
                print('nop')
        '\n        Add a new sensor to the container.\n\n        :type sensor: ``dict``\n        '
        sensor_id = self._get_sensor_id(sensor=sensor)
        if sensor_id in self._sensors:
            LOG.warning('Sensor %s already exists and running.', sensor_id)
            return False
        self._spawn_sensor_process(sensor=sensor)
        LOG.debug('Sensor %s started.', sensor_id)
        self._sensors[sensor_id] = sensor
        return True

    def remove_sensor(self, sensor):
        if False:
            return 10
        '\n        Remove an existing sensor from the container.\n\n        :type sensor: ``dict``\n        '
        sensor_id = self._get_sensor_id(sensor=sensor)
        if sensor_id not in self._sensors:
            LOG.warning("Sensor %s isn't running in this container.", sensor_id)
            return False
        self._stop_sensor_process(sensor_id=sensor_id)
        LOG.debug('Sensor %s stopped.', sensor_id)
        return True

    def _run_all_sensors(self):
        if False:
            for i in range(10):
                print('nop')
        sensor_ids = list(self._sensors.keys())
        for sensor_id in sensor_ids:
            sensor_obj = self._sensors[sensor_id]
            LOG.info('Running sensor %s', sensor_id)
            try:
                self._spawn_sensor_process(sensor=sensor_obj)
            except Exception as e:
                LOG.warning(six.text_type(e), exc_info=True)
                del self._sensors[sensor_id]
                continue
            LOG.info('Sensor %s started' % sensor_id)

    def _spawn_sensor_process(self, sensor):
        if False:
            return 10
        '\n        Spawn a new process for the provided sensor.\n\n        New process uses isolated Python binary from a virtual environment\n        belonging to the sensor pack.\n        '
        sensor_id = self._get_sensor_id(sensor=sensor)
        pack_ref = sensor['pack']
        virtualenv_path = get_sandbox_virtualenv_path(pack=pack_ref)
        python_path = get_sandbox_python_binary_path(pack=pack_ref)
        if virtualenv_path and (not os.path.isdir(virtualenv_path)):
            format_values = {'pack': sensor['pack'], 'virtualenv_path': virtualenv_path}
            msg = PACK_VIRTUALENV_DOESNT_EXIST % format_values
            raise Exception(msg)
        args = self._get_args_for_wrapper_script(python_binary=python_path, sensor=sensor)
        if self._enable_common_pack_libs:
            pack_common_libs_path = get_pack_common_libs_path_for_pack_ref(pack_ref=pack_ref)
        else:
            pack_common_libs_path = None
        env = os.environ.copy()
        sandbox_python_path = get_sandbox_python_path(inherit_from_parent=True, inherit_parent_virtualenv=True)
        if self._enable_common_pack_libs and pack_common_libs_path:
            env['PYTHONPATH'] = pack_common_libs_path + ':' + sandbox_python_path
        else:
            env['PYTHONPATH'] = sandbox_python_path
        if self._create_token:
            LOG.debug('Creating temporary auth token for sensor %s' % sensor['class_name'])
            ttl = cfg.CONF.auth.service_token_ttl
            metadata = {'service': 'sensors_container', 'sensor_path': sensor['file_path'], 'sensor_class': sensor['class_name']}
            temporary_token = create_token(username='sensors_container', ttl=ttl, metadata=metadata, service=True)
            env[API_URL_ENV_VARIABLE_NAME] = get_full_public_api_url()
            env[AUTH_TOKEN_ENV_VARIABLE_NAME] = temporary_token.token
        cmd = ' '.join(args)
        LOG.debug('Running sensor subprocess (cmd="%s")', cmd)
        try:
            process = subprocess.Popen(args=args, stdin=None, stdout=None, stderr=None, shell=False, env=env, preexec_fn=on_parent_exit('SIGTERM'))
        except Exception as e:
            cmd = ' '.join(args)
            message = 'Failed to spawn process for sensor %s ("%s"): %s' % (sensor_id, cmd, six.text_type(e))
            raise Exception(message)
        self._processes[sensor_id] = process
        self._sensors[sensor_id] = sensor
        self._sensor_start_times[sensor_id] = int(time.time())
        self._dispatch_trigger_for_sensor_spawn(sensor=sensor, process=process, cmd=cmd)
        return process

    def _stop_sensor_process(self, sensor_id, exit_timeout=PROCESS_EXIT_TIMEOUT):
        if False:
            for i in range(10):
                print('nop')
        "\n        Stop a sensor process for the provided sensor.\n\n        :param sensor_id: Sensor ID.\n        :type sensor_id: ``str``\n\n        :param exit_timeout: How long to wait for process to exit after\n                             sending SIGTERM signal. If the process doesn't\n                             exit in this amount of seconds, SIGKILL signal\n                             will be sent to the process.\n        :type exit__timeout: ``int``\n        "
        process = self._processes[sensor_id]
        self._delete_sensor(sensor_id)
        process.terminate()
        timeout = 0
        sleep_delay = 1
        while timeout < exit_timeout:
            status = process.poll()
            if status is not None:
                break
            timeout += sleep_delay
            time.sleep(sleep_delay)
        if status is None:
            process.kill()

    def _respawn_sensor(self, sensor_id, sensor, exit_code):
        if False:
            for i in range(10):
                print('nop')
        '\n        Method for respawning a sensor which died with a non-zero exit code.\n        '
        extra = {'sensor_id': sensor_id, 'sensor': sensor}
        if self._single_sensor_mode:
            LOG.info('Not respawning a sensor since running in single sensor mode', extra=extra)
            self._stopped = True
            self._exit_code = exit_code
            return
        if self._stopped:
            LOG.debug('Stopped, not respawning a dead sensor', extra=extra)
            return
        should_respawn = self._should_respawn_sensor(sensor_id=sensor_id, sensor=sensor, exit_code=exit_code)
        if not should_respawn:
            LOG.debug('Not respawning a dead sensor', extra=extra)
            return
        LOG.debug('Respawning dead sensor', extra=extra)
        self._sensor_respawn_counts[sensor_id] += 1
        sleep_delay = SENSOR_RESPAWN_DELAY * self._sensor_respawn_counts[sensor_id]
        concurrency.sleep(sleep_delay)
        try:
            self._spawn_sensor_process(sensor=sensor)
        except Exception as e:
            LOG.warning(six.text_type(e), exc_info=True)
            del self._sensors[sensor_id]

    def _should_respawn_sensor(self, sensor_id, sensor, exit_code):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return True if the provided sensor should be respawned, False otherwise.\n        '
        if exit_code == 0:
            return False
        respawn_count = self._sensor_respawn_counts[sensor_id]
        if respawn_count >= SENSOR_MAX_RESPAWN_COUNTS:
            LOG.debug('Sensor has already been respawned max times, giving up')
            return False
        return True

    def _get_args_for_wrapper_script(self, python_binary, sensor):
        if False:
            return 10
        '\n        Return CLI arguments passed to the sensor wrapper script.\n\n        :param python_binary: Python binary used to execute wrapper script.\n        :type python_binary: ``str``\n\n        :param sensor: Sensor object dictionary.\n        :type sensor: ``dict``\n\n        :rtype: ``list``\n        '
        trigger_type_refs = sensor['trigger_types'] or []
        trigger_type_refs = ','.join(trigger_type_refs)
        parent_args = json.dumps(sys.argv[1:])
        args = [python_binary, self._wrapper_script_path, '--pack=%s' % sensor['pack'], '--file-path=%s' % sensor['file_path'], '--class-name=%s' % sensor['class_name'], '--trigger-type-refs=%s' % trigger_type_refs, '--parent-args=%s' % parent_args]
        if sensor['poll_interval']:
            args.append('--poll-interval=%s' % sensor['poll_interval'])
        return args

    def _get_sensor_id(self, sensor):
        if False:
            print('Hello World!')
        '\n        Return unique identifier for the provider sensor dict.\n\n        :type sensor: ``dict``\n        '
        sensor_id = sensor['ref']
        return sensor_id

    def _dispatch_trigger_for_sensor_spawn(self, sensor, process, cmd):
        if False:
            return 10
        trigger = ResourceReference.to_string_reference(name=SENSOR_SPAWN_TRIGGER['name'], pack=SENSOR_SPAWN_TRIGGER['pack'])
        now = int(time.time())
        payload = {'id': sensor['class_name'], 'timestamp': now, 'pid': process.pid, 'cmd': cmd}
        self._dispatcher.dispatch(trigger, payload=payload)

    def _dispatch_trigger_for_sensor_exit(self, sensor, exit_code):
        if False:
            for i in range(10):
                print('nop')
        trigger = ResourceReference.to_string_reference(name=SENSOR_EXIT_TRIGGER['name'], pack=SENSOR_EXIT_TRIGGER['pack'])
        now = int(time.time())
        payload = {'id': sensor['class_name'], 'timestamp': now, 'exit_code': exit_code}
        self._dispatcher.dispatch(trigger, payload=payload)

    def _delete_sensor(self, sensor_id):
        if False:
            for i in range(10):
                print('nop')
        '\n        Delete / reset all the internal state about a particular sensor.\n        '
        for var in self._internal_sensor_state_variables:
            if sensor_id in var:
                del var[sensor_id]