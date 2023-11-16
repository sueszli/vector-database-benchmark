from __future__ import absolute_import
from st2common.util.monkey_patch import monkey_patch
monkey_patch()
import os
import sys
from oslo_config import cfg
from st2common import log as logging
from st2common.logging.misc import get_logger_name_for_module
from st2common.service_setup import setup as common_setup
from st2common.service_setup import teardown as common_teardown
from st2common.service_setup import deregister_service
from st2common.exceptions.sensors import SensorNotFoundException
from st2common.constants.exit_codes import FAILURE_EXIT_CODE
from st2reactor.sensor import config
from st2reactor.container.manager import SensorContainerManager
from st2reactor.container.partitioner_lookup import get_sensors_partitioner
__all__ = ['main']
LOGGER_NAME = get_logger_name_for_module(sys.modules[__name__])
LOG = logging.getLogger(LOGGER_NAME)
SENSOR_CONTAINER = 'sensorcontainer'

def _setup():
    if False:
        return 10
    capabilities = {'name': 'sensorcontainer', 'type': 'passive'}
    common_setup(service=SENSOR_CONTAINER, config=config, setup_db=True, register_mq_exchanges=True, register_signal_handlers=True, register_runners=False, service_registry=True, capabilities=capabilities)

def _teardown():
    if False:
        while True:
            i = 10
    common_teardown()

def main():
    if False:
        while True:
            i = 10
    try:
        _setup()
        single_sensor_mode = cfg.CONF.single_sensor_mode or cfg.CONF.sensorcontainer.single_sensor_mode
        if single_sensor_mode and (not cfg.CONF.sensor_ref):
            raise ValueError('--sensor-ref argument must be provided when running in single sensor mode')
        sensors_partitioner = get_sensors_partitioner()
        container_manager = SensorContainerManager(sensors_partitioner=sensors_partitioner, single_sensor_mode=single_sensor_mode)
        return container_manager.run_sensors()
    except SystemExit as exit_code:
        deregister_service(SENSOR_CONTAINER)
        return exit_code
    except SensorNotFoundException as e:
        LOG.exception(e)
        return 1
    except:
        LOG.exception('(PID:%s) SensorContainer quit due to exception.', os.getpid())
        return FAILURE_EXIT_CODE
    finally:
        _teardown()