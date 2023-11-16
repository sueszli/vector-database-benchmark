from __future__ import absolute_import
import copy
from oslo_config import cfg
from st2common import log as logging
from st2common.constants.sensors import DEFAULT_PARTITION_LOADER, KVSTORE_PARTITION_LOADER, FILE_PARTITION_LOADER, HASH_PARTITION_LOADER
from st2common.exceptions.sensors import SensorPartitionerNotSupportedException
from st2reactor.container.partitioners import DefaultPartitioner, KVStorePartitioner, FileBasedPartitioner, SingleSensorPartitioner
from st2reactor.container.hash_partitioner import HashPartitioner
__all__ = ['get_sensors_partitioner']
LOG = logging.getLogger(__name__)
PROVIDERS = {DEFAULT_PARTITION_LOADER: DefaultPartitioner, KVSTORE_PARTITION_LOADER: KVStorePartitioner, FILE_PARTITION_LOADER: FileBasedPartitioner, HASH_PARTITION_LOADER: HashPartitioner}

def get_sensors_partitioner():
    if False:
        while True:
            i = 10
    if cfg.CONF.sensor_ref:
        LOG.info('Running in single sensor mode, using a single sensor partitioner...')
        return SingleSensorPartitioner(sensor_ref=cfg.CONF.sensor_ref)
    partition_provider_config = copy.copy(cfg.CONF.sensorcontainer.partition_provider)
    partition_provider = partition_provider_config.pop('name')
    sensor_node_name = cfg.CONF.sensorcontainer.sensor_node_name
    provider = PROVIDERS.get(partition_provider.lower(), None)
    if not provider:
        raise SensorPartitionerNotSupportedException('Partition provider %s not found.' % partition_provider)
    LOG.info('Using partitioner %s with sensornode %s.', partition_provider, sensor_node_name)
    return provider(sensor_node_name=sensor_node_name, **partition_provider_config)