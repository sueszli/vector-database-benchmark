from __future__ import absolute_import
import os
from st2common.constants.pack import SYSTEM_PACK_NAME
from st2common.constants.sensors import MINIMUM_POLL_INTERVAL
from st2common.models.db.sensor import SensorTypeDB
from st2common.services import triggers as trigger_service
__all__ = ['to_sensor_db_model', 'get_sensor_entry_point', 'create_trigger_types']

def to_sensor_db_model(sensor_api_model=None):
    if False:
        print('Hello World!')
    '\n    Converts a SensorTypeAPI model to DB model.\n    Also, creates trigger type objects provided in SensorTypeAPI.\n\n    :param sensor_api_model: SensorTypeAPI object.\n    :type sensor_api_model: :class:`SensorTypeAPI`\n\n    :rtype: :class:`SensorTypeDB`\n    '
    class_name = getattr(sensor_api_model, 'class_name', None)
    pack = getattr(sensor_api_model, 'pack', None)
    entry_point = get_sensor_entry_point(sensor_api_model)
    artifact_uri = getattr(sensor_api_model, 'artifact_uri', None)
    description = getattr(sensor_api_model, 'description', None)
    trigger_types = getattr(sensor_api_model, 'trigger_types', [])
    poll_interval = getattr(sensor_api_model, 'poll_interval', None)
    enabled = getattr(sensor_api_model, 'enabled', True)
    metadata_file = getattr(sensor_api_model, 'metadata_file', None)
    poll_interval = getattr(sensor_api_model, 'poll_interval', None)
    if poll_interval and poll_interval < MINIMUM_POLL_INTERVAL:
        raise ValueError('Minimum possible poll_interval is %s seconds' % MINIMUM_POLL_INTERVAL)
    for trigger_type in trigger_types:
        trigger_type['pack'] = pack
        trigger_type['metadata_file'] = metadata_file
    trigger_type_refs = create_trigger_types(trigger_types)
    return _create_sensor_type(pack=pack, name=class_name, description=description, artifact_uri=artifact_uri, entry_point=entry_point, trigger_types=trigger_type_refs, poll_interval=poll_interval, enabled=enabled, metadata_file=metadata_file)

def create_trigger_types(trigger_types, metadata_file=None):
    if False:
        print('Hello World!')
    if not trigger_types:
        return []
    trigger_type_dbs = trigger_service.add_trigger_models(trigger_types=trigger_types)
    trigger_type_refs = []
    for (trigger_type_db, _) in trigger_type_dbs:
        ref_obj = trigger_type_db.get_reference()
        trigger_type_ref = ref_obj.ref
        trigger_type_refs.append(trigger_type_ref)
    return trigger_type_refs

def _create_sensor_type(pack=None, name=None, description=None, artifact_uri=None, entry_point=None, trigger_types=None, poll_interval=10, enabled=True, metadata_file=None):
    if False:
        return 10
    sensor_type = SensorTypeDB(pack=pack, name=name, description=description, artifact_uri=artifact_uri, entry_point=entry_point, poll_interval=poll_interval, enabled=enabled, trigger_types=trigger_types, metadata_file=metadata_file)
    return sensor_type

def get_sensor_entry_point(sensor_api_model):
    if False:
        for i in range(10):
            print('nop')
    file_path = getattr(sensor_api_model, 'artifact_uri', None)
    class_name = getattr(sensor_api_model, 'class_name', None)
    pack = getattr(sensor_api_model, 'pack', None)
    if pack == SYSTEM_PACK_NAME:
        entry_point = class_name
    else:
        module_path = file_path.split('/%s/' % pack)[1]
        module_path = module_path.replace(os.path.sep, '.')
        module_path = module_path.replace('.py', '')
        entry_point = '%s.%s' % (module_path, class_name)
    return entry_point