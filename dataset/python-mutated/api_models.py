import abc
DEFAULT_PACK_NAME = 'default'
NotificationSubSchemaAPI = {'type': 'object', 'properties': {'message': {'type': 'string', 'description': 'Message to use for notification'}, 'data': {'type': 'object', 'description': 'Data to be sent as part of notification'}, 'routes': {'type': 'array', 'description': 'Channels to post notifications to.'}, 'channels': {'type': 'array', 'description': 'Channels to post notifications to.'}}, 'additionalProperties': False}

def get_schema(**kwargs):
    if False:
        while True:
            i = 10
    return {}

class BaseAPI(abc.ABC):
    schema = abc.abstractproperty
    name = None

class ActionAPI(BaseAPI):
    """
    The system entity that represents a Stack Action/Automation in the system.
    """
    schema = {'title': 'Action', 'description': 'An activity that happens as a response to the external event.', 'type': 'object', 'properties': {'id': {'description': 'The unique identifier for the action.', 'type': 'string'}, 'ref': {'description': 'System computed user friendly reference for the action.                                 Provided value will be overridden by computed value.', 'type': 'string'}, 'uid': {'type': 'string'}, 'name': {'description': 'The name of the action.', 'type': 'string', 'required': True}, 'description': {'description': 'The description of the action.', 'type': 'string'}, 'enabled': {'description': 'Enable or disable the action from invocation.', 'type': 'boolean', 'default': True}, 'runner_type': {'description': 'The type of runner that executes the action.', 'type': 'string', 'required': True}, 'entry_point': {'description': 'The entry point for the action.', 'type': 'string', 'default': ''}, 'pack': {'description': 'The content pack this action belongs to.', 'type': 'string', 'default': DEFAULT_PACK_NAME}, 'parameters': {'description': 'Input parameters for the action.', 'type': 'object', 'patternProperties': {'^\\w+$': get_schema()}, 'additionalProperties': False, 'default': {}}, 'output_schema': get_schema(description='Action Output Schema'), 'tags': {'description': 'User associated metadata assigned to this object.', 'type': 'array', 'items': {'type': 'object'}}, 'notify': {'description': 'Notification settings for action.', 'type': 'object', 'properties': {'on-complete': NotificationSubSchemaAPI, 'on-failure': NotificationSubSchemaAPI, 'on-success': NotificationSubSchemaAPI}, 'additionalProperties': False}, 'metadata_file': {'description': 'Path to the metadata file relative to the pack directory.', 'type': 'string', 'default': ''}}, 'additionalProperties': False}

class TriggerTypeAPI(BaseAPI):
    schema = {'type': 'object', 'properties': {'id': {'type': 'string', 'default': None}, 'ref': {'type': 'string'}, 'uid': {'type': 'string'}, 'name': {'type': 'string', 'required': True}, 'pack': {'type': 'string'}, 'description': {'type': 'string'}, 'payload_schema': {'type': 'object', 'default': {}}, 'parameters_schema': {'type': 'object', 'default': {}}, 'tags': {'description': 'User associated metadata assigned to this object.', 'type': 'array', 'items': {'type': 'object'}}, 'metadata_file': {'description': 'Path to the metadata file relative to the pack directory.', 'type': 'string', 'default': ''}}, 'additionalProperties': False}