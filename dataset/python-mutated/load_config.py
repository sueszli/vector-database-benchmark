"""Message bus configuration loader.

The message bus event handler and client use basically the same configuration.
This code is re-used in both to load config values.
"""
from collections import namedtuple
from mycroft.configuration import Configuration
from mycroft.util.log import LOG
MessageBusConfig = namedtuple('MessageBusConfig', ['host', 'port', 'route', 'ssl'])

def load_message_bus_config(**overrides):
    if False:
        while True:
            i = 10
    'Load the bits of device configuration needed to run the message bus.'
    LOG.info('Loading message bus configs')
    config = Configuration.get()
    try:
        websocket_configs = config['websocket']
    except KeyError as ke:
        LOG.error('No websocket configs found ({})'.format(repr(ke)))
        raise
    else:
        mb_config = MessageBusConfig(host=overrides.get('host') or websocket_configs.get('host'), port=overrides.get('port') or websocket_configs.get('port'), route=overrides.get('route') or websocket_configs.get('route'), ssl=overrides.get('ssl') or config.get('ssl'))
        if not all([mb_config.host, mb_config.port, mb_config.route]):
            error_msg = 'Missing one or more websocket configs'
            LOG.error(error_msg)
            raise ValueError(error_msg)
    return mb_config