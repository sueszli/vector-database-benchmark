"""Manage the Glances ports list (Ports plugin)."""
from glances.logger import logger
from glances.globals import BSD
if not BSD:
    try:
        import netifaces
        netifaces_tag = True
    except ImportError:
        netifaces_tag = False
else:
    netifaces_tag = False

class GlancesPortsList(object):
    """Manage the ports list for the ports plugin."""
    _section = 'ports'
    _default_refresh = 60
    _default_timeout = 3

    def __init__(self, config=None, args=None):
        if False:
            while True:
                i = 10
        self._ports_list = self.load(config)

    def load(self, config):
        if False:
            print('Hello World!')
        'Load the ports list from the configuration file.'
        ports_list = []
        if config is None:
            logger.debug('No configuration file available. Cannot load ports list.')
        elif not config.has_section(self._section):
            logger.debug('No [%s] section in the configuration file. Cannot load ports list.' % self._section)
        else:
            logger.debug('Start reading the [%s] section in the configuration file' % self._section)
            refresh = int(config.get_value(self._section, 'refresh', default=self._default_refresh))
            timeout = int(config.get_value(self._section, 'timeout', default=self._default_timeout))
            default_gateway = config.get_value(self._section, 'port_default_gateway', default='False')
            if default_gateway.lower().startswith('true') and netifaces_tag:
                new_port = {}
                try:
                    new_port['host'] = netifaces.gateways()['default'][netifaces.AF_INET][0]
                except KeyError:
                    new_port['host'] = None
                new_port['port'] = 0
                new_port['description'] = 'DefaultGateway'
                new_port['refresh'] = refresh
                new_port['timeout'] = timeout
                new_port['status'] = None
                new_port['rtt_warning'] = None
                new_port['indice'] = str('port_0')
                logger.debug('Add default gateway %s to the static list' % new_port['host'])
                ports_list.append(new_port)
            for i in range(1, 256):
                new_port = {}
                postfix = 'port_%s_' % str(i)
                new_port['host'] = config.get_value(self._section, '%s%s' % (postfix, 'host'))
                if new_port['host'] is None:
                    continue
                new_port['port'] = config.get_value(self._section, '%s%s' % (postfix, 'port'), 0)
                new_port['description'] = config.get_value(self._section, '%sdescription' % postfix, default='%s:%s' % (new_port['host'], new_port['port']))
                new_port['status'] = None
                new_port['refresh'] = refresh
                new_port['timeout'] = int(config.get_value(self._section, '%stimeout' % postfix, default=timeout))
                new_port['rtt_warning'] = config.get_value(self._section, '%srtt_warning' % postfix, default=None)
                if new_port['rtt_warning'] is not None:
                    new_port['rtt_warning'] = int(new_port['rtt_warning']) / 1000.0
                new_port['indice'] = 'port_' + str(i)
                logger.debug('Add port %s:%s to the static list' % (new_port['host'], new_port['port']))
                ports_list.append(new_port)
            logger.debug('Ports list loaded: %s' % ports_list)
        return ports_list

    def get_ports_list(self):
        if False:
            return 10
        'Return the current server list (dict of dict).'
        return self._ports_list

    def set_server(self, pos, key, value):
        if False:
            print('Hello World!')
        'Set the key to the value for the pos (position in the list).'
        self._ports_list[pos][key] = value