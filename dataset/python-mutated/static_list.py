"""Manage the Glances server static list."""
from socket import gaierror, gethostbyname
from glances.logger import logger

class GlancesStaticServer(object):
    """Manage the static servers list for the client browser."""
    _section = 'serverlist'

    def __init__(self, config=None, args=None):
        if False:
            while True:
                i = 10
        self._server_list = self.load(config)

    def load(self, config):
        if False:
            return 10
        'Load the server list from the configuration file.'
        server_list = []
        if config is None:
            logger.debug('No configuration file available. Cannot load server list.')
        elif not config.has_section(self._section):
            logger.warning('No [%s] section in the configuration file. Cannot load server list.' % self._section)
        else:
            logger.info('Start reading the [%s] section in the configuration file' % self._section)
            for i in range(1, 256):
                new_server = {}
                postfix = 'server_%s_' % str(i)
                for s in ['name', 'port', 'alias']:
                    new_server[s] = config.get_value(self._section, '%s%s' % (postfix, s))
                if new_server['name'] is not None:
                    if new_server['port'] is None:
                        new_server['port'] = '61209'
                    new_server['username'] = 'glances'
                    new_server['password'] = ''
                    try:
                        new_server['ip'] = gethostbyname(new_server['name'])
                    except gaierror as e:
                        logger.error('Cannot get IP address for server %s (%s)' % (new_server['name'], e))
                        continue
                    new_server['key'] = new_server['name'] + ':' + new_server['port']
                    new_server['status'] = 'UNKNOWN'
                    new_server['type'] = 'STATIC'
                    logger.debug('Add server %s to the static list' % new_server['name'])
                    server_list.append(new_server)
            logger.info('%s server(s) loaded from the configuration file' % len(server_list))
            logger.debug('Static server list: %s' % server_list)
        return server_list

    def get_servers_list(self):
        if False:
            i = 10
            return i + 15
        'Return the current server list (list of dict).'
        return self._server_list

    def set_server(self, server_pos, key, value):
        if False:
            i = 10
            return i + 15
        'Set the key to the value for the server_pos (position in the list).'
        self._server_list[server_pos][key] = value