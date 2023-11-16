"""Manage the Glances passwords list."""
from glances.logger import logger
from glances.password import GlancesPassword

class GlancesPasswordList(GlancesPassword):
    """Manage the Glances passwords list for the client|browser/server."""
    _section = 'passwords'

    def __init__(self, config=None, args=None):
        if False:
            for i in range(10):
                print('nop')
        super(GlancesPasswordList, self).__init__()
        self._password_dict = self.load(config)

    def load(self, config):
        if False:
            i = 10
            return i + 15
        'Load the password from the configuration file.'
        password_dict = {}
        if config is None:
            logger.warning('No configuration file available. Cannot load password list.')
        elif not config.has_section(self._section):
            logger.warning('No [%s] section in the configuration file. Cannot load password list.' % self._section)
        else:
            logger.info('Start reading the [%s] section in the configuration file' % self._section)
            password_dict = dict(config.items(self._section))
            logger.info('%s password(s) loaded from the configuration file' % len(password_dict))
        return password_dict

    def get_password(self, host=None):
        if False:
            i = 10
            return i + 15
        "Get the password from a Glances client or server.\n\n        If host=None, return the current server list (dict).\n        Else, return the host's password (or the default one if defined or None)\n        "
        if host is None:
            return self._password_dict
        else:
            try:
                return self._password_dict[host]
            except (KeyError, TypeError):
                try:
                    return self._password_dict['default']
                except (KeyError, TypeError):
                    return None

    def set_password(self, host, password):
        if False:
            return 10
        'Set a password for a specific host.'
        self._password_dict[host] = password