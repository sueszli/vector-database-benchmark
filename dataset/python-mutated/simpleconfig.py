import configparser
import logging
import os
import shutil
logger = logging.getLogger(__name__)

class ConfigEntry(object):
    """ Simple config entry representation """

    def __init__(self, section, key, value):
        if False:
            return 10
        ' Create new config entry\n        :param str section: section name\n        :param str key: config entry name\n        :param value: config entry value\n        '
        self._key = key
        self._value = value
        self._section = section
        if value is None:
            self._value_type = lambda _: None
        else:
            self._value_type = type(value)

    def section(self):
        if False:
            print('Hello World!')
        ' Return config entry section '
        return self._section

    def key(self):
        if False:
            return 10
        ' Return config entry name '
        return self._key

    def value(self):
        if False:
            for i in range(10):
                print('nop')
        ' Return config entry value '
        return self._value

    def set_key(self, k):
        if False:
            while True:
                i = 10
        self._key = k

    def set_value(self, v):
        if False:
            while True:
                i = 10
        self._value = v

    def set_value_from_str(self, val):
        if False:
            print('Hello World!')
        'Change string to the value type and save it as a value\n        :param str val: string to be converse to value\n        '
        value_type = self._value_type(val)
        logger.debug('set_value_from_str(%(val)r). value_type=%(value_type)r', {'val': val, 'value_type': value_type})
        self.set_value(value_type)

    @classmethod
    def create_property(cls, section, key, value, other, prop_name):
        if False:
            while True:
                i = 10
        'Create new property: config entry with getter and setter method\n           for this property in other object. Append this entry to property\n           list in other object.\n        :param str section: config entry section name\n        :param str key: config entry name\n        :param value: config entry value\n        :param other: object instance for which new setter, getter and\n                      property entry should be created\n        :param str prop_name: property name\n        :return:\n        '
        property_ = ConfigEntry(section, key, value)
        getter_name = 'get_{}'.format(prop_name)
        setter_name = 'set_{}'.format(prop_name)

        def get_prop(_self):
            if False:
                print('Hello World!')
            return getattr(_self, prop_name).value()

        def set_prop(_self, val):
            if False:
                for i in range(10):
                    print('nop')
            return getattr(_self, prop_name).set_value(val)

        def get_properties(_self):
            if False:
                print('Hello World!')
            return getattr(_self, '_properties')
        setattr(other, prop_name, property_)
        setattr(other.__class__, getter_name, get_prop)
        setattr(other.__class__, setter_name, set_prop)
        if not hasattr(other, '_properties'):
            setattr(other, '_properties', [])
        if not hasattr(other.__class__, 'properties'):
            setattr(other.__class__, 'properties', get_properties)
        other._properties.append(property_)

class SimpleConfig(object):
    """ Simple configuration manager"""

    def __init__(self, node_config, cfg_file, refresh=False, keep_old=True):
        if False:
            while True:
                i = 10
        "Read existing configuration or create new one if it doesn't exist\n           or refresh option is set to True.\n        :param node_config: node specific configuration\n        :param str cfg_file: configuration file name\n        :param bool refresh: *Default: False*  if set to True, than\n                             configuration for given node should be written\n                             even if it already exists.\n        "
        self._node_config = node_config
        logger_msg = 'Reading config from file {}'.format(cfg_file)
        try:
            write_config = True
            cfg = configparser.ConfigParser()
            files = cfg.read(cfg_file)
            if files:
                if self._node_config.section() in cfg.sections():
                    if refresh:
                        cfg.remove_section(self._node_config.section())
                        cfg.add_section(self._node_config.section())
                    else:
                        self.__read_options(cfg)
                        if not keep_old:
                            self.__remove_old_options(cfg)
                else:
                    cfg.add_section(self._node_config.section())
                logger.info('%s ... successfully', logger_msg)
            else:
                logger.info('%s ... failed', logger_msg)
                cfg = self.__create_fresh_config()
            if write_config:
                logger.info("Writing %r's configuration to %r", self.get_node_config().section(), cfg_file)
                self.__write_config(cfg, cfg_file)
        except Exception:
            logger.exception('%r ... failed with an exception', logger_msg)
            logger.info('Failed to write configuration file. Creating fresh config.')
            self.__write_config(self.__create_fresh_config(), cfg_file)

    def get_node_config(self):
        if False:
            i = 10
            return i + 15
        ' Return node specific configuration '
        return self._node_config

    def __create_fresh_config(self):
        if False:
            while True:
                i = 10
        cfg = configparser.ConfigParser()
        cfg.add_section(self.get_node_config().section())
        return cfg

    def __write_config(self, cfg, cfg_file):
        if False:
            for i in range(10):
                print('nop')
        self.__write_options(cfg)
        if os.path.exists(cfg_file):
            backup_file_name = '{}.bak'.format(cfg_file)
            logger.info('Creating backup configuration file %r', backup_file_name)
            shutil.copy(cfg_file, backup_file_name)
        elif not os.path.exists(os.path.dirname(cfg_file)):
            os.makedirs(os.path.dirname(cfg_file))
        with open(cfg_file, 'w') as f:
            cfg.write(f)

    @staticmethod
    def __read_option(cfg, property_):
        if False:
            i = 10
            return i + 15
        return cfg.get(property_.section(), property_.key())

    @staticmethod
    def __write_option(cfg, property_):
        if False:
            i = 10
            return i + 15
        return cfg.set(property_.section(), property_.key(), str(property_.value()))

    def __read_options(self, cfg):
        if False:
            i = 10
            return i + 15
        for prop in self.get_node_config().properties():
            try:
                prop.set_value_from_str(self.__read_option(cfg, prop))
            except configparser.NoOptionError:
                logger.info('Adding new config option: %r (%r)', prop.key(), prop.value())

    def __write_options(self, cfg):
        if False:
            for i in range(10):
                print('nop')
        current_cfg = self.get_node_config()
        logger.debug('writing config, old = %s, new = %s', current_cfg, cfg)
        if not hasattr(current_cfg, '_properties'):
            return
        for prop in current_cfg.properties():
            self.__write_option(cfg, prop)

    def __remove_old_options(self, cfg):
        if False:
            print('Hello World!')
        props = [p.key() for p in self.get_node_config().properties()]
        for opt in cfg.options('Node'):
            if opt not in props:
                cfg.remove_option('Node', opt)