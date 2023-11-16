from . import config
from . import ConfigBase
from . import CONFIG_FORMATS
from . import URLBase
from .AppriseAsset import AppriseAsset
from . import common
from .utils import GET_SCHEMA_RE
from .utils import parse_list
from .utils import is_exclusive_match
from .logger import logger

class AppriseConfig:
    """
    Our Apprise Configuration File Manager

        - Supports a list of URLs defined one after another (text format)
        - Supports a destinct YAML configuration format

    """

    def __init__(self, paths=None, asset=None, cache=True, recursion=0, insecure_includes=False, **kwargs):
        if False:
            print('Hello World!')
        "\n        Loads all of the paths specified (if any).\n\n        The path can either be a single string identifying one explicit\n        location, otherwise you can pass in a series of locations to scan\n        via a list.\n\n        If no path is specified then a default list is used.\n\n        By default we cache our responses so that subsiquent calls does not\n        cause the content to be retrieved again. Setting this to False does\n        mean more then one call can be made to retrieve the (same) data.  This\n        method can be somewhat inefficient if disabled and you're set up to\n        make remote calls.  Only disable caching if you understand the\n        consequences.\n\n        You can alternatively set the cache value to an int identifying the\n        number of seconds the previously retrieved can exist for before it\n        should be considered expired.\n\n        It's also worth nothing that the cache value is only set to elements\n        that are not already of subclass ConfigBase()\n\n        recursion defines how deep we recursively handle entries that use the\n        `import` keyword. This keyword requires us to fetch more configuration\n        from another source and add it to our existing compilation. If the\n        file we remotely retrieve also has an `import` reference, we will only\n        advance through it if recursion is set to 2 deep.  If set to zero\n        it is off.  There is no limit to how high you set this value. It would\n        be recommended to keep it low if you do intend to use it.\n\n        insecure includes by default are disabled. When set to True, all\n        Apprise Config files marked to be in STRICT mode are treated as being\n        in ALWAYS mode.\n\n        Take a file:// based configuration for example, only a file:// based\n        configuration can import another file:// based one. because it is set\n        to STRICT mode. If an http:// based configuration file attempted to\n        import a file:// one it woul fail. However this import would be\n        possible if insecure_includes is set to True.\n\n        There are cases where a self hosting apprise developer may wish to load\n        configuration from memory (in a string format) that contains import\n        entries (even file:// based ones).  In these circumstances if you want\n        these includes to be honored, this value must be set to True.\n        "
        self.configs = list()
        self.asset = asset if isinstance(asset, AppriseAsset) else AppriseAsset()
        self.cache = cache
        self.recursion = recursion
        self.insecure_includes = insecure_includes
        if paths is not None:
            self.add(paths)
        return

    def add(self, configs, asset=None, tag=None, cache=True, recursion=None, insecure_includes=None):
        if False:
            for i in range(10):
                print('nop')
        "\n        Adds one or more config URLs into our list.\n\n        You can override the global asset if you wish by including it with the\n        config(s) that you add.\n\n        By default we cache our responses so that subsiquent calls does not\n        cause the content to be retrieved again. Setting this to False does\n        mean more then one call can be made to retrieve the (same) data.  This\n        method can be somewhat inefficient if disabled and you're set up to\n        make remote calls.  Only disable caching if you understand the\n        consequences.\n\n        You can alternatively set the cache value to an int identifying the\n        number of seconds the previously retrieved can exist for before it\n        should be considered expired.\n\n        It's also worth nothing that the cache value is only set to elements\n        that are not already of subclass ConfigBase()\n\n        Optionally override the default recursion value.\n\n        Optionally override the insecure_includes flag.\n        if insecure_includes is set to True then all plugins that are\n        set to a STRICT mode will be a treated as ALWAYS.\n        "
        return_status = True
        cache = cache if cache is not None else self.cache
        recursion = recursion if recursion is not None else self.recursion
        insecure_includes = insecure_includes if insecure_includes is not None else self.insecure_includes
        if asset is None:
            asset = self.asset
        if isinstance(configs, ConfigBase):
            self.configs.append(configs)
            return True
        elif isinstance(configs, str):
            configs = (configs,)
        elif not isinstance(configs, (tuple, set, list)):
            logger.error('An invalid configuration path (type={}) was specified.'.format(type(configs)))
            return False
        for _config in configs:
            if isinstance(_config, ConfigBase):
                self.configs.append(_config)
                continue
            elif not isinstance(_config, str):
                logger.warning('An invalid configuration (type={}) was specified.'.format(type(_config)))
                return_status = False
                continue
            logger.debug('Loading configuration: {}'.format(_config))
            instance = AppriseConfig.instantiate(_config, asset=asset, tag=tag, cache=cache, recursion=recursion, insecure_includes=insecure_includes)
            if not isinstance(instance, ConfigBase):
                return_status = False
                continue
            self.configs.append(instance)
        return return_status

    def add_config(self, content, asset=None, tag=None, format=None, recursion=None, insecure_includes=None):
        if False:
            return 10
        "\n        Adds one configuration file in it's raw format. Content gets loaded as\n        a memory based object and only exists for the life of this\n        AppriseConfig object it was loaded into.\n\n        If you know the format ('yaml' or 'text') you can specify\n        it for slightly less overhead during this call.  Otherwise the\n        configuration is auto-detected.\n\n        Optionally override the default recursion value.\n\n        Optionally override the insecure_includes flag.\n        if insecure_includes is set to True then all plugins that are\n        set to a STRICT mode will be a treated as ALWAYS.\n        "
        recursion = recursion if recursion is not None else self.recursion
        insecure_includes = insecure_includes if insecure_includes is not None else self.insecure_includes
        if asset is None:
            asset = self.asset
        if not isinstance(content, str):
            logger.warning('An invalid configuration (type={}) was specified.'.format(type(content)))
            return False
        logger.debug('Loading raw configuration: {}'.format(content))
        instance = config.ConfigMemory(content=content, format=format, asset=asset, tag=tag, recursion=recursion, insecure_includes=insecure_includes)
        if instance.config_format not in CONFIG_FORMATS:
            logger.warning('The format of the configuration could not be deteced.')
            return False
        self.configs.append(instance)
        return True

    def servers(self, tag=common.MATCH_ALL_TAG, match_always=True, *args, **kwargs):
        if False:
            print('Hello World!')
        '\n        Returns all of our servers dynamically build based on parsed\n        configuration.\n\n        If a tag is specified, it applies to the configuration sources\n        themselves and not the notification services inside them.\n\n        This is for filtering the configuration files polled for\n        results.\n\n        If the anytag is set, then any notification that is found\n        set with that tag are included in the response.\n\n        '
        match_always = common.MATCH_ALWAYS_TAG if match_always else None
        response = list()
        for entry in self.configs:
            if is_exclusive_match(logic=tag, data=entry.tags, match_all=common.MATCH_ALL_TAG, match_always=match_always):
                response.extend(entry.servers())
        return response

    @staticmethod
    def instantiate(url, asset=None, tag=None, cache=None, recursion=0, insecure_includes=False, suppress_exceptions=True):
        if False:
            i = 10
            return i + 15
        '\n        Returns the instance of a instantiated configuration plugin based on\n        the provided Config URL.  If the url fails to be parsed, then None\n        is returned.\n\n        '
        schema = GET_SCHEMA_RE.match(url)
        if schema is None:
            schema = config.ConfigFile.protocol
            url = '{}://{}'.format(schema, URLBase.quote(url))
        else:
            schema = schema.group('schema').lower()
            if schema not in common.CONFIG_SCHEMA_MAP:
                logger.warning('Unsupported schema {}.'.format(schema))
                return None
        results = common.CONFIG_SCHEMA_MAP[schema].parse_url(url)
        if not results:
            logger.warning('Unparseable URL {}.'.format(url))
            return None
        results['tag'] = set(parse_list(tag))
        results['asset'] = asset if isinstance(asset, AppriseAsset) else AppriseAsset()
        if cache is not None:
            results['cache'] = cache
        results['recursion'] = recursion
        results['insecure_includes'] = insecure_includes
        if suppress_exceptions:
            try:
                cfg_plugin = common.CONFIG_SCHEMA_MAP[results['schema']](**results)
            except Exception:
                logger.warning('Could not load URL: %s' % url)
                return None
        else:
            cfg_plugin = common.CONFIG_SCHEMA_MAP[results['schema']](**results)
        return cfg_plugin

    def clear(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Empties our configuration list\n\n        '
        self.configs[:] = []

    def server_pop(self, index):
        if False:
            return 10
        '\n        Removes an indexed Apprise Notification from the servers\n        '
        prev_offset = -1
        offset = prev_offset
        for entry in self.configs:
            servers = entry.servers(cache=True)
            if len(servers) > 0:
                offset = prev_offset + len(servers)
                if offset >= index:
                    return entry.pop(index if prev_offset == -1 else index - prev_offset - 1)
                prev_offset = offset
        raise IndexError('list index out of range')

    def pop(self, index=-1):
        if False:
            for i in range(10):
                print('nop')
        '\n        Removes an indexed Apprise Configuration from the stack and returns it.\n\n        By default, the last element is removed from the list\n        '
        return self.configs.pop(index)

    def __getitem__(self, index):
        if False:
            return 10
        '\n        Returns the indexed config entry of a loaded apprise configuration\n        '
        return self.configs[index]

    def __bool__(self):
        if False:
            i = 10
            return i + 15
        "\n        Allows the Apprise object to be wrapped in an 'if statement'.\n        True is returned if at least one service has been loaded.\n        "
        return True if self.configs else False

    def __iter__(self):
        if False:
            i = 10
            return i + 15
        '\n        Returns an iterator to our config list\n        '
        return iter(self.configs)

    def __len__(self):
        if False:
            return 10
        '\n        Returns the number of config entries loaded\n        '
        return len(self.configs)