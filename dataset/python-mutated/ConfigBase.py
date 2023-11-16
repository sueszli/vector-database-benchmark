import os
import re
import yaml
import time
from .. import plugins
from .. import common
from ..AppriseAsset import AppriseAsset
from ..URLBase import URLBase
from ..utils import GET_SCHEMA_RE
from ..utils import parse_list
from ..utils import parse_bool
from ..utils import parse_urls
from ..utils import cwe312_url
VALID_TOKEN = re.compile('(?P<token>[a-z0-9][a-z0-9_]+)', re.I)

class ConfigBase(URLBase):
    """
    This is the base class for all supported configuration sources
    """
    encoding = 'utf-8'
    default_config_format = common.ConfigFormat.TEXT
    config_format = None
    max_buffer_size = 131072
    allow_cross_includes = common.ContentIncludeMode.NEVER
    config_path = os.getcwd()

    def __init__(self, cache=True, recursion=0, insecure_includes=False, **kwargs):
        if False:
            print('Hello World!')
        "\n        Initialize some general logging and common server arguments that will\n        keep things consistent when working with the configurations that\n        inherit this class.\n\n        By default we cache our responses so that subsiquent calls does not\n        cause the content to be retrieved again.  For local file references\n        this makes no difference at all.  But for remote content, this does\n        mean more then one call can be made to retrieve the (same) data.  This\n        method can be somewhat inefficient if disabled.  Only disable caching\n        if you understand the consequences.\n\n        You can alternatively set the cache value to an int identifying the\n        number of seconds the previously retrieved can exist for before it\n        should be considered expired.\n\n        recursion defines how deep we recursively handle entries that use the\n        `include` keyword. This keyword requires us to fetch more configuration\n        from another source and add it to our existing compilation. If the\n        file we remotely retrieve also has an `include` reference, we will only\n        advance through it if recursion is set to 2 deep.  If set to zero\n        it is off.  There is no limit to how high you set this value. It would\n        be recommended to keep it low if you do intend to use it.\n\n        insecure_include by default are disabled. When set to True, all\n        Apprise Config files marked to be in STRICT mode are treated as being\n        in ALWAYS mode.\n\n        Take a file:// based configuration for example, only a file:// based\n        configuration can include another file:// based one. because it is set\n        to STRICT mode. If an http:// based configuration file attempted to\n        include a file:// one it woul fail. However this include would be\n        possible if insecure_includes is set to True.\n\n        There are cases where a self hosting apprise developer may wish to load\n        configuration from memory (in a string format) that contains 'include'\n        entries (even file:// based ones).  In these circumstances if you want\n        these 'include' entries to be honored, this value must be set to True.\n        "
        super().__init__(**kwargs)
        self._cached_time = None
        self._cached_servers = None
        self.recursion = recursion
        self.insecure_includes = insecure_includes
        if 'encoding' in kwargs:
            self.encoding = kwargs.get('encoding')
        if 'format' in kwargs and isinstance(kwargs['format'], str):
            self.config_format = kwargs.get('format').lower()
            if self.config_format not in common.CONFIG_FORMATS:
                err = 'An invalid config format ({}) was specified.'.format(self.config_format)
                self.logger.warning(err)
                raise TypeError(err)
        try:
            self.cache = cache if isinstance(cache, bool) else int(cache)
            if self.cache < 0:
                err = 'A negative cache value ({}) was specified.'.format(cache)
                self.logger.warning(err)
                raise TypeError(err)
        except (ValueError, TypeError):
            err = 'An invalid cache value ({}) was specified.'.format(cache)
            self.logger.warning(err)
            raise TypeError(err)
        return

    def servers(self, asset=None, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Performs reads loaded configuration and returns all of the services\n        that could be parsed and loaded.\n\n        '
        if not self.expired():
            return self._cached_servers
        self._cached_servers = list()
        content = self.read(**kwargs)
        if not isinstance(content, str):
            self._cached_time = time.time()
            return self._cached_servers
        config_format = self.default_config_format if self.config_format is None else self.config_format
        fn = getattr(ConfigBase, 'config_parse_{}'.format(config_format))
        asset = asset if isinstance(asset, AppriseAsset) else self.asset
        (servers, configs) = fn(content=content, asset=asset)
        self._cached_servers.extend(servers)
        for url in configs:
            if self.recursion > 0:
                schema = GET_SCHEMA_RE.match(url)
                if schema is None:
                    schema = 'file'
                    if not os.path.isabs(url):
                        url = os.path.join(self.config_path, url)
                    url = '{}://{}'.format(schema, URLBase.quote(url))
                else:
                    schema = schema.group('schema').lower()
                    if schema not in common.CONFIG_SCHEMA_MAP:
                        ConfigBase.logger.warning('Unsupported include schema {}.'.format(schema))
                        continue
                loggable_url = url if not asset.secure_logging else cwe312_url(url)
                results = common.CONFIG_SCHEMA_MAP[schema].parse_url(url)
                if not results:
                    self.logger.warning('Unparseable include URL {}'.format(loggable_url))
                    continue
                if common.CONFIG_SCHEMA_MAP[schema].allow_cross_includes == common.ContentIncludeMode.STRICT and schema not in self.schemas() and (not self.insecure_includes) or common.CONFIG_SCHEMA_MAP[schema].allow_cross_includes == common.ContentIncludeMode.NEVER:
                    ConfigBase.logger.warning('Including {}:// based configuration is prohibited. Ignoring URL {}'.format(schema, loggable_url))
                    continue
                results['asset'] = asset
                results['cache'] = False
                results['recursion'] = self.recursion - 1
                results['insecure_includes'] = self.insecure_includes
                try:
                    cfg_plugin = common.CONFIG_SCHEMA_MAP[results['schema']](**results)
                except Exception as e:
                    self.logger.warning('Could not load include URL: {}'.format(loggable_url))
                    self.logger.debug('Loading Exception: {}'.format(str(e)))
                    continue
                self._cached_servers.extend(cfg_plugin.servers(asset=asset))
                del cfg_plugin
            else:
                loggable_url = url if not asset.secure_logging else cwe312_url(url)
                self.logger.debug('Recursion limit reached; ignoring Include URL: %s', loggable_url)
        if self._cached_servers:
            self.logger.info('Loaded {} entries from {}'.format(len(self._cached_servers), self.url(privacy=asset.secure_logging)))
        else:
            self.logger.warning('Failed to load Apprise configuration from {}'.format(self.url(privacy=asset.secure_logging)))
        self._cached_time = time.time()
        return self._cached_servers

    def read(self):
        if False:
            print('Hello World!')
        '\n        This object should be implimented by the child classes\n\n        '
        return None

    def expired(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Simply returns True if the configuration should be considered\n        as expired or False if content should be retrieved.\n        '
        if isinstance(self._cached_servers, list) and self.cache:
            if self.cache is True:
                return False
            age_in_sec = time.time() - self._cached_time
            if age_in_sec <= self.cache:
                return False
        return True

    @staticmethod
    def __normalize_tag_groups(group_tags):
        if False:
            while True:
                i = 10
        "\n        Used to normalize a tag assign map which looks like:\n          {\n             'group': set('{tag1}', '{group1}', '{tag2}'),\n             'group1': set('{tag2}','{tag3}'),\n          }\n\n          Then normalized it (merging groups); with respect to the above, the\n          output would be:\n          {\n             'group': set('{tag1}', '{tag2}', '{tag3}),\n             'group1': set('{tag2}','{tag3}'),\n          }\n\n        "
        tag_groups = set([str(x) for x in group_tags.keys()])

        def _expand(tags, ignore=None):
            if False:
                for i in range(10):
                    print('nop')
            '\n            Expands based on tag provided and returns a set\n\n            this also updates the group_tags while it goes\n            '
            results = set()
            ignore = set() if ignore is None else ignore
            groups = set()
            for tag in tags:
                if tag in ignore:
                    continue
                groups.add(tag)
                if tag not in group_tags:
                    group_tags[tag] = set()
                results |= group_tags[tag] - tag_groups
                found = group_tags[tag] & tag_groups
                if not found:
                    continue
                for gtag in found:
                    if gtag in ignore:
                        continue
                    ignore.add(tag)
                    group_tags[gtag] = _expand(set([gtag]), ignore=ignore)
                    results |= group_tags[gtag]
                    ignore.remove(tag)
            return results
        for tag in tag_groups:
            group_tags[tag] |= _expand(set([tag]))
            if not group_tags[tag]:
                ConfigBase.logger.warning('The group {} has no tags assigned to it'.format(tag))
                del group_tags[tag]

    @staticmethod
    def parse_url(url, verify_host=True):
        if False:
            for i in range(10):
                print('nop')
        'Parses the URL and returns it broken apart into a dictionary.\n\n        This is very specific and customized for Apprise.\n\n        Args:\n            url (str): The URL you want to fully parse.\n            verify_host (:obj:`bool`, optional): a flag kept with the parsed\n                 URL which some child classes will later use to verify SSL\n                 keys (if SSL transactions take place).  Unless under very\n                 specific circumstances, it is strongly recomended that\n                 you leave this default value set to True.\n\n        Returns:\n            A dictionary is returned containing the URL fully parsed if\n            successful, otherwise None is returned.\n        '
        results = URLBase.parse_url(url, verify_host=verify_host)
        if not results:
            return results
        if 'format' in results['qsd']:
            results['format'] = results['qsd'].get('format')
            if results['format'] not in common.CONFIG_FORMATS:
                URLBase.logger.warning('Unsupported format specified {}'.format(results['format']))
                del results['format']
        if 'encoding' in results['qsd']:
            results['encoding'] = results['qsd'].get('encoding')
        if 'cache' in results['qsd']:
            try:
                results['cache'] = int(results['qsd']['cache'])
            except (ValueError, TypeError):
                results['cache'] = parse_bool(results['qsd']['cache'])
        return results

    @staticmethod
    def detect_config_format(content, **kwargs):
        if False:
            print('Hello World!')
        '\n        Takes the specified content and attempts to detect the format type\n\n        The function returns the actual format type if detected, otherwise\n        it returns None\n        '
        valid_line_re = re.compile('^\\s*(?P<line>([;#]+(?P<comment>.*))|(?P<text>((?P<tag>[ \\t,a-z0-9_-]+)=)?[a-z0-9]+://.*)|((?P<yaml>[a-z0-9]+):.*))?$', re.I)
        try:
            content = re.split('\\r*\\n', content)
        except TypeError:
            ConfigBase.logger.error('Invalid Apprise configuration specified.')
            return None
        config_format = None
        for (line, entry) in enumerate(content, start=1):
            result = valid_line_re.match(entry)
            if not result:
                ConfigBase.logger.error('Undetectable Apprise configuration found based on line {}.'.format(line))
                return None
            if result.group('yaml'):
                config_format = common.ConfigFormat.YAML
                ConfigBase.logger.debug('Detected YAML configuration based on line {}.'.format(line))
                break
            elif result.group('text'):
                config_format = common.ConfigFormat.TEXT
                ConfigBase.logger.debug('Detected TEXT configuration based on line {}.'.format(line))
                break
            config_format = common.ConfigFormat.TEXT
        return config_format

    @staticmethod
    def config_parse(content, asset=None, config_format=None, **kwargs):
        if False:
            print('Hello World!')
        "\n        Takes the specified config content and loads it based on the specified\n        config_format. If a format isn't specified, then it is auto detected.\n\n        "
        if config_format is None:
            config_format = ConfigBase.detect_config_format(content)
            if not config_format:
                ConfigBase.logger.error('Could not detect configuration')
                return (list(), list())
        if config_format not in common.CONFIG_FORMATS:
            ConfigBase.logger.error('An invalid configuration format ({}) was specified'.format(config_format))
            return (list(), list())
        fn = getattr(ConfigBase, 'config_parse_{}'.format(config_format))
        return fn(content=content, asset=asset)

    @staticmethod
    def config_parse_text(content, asset=None):
        if False:
            for i in range(10):
                print('nop')
        "\n        Parse the specified content as though it were a simple text file only\n        containing a list of URLs.\n\n        Return a tuple that looks like (servers, configs) where:\n          - servers contains a list of loaded notification plugins\n          - configs contains a list of additional configuration files\n            referenced.\n\n        You may also optionally associate an asset with the notification.\n\n        The file syntax is:\n\n            #\n            # pound/hashtag allow for line comments\n            #\n            # One or more tags can be idenified using comma's (,) to separate\n            # them.\n            <Tag(s)>=<URL>\n\n            # Or you can use this format (no tags associated)\n            <URL>\n\n            # you can also use the keyword 'include' and identify a\n            # configuration location (like this file) which will be included\n            # as additional configuration entries when loaded.\n            include <ConfigURL>\n\n            # Assign tag contents to a group identifier\n            <Group(s)>=<Tag(s)>\n\n        "
        servers = list()
        configs = list()
        group_tags = {}
        preloaded = []
        asset = asset if isinstance(asset, AppriseAsset) else AppriseAsset()
        valid_line_re = re.compile('^\\s*(?P<line>([;#]+(?P<comment>.*))|(\\s*(?P<tags>[a-z0-9, \\t_-]+)\\s*=|=)?\\s*((?P<url>[a-z0-9]{1,12}://.*)|(?P<assign>[a-z0-9, \\t_-]+))|include\\s+(?P<config>.+))?\\s*$', re.I)
        try:
            content = re.split('\\r*\\n', content)
        except TypeError:
            ConfigBase.logger.error('Invalid Apprise TEXT based configuration specified.')
            return (list(), list())
        for (line, entry) in enumerate(content, start=1):
            result = valid_line_re.match(entry)
            if not result:
                ConfigBase.logger.error('Invalid Apprise TEXT configuration format found {} on line {}.'.format(entry, line))
                return (list(), list())
            (url, assign, config) = (result.group('url'), result.group('assign'), result.group('config'))
            if not (url or config or assign):
                continue
            if config:
                loggable_url = config if not asset.secure_logging else cwe312_url(config)
                ConfigBase.logger.debug('Include URL: {}'.format(loggable_url))
                configs.append(config.strip())
                continue
            loggable_url = url if not asset.secure_logging else cwe312_url(url)
            if assign:
                groups = set(parse_list(result.group('tags'), cast=str))
                if not groups:
                    ConfigBase.logger.warning('Unparseable tag assignment - no group(s) on line {}'.format(line))
                    continue
                tags = set(parse_list(assign, cast=str))
                if not tags:
                    ConfigBase.logger.warning('Unparseable tag assignment - no tag(s) to assign on line {}'.format(line))
                    continue
                for tag_group in groups:
                    if tag_group not in group_tags:
                        group_tags[tag_group] = set()
                    group_tags[tag_group] |= tags - set([tag_group])
                continue
            results = plugins.url_to_dict(url, secure_logging=asset.secure_logging)
            if results is None:
                ConfigBase.logger.warning('Unparseable URL {} on line {}.'.format(loggable_url, line))
                continue
            results['tag'] = set(parse_list(result.group('tags'), cast=str))
            results['asset'] = asset
            preloaded.append({'results': results, 'line': line, 'loggable_url': loggable_url})
        ConfigBase.__normalize_tag_groups(group_tags)
        for entry in preloaded:
            results = entry['results']
            for (group, tags) in group_tags.items():
                if next((True for tag in results['tag'] if tag in tags), False):
                    results['tag'].add(group)
            try:
                plugin = common.NOTIFY_SCHEMA_MAP[results['schema']](**results)
                ConfigBase.logger.debug('Loaded URL: %s', plugin.url(privacy=results['asset'].secure_logging))
            except Exception as e:
                ConfigBase.logger.warning('Could not load URL {} on line {}.'.format(entry['loggable_url'], entry['line']))
                ConfigBase.logger.debug('Loading Exception: %s' % str(e))
                continue
            servers.append(plugin)
        return (servers, configs)

    @staticmethod
    def config_parse_yaml(content, asset=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Parse the specified content as though it were a yaml file\n        specifically formatted for Apprise.\n\n        Return a tuple that looks like (servers, configs) where:\n          - servers contains a list of loaded notification plugins\n          - configs contains a list of additional configuration files\n            referenced.\n\n        You may optionally associate an asset with the notification.\n\n        '
        servers = list()
        configs = list()
        group_tags = {}
        preloaded = []
        try:
            result = yaml.load(content, Loader=yaml.SafeLoader)
        except (AttributeError, yaml.parser.ParserError, yaml.error.MarkedYAMLError) as e:
            ConfigBase.logger.error('Invalid Apprise YAML data specified.')
            ConfigBase.logger.debug('YAML Exception:{}{}'.format(os.linesep, e))
            return (list(), list())
        if not isinstance(result, dict):
            ConfigBase.logger.error('Invalid Apprise YAML based configuration specified.')
            return (list(), list())
        version = result.get('version', 1)
        if version != 1:
            ConfigBase.logger.error('Invalid Apprise YAML version specified {}.'.format(version))
            return (list(), list())
        asset = asset if isinstance(asset, AppriseAsset) else AppriseAsset()
        tokens = result.get('asset', None)
        if tokens and isinstance(tokens, dict):
            for (k, v) in tokens.items():
                if k.startswith('_') or k.endswith('_'):
                    ConfigBase.logger.warning('Ignored asset key "{}".'.format(k))
                    continue
                if not (hasattr(asset, k) and isinstance(getattr(asset, k), (bool, str))):
                    ConfigBase.logger.warning('Invalid asset key "{}".'.format(k))
                    continue
                if v is None:
                    v = ''
                if isinstance(v, (bool, str)) and isinstance(getattr(asset, k), bool):
                    setattr(asset, k, parse_bool(v))
                elif isinstance(v, str):
                    setattr(asset, k, v.strip())
                else:
                    ConfigBase.logger.warning('Invalid asset value to "{}".'.format(k))
                    continue
        global_tags = set()
        tags = result.get('tag', None)
        if tags and isinstance(tags, (list, tuple, str)):
            global_tags = set(parse_list(tags, cast=str))
        groups = result.get('groups', None)
        if isinstance(groups, dict):
            for (_groups, tags) in groups.items():
                for group in parse_list(_groups, cast=str):
                    if isinstance(tags, (list, tuple)):
                        _tags = set()
                        for e in tags:
                            if isinstance(e, dict):
                                _tags |= set(e.keys())
                            else:
                                _tags |= set(parse_list(e, cast=str))
                        tags = _tags
                    else:
                        tags = set(parse_list(tags, cast=str))
                    if group not in group_tags:
                        group_tags[group] = tags
                    else:
                        group_tags[group] |= tags
        elif isinstance(groups, (list, tuple)):
            for (no, entry) in enumerate(groups):
                if not isinstance(entry, dict):
                    ConfigBase.logger.warning('No assignment for group {}, entry #{}'.format(entry, no + 1))
                    continue
                for (_groups, tags) in entry.items():
                    for group in parse_list(_groups, cast=str):
                        if isinstance(tags, (list, tuple)):
                            _tags = set()
                            for e in tags:
                                if isinstance(e, dict):
                                    _tags |= set(e.keys())
                                else:
                                    _tags |= set(parse_list(e, cast=str))
                            tags = _tags
                        else:
                            tags = set(parse_list(tags, cast=str))
                        if group not in group_tags:
                            group_tags[group] = tags
                        else:
                            group_tags[group] |= tags
        includes = result.get('include', None)
        if isinstance(includes, str):
            includes = parse_urls(includes)
        elif not isinstance(includes, (list, tuple)):
            includes = list()
        for (no, url) in enumerate(includes):
            if isinstance(url, str):
                configs.extend(parse_urls(url))
            elif isinstance(url, dict):
                configs.extend((u for u in url.keys()))
        urls = result.get('urls', None)
        if not isinstance(urls, (list, tuple)):
            urls = list()
        for (no, url) in enumerate(urls):
            results = list()
            loggable_url = url if not asset.secure_logging else cwe312_url(url)
            if isinstance(url, str):
                schema = GET_SCHEMA_RE.match(url)
                if schema is None:
                    ConfigBase.logger.warning('Invalid URL {}, entry #{}'.format(loggable_url, no + 1))
                    continue
                _results = plugins.url_to_dict(url, secure_logging=asset.secure_logging)
                if _results is None:
                    ConfigBase.logger.warning('Unparseable URL {}, entry #{}'.format(loggable_url, no + 1))
                    continue
                results.append(_results)
            elif isinstance(url, dict):
                it = iter(url.items())
                _url = None
                schema = None
                for (key, tokens) in it:
                    _schema = GET_SCHEMA_RE.match(key)
                    if _schema is None:
                        ConfigBase.logger.warning('Ignored entry {} found under urls, entry #{}'.format(key, no + 1))
                        continue
                    schema = _schema.group('schema').lower()
                    _url = key
                if _url is None:
                    ConfigBase.logger.warning('Unsupported URL, entry #{}'.format(no + 1))
                    continue
                _results = plugins.url_to_dict(_url, secure_logging=asset.secure_logging)
                if _results is None:
                    _results = {'schema': schema}
                if isinstance(tokens, (list, tuple, set)):
                    for entries in tokens:
                        r = _results.copy()
                        if isinstance(entries, dict):
                            (_url, tokens) = next(iter(url.items()))
                            if 'schema' in entries:
                                del entries['schema']
                            if schema in common.NOTIFY_SCHEMA_MAP:
                                entries = ConfigBase._special_token_handler(schema, entries)
                            r.update(entries)
                            results.append(r)
                elif isinstance(tokens, dict):
                    if schema in common.NOTIFY_SCHEMA_MAP:
                        tokens = ConfigBase._special_token_handler(schema, tokens)
                    r = _results.copy()
                    r.update(tokens)
                    results.append(r)
                else:
                    results.append(_results)
            else:
                ConfigBase.logger.warning('Unsupported Apprise YAML entry #{}'.format(no + 1))
                continue
            entry = 0
            while len(results):
                entry += 1
                _results = results.pop(0)
                if _results['schema'] not in common.NOTIFY_SCHEMA_MAP:
                    ConfigBase.logger.warning('An invalid Apprise schema ({}) in YAML configuration entry #{}, item #{}'.format(_results['schema'], no + 1, entry))
                    continue
                if 'tag' in _results:
                    _results['tag'] = set(parse_list(_results['tag'], cast=str)) | global_tags
                else:
                    _results['tag'] = global_tags
                for key in list(_results.keys()):
                    match = VALID_TOKEN.match(key)
                    if not match:
                        ConfigBase.logger.warning('Ignoring invalid token ({}) found in YAML configuration entry #{}, item #{}'.format(key, no + 1, entry))
                        del _results[key]
                ConfigBase.logger.trace('URL #{}: {} unpacked as:{}{}'.format(no + 1, url, os.linesep, os.linesep.join(['{}="{}"'.format(k, a) for (k, a) in _results.items()])))
                _results['asset'] = asset
                preloaded.append({'results': _results, 'entry': no + 1, 'item': entry})
        ConfigBase.__normalize_tag_groups(group_tags)
        for entry in preloaded:
            results = entry['results']
            for (group, tags) in group_tags.items():
                if next((True for tag in results['tag'] if tag in tags), False):
                    results['tag'].add(group)
            try:
                plugin = common.NOTIFY_SCHEMA_MAP[results['schema']](**results)
                ConfigBase.logger.debug('Loaded URL: %s', plugin.url(privacy=results['asset'].secure_logging))
            except Exception as e:
                ConfigBase.logger.warning('Could not load Apprise YAML configuration entry #{}, item #{}'.format(entry['entry'], entry['item']))
                ConfigBase.logger.debug('Loading Exception: %s' % str(e))
                continue
            servers.append(plugin)
        return (servers, configs)

    def pop(self, index=-1):
        if False:
            return 10
        '\n        Removes an indexed Notification Service from the stack and returns it.\n\n        By default, the last element of the list is removed.\n        '
        if not isinstance(self._cached_servers, list):
            self.servers()
        return self._cached_servers.pop(index)

    @staticmethod
    def _special_token_handler(schema, tokens):
        if False:
            i = 10
            return i + 15
        "\n        This function takes a list of tokens and updates them to no longer\n        include any special tokens such as +,-, and :\n\n        - schema must be a valid schema of a supported plugin type\n        - tokens must be a dictionary containing the yaml entries parsed.\n\n        The idea here is we can post process a set of tokens provided in\n        a YAML file where the user provided some of the special keywords.\n\n        We effectivley look up what these keywords map to their appropriate\n        value they're expected\n        "
        tokens = tokens.copy()
        for (kw, meta) in common.NOTIFY_SCHEMA_MAP[schema].template_kwargs.items():
            prefix = meta.get('prefix', '+')
            matches = {k[1:]: str(v) for (k, v) in tokens.items() if k.startswith(prefix)}
            if not matches:
                continue
            if not isinstance(tokens.get(kw), dict):
                tokens[kw] = dict()
            tokens = {k: v for (k, v) in tokens.items() if not k.startswith(prefix)}
            tokens[kw].update(matches)
        class_templates = plugins.details(common.NOTIFY_SCHEMA_MAP[schema])
        for key in list(tokens.keys()):
            if key not in class_templates['args']:
                continue
            map_to = class_templates['args'][key].get('alias_of', class_templates['args'][key].get('map_to', ''))
            if map_to == key:
                continue
            if map_to in class_templates['tokens']:
                meta = class_templates['tokens'][map_to]
            else:
                meta = class_templates['args'].get(map_to, class_templates['args'][key])
            value = tokens[key]
            del tokens[key]
            is_list = re.search('^list:.*', meta.get('type'), re.IGNORECASE)
            if map_to not in tokens:
                tokens[map_to] = [] if is_list else meta.get('default')
            elif is_list and (not isinstance(tokens.get(map_to), list)):
                tokens[map_to] = [tokens[map_to]]
            if re.search('^(choice:)?string', meta.get('type'), re.IGNORECASE) and (not isinstance(value, str)):
                value = str(value)
            abs_map = meta.get('map_to', map_to)
            if isinstance(tokens.get(map_to), list):
                tokens[abs_map].append(value)
            else:
                tokens[abs_map] = value
        return tokens

    def __getitem__(self, index):
        if False:
            while True:
                i = 10
        '\n        Returns the indexed server entry associated with the loaded\n        notification servers\n        '
        if not isinstance(self._cached_servers, list):
            self.servers()
        return self._cached_servers[index]

    def __iter__(self):
        if False:
            print('Hello World!')
        '\n        Returns an iterator to our server list\n        '
        if not isinstance(self._cached_servers, list):
            self.servers()
        return iter(self._cached_servers)

    def __len__(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns the total number of servers loaded\n        '
        if not isinstance(self._cached_servers, list):
            self.servers()
        return len(self._cached_servers)

    def __bool__(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Allows the Apprise object to be wrapped in an 'if statement'.\n        True is returned if our content was downloaded correctly.\n        "
        if not isinstance(self._cached_servers, list):
            self.servers()
        return True if self._cached_servers else False