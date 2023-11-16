import asyncio
import concurrent.futures as cf
import os
from itertools import chain
from . import common
from .conversion import convert_between
from .utils import is_exclusive_match
from .utils import parse_list
from .utils import parse_urls
from .utils import cwe312_url
from .logger import logger
from .AppriseAsset import AppriseAsset
from .AppriseConfig import AppriseConfig
from .AppriseAttachment import AppriseAttachment
from .AppriseLocale import AppriseLocale
from .config.ConfigBase import ConfigBase
from .plugins.NotifyBase import NotifyBase
from . import plugins
from . import __version__

class Apprise:
    """
    Our Notification Manager

    """

    def __init__(self, servers=None, asset=None, location=None, debug=False):
        if False:
            return 10
        '\n        Loads a set of server urls while applying the Asset() module to each\n        if specified.\n\n        If no asset is provided, then the default asset is used.\n\n        Optionally specify a global ContentLocation for a more strict means\n        of handling Attachments.\n        '
        self.servers = list()
        self.asset = asset if isinstance(asset, AppriseAsset) else AppriseAsset()
        if servers:
            self.add(servers)
        self.locale = AppriseLocale()
        self.debug = debug
        self.location = location

    @staticmethod
    def instantiate(url, asset=None, tag=None, suppress_exceptions=True):
        if False:
            for i in range(10):
                print('nop')
        "\n        Returns the instance of a instantiated plugin based on the provided\n        Server URL.  If the url fails to be parsed, then None is returned.\n\n        The specified url can be either a string (the URL itself) or a\n        dictionary containing all of the components needed to istantiate\n        the notification service.  If identifying a dictionary, at the bare\n        minimum, one must specify the schema.\n\n        An example of a url dictionary object might look like:\n          {\n            schema: 'mailto',\n            host: 'google.com',\n            user: 'myuser',\n            password: 'mypassword',\n          }\n\n        Alternatively the string is much easier to specify:\n          mailto://user:mypassword@google.com\n\n        The dictionary works well for people who are calling details() to\n        extract the components they need to build the URL manually.\n        "
        results = None
        asset = asset if isinstance(asset, AppriseAsset) else AppriseAsset()
        if isinstance(url, str):
            results = plugins.url_to_dict(url, secure_logging=asset.secure_logging)
            if results is None:
                return None
        elif isinstance(url, dict):
            results = url
            if results.get('schema') not in common.NOTIFY_SCHEMA_MAP:
                logger.error('Dictionary does not include a "schema" entry.')
                logger.trace('Invalid dictionary unpacked as:{}{}'.format(os.linesep, os.linesep.join(['{}="{}"'.format(k, v) for (k, v) in results.items()])))
                return None
            logger.trace('Dictionary unpacked as:{}{}'.format(os.linesep, os.linesep.join(['{}="{}"'.format(k, v) for (k, v) in results.items()])))
        else:
            logger.error('An invalid URL type (%s) was specified for instantiation', type(url))
            return None
        if not common.NOTIFY_SCHEMA_MAP[results['schema']].enabled:
            logger.error('%s:// is disabled on this system.', results['schema'])
            return None
        results['tag'] = set(parse_list(tag))
        results['asset'] = asset
        if suppress_exceptions:
            try:
                plugin = common.NOTIFY_SCHEMA_MAP[results['schema']](**results)
                logger.debug('Loaded {} URL: {}'.format(common.NOTIFY_SCHEMA_MAP[results['schema']].service_name, plugin.url(privacy=asset.secure_logging)))
            except Exception:
                loggable_url = url if not asset.secure_logging else cwe312_url(url)
                logger.error('Could not load {} URL: {}'.format(common.NOTIFY_SCHEMA_MAP[results['schema']].service_name, loggable_url))
                return None
        else:
            plugin = common.NOTIFY_SCHEMA_MAP[results['schema']](**results)
        if not plugin.enabled:
            logger.error('%s:// has become disabled on this system.', results['schema'])
            return None
        return plugin

    def add(self, servers, asset=None, tag=None):
        if False:
            i = 10
            return i + 15
        '\n        Adds one or more server URLs into our list.\n\n        You can override the global asset if you wish by including it with the\n        server(s) that you add.\n\n        The tag allows you to associate 1 or more tag values to the server(s)\n        being added. tagging a service allows you to exclusively access them\n        when calling the notify() function.\n        '
        return_status = True
        if asset is None:
            asset = self.asset
        if isinstance(servers, str):
            servers = parse_urls(servers)
            if len(servers) == 0:
                return False
        elif isinstance(servers, dict):
            servers = [servers]
        elif isinstance(servers, (ConfigBase, NotifyBase, AppriseConfig)):
            self.servers.append(servers)
            return True
        elif not isinstance(servers, (tuple, set, list)):
            logger.error('An invalid notification (type={}) was specified.'.format(type(servers)))
            return False
        for _server in servers:
            if isinstance(_server, (ConfigBase, NotifyBase, AppriseConfig)):
                self.servers.append(_server)
                continue
            elif not isinstance(_server, (str, dict)):
                logger.error('An invalid notification (type={}) was specified.'.format(type(_server)))
                return_status = False
                continue
            instance = Apprise.instantiate(_server, asset=asset, tag=tag)
            if not isinstance(instance, NotifyBase):
                return_status = False
                continue
            self.servers.append(instance)
        return return_status

    def clear(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Empties our server list\n\n        '
        self.servers[:] = []

    def find(self, tag=common.MATCH_ALL_TAG, match_always=True):
        if False:
            print('Hello World!')
        '\n        Returns a list of all servers matching against the tag specified.\n\n        '
        match_always = common.MATCH_ALWAYS_TAG if match_always else None
        for entry in self.servers:
            if isinstance(entry, (ConfigBase, AppriseConfig)):
                servers = entry.servers()
            else:
                servers = [entry]
            for server in servers:
                if is_exclusive_match(logic=tag, data=server.tags, match_all=common.MATCH_ALL_TAG, match_always=match_always):
                    yield server
        return

    def notify(self, body, title='', notify_type=common.NotifyType.INFO, body_format=None, tag=common.MATCH_ALL_TAG, match_always=True, attach=None, interpret_escapes=None):
        if False:
            i = 10
            return i + 15
        '\n        Send a notification to all the plugins previously loaded.\n\n        If the body_format specified is NotifyFormat.MARKDOWN, it will\n        be converted to HTML if the Notification type expects this.\n\n        if the tag is specified (either a string or a set/list/tuple\n        of strings), then only the notifications flagged with that\n        tagged value are notified.  By default, all added services\n        are notified (tag=MATCH_ALL_TAG)\n\n        This function returns True if all notifications were successfully\n        sent, False if even just one of them fails, and None if no\n        notifications were sent at all as a result of tag filtering and/or\n        simply having empty configuration files that were read.\n\n        Attach can contain a list of attachment URLs.  attach can also be\n        represented by an AttachBase() (or list of) object(s). This\n        identifies the products you wish to notify\n\n        Set interpret_escapes to True if you want to pre-escape a string\n        such as turning a \n into an actual new line, etc.\n        '
        try:
            (sequential_calls, parallel_calls) = self._create_notify_calls(body, title, notify_type=notify_type, body_format=body_format, tag=tag, match_always=match_always, attach=attach, interpret_escapes=interpret_escapes)
        except TypeError:
            return False
        if not sequential_calls and (not parallel_calls):
            return None
        sequential_result = Apprise._notify_sequential(*sequential_calls)
        parallel_result = Apprise._notify_parallel_threadpool(*parallel_calls)
        return sequential_result and parallel_result

    async def async_notify(self, *args, **kwargs):
        """
        Send a notification to all the plugins previously loaded, for
        asynchronous callers.

        The arguments are identical to those of Apprise.notify().

        """
        try:
            (sequential_calls, parallel_calls) = self._create_notify_calls(*args, **kwargs)
        except TypeError:
            return False
        if not sequential_calls and (not parallel_calls):
            return None
        sequential_result = Apprise._notify_sequential(*sequential_calls)
        parallel_result = await Apprise._notify_parallel_asyncio(*parallel_calls)
        return sequential_result and parallel_result

    def _create_notify_calls(self, *args, **kwargs):
        if False:
            print('Hello World!')
        '\n        Creates notifications for all the plugins loaded.\n\n        Returns a list of (server, notify() kwargs) tuples for plugins with\n        parallelism disabled and another list for plugins with parallelism\n        enabled.\n        '
        all_calls = list(self._create_notify_gen(*args, **kwargs))
        (sequential, parallel) = ([], [])
        for (server, notify_kwargs) in all_calls:
            if server.asset.async_mode:
                parallel.append((server, notify_kwargs))
            else:
                sequential.append((server, notify_kwargs))
        return (sequential, parallel)

    def _create_notify_gen(self, body, title='', notify_type=common.NotifyType.INFO, body_format=None, tag=common.MATCH_ALL_TAG, match_always=True, attach=None, interpret_escapes=None):
        if False:
            print('Hello World!')
        '\n        Internal generator function for _create_notify_calls().\n        '
        if len(self) == 0:
            msg = 'There are no service(s) to notify'
            logger.error(msg)
            raise TypeError(msg)
        if not (title or body or attach):
            msg = 'No message content specified to deliver'
            logger.error(msg)
            raise TypeError(msg)
        try:
            if title and isinstance(title, bytes):
                title = title.decode(self.asset.encoding)
            if body and isinstance(body, bytes):
                body = body.decode(self.asset.encoding)
        except UnicodeDecodeError:
            msg = 'The content passed into Apprise was not of encoding type: {}'.format(self.asset.encoding)
            logger.error(msg)
            raise TypeError(msg)
        conversion_body_map = dict()
        conversion_title_map = dict()
        if attach is not None and (not isinstance(attach, AppriseAttachment)):
            attach = AppriseAttachment(attach, asset=self.asset, location=self.location)
        body_format = self.asset.body_format if body_format is None else body_format
        interpret_escapes = self.asset.interpret_escapes if interpret_escapes is None else interpret_escapes
        for server in self.find(tag, match_always=match_always):
            key = server.notify_format if server.title_maxlen > 0 else f'_{server.notify_format}'
            if key not in conversion_title_map:
                conversion_title_map[key] = '' if not title else title
                if conversion_title_map[key] and server.title_maxlen <= 0:
                    conversion_title_map[key] = convert_between(body_format, server.notify_format, content=conversion_title_map[key])
                conversion_body_map[key] = convert_between(body_format, server.notify_format, content=body)
                if interpret_escapes:
                    try:
                        conversion_body_map[key] = conversion_body_map[key].encode('ascii', 'backslashreplace').decode('unicode-escape')
                        conversion_title_map[key] = conversion_title_map[key].encode('ascii', 'backslashreplace').decode('unicode-escape')
                    except AttributeError:
                        msg = 'Failed to escape message body'
                        logger.error(msg)
                        raise TypeError(msg)
            kwargs = dict(body=conversion_body_map[key], title=conversion_title_map[key], notify_type=notify_type, attach=attach, body_format=body_format)
            yield (server, kwargs)

    @staticmethod
    def _notify_sequential(*servers_kwargs):
        if False:
            return 10
        '\n        Process a list of notify() calls sequentially and synchronously.\n        '
        success = True
        for (server, kwargs) in servers_kwargs:
            try:
                result = server.notify(**kwargs)
                success = success and result
            except TypeError:
                success = False
            except Exception:
                logger.exception('Unhandled Notification Exception')
                success = False
        return success

    @staticmethod
    def _notify_parallel_threadpool(*servers_kwargs):
        if False:
            print('Hello World!')
        '\n        Process a list of notify() calls in parallel and synchronously.\n        '
        n_calls = len(servers_kwargs)
        if n_calls == 0:
            return True
        if n_calls == 1:
            return Apprise._notify_sequential(servers_kwargs[0])
        logger.info('Notifying %d service(s) with threads.', len(servers_kwargs))
        with cf.ThreadPoolExecutor() as executor:
            success = True
            futures = [executor.submit(server.notify, **kwargs) for (server, kwargs) in servers_kwargs]
            for future in cf.as_completed(futures):
                try:
                    result = future.result()
                    success = success and result
                except TypeError:
                    success = False
                except Exception:
                    logger.exception('Unhandled Notification Exception')
                    success = False
            return success

    @staticmethod
    async def _notify_parallel_asyncio(*servers_kwargs):
        """
        Process a list of async_notify() calls in parallel and asynchronously.
        """
        n_calls = len(servers_kwargs)
        if n_calls == 0:
            return True
        logger.info('Notifying %d service(s) asynchronously.', len(servers_kwargs))

        async def do_call(server, kwargs):
            return await server.async_notify(**kwargs)
        cors = (do_call(server, kwargs) for (server, kwargs) in servers_kwargs)
        results = await asyncio.gather(*cors, return_exceptions=True)
        if any((isinstance(status, Exception) and (not isinstance(status, TypeError)) for status in results)):
            logger.exception('Unhandled Notification Exception')
            return False
        if any((isinstance(status, TypeError) for status in results)):
            return False
        return all(results)

    def details(self, lang=None, show_requirements=False, show_disabled=False):
        if False:
            i = 10
            return i + 15
        '\n        Returns the details associated with the Apprise object\n\n        '
        response = {'version': __version__, 'schemas': [], 'asset': self.asset.details()}
        for plugin in set(common.NOTIFY_SCHEMA_MAP.values()):
            content = {'service_name': getattr(plugin, 'service_name', None), 'service_url': getattr(plugin, 'service_url', None), 'setup_url': getattr(plugin, 'setup_url', None), 'details': None, 'attachment_support': getattr(plugin, 'attachment_support', False), 'category': getattr(plugin, 'category', None)}
            enabled = getattr(plugin, 'enabled', True)
            if not show_disabled and (not enabled):
                continue
            elif show_disabled:
                content['enabled'] = enabled
            protocols = getattr(plugin, 'protocol', None)
            if isinstance(protocols, str):
                protocols = (protocols,)
            secure_protocols = getattr(plugin, 'secure_protocol', None)
            if isinstance(secure_protocols, str):
                secure_protocols = (secure_protocols,)
            content.update({'protocols': protocols, 'secure_protocols': secure_protocols})
            if not lang:
                content['details'] = plugins.details(plugin)
                if show_requirements:
                    content['requirements'] = plugins.requirements(plugin)
            else:
                with self.locale.lang_at(lang):
                    content['details'] = plugins.details(plugin)
                    if show_requirements:
                        content['requirements'] = plugins.requirements(plugin)
            response['schemas'].append(content)
        return response

    def urls(self, privacy=False):
        if False:
            while True:
                i = 10
        '\n        Returns all of the loaded URLs defined in this apprise object.\n        '
        return [x.url(privacy=privacy) for x in self.servers]

    def pop(self, index):
        if False:
            return 10
        '\n        Removes an indexed Notification Service from the stack and returns it.\n\n        The thing is we can never pop AppriseConfig() entries, only what was\n        loaded within them. So pop needs to carefully iterate over our list\n        and only track actual entries.\n        '
        prev_offset = -1
        offset = prev_offset
        for (idx, s) in enumerate(self.servers):
            if isinstance(s, (ConfigBase, AppriseConfig)):
                servers = s.servers()
                if len(servers) > 0:
                    offset = prev_offset + len(servers)
                    if offset >= index:
                        fn = s.pop if isinstance(s, ConfigBase) else s.server_pop
                        return fn(index if prev_offset == -1 else index - prev_offset - 1)
            else:
                offset = prev_offset + 1
                if offset == index:
                    return self.servers.pop(idx)
            prev_offset = offset
        raise IndexError('list index out of range')

    def __getitem__(self, index):
        if False:
            return 10
        '\n        Returns the indexed server entry of a loaded notification server\n        '
        prev_offset = -1
        offset = prev_offset
        for (idx, s) in enumerate(self.servers):
            if isinstance(s, (ConfigBase, AppriseConfig)):
                servers = s.servers()
                if len(servers) > 0:
                    offset = prev_offset + len(servers)
                    if offset >= index:
                        return servers[index if prev_offset == -1 else index - prev_offset - 1]
            else:
                offset = prev_offset + 1
                if offset == index:
                    return self.servers[idx]
            prev_offset = offset
        raise IndexError('list index out of range')

    def __getstate__(self):
        if False:
            while True:
                i = 10
        '\n        Pickle Support dumps()\n        '
        attributes = {'asset': self.asset, 'urls': [{'url': server.url(privacy=False), 'tag': server.tags if server.tags else None, 'asset': server.asset} for server in self.servers], 'locale': self.locale, 'debug': self.debug, 'location': self.location}
        return attributes

    def __setstate__(self, state):
        if False:
            for i in range(10):
                print('nop')
        '\n        Pickle Support loads()\n        '
        self.servers = list()
        self.asset = state['asset']
        self.locale = state['locale']
        self.location = state['location']
        for entry in state['urls']:
            self.add(entry['url'], asset=entry['asset'], tag=entry['tag'])

    def __bool__(self):
        if False:
            i = 10
            return i + 15
        "\n        Allows the Apprise object to be wrapped in an 'if statement'.\n        True is returned if at least one service has been loaded.\n        "
        return len(self) > 0

    def __iter__(self):
        if False:
            print('Hello World!')
        '\n        Returns an iterator to each of our servers loaded. This includes those\n        found inside configuration.\n        '
        return chain(*[[s] if not isinstance(s, (ConfigBase, AppriseConfig)) else iter(s.servers()) for s in self.servers])

    def __len__(self):
        if False:
            print('Hello World!')
        '\n        Returns the number of servers loaded; this includes those found within\n        loaded configuration. This funtion nnever actually counts the\n        Config entry themselves (if they exist), only what they contain.\n        '
        return sum([1 if not isinstance(s, (ConfigBase, AppriseConfig)) else len(s.servers()) for s in self.servers])