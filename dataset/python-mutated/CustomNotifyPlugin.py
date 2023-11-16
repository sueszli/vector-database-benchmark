from ..plugins.NotifyBase import NotifyBase
from ..utils import URL_DETAILS_RE
from ..utils import parse_url
from ..utils import url_assembly
from ..utils import dict_full_update
from .. import common
from ..logger import logger
import inspect

class CustomNotifyPlugin(NotifyBase):
    """
    Apprise Custom Plugin Hook

    This gets initialized based on @notify decorator definitions

    """
    service_url = 'https://github.com/caronc/apprise/wiki/Custom_Notification'
    category = 'custom'
    templates = ('{schema}://',)

    @staticmethod
    def parse_url(url):
        if False:
            for i in range(10):
                print('nop')
        '\n        Parses the URL and returns arguments retrieved\n\n        '
        return parse_url(url, verify_host=False, simple=True)

    def url(self, privacy=False, *args, **kwargs):
        if False:
            return 10
        '\n        General URL assembly\n        '
        return '{schema}://'.format(schema=self.secure_protocol)

    @staticmethod
    def instantiate_plugin(url, send_func, name=None):
        if False:
            while True:
                i = 10
        '\n        The function used to add a new notification plugin based on the schema\n        parsed from the provided URL into our supported matrix structure.\n        '
        if not isinstance(url, str):
            msg = 'An invalid custom notify url/schema ({}) provided in function {}.'.format(url, send_func.__name__)
            logger.warning(msg)
            return None
        re_match = URL_DETAILS_RE.match(url)
        if not re_match:
            msg = 'An invalid custom notify url/schema ({}) provided in function {}.'.format(url, send_func.__name__)
            logger.warning(msg)
            return None
        plugin_name = re_match.group('schema').lower()
        if not re_match.group('base'):
            url = '{}://'.format(plugin_name)
        base_args = parse_url(url, default_schema=plugin_name, verify_host=False, simple=True)
        if plugin_name in common.NOTIFY_SCHEMA_MAP:
            msg = 'The schema ({}) is already defined and could not be loaded from custom notify function {}.'.format(url, send_func.__name__)
            logger.warning(msg)
            return None

        class CustomNotifyPluginWrapper(CustomNotifyPlugin):
            service_name = name if isinstance(name, str) and name else 'Custom - {}'.format(plugin_name)
            secure_protocol = plugin_name
            requirements = {'details': 'Source: {}'.format(inspect.getfile(send_func))}
            __send = staticmethod(send_func)
            _base_args = base_args

            def __init__(self, **kwargs):
                if False:
                    i = 10
                    return i + 15
                '\n                Our initialization\n\n                '
                super().__init__(**kwargs)
                self._default_args = {}
                dict_full_update(self._default_args, self._base_args)
                dict_full_update(self._default_args, kwargs)
                self._default_args['url'] = url_assembly(**self._default_args)

            def send(self, body, title='', notify_type=common.NotifyType.INFO, *args, **kwargs):
                if False:
                    for i in range(10):
                        print('nop')
                '\n                Our send() call which triggers our hook\n                '
                response = False
                try:
                    result = self.__send(body, title, notify_type, *args, meta=self._default_args, **kwargs)
                    if result is None:
                        response = True
                    else:
                        response = True if result else False
                except Exception as e:
                    self.logger.warning('An exception occured sending a %s notification.', common.NOTIFY_SCHEMA_MAP[self.secure_protocol].service_name)
                    self.logger.debug('%s Exception: %s', common.NOTIFY_SCHEMA_MAP[self.secure_protocol], str(e))
                    return False
                if response:
                    self.logger.info('Sent %s notification.', common.NOTIFY_SCHEMA_MAP[self.secure_protocol].service_name)
                else:
                    self.logger.warning('Failed to send %s notification.', common.NOTIFY_SCHEMA_MAP[self.secure_protocol].service_name)
                return response
        common.NOTIFY_SCHEMA_MAP[plugin_name] = CustomNotifyPluginWrapper
        module_pyname = str(send_func.__module__)
        if module_pyname not in common.NOTIFY_CUSTOM_MODULE_MAP:
            common.NOTIFY_CUSTOM_MODULE_MAP[module_pyname] = {'path': inspect.getfile(send_func), 'notify': {}}
        common.NOTIFY_CUSTOM_MODULE_MAP[module_pyname]['notify'][plugin_name] = {'name': CustomNotifyPluginWrapper.service_name, 'fn_name': send_func.__name__, 'url': url, 'plugin': CustomNotifyPluginWrapper}
        return common.NOTIFY_SCHEMA_MAP[plugin_name]