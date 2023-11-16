from .NotifyBase import NotifyBase
from ..URLBase import PrivacyMode
from ..common import NotifyImageSize
from ..common import NotifyType
from ..utils import parse_bool
from ..AppriseLocale import gettext_lazy as _
NOTIFY_GROWL_SUPPORT_ENABLED = False
try:
    import gntp.notifier
    NOTIFY_GROWL_SUPPORT_ENABLED = True
except ImportError:
    pass

class GrowlPriority:
    LOW = -2
    MODERATE = -1
    NORMAL = 0
    HIGH = 1
    EMERGENCY = 2
GROWL_PRIORITIES = {GrowlPriority.LOW: 'low', GrowlPriority.MODERATE: 'moderate', GrowlPriority.NORMAL: 'normal', GrowlPriority.HIGH: 'high', GrowlPriority.EMERGENCY: 'emergency'}
GROWL_PRIORITY_MAP = {'l': GrowlPriority.LOW, 'm': GrowlPriority.MODERATE, 'n': GrowlPriority.NORMAL, 'h': GrowlPriority.HIGH, 'e': GrowlPriority.EMERGENCY, '-2': GrowlPriority.LOW, '-1': GrowlPriority.MODERATE, '0': GrowlPriority.NORMAL, '1': GrowlPriority.HIGH, '2': GrowlPriority.EMERGENCY}

class NotifyGrowl(NotifyBase):
    """
    A wrapper to Growl Notifications

    """
    enabled = NOTIFY_GROWL_SUPPORT_ENABLED
    requirements = {'packages_required': 'gntp'}
    service_name = 'Growl'
    service_url = 'http://growl.info/'
    protocol = 'growl'
    setup_url = 'https://github.com/caronc/apprise/wiki/Notify_growl'
    image_size = NotifyImageSize.XY_72
    request_rate_per_sec = 0
    body_max_line_count = 2
    default_port = 23053
    growl_notification_type = 'New Messages'
    templates = ('{schema}://{host}', '{schema}://{host}:{port}', '{schema}://{password}@{host}', '{schema}://{password}@{host}:{port}')
    template_tokens = dict(NotifyBase.template_tokens, **{'host': {'name': _('Hostname'), 'type': 'string', 'required': True}, 'port': {'name': _('Port'), 'type': 'int', 'min': 1, 'max': 65535}, 'password': {'name': _('Password'), 'type': 'string', 'private': True}})
    template_args = dict(NotifyBase.template_args, **{'priority': {'name': _('Priority'), 'type': 'choice:int', 'values': GROWL_PRIORITIES, 'default': GrowlPriority.NORMAL}, 'version': {'name': _('Version'), 'type': 'choice:int', 'values': (1, 2), 'default': 2}, 'image': {'name': _('Include Image'), 'type': 'bool', 'default': True, 'map_to': 'include_image'}, 'sticky': {'name': _('Sticky'), 'type': 'bool', 'default': True, 'map_to': 'sticky'}})

    def __init__(self, priority=None, version=2, include_image=True, sticky=False, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Initialize Growl Object\n        '
        super().__init__(**kwargs)
        if not self.port:
            self.port = self.default_port
        self.priority = NotifyGrowl.template_args['priority']['default'] if not priority else next((v for (k, v) in GROWL_PRIORITY_MAP.items() if str(priority).lower().startswith(k)), NotifyGrowl.template_args['priority']['default'])
        self.growl = None
        self.sticky = sticky
        self.version = version
        self.include_image = include_image
        return

    def register(self):
        if False:
            i = 10
            return i + 15
        '\n        Registers with the Growl server\n        '
        payload = {'applicationName': self.app_id, 'notifications': [self.growl_notification_type], 'defaultNotifications': [self.growl_notification_type], 'hostname': self.host, 'port': self.port}
        if self.password is not None:
            payload['password'] = self.password
        self.logger.debug('Growl Registration Payload: %s' % str(payload))
        self.growl = gntp.notifier.GrowlNotifier(**payload)
        try:
            self.growl.register()
        except gntp.errors.NetworkError:
            msg = 'A network error error occurred registering with Growl at {}.'.format(self.host)
            self.logger.warning(msg)
            return False
        except gntp.errors.ParseError:
            msg = 'A parsing error error occurred registering with Growl at {}.'.format(self.host)
            self.logger.warning(msg)
            return False
        except gntp.errors.AuthError:
            msg = 'An authentication error error occurred registering with Growl at {}.'.format(self.host)
            self.logger.warning(msg)
            return False
        except gntp.errors.UnsupportedError:
            msg = 'An unsupported error occurred registering with Growl at {}.'.format(self.host)
            self.logger.warning(msg)
            return False
        self.logger.debug('Growl server registration completed successfully.')
        return True

    def send(self, body, title='', notify_type=NotifyType.INFO, **kwargs):
        if False:
            return 10
        '\n        Perform Growl Notification\n        '
        if not self.growl and (not self.register()):
            return False
        icon = None
        if self.version >= 2:
            icon = None if not self.include_image else self.image_url(notify_type)
        else:
            icon = None if not self.include_image else self.image_raw(notify_type)
        payload = {'noteType': self.growl_notification_type, 'title': title, 'description': body, 'icon': icon is not None, 'sticky': self.sticky, 'priority': self.priority}
        self.logger.debug('Growl Payload: %s' % str(payload))
        payload['icon'] = icon
        self.throttle()
        try:
            response = self.growl.notify(**payload)
            if not isinstance(response, bool):
                self.logger.warning('Growl notification failed to send with response: %s' % str(response))
            else:
                self.logger.info('Sent Growl notification.')
        except gntp.errors.BaseError as e:
            self.logger.warning('A Connection error occurred sending Growl notification to %s.' % self.host)
            self.logger.debug('Growl Exception: %s' % str(e))
            return False
        return True

    def url(self, privacy=False, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Returns the URL built dynamically based on specified arguments.\n        '
        params = {'image': 'yes' if self.include_image else 'no', 'sticky': 'yes' if self.sticky else 'no', 'priority': GROWL_PRIORITIES[self.template_args['priority']['default']] if self.priority not in GROWL_PRIORITIES else GROWL_PRIORITIES[self.priority], 'version': self.version}
        params.update(self.url_parameters(*args, privacy=privacy, **kwargs))
        auth = ''
        if self.user:
            auth = '{password}@'.format(password=self.pprint(self.user, privacy, mode=PrivacyMode.Secret, safe=''))
        return '{schema}://{auth}{hostname}{port}/?{params}'.format(schema=self.secure_protocol if self.secure else self.protocol, auth=auth, hostname=self.host, port='' if self.port is None or self.port == self.default_port else ':{}'.format(self.port), params=NotifyGrowl.urlencode(params))

    @staticmethod
    def parse_url(url):
        if False:
            for i in range(10):
                print('nop')
        '\n        Parses the URL and returns enough arguments that can allow\n        us to re-instantiate this object.\n\n        '
        results = NotifyBase.parse_url(url)
        if not results:
            return results
        version = None
        if 'version' in results['qsd'] and len(results['qsd']['version']):
            try:
                version = int(NotifyGrowl.unquote(results['qsd']['version']).strip().split('.')[0])
            except (AttributeError, IndexError, TypeError, ValueError):
                NotifyGrowl.logger.warning('An invalid Growl version of "%s" was specified and will be ignored.' % results['qsd']['version'])
                pass
        if 'priority' in results['qsd'] and len(results['qsd']['priority']):
            results['priority'] = NotifyGrowl.unquote(results['qsd']['priority'])
        if results.get('password', None) is None:
            results['password'] = results.get('user', None)
        results['include_image'] = parse_bool(results['qsd'].get('image', NotifyGrowl.template_args['image']['default']))
        results['sticky'] = parse_bool(results['qsd'].get('sticky', NotifyGrowl.template_args['sticky']['default']))
        if version:
            results['version'] = version
        return results