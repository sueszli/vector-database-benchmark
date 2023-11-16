import re
import requests
from .NotifyBase import NotifyBase
from ..common import NotifyImageSize
from ..common import NotifyType
from ..utils import parse_list
from ..utils import parse_bool
from ..utils import validate_regex
from ..AppriseLocale import gettext_lazy as _
JOIN_HTTP_ERROR_MAP = {401: 'Unauthorized - Invalid Token.'}
IS_DEVICE_RE = re.compile('^[a-z0-9]{32}$', re.I)
IS_GROUP_RE = re.compile('(group\\.)?(?P<name>(all|android|chrome|windows10|phone|tablet|pc))', re.IGNORECASE)
JOIN_IMAGE_XY = NotifyImageSize.XY_72

class JoinPriority:
    LOW = -2
    MODERATE = -1
    NORMAL = 0
    HIGH = 1
    EMERGENCY = 2
JOIN_PRIORITIES = {JoinPriority.LOW: 'low', JoinPriority.MODERATE: 'moderate', JoinPriority.NORMAL: 'normal', JoinPriority.HIGH: 'high', JoinPriority.EMERGENCY: 'emergency'}
JOIN_PRIORITY_MAP = {'l': JoinPriority.LOW, 'm': JoinPriority.MODERATE, 'n': JoinPriority.NORMAL, 'h': JoinPriority.HIGH, 'e': JoinPriority.EMERGENCY, '-2': JoinPriority.LOW, '-1': JoinPriority.MODERATE, '0': JoinPriority.NORMAL, '1': JoinPriority.HIGH, '2': JoinPriority.EMERGENCY}

class NotifyJoin(NotifyBase):
    """
    A wrapper for Join Notifications
    """
    service_name = 'Join'
    service_url = 'https://joaoapps.com/join/'
    secure_protocol = 'join'
    setup_url = 'https://github.com/caronc/apprise/wiki/Notify_join'
    notify_url = 'https://joinjoaomgcd.appspot.com/_ah/api/messaging/v1/sendPush'
    image_size = NotifyImageSize.XY_72
    body_maxlen = 1000
    default_join_group = 'group.all'
    templates = ('{schema}://{apikey}/{targets}',)
    template_tokens = dict(NotifyBase.template_tokens, **{'apikey': {'name': _('API Key'), 'type': 'string', 'regex': ('^[a-z0-9]{32}$', 'i'), 'private': True, 'required': True}, 'device': {'name': _('Device ID'), 'type': 'string', 'regex': ('^[a-z0-9]{32}$', 'i'), 'map_to': 'targets'}, 'device_name': {'name': _('Device Name'), 'type': 'string', 'map_to': 'targets'}, 'group': {'name': _('Group'), 'type': 'choice:string', 'values': ('all', 'android', 'chrome', 'windows10', 'phone', 'tablet', 'pc'), 'map_to': 'targets'}, 'targets': {'name': _('Targets'), 'type': 'list:string'}})
    template_args = dict(NotifyBase.template_args, **{'image': {'name': _('Include Image'), 'type': 'bool', 'default': False, 'map_to': 'include_image'}, 'priority': {'name': _('Priority'), 'type': 'choice:int', 'values': JOIN_PRIORITIES, 'default': JoinPriority.NORMAL}, 'to': {'alias_of': 'targets'}})

    def __init__(self, apikey, targets=None, include_image=True, priority=None, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Initialize Join Object\n        '
        super().__init__(**kwargs)
        self.include_image = include_image
        self.apikey = validate_regex(apikey, *self.template_tokens['apikey']['regex'])
        if not self.apikey:
            msg = 'An invalid Join API Key ({}) was specified.'.format(apikey)
            self.logger.warning(msg)
            raise TypeError(msg)
        self.priority = int(NotifyJoin.template_args['priority']['default'] if priority is None else next((v for (k, v) in JOIN_PRIORITY_MAP.items() if str(priority).lower().startswith(k)), NotifyJoin.template_args['priority']['default']))
        self.targets = list()
        targets = parse_list(targets)
        if len(targets) == 0:
            self.targets.append(self.default_join_group)
            return
        while len(targets):
            target = targets.pop(0)
            group_re = IS_GROUP_RE.match(target)
            if group_re:
                self.targets.append('group.{}'.format(group_re.group('name').lower()))
                continue
            self.targets.append(target)
        return

    def send(self, body, title='', notify_type=NotifyType.INFO, **kwargs):
        if False:
            return 10
        '\n        Perform Join Notification\n        '
        headers = {'User-Agent': self.app_id, 'Content-Type': 'application/x-www-form-urlencoded'}
        has_error = False
        targets = list(self.targets)
        while len(targets):
            target = targets.pop(0)
            url_args = {'apikey': self.apikey, 'priority': str(self.priority), 'title': title, 'text': body}
            if IS_GROUP_RE.match(target) or IS_DEVICE_RE.match(target):
                url_args['deviceId'] = target
            else:
                url_args['deviceNames'] = target
            image_url = None if not self.include_image else self.image_url(notify_type)
            if image_url:
                url_args['icon'] = image_url
            payload = {}
            url = '%s?%s' % (self.notify_url, NotifyJoin.urlencode(url_args))
            self.logger.debug('Join POST URL: %s (cert_verify=%r)' % (url, self.verify_certificate))
            self.logger.debug('Join Payload: %s' % str(payload))
            self.throttle()
            try:
                r = requests.post(url, data=payload, headers=headers, verify=self.verify_certificate, timeout=self.request_timeout)
                if r.status_code != requests.codes.ok:
                    status_str = NotifyJoin.http_response_code_lookup(r.status_code, JOIN_HTTP_ERROR_MAP)
                    self.logger.warning('Failed to send Join notification to {}: {}{}error={}.'.format(target, status_str, ', ' if status_str else '', r.status_code))
                    self.logger.debug('Response Details:\r\n{}'.format(r.content))
                    has_error = True
                    continue
                else:
                    self.logger.info('Sent Join notification to %s.' % target)
            except requests.RequestException as e:
                self.logger.warning('A Connection error occurred sending Join:%s notification.' % target)
                self.logger.debug('Socket Exception: %s' % str(e))
                has_error = True
                continue
        return not has_error

    def url(self, privacy=False, *args, **kwargs):
        if False:
            return 10
        '\n        Returns the URL built dynamically based on specified arguments.\n        '
        params = {'priority': JOIN_PRIORITIES[self.template_args['priority']['default']] if self.priority not in JOIN_PRIORITIES else JOIN_PRIORITIES[self.priority], 'image': 'yes' if self.include_image else 'no'}
        params.update(self.url_parameters(*args, privacy=privacy, **kwargs))
        return '{schema}://{apikey}/{targets}/?{params}'.format(schema=self.secure_protocol, apikey=self.pprint(self.apikey, privacy, safe=''), targets='/'.join([NotifyJoin.quote(x, safe='') for x in self.targets]), params=NotifyJoin.urlencode(params))

    def __len__(self):
        if False:
            print('Hello World!')
        '\n        Returns the number of targets associated with this notification\n        '
        return len(self.targets)

    @staticmethod
    def parse_url(url):
        if False:
            i = 10
            return i + 15
        '\n        Parses the URL and returns enough arguments that can allow\n        us to re-instantiate this object.\n\n        '
        results = NotifyBase.parse_url(url, verify_host=False)
        if not results:
            return results
        results['apikey'] = results['user'] if results['user'] else results['host']
        results['apikey'] = NotifyJoin.unquote(results['apikey'])
        if 'priority' in results['qsd'] and len(results['qsd']['priority']):
            results['priority'] = NotifyJoin.unquote(results['qsd']['priority'])
        results['targets'] = list()
        if results['user']:
            results['targets'].append(NotifyJoin.unquote(results['host']))
        results['targets'].extend(NotifyJoin.split_path(results['fullpath']))
        if 'to' in results['qsd'] and len(results['qsd']['to']):
            results['targets'] += NotifyJoin.parse_list(results['qsd']['to'])
        results['include_image'] = parse_bool(results['qsd'].get('image', True))
        return results