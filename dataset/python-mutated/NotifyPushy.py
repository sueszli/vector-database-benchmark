import re
import requests
from itertools import chain
from json import dumps, loads
from .NotifyBase import NotifyBase
from ..common import NotifyType
from ..utils import parse_list
from ..utils import validate_regex
from ..AppriseLocale import gettext_lazy as _
VALIDATE_DEVICE = re.compile('^@(?P<device>[a-z0-9]+)$', re.I)
VALIDATE_TOPIC = re.compile('^[#]?(?P<topic>[a-z0-9]+)$', re.I)
PUSHY_HTTP_ERROR_MAP = {401: 'Unauthorized - Invalid Token.'}

class NotifyPushy(NotifyBase):
    """
    A wrapper for Pushy Notifications
    """
    service_name = 'Pushy'
    service_url = 'https://pushy.me/'
    secure_protocol = 'pushy'
    setup_url = 'https://github.com/caronc/apprise/wiki/Notify_pushy'
    notify_url = 'https://api.pushy.me/push?api_key={apikey}'
    body_maxlen = 4096
    templates = ('{schema}://{apikey}/{targets}',)
    template_tokens = dict(NotifyBase.template_tokens, **{'apikey': {'name': _('Secret API Key'), 'type': 'string', 'private': True, 'required': True}, 'target_device': {'name': _('Target Device'), 'type': 'string', 'prefix': '@', 'map_to': 'targets'}, 'target_topic': {'name': _('Target Topic'), 'type': 'string', 'prefix': '#', 'map_to': 'targets'}, 'targets': {'name': _('Targets'), 'type': 'list:string', 'required': True}})
    template_args = dict(NotifyBase.template_args, **{'sound': {'name': _('Sound'), 'type': 'string'}, 'badge': {'name': _('Badge'), 'type': 'int', 'min': 0}, 'to': {'alias_of': 'targets'}, 'key': {'alias_of': 'apikey'}})

    def __init__(self, apikey, targets=None, sound=None, badge=None, **kwargs):
        if False:
            return 10
        '\n        Initialize Pushy Object\n        '
        super().__init__(**kwargs)
        self.apikey = validate_regex(apikey)
        if not self.apikey:
            msg = 'An invalid Pushy Secret API Key ({}) was specified.'.format(apikey)
            self.logger.warning(msg)
            raise TypeError(msg)
        self.devices = []
        self.topics = []
        for target in parse_list(targets):
            result = VALIDATE_TOPIC.match(target)
            if result:
                self.topics.append(result.group('topic'))
                continue
            result = VALIDATE_DEVICE.match(target)
            if result:
                self.devices.append(result.group('device'))
                continue
            self.logger.warning('Dropped invalid topic/device  ({}) specified.'.format(target))
        self.sound = sound
        try:
            self.badge = int(badge)
            if self.badge < 0:
                raise ValueError()
        except TypeError:
            self.badge = None
        except ValueError:
            self.badge = None
            self.logger.warning('The specified Pushy badge ({}) is not valid ', badge)
        return

    def send(self, body, title='', notify_type=NotifyType.INFO, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Perform Pushy Notification\n        '
        if len(self.topics) + len(self.devices) == 0:
            self.logger.warning('There were no Pushy targets to notify.')
            return False
        has_error = False
        headers = {'User-Agent': self.app_id, 'Content-Type': 'application/json', 'Accepts': 'application/json'}
        notify_url = self.notify_url.format(apikey=self.apikey)
        content = {}
        targets = list(self.topics) + list(self.devices)
        while len(targets):
            target = targets.pop(0)
            payload = {'to': target, 'data': {'message': body}, 'notification': {'body': body}}
            if title:
                payload['notification']['title'] = title
            if self.sound:
                payload['notification']['sound'] = self.sound
            if self.badge is not None:
                payload['notification']['badge'] = self.badge
            self.logger.debug('Pushy POST URL: %s (cert_verify=%r)' % (notify_url, self.verify_certificate))
            self.logger.debug('Pushy Payload: %s' % str(payload))
            self.throttle()
            try:
                r = requests.post(notify_url, data=dumps(payload), headers=headers, verify=self.verify_certificate, timeout=self.request_timeout)
                try:
                    content = loads(r.content)
                except (AttributeError, TypeError, ValueError):
                    content = {'success': False, 'id': '', 'info': {}}
                if r.status_code != requests.codes.ok or not content.get('success'):
                    status_str = NotifyPushy.http_response_code_lookup(r.status_code, PUSHY_HTTP_ERROR_MAP)
                    self.logger.warning('Failed to send Pushy notification to {}: {}{}error={}.'.format(target, status_str, ', ' if status_str else '', r.status_code))
                    self.logger.debug('Response Details:\r\n{}'.format(r.content))
                    has_error = True
                    continue
                else:
                    self.logger.info('Sent Pushy notification to %s.' % target)
            except requests.RequestException as e:
                self.logger.warning('A Connection error occurred sending Pushy:%s notification', target)
                self.logger.debug('Socket Exception: %s' % str(e))
                has_error = True
                continue
        return not has_error

    def url(self, privacy=False, *args, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Returns the URL built dynamically based on specified arguments.\n        '
        params = {}
        if self.sound:
            params['sound'] = self.sound
        if self.badge is not None:
            params['badge'] = str(self.badge)
        params.update(self.url_parameters(*args, privacy=privacy, **kwargs))
        return '{schema}://{apikey}/{targets}/?{params}'.format(schema=self.secure_protocol, apikey=self.pprint(self.apikey, privacy, safe=''), targets='/'.join([NotifyPushy.quote(x, safe='@#') for x in chain(['#{}'.format(x) for x in self.topics], ['@{}'.format(x) for x in self.devices])]), params=NotifyPushy.urlencode(params))

    def __len__(self):
        if False:
            print('Hello World!')
        '\n        Returns the number of targets associated with this notification\n        '
        return len(self.topics) + len(self.devices)

    @staticmethod
    def parse_url(url):
        if False:
            i = 10
            return i + 15
        '\n        Parses the URL and returns enough arguments that can allow\n        us to re-instantiate this object.\n\n        '
        results = NotifyBase.parse_url(url, verify_host=False)
        if not results:
            return results
        results['apikey'] = NotifyPushy.unquote(results['host'])
        results['targets'] = NotifyPushy.split_path(results['fullpath'])
        if 'sound' in results['qsd'] and len(results['qsd']['sound']):
            results['sound'] = NotifyPushy.unquote(results['qsd']['sound'])
        if 'badge' in results['qsd'] and results['qsd']['badge']:
            results['badge'] = NotifyPushy.unquote(results['qsd']['badge'].strip())
        if 'key' in results['qsd'] and len(results['qsd']['key']):
            results['apikey'] = results['qsd']['key']
        if 'to' in results['qsd'] and len(results['qsd']['to']):
            results['targets'] += NotifyPushy.parse_list(results['qsd']['to'])
        return results