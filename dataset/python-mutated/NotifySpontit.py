import re
import requests
from json import loads
from .NotifyBase import NotifyBase
from ..common import NotifyType
from ..utils import parse_list
from ..utils import validate_regex
from ..AppriseLocale import gettext_lazy as _
CHANNEL_REGEX = re.compile('^\\s*(\\#|\\%23)?((\\@|\\%40)?(?P<user>[a-z0-9_]+)([/\\\\]|\\%2F))?(?P<channel>[a-z0-9_-]+)\\s*$', re.I)

class NotifySpontit(NotifyBase):
    """
    A wrapper for Spontit Notifications
    """
    service_name = 'Spontit'
    service_url = 'https://spontit.com/'
    secure_protocol = 'spontit'
    request_rate_per_sec = 0.2
    setup_url = 'https://github.com/caronc/apprise/wiki/Notify_spontit'
    notify_url = 'https://api.spontit.com/v3/push'
    body_maxlen = 5000
    title_maxlen = 100
    spontit_body_minlen = 100
    spontit_subtitle_maxlen = 20
    templates = ('{schema}://{user}@{apikey}', '{schema}://{user}@{apikey}/{targets}')
    template_tokens = dict(NotifyBase.template_tokens, **{'user': {'name': _('User ID'), 'type': 'string', 'required': True, 'regex': ('^[a-z0-9_-]+$', 'i')}, 'apikey': {'name': _('API Key'), 'type': 'string', 'required': True, 'private': True, 'regex': ('^[a-z0-9]+$', 'i')}, 'target_channel': {'name': _('Target Channel ID'), 'type': 'string', 'prefix': '#', 'regex': ('^[0-9\\s)(+-]+$', 'i'), 'map_to': 'targets'}, 'targets': {'name': _('Targets'), 'type': 'list:string'}})
    template_args = dict(NotifyBase.template_args, **{'to': {'alias_of': 'targets'}, 'subtitle': {'name': _('Subtitle'), 'type': 'string'}})

    def __init__(self, apikey, targets=None, subtitle=None, **kwargs):
        if False:
            return 10
        '\n        Initialize Spontit Object\n        '
        super().__init__(**kwargs)
        user = validate_regex(self.user, *self.template_tokens['user']['regex'])
        if not user:
            msg = 'An invalid Spontit User ID ({}) was specified.'.format(self.user)
            self.logger.warning(msg)
            raise TypeError(msg)
        self.user = user
        self.apikey = validate_regex(apikey, *self.template_tokens['apikey']['regex'])
        if not self.apikey:
            msg = 'An invalid Spontit API Key ({}) was specified.'.format(apikey)
            self.logger.warning(msg)
            raise TypeError(msg)
        self.subtitle = subtitle
        self.targets = list()
        for target in parse_list(targets):
            result = CHANNEL_REGEX.match(target)
            if result:
                self.targets.append('{}'.format(result.group('channel')))
                continue
            self.logger.warning('Dropped invalid channel/user ({}) specified.'.format(target))
        return

    def send(self, body, title='', notify_type=NotifyType.INFO, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Sends Message\n        '
        has_error = False
        headers = {'User-Agent': self.app_id, 'Content-Type': 'application/json', 'X-Authorization': self.apikey, 'X-UserId': self.user}
        targets = list(self.targets)
        if not len(targets):
            targets = [None]
        while len(targets):
            target = targets.pop(0)
            payload = {'message': body}
            if len(body) > self.spontit_body_minlen:
                payload['message'] = '{}...'.format(body[:self.spontit_body_minlen - 3])
                payload['body'] = body
            if self.subtitle:
                payload['subtitle'] = self.subtitle[:self.spontit_subtitle_maxlen]
            elif self.app_desc:
                payload['subtitle'] = self.app_desc[:self.spontit_subtitle_maxlen]
            elif self.app_id:
                payload['subtitle'] = self.app_id[:self.spontit_subtitle_maxlen]
            if title:
                payload['pushTitle'] = title
            if target is not None:
                payload['channelName'] = target
            self.logger.debug('Spontit POST URL: {} (cert_verify={})'.format(self.notify_url, self.verify_certificate))
            self.logger.debug('Spontit Payload: {}'.format(payload))
            self.throttle()
            try:
                r = requests.post(self.notify_url, params=payload, headers=headers, verify=self.verify_certificate, timeout=self.request_timeout)
                if r.status_code not in (requests.codes.created, requests.codes.ok):
                    status_str = NotifyBase.http_response_code_lookup(r.status_code)
                    try:
                        json_response = loads(r.content)
                        status_str = json_response.get('message', status_str)
                    except (AttributeError, TypeError, ValueError):
                        pass
                    self.logger.warning('Failed to send Spontit notification to {}: {}{}error={}.'.format(target, status_str, ', ' if status_str else '', r.status_code))
                    self.logger.debug('Response Details:\r\n{}'.format(r.content))
                    has_error = True
                    continue
                self.logger.info('Sent Spontit notification to {}.'.format(target))
                self.logger.debug('Response Details:\r\n{}'.format(r.content))
            except requests.RequestException as e:
                self.logger.warning('A Connection error occurred sending Spontit:%s ' % ', '.join(self.targets) + 'notification.')
                self.logger.debug('Socket Exception: %s' % str(e))
                has_error = True
                continue
        return not has_error

    def url(self, privacy=False, *args, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Returns the URL built dynamically based on specified arguments.\n        '
        params = self.url_parameters(*args, privacy=privacy, **kwargs)
        if self.subtitle:
            params['subtitle'] = self.subtitle
        return '{schema}://{userid}@{apikey}/{targets}?{params}'.format(schema=self.secure_protocol, userid=self.user, apikey=self.pprint(self.apikey, privacy, safe=''), targets='/'.join([NotifySpontit.quote(x, safe='') for x in self.targets]), params=NotifySpontit.urlencode(params))

    def __len__(self):
        if False:
            print('Hello World!')
        '\n        Returns the number of targets associated with this notification\n        '
        targets = len(self.targets)
        return targets if targets > 0 else 1

    @staticmethod
    def parse_url(url):
        if False:
            while True:
                i = 10
        '\n        Parses the URL and returns enough arguments that can allow\n        us to re-instantiate this object.\n\n        '
        results = NotifyBase.parse_url(url, verify_host=False)
        if not results:
            return results
        results['targets'] = NotifySpontit.split_path(results['fullpath'])
        results['apikey'] = NotifySpontit.unquote(results['host'])
        if 'subtitle' in results['qsd'] and len(results['qsd']['subtitle']):
            results['subtitle'] = NotifySpontit.unquote(results['qsd']['subtitle'])
        if 'to' in results['qsd'] and len(results['qsd']['to']):
            results['targets'] += NotifySpontit.parse_list(results['qsd']['to'])
        return results