import re
import requests
import hmac
from json import dumps
from time import time
from hashlib import sha1
from itertools import chain
try:
    from urlparse import urlparse
except ImportError:
    from urllib.parse import urlparse
from .NotifyBase import NotifyBase
from ..URLBase import PrivacyMode
from ..utils import parse_bool
from ..utils import parse_list
from ..utils import validate_regex
from ..common import NotifyType
from ..common import NotifyImageSize
from ..AppriseLocale import gettext_lazy as _
DEFAULT_TAG = '@all'
IS_TAG = re.compile('^[@]?(?P<name>[A-Z0-9]{1,63})$', re.I)
IS_DEVICETOKEN = re.compile('^[A-Z0-9]{64}$', re.I)
TAGS_LIST_DELIM = re.compile('[ \\t\\r\\n,\\\\/]+')

class NotifyBoxcar(NotifyBase):
    """
    A wrapper for Boxcar Notifications
    """
    service_name = 'Boxcar'
    service_url = 'https://boxcar.io/'
    secure_protocol = 'boxcar'
    setup_url = 'https://github.com/caronc/apprise/wiki/Notify_boxcar'
    notify_url = 'https://boxcar-api.io/api/push/'
    image_size = NotifyImageSize.XY_72
    body_maxlen = 10000
    templates = ('{schema}://{access_key}/{secret_key}/', '{schema}://{access_key}/{secret_key}/{targets}')
    template_tokens = dict(NotifyBase.template_tokens, **{'access_key': {'name': _('Access Key'), 'type': 'string', 'private': True, 'required': True, 'regex': ('^[A-Z0-9_-]{64}$', 'i'), 'map_to': 'access'}, 'secret_key': {'name': _('Secret Key'), 'type': 'string', 'private': True, 'required': True, 'regex': ('^[A-Z0-9_-]{64}$', 'i'), 'map_to': 'secret'}, 'target_tag': {'name': _('Target Tag ID'), 'type': 'string', 'prefix': '@', 'regex': ('^[A-Z0-9]{1,63}$', 'i'), 'map_to': 'targets'}, 'target_device': {'name': _('Target Device ID'), 'type': 'string', 'regex': ('^[A-Z0-9]{64}$', 'i'), 'map_to': 'targets'}, 'targets': {'name': _('Targets'), 'type': 'list:string'}})
    template_args = dict(NotifyBase.template_args, **{'image': {'name': _('Include Image'), 'type': 'bool', 'default': True, 'map_to': 'include_image'}, 'to': {'alias_of': 'targets'}, 'access': {'alias_of': 'access_key'}, 'secret': {'alias_of': 'secret_key'}})

    def __init__(self, access, secret, targets=None, include_image=True, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Initialize Boxcar Object\n        '
        super().__init__(**kwargs)
        self._tags = list()
        self.device_tokens = list()
        self.access = validate_regex(access, *self.template_tokens['access_key']['regex'])
        if not self.access:
            msg = 'An invalid Boxcar Access Key ({}) was specified.'.format(access)
            self.logger.warning(msg)
            raise TypeError(msg)
        self.secret = validate_regex(secret, *self.template_tokens['secret_key']['regex'])
        if not self.secret:
            msg = 'An invalid Boxcar Secret Key ({}) was specified.'.format(secret)
            self.logger.warning(msg)
            raise TypeError(msg)
        if not targets:
            self._tags.append(DEFAULT_TAG)
            targets = []
        for target in parse_list(targets):
            result = IS_TAG.match(target)
            if result:
                self._tags.append(result.group('name'))
                continue
            result = IS_DEVICETOKEN.match(target)
            if result:
                self.device_tokens.append(target)
                continue
            self.logger.warning('Dropped invalid tag/alias/device_token ({}) specified.'.format(target))
        self.include_image = include_image
        return

    def send(self, body, title='', notify_type=NotifyType.INFO, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Perform Boxcar Notification\n        '
        headers = {'User-Agent': self.app_id, 'Content-Type': 'application/json'}
        payload = {'aps': {'badge': 'auto', 'alert': ''}, 'expires': str(int(time() + 30))}
        if title:
            payload['aps']['@title'] = title
        payload['aps']['alert'] = body
        if self._tags:
            payload['tags'] = {'or': self._tags}
        if self.device_tokens:
            payload['device_tokens'] = self.device_tokens
        image_url = None if not self.include_image else self.image_url(notify_type)
        if image_url:
            payload['@img'] = image_url
        host = urlparse(self.notify_url).hostname
        str_to_sign = '%s\n%s\n%s\n%s' % ('POST', host, '/api/push', dumps(payload))
        h = hmac.new(bytearray(self.secret, 'utf-8'), bytearray(str_to_sign, 'utf-8'), sha1)
        params = NotifyBoxcar.urlencode({'publishkey': self.access, 'signature': h.hexdigest()})
        notify_url = '%s?%s' % (self.notify_url, params)
        self.logger.debug('Boxcar POST URL: %s (cert_verify=%r)' % (notify_url, self.verify_certificate))
        self.logger.debug('Boxcar Payload: %s' % str(payload))
        self.throttle()
        try:
            r = requests.post(notify_url, data=dumps(payload), headers=headers, verify=self.verify_certificate, timeout=self.request_timeout)
            if r.status_code != requests.codes.created:
                status_str = NotifyBoxcar.http_response_code_lookup(r.status_code)
                self.logger.warning('Failed to send Boxcar notification: {}{}error={}.'.format(status_str, ', ' if status_str else '', r.status_code))
                self.logger.debug('Response Details:\r\n{}'.format(r.content))
                return False
            else:
                self.logger.info('Sent Boxcar notification.')
        except requests.RequestException as e:
            self.logger.warning('A Connection error occurred sending Boxcar notification to %s.' % host)
            self.logger.debug('Socket Exception: %s' % str(e))
            return False
        return True

    def url(self, privacy=False, *args, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Returns the URL built dynamically based on specified arguments.\n        '
        params = {'image': 'yes' if self.include_image else 'no'}
        params.update(self.url_parameters(*args, privacy=privacy, **kwargs))
        return '{schema}://{access}/{secret}/{targets}?{params}'.format(schema=self.secure_protocol, access=self.pprint(self.access, privacy, safe=''), secret=self.pprint(self.secret, privacy, mode=PrivacyMode.Secret, safe=''), targets='/'.join([NotifyBoxcar.quote(x, safe='') for x in chain(self._tags, self.device_tokens) if x != DEFAULT_TAG]), params=NotifyBoxcar.urlencode(params))

    def __len__(self):
        if False:
            while True:
                i = 10
        '\n        Returns the number of targets associated with this notification\n        '
        targets = len(self._tags) + len(self.device_tokens)
        return targets if targets > 0 else 1

    @staticmethod
    def parse_url(url):
        if False:
            print('Hello World!')
        '\n        Parses the URL and returns it broken apart into a dictionary.\n\n        '
        results = NotifyBase.parse_url(url, verify_host=False)
        if not results:
            return None
        results['access'] = NotifyBoxcar.unquote(results['host'])
        entries = NotifyBoxcar.split_path(results['fullpath'])
        results['secret'] = entries.pop(0) if entries else None
        results['targets'] = entries
        if 'to' in results['qsd'] and len(results['qsd']['to']):
            results['targets'] += NotifyBoxcar.parse_list(results['qsd'].get('to'))
        if 'access' in results['qsd'] and results['qsd']['access']:
            results['access'] = NotifyBoxcar.unquote(results['qsd']['access'].strip())
        if 'secret' in results['qsd'] and results['qsd']['secret']:
            results['secret'] = NotifyBoxcar.unquote(results['qsd']['secret'].strip())
        results['include_image'] = parse_bool(results['qsd'].get('image', True))
        return results