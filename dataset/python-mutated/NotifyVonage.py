import requests
from .NotifyBase import NotifyBase
from ..URLBase import PrivacyMode
from ..common import NotifyType
from ..utils import is_phone_no
from ..utils import parse_phone_no
from ..utils import validate_regex
from ..AppriseLocale import gettext_lazy as _

class NotifyVonage(NotifyBase):
    """
    A wrapper for Vonage Notifications
    """
    service_name = 'Vonage'
    service_url = 'https://dashboard.nexmo.com/'
    secure_protocol = ('vonage', 'nexmo')
    setup_url = 'https://github.com/caronc/apprise/wiki/Notify_nexmo'
    notify_url = 'https://rest.nexmo.com/sms/json'
    body_maxlen = 160
    title_maxlen = 0
    templates = ('{schema}://{apikey}:{secret}@{from_phone}', '{schema}://{apikey}:{secret}@{from_phone}/{targets}')
    template_tokens = dict(NotifyBase.template_tokens, **{'apikey': {'name': _('API Key'), 'type': 'string', 'required': True, 'regex': ('^[a-z0-9]+$', 'i'), 'private': True}, 'secret': {'name': _('API Secret'), 'type': 'string', 'private': True, 'required': True, 'regex': ('^[a-z0-9]+$', 'i')}, 'from_phone': {'name': _('From Phone No'), 'type': 'string', 'required': True, 'regex': ('^\\+?[0-9\\s)(+-]+$', 'i'), 'map_to': 'source'}, 'target_phone': {'name': _('Target Phone No'), 'type': 'string', 'prefix': '+', 'regex': ('^[0-9\\s)(+-]+$', 'i'), 'map_to': 'targets'}, 'targets': {'name': _('Targets'), 'type': 'list:string'}})
    template_args = dict(NotifyBase.template_args, **{'to': {'alias_of': 'targets'}, 'from': {'alias_of': 'from_phone'}, 'key': {'alias_of': 'apikey'}, 'secret': {'alias_of': 'secret'}, 'ttl': {'name': _('ttl'), 'type': 'int', 'default': 900000, 'min': 20000, 'max': 604800000}})

    def __init__(self, apikey, secret, source, targets=None, ttl=None, **kwargs):
        if False:
            return 10
        '\n        Initialize Vonage Object\n        '
        super().__init__(**kwargs)
        self.apikey = validate_regex(apikey, *self.template_tokens['apikey']['regex'])
        if not self.apikey:
            msg = 'An invalid Vonage API Key ({}) was specified.'.format(apikey)
            self.logger.warning(msg)
            raise TypeError(msg)
        self.secret = validate_regex(secret, *self.template_tokens['secret']['regex'])
        if not self.secret:
            msg = 'An invalid Vonage API Secret ({}) was specified.'.format(secret)
            self.logger.warning(msg)
            raise TypeError(msg)
        self.ttl = self.template_args['ttl']['default']
        try:
            self.ttl = int(ttl)
        except (ValueError, TypeError):
            pass
        if self.ttl < self.template_args['ttl']['min'] or self.ttl > self.template_args['ttl']['max']:
            msg = 'The Vonage TTL specified ({}) is out of range.'.format(self.ttl)
            self.logger.warning(msg)
            raise TypeError(msg)
        self.source = source
        result = is_phone_no(source)
        if not result:
            msg = 'The Account (From) Phone # specified ({}) is invalid.'.format(source)
            self.logger.warning(msg)
            raise TypeError(msg)
        self.source = result['full']
        self.targets = list()
        for target in parse_phone_no(targets):
            result = is_phone_no(target)
            if not result:
                self.logger.warning('Dropped invalid phone # ({}) specified.'.format(target))
                continue
            self.targets.append(result['full'])
        return

    def send(self, body, title='', notify_type=NotifyType.INFO, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Perform Vonage Notification\n        '
        has_error = False
        headers = {'User-Agent': self.app_id, 'Content-Type': 'application/x-www-form-urlencoded'}
        payload = {'api_key': self.apikey, 'api_secret': self.secret, 'ttl': self.ttl, 'from': self.source, 'text': body, 'to': None}
        targets = list(self.targets)
        if len(targets) == 0:
            targets.append(self.source)
        while len(targets):
            target = targets.pop(0)
            payload['to'] = target
            self.logger.debug('Vonage POST URL: {} (cert_verify={})'.format(self.notify_url, self.verify_certificate))
            self.logger.debug('Vonage Payload: {}'.format(payload))
            self.throttle()
            try:
                r = requests.post(self.notify_url, data=payload, headers=headers, verify=self.verify_certificate, timeout=self.request_timeout)
                if r.status_code != requests.codes.ok:
                    status_str = NotifyVonage.http_response_code_lookup(r.status_code)
                    self.logger.warning('Failed to send Vonage notification to {}: {}{}error={}.'.format(target, status_str, ', ' if status_str else '', r.status_code))
                    self.logger.debug('Response Details:\r\n{}'.format(r.content))
                    has_error = True
                    continue
                else:
                    self.logger.info('Sent Vonage notification to %s.' % target)
            except requests.RequestException as e:
                self.logger.warning('A Connection error occurred sending Vonage:%s notification.' % target)
                self.logger.debug('Socket Exception: %s' % str(e))
                has_error = True
                continue
        return not has_error

    def url(self, privacy=False, *args, **kwargs):
        if False:
            print('Hello World!')
        '\n        Returns the URL built dynamically based on specified arguments.\n        '
        params = {'ttl': str(self.ttl)}
        params.update(self.url_parameters(*args, privacy=privacy, **kwargs))
        return '{schema}://{key}:{secret}@{source}/{targets}/?{params}'.format(schema=self.secure_protocol[0], key=self.pprint(self.apikey, privacy, safe=''), secret=self.pprint(self.secret, privacy, mode=PrivacyMode.Secret, safe=''), source=NotifyVonage.quote(self.source, safe=''), targets='/'.join([NotifyVonage.quote(x, safe='') for x in self.targets]), params=NotifyVonage.urlencode(params))

    def __len__(self):
        if False:
            while True:
                i = 10
        '\n        Returns the number of targets associated with this notification\n        '
        targets = len(self.targets)
        return targets if targets > 0 else 1

    @staticmethod
    def parse_url(url):
        if False:
            i = 10
            return i + 15
        '\n        Parses the URL and returns enough arguments that can allow\n        us to re-instantiate this object.\n\n        '
        results = NotifyBase.parse_url(url, verify_host=False)
        if not results:
            return results
        results['targets'] = NotifyVonage.split_path(results['fullpath'])
        results['source'] = NotifyVonage.unquote(results['host'])
        results['apikey'] = NotifyVonage.unquote(results['user'])
        results['secret'] = NotifyVonage.unquote(results['password'])
        if 'key' in results['qsd'] and len(results['qsd']['key']):
            results['apikey'] = NotifyVonage.unquote(results['qsd']['key'])
        if 'secret' in results['qsd'] and len(results['qsd']['secret']):
            results['secret'] = NotifyVonage.unquote(results['qsd']['secret'])
        if 'from' in results['qsd'] and len(results['qsd']['from']):
            results['source'] = NotifyVonage.unquote(results['qsd']['from'])
        if 'source' in results['qsd'] and len(results['qsd']['source']):
            results['source'] = NotifyVonage.unquote(results['qsd']['source'])
        if 'ttl' in results['qsd'] and len(results['qsd']['ttl']):
            results['ttl'] = NotifyVonage.unquote(results['qsd']['ttl'])
        if 'to' in results['qsd'] and len(results['qsd']['to']):
            results['targets'] += NotifyVonage.parse_phone_no(results['qsd']['to'])
        return results