import requests
from json import dumps
from .NotifyBase import NotifyBase
from ..URLBase import PrivacyMode
from ..common import NotifyType
from ..utils import validate_regex
from ..AppriseLocale import gettext_lazy as _

class NotifyPushjet(NotifyBase):
    """
    A wrapper for Pushjet Notifications
    """
    service_name = 'Pushjet'
    protocol = 'pjet'
    secure_protocol = 'pjets'
    setup_url = 'https://github.com/caronc/apprise/wiki/Notify_pushjet'
    request_rate_per_sec = 0
    templates = ('{schema}://{host}:{port}/{secret_key}', '{schema}://{host}/{secret_key}', '{schema}://{user}:{password}@{host}:{port}/{secret_key}', '{schema}://{user}:{password}@{host}/{secret_key}')
    template_tokens = dict(NotifyBase.template_tokens, **{'host': {'name': _('Hostname'), 'type': 'string', 'required': True}, 'port': {'name': _('Port'), 'type': 'int', 'min': 1, 'max': 65535}, 'secret_key': {'name': _('Secret Key'), 'type': 'string', 'required': True, 'private': True}, 'user': {'name': _('Username'), 'type': 'string'}, 'password': {'name': _('Password'), 'type': 'string', 'private': True}})
    template_args = dict(NotifyBase.template_args, **{'secret': {'alias_of': 'secret_key'}})

    def __init__(self, secret_key, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Initialize Pushjet Object\n        '
        super().__init__(**kwargs)
        self.secret_key = validate_regex(secret_key)
        if not self.secret_key:
            msg = 'An invalid Pushjet Secret Key ({}) was specified.'.format(secret_key)
            self.logger.warning(msg)
            raise TypeError(msg)
        return

    def url(self, privacy=False, *args, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Returns the URL built dynamically based on specified arguments.\n        '
        params = self.url_parameters(*args, privacy=privacy, **kwargs)
        default_port = 443 if self.secure else 80
        auth = ''
        if self.user and self.password:
            auth = '{user}:{password}@'.format(user=NotifyPushjet.quote(self.user, safe=''), password=self.pprint(self.password, privacy, mode=PrivacyMode.Secret, safe=''))
        return '{schema}://{auth}{hostname}{port}/{secret}/?{params}'.format(schema=self.secure_protocol if self.secure else self.protocol, auth=auth, hostname=self.host, port='' if self.port is None or self.port == default_port else ':{}'.format(self.port), secret=self.pprint(self.secret_key, privacy, mode=PrivacyMode.Secret, safe=''), params=NotifyPushjet.urlencode(params))

    def send(self, body, title='', notify_type=NotifyType.INFO, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Perform Pushjet Notification\n        '
        params = {'secret': self.secret_key}
        payload = {'message': body, 'title': title, 'link': None, 'level': None}
        headers = {'User-Agent': self.app_id, 'Content-Type': 'application/x-www-form-urlencoded; charset=utf-8'}
        auth = None
        if self.user:
            auth = (self.user, self.password)
        notify_url = '{schema}://{host}{port}/message/'.format(schema='https' if self.secure else 'http', host=self.host, port=':{}'.format(self.port) if self.port else '')
        self.logger.debug('Pushjet POST URL: %s (cert_verify=%r)' % (notify_url, self.verify_certificate))
        self.logger.debug('Pushjet Payload: %s' % str(payload))
        self.throttle()
        try:
            r = requests.post(notify_url, params=params, data=dumps(payload), headers=headers, auth=auth, verify=self.verify_certificate, timeout=self.request_timeout)
            if r.status_code != requests.codes.ok:
                status_str = NotifyPushjet.http_response_code_lookup(r.status_code)
                self.logger.warning('Failed to send Pushjet notification: {}{}error={}.'.format(status_str, ', ' if status_str else '', r.status_code))
                self.logger.debug('Response Details:\r\n{}'.format(r.content))
                return False
            else:
                self.logger.info('Sent Pushjet notification.')
        except requests.RequestException as e:
            self.logger.warning('A Connection error occurred sending Pushjet notification to %s.' % self.host)
            self.logger.debug('Socket Exception: %s' % str(e))
            return False
        return True

    @staticmethod
    def parse_url(url):
        if False:
            while True:
                i = 10
        '\n        Parses the URL and returns enough arguments that can allow\n        us to re-instantiate this object.\n\n        Syntax:\n           pjet://hostname/secret_key\n           pjet://hostname:port/secret_key\n           pjet://user:pass@hostname/secret_key\n           pjet://user:pass@hostname:port/secret_key\n           pjets://hostname/secret_key\n           pjets://hostname:port/secret_key\n           pjets://user:pass@hostname/secret_key\n           pjets://user:pass@hostname:port/secret_key\n        '
        results = NotifyBase.parse_url(url)
        if not results:
            return results
        try:
            results['secret_key'] = NotifyPushjet.split_path(results['fullpath'])[0]
        except IndexError:
            results['secret_key'] = None
        if 'secret' in results['qsd'] and len(results['qsd']['secret']):
            results['secret_key'] = NotifyPushjet.unquote(results['qsd']['secret'])
        return results