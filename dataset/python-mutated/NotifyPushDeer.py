import requests
from ..common import NotifyType
from .NotifyBase import NotifyBase
from ..utils import validate_regex
from ..AppriseLocale import gettext_lazy as _

class NotifyPushDeer(NotifyBase):
    """
    A wrapper for PushDeer Notifications
    """
    service_name = 'PushDeer'
    service_url = 'https://www.pushdeer.com/'
    protocol = 'pushdeer'
    secure_protocol = 'pushdeers'
    setup_url = 'https://github.com/caronc/apprise/wiki/Notify_PushDeer'
    default_hostname = 'api2.pushdeer.com'
    notify_url = '{schema}://{host}:{port}/message/push?pushkey={pushKey}'
    templates = ('{schema}://{pushkey}', '{schema}://{host}/{pushkey}', '{schema}://{host}:{port}/{pushkey}')
    template_tokens = dict(NotifyBase.template_tokens, **{'host': {'name': _('Hostname'), 'type': 'string'}, 'port': {'name': _('Port'), 'type': 'int', 'min': 1, 'max': 65535}, 'pushkey': {'name': _('Pushkey'), 'type': 'string', 'private': True, 'required': True, 'regex': ('^[a-z0-9]+$', 'i')}})

    def __init__(self, pushkey, **kwargs):
        if False:
            print('Hello World!')
        '\n        Initialize PushDeer Object\n        '
        super().__init__(**kwargs)
        self.push_key = validate_regex(pushkey, *self.template_tokens['pushkey']['regex'])
        if not self.push_key:
            msg = 'An invalid PushDeer API Pushkey ({}) was specified.'.format(pushkey)
            self.logger.warning(msg)
            raise TypeError(msg)

    def send(self, body, title='', notify_type=NotifyType.INFO, **kwargs):
        if False:
            print('Hello World!')
        '\n        Perform PushDeer Notification\n        '
        payload = {'text': title if title else body, 'type': 'text', 'desp': body if title else ''}
        schema = 'https' if self.secure else 'http'
        host = self.default_hostname
        if self.host:
            host = self.host
        port = 443 if self.secure else 80
        if self.port:
            port = self.port
        notify_url = self.notify_url.format(schema=schema, host=host, port=port, pushKey=self.push_key)
        self.logger.debug('PushDeer URL: {} (cert_verify={})'.format(notify_url, self.verify_certificate))
        self.logger.debug('PushDeer Payload: {}'.format(payload))
        self.throttle()
        try:
            r = requests.post(notify_url, data=payload, timeout=self.request_timeout)
            if r.status_code != requests.codes.ok:
                status_str = NotifyPushDeer.http_response_code_lookup(r.status_code)
                self.logger.warning('Failed to send PushDeer notification: {}{}error={}.'.format(status_str, ', ' if status_str else '', r.status_code))
                self.logger.debug('Response Details:\r\n{}'.format(r.content))
                return False
            else:
                self.logger.info('Sent PushDeer notification.')
        except requests.RequestException as e:
            self.logger.warning('A Connection error occured sending PushDeer notification.')
            self.logger.debug('Socket Exception: %s' % str(e))
            return False
        return True

    def url(self, privacy=False):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns the URL built dynamically based on specified arguments.\n        '
        if self.host:
            url = '{schema}://{host}{port}/{pushkey}'
        else:
            url = '{schema}://{pushkey}'
        return url.format(schema=self.secure_protocol if self.secure else self.protocol, host=self.host, port='' if not self.port else ':{}'.format(self.port), pushkey=self.pprint(self.push_key, privacy, safe=''))

    @staticmethod
    def parse_url(url):
        if False:
            return 10
        '\n        Parses the URL and returns enough arguments that can allow\n        us to substantiate this object.\n        '
        results = NotifyBase.parse_url(url, verify_host=False)
        if not results:
            return results
        fullpaths = NotifyPushDeer.split_path(results['fullpath'])
        if len(fullpaths) == 0:
            results['pushkey'] = results['host']
            results['host'] = None
        else:
            results['pushkey'] = fullpaths.pop()
        return results