import requests
from json import dumps
from base64 import b64encode
from .NotifyBase import NotifyBase
from ..URLBase import PrivacyMode
from ..common import NotifyType
from ..utils import is_phone_no
from ..utils import parse_phone_no
from ..utils import parse_bool
from ..AppriseLocale import gettext_lazy as _
CLICKSEND_HTTP_ERROR_MAP = {401: 'Unauthorized - Invalid Token.'}

class NotifyClickSend(NotifyBase):
    """
    A wrapper for ClickSend Notifications
    """
    service_name = 'ClickSend'
    service_url = 'https://clicksend.com/'
    secure_protocol = 'clicksend'
    setup_url = 'https://github.com/caronc/apprise/wiki/Notify_clicksend'
    notify_url = 'https://rest.clicksend.com/v3/sms/send'
    body_maxlen = 160
    title_maxlen = 0
    default_batch_size = 1000
    templates = ('{schema}://{user}:{password}@{targets}',)
    template_tokens = dict(NotifyBase.template_tokens, **{'user': {'name': _('User Name'), 'type': 'string', 'required': True}, 'password': {'name': _('Password'), 'type': 'string', 'private': True, 'required': True}, 'target_phone': {'name': _('Target Phone No'), 'type': 'string', 'prefix': '+', 'regex': ('^[0-9\\s)(+-]+$', 'i'), 'map_to': 'targets'}, 'targets': {'name': _('Targets'), 'type': 'list:string', 'required': True}})
    template_args = dict(NotifyBase.template_args, **{'to': {'alias_of': 'targets'}, 'batch': {'name': _('Batch Mode'), 'type': 'bool', 'default': False}})

    def __init__(self, targets=None, batch=False, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Initialize ClickSend Object\n        '
        super().__init__(**kwargs)
        self.batch = batch
        self.targets = list()
        if not (self.user and self.password):
            msg = 'A ClickSend user/pass was not provided.'
            self.logger.warning(msg)
            raise TypeError(msg)
        for target in parse_phone_no(targets):
            result = is_phone_no(target)
            if not result:
                self.logger.warning('Dropped invalid phone # ({}) specified.'.format(target))
                continue
            self.targets.append(result['full'])

    def send(self, body, title='', notify_type=NotifyType.INFO, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Perform ClickSend Notification\n        '
        if len(self.targets) == 0:
            self.logger.warning('There were no ClickSend targets to notify.')
            return False
        headers = {'User-Agent': self.app_id, 'Content-Type': 'application/json; charset=utf-8', 'Authorization': 'Basic {}'.format(b64encode('{}:{}'.format(self.user, self.password).encode('utf-8')))}
        has_error = False
        payload = {'messages': []}
        default_batch_size = 1 if not self.batch else self.default_batch_size
        for index in range(0, len(self.targets), default_batch_size):
            payload['messages'] = [{'source': 'php', 'body': body, 'to': '+{}'.format(to)} for to in self.targets[index:index + default_batch_size]]
            self.logger.debug('ClickSend POST URL: %s (cert_verify=%r)' % (self.notify_url, self.verify_certificate))
            self.logger.debug('ClickSend Payload: %s' % str(payload))
            self.throttle()
            try:
                r = requests.post(self.notify_url, data=dumps(payload), headers=headers, verify=self.verify_certificate, timeout=self.request_timeout)
                if r.status_code != requests.codes.ok:
                    status_str = NotifyClickSend.http_response_code_lookup(r.status_code, CLICKSEND_HTTP_ERROR_MAP)
                    self.logger.warning('Failed to send {} ClickSend notification{}: {}{}error={}.'.format(len(payload['messages']), ' to {}'.format(self.targets[index]) if default_batch_size == 1 else '(s)', status_str, ', ' if status_str else '', r.status_code))
                    self.logger.debug('Response Details:\r\n{}'.format(r.content))
                    has_error = True
                    continue
                else:
                    self.logger.info('Sent {} ClickSend notification{}.'.format(len(payload['messages']), ' to {}'.format(self.targets[index]) if default_batch_size == 1 else '(s)'))
            except requests.RequestException as e:
                self.logger.warning('A Connection error occurred sending {} ClickSend notification(s).'.format(len(payload['messages'])))
                self.logger.debug('Socket Exception: %s' % str(e))
                has_error = True
                continue
        return not has_error

    def url(self, privacy=False, *args, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Returns the URL built dynamically based on specified arguments.\n        '
        params = {'batch': 'yes' if self.batch else 'no'}
        params.update(self.url_parameters(*args, privacy=privacy, **kwargs))
        auth = '{user}:{password}@'.format(user=NotifyClickSend.quote(self.user, safe=''), password=self.pprint(self.password, privacy, mode=PrivacyMode.Secret, safe=''))
        return '{schema}://{auth}{targets}?{params}'.format(schema=self.secure_protocol, auth=auth, targets='/'.join([NotifyClickSend.quote(x, safe='') for x in self.targets]), params=NotifyClickSend.urlencode(params))

    def __len__(self):
        if False:
            while True:
                i = 10
        '\n        Returns the number of targets associated with this notification\n        '
        batch_size = 1 if not self.batch else self.default_batch_size
        targets = len(self.targets)
        if batch_size > 1:
            targets = int(targets / batch_size) + (1 if targets % batch_size else 0)
        return targets

    @staticmethod
    def parse_url(url):
        if False:
            i = 10
            return i + 15
        '\n        Parses the URL and returns enough arguments that can allow\n        us to re-instantiate this object.\n\n        '
        results = NotifyBase.parse_url(url, verify_host=False)
        if not results:
            return results
        results['targets'] = [NotifyClickSend.unquote(results['host'])]
        results['targets'].extend(NotifyClickSend.split_path(results['fullpath']))
        results['batch'] = parse_bool(results['qsd'].get('batch', False))
        if 'to' in results['qsd'] and len(results['qsd']['to']):
            results['targets'] += NotifyClickSend.parse_phone_no(results['qsd']['to'])
        return results