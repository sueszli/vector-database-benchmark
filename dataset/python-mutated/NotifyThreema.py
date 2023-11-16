import requests
from itertools import chain
from .NotifyBase import NotifyBase
from ..common import NotifyType
from ..utils import is_phone_no
from ..utils import validate_regex
from ..utils import is_email
from ..URLBase import PrivacyMode
from ..utils import parse_list
from ..AppriseLocale import gettext_lazy as _

class ThreemaRecipientTypes:
    """
    The supported recipient specifiers
    """
    THREEMA_ID = 'to'
    PHONE = 'phone'
    EMAIL = 'email'

class NotifyThreema(NotifyBase):
    """
    A wrapper for Threema Gateway Notifications
    """
    service_name = 'Threema Gateway'
    service_url = 'https://gateway.threema.ch/'
    secure_protocol = 'threema'
    setup_url = 'https://github.com/caronc/apprise/wiki/Notify_threema'
    notify_url = 'https://msgapi.threema.ch/send_simple'
    body_maxlen = 3500
    title_maxlen = 0
    templates = ('{schema}://{gateway_id}@{secret}/{targets}',)
    template_tokens = dict(NotifyBase.template_tokens, **{'gateway_id': {'name': _('Gateway ID'), 'type': 'string', 'private': True, 'required': True, 'map_to': 'user'}, 'secret': {'name': _('API Secret'), 'type': 'string', 'private': True, 'required': True}, 'target_phone': {'name': _('Target Phone No'), 'type': 'string', 'prefix': '+', 'regex': ('^[0-9\\s)(+-]+$', 'i'), 'map_to': 'targets'}, 'target_email': {'name': _('Target Email'), 'type': 'string', 'map_to': 'targets'}, 'target_threema_id': {'name': _('Target Threema ID'), 'type': 'string', 'map_to': 'targets'}, 'targets': {'name': _('Targets'), 'type': 'list:string', 'required': True}})
    template_args = dict(NotifyBase.template_args, **{'to': {'alias_of': 'targets'}, 'from': {'alias_of': 'gateway_id'}, 'gwid': {'alias_of': 'gateway_id'}, 'secret': {'alias_of': 'secret'}})

    def __init__(self, secret=None, targets=None, **kwargs):
        if False:
            print('Hello World!')
        '\n        Initialize Threema Gateway Object\n        '
        super().__init__(**kwargs)
        if not self.user:
            msg = 'Threema Gateway ID must be specified'
            self.logger.warning(msg)
            raise TypeError(msg)
        if len(self.user) != 8:
            msg = 'Threema Gateway ID must be 8 characters in length'
            self.logger.warning(msg)
            raise TypeError(msg)
        self.secret = validate_regex(secret)
        if not self.secret:
            msg = 'An invalid Threema API Secret ({}) was specified'.format(secret)
            self.logger.warning(msg)
            raise TypeError(msg)
        self.targets = list()
        self.invalid_targets = list()
        for target in parse_list(targets, allow_whitespace=False):
            if len(target) == 8:
                self.targets.append((ThreemaRecipientTypes.THREEMA_ID, target))
                continue
            result = is_email(target)
            if result:
                self.targets.append((ThreemaRecipientTypes.EMAIL, result['full_email']))
                continue
            result = is_phone_no(target)
            if result:
                self.targets.append((ThreemaRecipientTypes.PHONE, result['full']))
                continue
            self.logger.warning('Dropped invalid user/email/phone ({}) specified'.format(target))
            self.invalid_targets.append(target)
        return

    def send(self, body, title='', notify_type=NotifyType.INFO, **kwargs):
        if False:
            return 10
        '\n        Perform Threema Gateway Notification\n        '
        if len(self.targets) == 0:
            self.logger.warning('There were no Threema Gateway targets to notify')
            return False
        has_error = False
        headers = {'User-Agent': self.app_id, 'Content-Type': 'application/x-www-form-urlencoded; charset=utf-8', 'Accept': '*/*'}
        _payload = {'secret': self.secret, 'from': self.user, 'text': body.encode('utf-8')}
        targets = list(self.targets)
        while len(targets):
            (key, target) = targets.pop(0)
            payload = _payload.copy()
            payload[key] = target
            self.logger.debug('Threema Gateway GET URL: {} (cert_verify={})'.format(self.notify_url, self.verify_certificate))
            self.logger.debug('Threema Gateway Payload: {}'.format(payload))
            self.throttle()
            try:
                r = requests.post(self.notify_url, params=payload, headers=headers, verify=self.verify_certificate, timeout=self.request_timeout)
                if r.status_code != requests.codes.ok:
                    status_str = NotifyThreema.http_response_code_lookup(r.status_code)
                    self.logger.warning('Failed to send Threema Gateway notification to {}: {}{}error={}'.format(target, status_str, ', ' if status_str else '', r.status_code))
                    self.logger.debug('Response Details:\r\n{}'.format(r.content))
                    has_error = True
                    continue
                self.logger.info('Sent Threema Gateway notification to %s' % target)
            except requests.RequestException as e:
                self.logger.warning('A Connection error occurred sending Threema Gateway:%s notification' % target)
                self.logger.debug('Socket Exception: %s' % str(e))
                has_error = True
                continue
        return not has_error

    def url(self, privacy=False, *args, **kwargs):
        if False:
            return 10
        '\n        Returns the URL built dynamically based on specified arguments.\n        '
        params = self.url_parameters(*args, privacy=privacy, **kwargs)
        schemaStr = '{schema}://{gatewayid}@{secret}/{targets}?{params}'
        return schemaStr.format(schema=self.secure_protocol, gatewayid=NotifyThreema.quote(self.user), secret=self.pprint(self.secret, privacy, mode=PrivacyMode.Secret, safe=''), targets='/'.join(chain([NotifyThreema.quote(x[1], safe='@+') for x in self.targets], [NotifyThreema.quote(x, safe='@+') for x in self.invalid_targets])), params=NotifyThreema.urlencode(params))

    def __len__(self):
        if False:
            return 10
        '\n        Returns the number of targets associated with this notification\n        '
        targets = len(self.targets)
        return targets if targets > 0 else 1

    @staticmethod
    def parse_url(url):
        if False:
            for i in range(10):
                print('nop')
        '\n        Parses the URL and returns enough arguments that can allow\n        us to re-instantiate this object.\n        '
        results = NotifyBase.parse_url(url, verify_host=False)
        if not results:
            return results
        results['targets'] = list()
        if 'secret' in results['qsd'] and len(results['qsd']['secret']):
            results['secret'] = NotifyThreema.unquote(results['qsd']['secret'])
        else:
            results['secret'] = NotifyThreema.unquote(results['host'])
        results['targets'] += NotifyThreema.split_path(results['fullpath'])
        if 'from' in results['qsd'] and len(results['qsd']['from']):
            results['user'] = NotifyThreema.unquote(results['qsd']['from'])
        elif 'gwid' in results['qsd'] and len(results['qsd']['gwid']):
            results['user'] = NotifyThreema.unquote(results['qsd']['gwid'])
        if 'to' in results['qsd'] and len(results['qsd']['to']):
            results['targets'] += NotifyThreema.parse_list(results['qsd']['to'], allow_whitespace=False)
        return results