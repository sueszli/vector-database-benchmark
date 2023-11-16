import requests
from json import loads
from .NotifyBase import NotifyBase
from ..common import NotifyType
from ..utils import is_phone_no
from ..utils import is_email
from ..utils import parse_phone_no
from ..AppriseLocale import gettext_lazy as _

class NotifyVoipms(NotifyBase):
    """
    A wrapper for Voipms Notifications
    """
    service_name = 'VoIPms'
    service_url = 'https://voip.ms'
    secure_protocol = 'voipms'
    setup_url = 'https://github.com/caronc/apprise/wiki/Notify_voipms'
    notify_url = 'https://voip.ms/api/v1/rest.php'
    body_maxlen = 160
    title_maxlen = 0
    templates = ('{schema}://{password}:{email}/{from_phone}/{targets}',)
    template_tokens = dict(NotifyBase.template_tokens, **{'email': {'name': _('User Email'), 'type': 'string', 'required': True}, 'password': {'name': _('Password'), 'type': 'string', 'private': True, 'required': True}, 'from_phone': {'name': _('From Phone No'), 'type': 'string', 'regex': ('^\\+?[0-9\\s)(+-]+$', 'i'), 'map_to': 'source'}, 'target_phone': {'name': _('Target Phone No'), 'type': 'string', 'prefix': '+', 'regex': ('^[0-9\\s)(+-]+$', 'i'), 'map_to': 'targets'}, 'targets': {'name': _('Targets'), 'type': 'list:string', 'required': True}})
    template_args = dict(NotifyBase.template_args, **{'to': {'alias_of': 'targets'}, 'from': {'alias_of': 'from_phone'}})

    def __init__(self, email, source=None, targets=None, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Initialize Voipms Object\n        '
        super().__init__(**kwargs)
        if self.password is None:
            msg = 'Password has to be specified.'
            self.logger.warning(msg)
            raise TypeError(msg)
        result = is_email(email)
        if not result:
            msg = 'An invalid Voipms user email: ({}) was specified.'.format(email)
            self.logger.warning(msg)
            raise TypeError(msg)
        self.email = result['full_email']
        result = is_phone_no(source)
        if not result:
            msg = 'An invalid Voipms source phone # ({}) was specified.'.format(source)
            self.logger.warning(msg)
            raise TypeError(msg)
        if result['country'] and result['country'] != '1':
            msg = 'Voipms only supports +1 country code ({}) was specified.'.format(source)
            self.logger.warning(msg)
            raise TypeError(msg)
        self.source = result['area'] + result['line']
        self.targets = list()
        if targets:
            for target in parse_phone_no(targets):
                result = is_phone_no(target)
                if result['country'] != '1':
                    self.logger.warning('Dropped invalid phone # ({}) specified.'.format(target))
                    continue
                self.targets.append(result['area'] + result['line'])
        else:
            self.targets.append(self.source)
        return

    def send(self, body, title='', notify_type=NotifyType.INFO, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Perform Voipms Notification\n        '
        if len(self.targets) == 0:
            self.logger.warning('There were no Voipms targets to notify.')
            return False
        has_error = False
        headers = {'User-Agent': self.app_id, 'Content-Type': 'application/x-www-form-urlencoded'}
        payload = {'api_username': self.email, 'api_password': self.password, 'did': self.source, 'message': body, 'method': 'sendSMS', 'dst': None}
        targets = list(self.targets)
        while len(targets):
            target = targets.pop(0)
            payload['dst'] = target
            self.logger.debug('Voipms GET URL: {} (cert_verify={})'.format(self.notify_url, self.verify_certificate))
            self.logger.debug('Voipms Payload: {}'.format(payload))
            self.throttle()
            response = {'status': 'unknown', 'message': ''}
            try:
                r = requests.get(self.notify_url, params=payload, headers=headers, verify=self.verify_certificate, timeout=self.request_timeout)
                try:
                    response = loads(r.content)
                except (AttributeError, TypeError, ValueError):
                    pass
                if r.status_code != requests.codes.ok:
                    status_str = NotifyVoipms.http_response_code_lookup(r.status_code)
                    self.logger.warning('Failed to send Voipms notification to {}: {}{}error={}.'.format(target, status_str, ', ' if status_str else '', r.status_code))
                    self.logger.debug('Response Details:\r\n{}'.format(r.content))
                    has_error = True
                    continue
                if response is not None and response['status'] != 'success':
                    self.logger.warning('Failed to send Voipms notification to {}: status: {}, message: {}'.format(target, response['status'], response['message']))
                    has_error = True
                    continue
                else:
                    self.logger.info('Sent Voipms notification to %s' % target)
            except requests.RequestException as e:
                self.logger.warning('A Connection error occurred sending Voipms:%s notification.' % target)
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
        schemaStr = '{schema}://{password}:{email}/{from_phone}/{targets}/?{params}'
        return schemaStr.format(schema=self.secure_protocol, email=self.email, password=self.pprint(self.password, privacy, safe=''), from_phone='1' + self.pprint(self.source, privacy, safe=''), targets='/'.join(['1' + NotifyVoipms.quote(x, safe='') for x in self.targets]), params=NotifyVoipms.urlencode(params))

    def __len__(self):
        if False:
            for i in range(10):
                print('nop')
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
        results['targets'] = NotifyVoipms.split_path(results['fullpath'])
        if 'from' in results['qsd'] and len(results['qsd']['from']):
            results['source'] = NotifyVoipms.unquote(results['qsd']['from'])
        elif results['targets']:
            results['source'] = results['targets'].pop(0)
        user = results['password']
        password = results['user']
        results['password'] = password
        results['user'] = user
        results['email'] = '{}@{}'.format(NotifyVoipms.unquote(user), NotifyVoipms.unquote(results['host']))
        if 'to' in results['qsd'] and len(results['qsd']['to']):
            results['targets'] += NotifyVoipms.parse_phone_no(results['qsd']['to'])
        return results