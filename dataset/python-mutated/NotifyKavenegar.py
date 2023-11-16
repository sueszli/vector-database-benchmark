import requests
from json import loads
from .NotifyBase import NotifyBase
from ..common import NotifyType
from ..utils import is_phone_no
from ..utils import parse_phone_no
from ..utils import validate_regex
from ..AppriseLocale import gettext_lazy as _
KAVENEGAR_HTTP_ERROR_MAP = {200: 'The request was approved', 400: 'Parameters are incomplete', 401: 'Account has been disabled', 402: 'The operation failed', 403: 'The API Key is invalid', 404: 'The method is unknown', 405: 'The GET/POST request is wrong', 406: 'Invalid mandatory parameters sent', 407: 'You canot access the information you want', 409: 'The server is unable to response', 411: 'The recipient is invalid', 412: 'The sender is invalid', 413: 'Message empty or message length exceeded', 414: 'The number of recipients is more than 200', 415: 'The start index is larger then the total', 416: 'The source IP of the service does not match the settings', 417: 'The submission date is incorrect, either expired or not in the correct format', 418: 'Your account credit is insufficient', 422: 'Data cannot be processed due to invalid characters', 501: 'SMS can only be sent to the account holder number'}

class NotifyKavenegar(NotifyBase):
    """
    A wrapper for Kavenegar Notifications
    """
    service_name = 'Kavenegar'
    service_url = 'https://kavenegar.com/'
    secure_protocol = 'kavenegar'
    request_rate_per_sec = 0.2
    setup_url = 'https://github.com/caronc/apprise/wiki/Notify_kavenegar'
    notify_url = 'http://api.kavenegar.com/v1/{apikey}/sms/send.json'
    body_maxlen = 160
    title_maxlen = 0
    templates = ('{schema}://{apikey}/{targets}', '{schema}://{source}@{apikey}/{targets}')
    template_tokens = dict(NotifyBase.template_tokens, **{'apikey': {'name': _('API Key'), 'type': 'string', 'required': True, 'private': True, 'regex': ('^[a-z0-9]+$', 'i')}, 'source': {'name': _('Source Phone No'), 'type': 'string', 'prefix': '+', 'regex': ('^[0-9\\s)(+-]+$', 'i')}, 'target_phone': {'name': _('Target Phone No'), 'type': 'string', 'prefix': '+', 'regex': ('^[0-9\\s)(+-]+$', 'i'), 'map_to': 'targets'}, 'targets': {'name': _('Targets'), 'type': 'list:string', 'required': True}})
    template_args = dict(NotifyBase.template_args, **{'to': {'alias_of': 'targets'}, 'from': {'alias_of': 'source'}})

    def __init__(self, apikey, source=None, targets=None, **kwargs):
        if False:
            print('Hello World!')
        '\n        Initialize Kavenegar Object\n        '
        super().__init__(**kwargs)
        self.apikey = validate_regex(apikey, *self.template_tokens['apikey']['regex'])
        if not self.apikey:
            msg = 'An invalid Kavenegar API Key ({}) was specified.'.format(apikey)
            self.logger.warning(msg)
            raise TypeError(msg)
        self.source = None
        if source is not None:
            result = is_phone_no(source)
            if not result:
                msg = 'The Kavenegar source specified ({}) is invalid.'.format(source)
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
            return 10
        '\n        Sends SMS Message\n        '
        if len(self.targets) == 0:
            self.logger.warning('There were no Kavenegar targets to notify.')
            return False
        has_error = False
        headers = {'User-Agent': self.app_id, 'Accept': 'application/json'}
        url = self.notify_url.format(apikey=self.apikey)
        targets = list(self.targets)
        while len(targets):
            target = targets.pop(0)
            payload = {'receptor': target, 'message': body}
            if self.source:
                payload['sender'] = self.source
            self.logger.debug('Kavenegar POST URL: {} (cert_verify={})'.format(url, self.verify_certificate))
            self.logger.debug('Kavenegar Payload: {}'.format(payload))
            self.throttle()
            try:
                r = requests.post(url, params=payload, headers=headers, verify=self.verify_certificate, timeout=self.request_timeout)
                if r.status_code not in (requests.codes.created, requests.codes.ok):
                    status_str = NotifyBase.http_response_code_lookup(r.status_code, KAVENEGAR_HTTP_ERROR_MAP)
                    try:
                        json_response = loads(r.content)
                        status_str = json_response.get('message', status_str)
                    except (AttributeError, TypeError, ValueError):
                        pass
                    self.logger.warning('Failed to send Kavenegar SMS notification to {}: {}{}error={}.'.format(target, status_str, ', ' if status_str else '', r.status_code))
                    self.logger.debug('Response Details:\r\n{}'.format(r.content))
                    has_error = True
                    continue
                self.logger.info('Sent Kavenegar SMS notification to {}.'.format(target))
                self.logger.debug('Response Details:\r\n{}'.format(r.content))
            except requests.RequestException as e:
                self.logger.warning('A Connection error occurred sending Kavenegar:%s ' % ', '.join(self.targets) + 'notification.')
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
        return '{schema}://{source}{apikey}/{targets}?{params}'.format(schema=self.secure_protocol, source='' if not self.source else '{}@'.format(self.source), apikey=self.pprint(self.apikey, privacy, safe=''), targets='/'.join([NotifyKavenegar.quote(x, safe='') for x in self.targets]), params=NotifyKavenegar.urlencode(params))

    def __len__(self):
        if False:
            for i in range(10):
                print('nop')
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
        if results.get('user', None):
            results['source'] = results['user']
        results['targets'] = NotifyKavenegar.split_path(results['fullpath'])
        results['apikey'] = NotifyKavenegar.unquote(results['host'])
        if 'to' in results['qsd'] and len(results['qsd']['to']):
            results['targets'] += NotifyKavenegar.parse_phone_no(results['qsd']['to'])
        if 'from' in results['qsd'] and len(results['qsd']['from']):
            results['source'] = NotifyKavenegar.unquote(results['qsd']['from'])
        return results