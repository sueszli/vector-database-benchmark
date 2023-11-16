import requests
from .NotifyBase import NotifyBase
from ..common import NotifyType
from ..utils import is_phone_no
from ..utils import parse_phone_no
from ..utils import validate_regex
from ..AppriseLocale import gettext_lazy as _

class NotifyMessageBird(NotifyBase):
    """
    A wrapper for MessageBird Notifications
    """
    service_name = 'MessageBird'
    service_url = 'https://messagebird.com'
    secure_protocol = 'msgbird'
    setup_url = 'https://github.com/caronc/apprise/wiki/Notify_messagebird'
    notify_url = 'https://rest.messagebird.com/messages'
    body_maxlen = 160
    title_maxlen = 0
    templates = ('{schema}://{apikey}/{source}', '{schema}://{apikey}/{source}/{targets}')
    template_tokens = dict(NotifyBase.template_tokens, **{'apikey': {'name': _('API Key'), 'type': 'string', 'required': True, 'private': True, 'regex': ('^[a-z0-9]{25}$', 'i')}, 'source': {'name': _('Source Phone No'), 'type': 'string', 'prefix': '+', 'required': True, 'regex': ('^[0-9\\s)(+-]+$', 'i')}, 'target_phone': {'name': _('Target Phone No'), 'type': 'string', 'prefix': '+', 'regex': ('^[0-9\\s)(+-]+$', 'i'), 'map_to': 'targets'}, 'targets': {'name': _('Targets'), 'type': 'list:string'}})
    template_args = dict(NotifyBase.template_args, **{'to': {'alias_of': 'targets'}, 'from': {'alias_of': 'source'}})

    def __init__(self, apikey, source, targets=None, **kwargs):
        if False:
            return 10
        '\n        Initialize MessageBird Object\n        '
        super().__init__(**kwargs)
        self.apikey = validate_regex(apikey, *self.template_tokens['apikey']['regex'])
        if not self.apikey:
            msg = 'An invalid MessageBird API Key ({}) was specified.'.format(apikey)
            self.logger.warning(msg)
            raise TypeError(msg)
        result = is_phone_no(source)
        if not result:
            msg = 'The MessageBird source specified ({}) is invalid.'.format(source)
            self.logger.warning(msg)
            raise TypeError(msg)
        self.source = result['full']
        self.targets = list()
        targets = parse_phone_no(targets)
        if not targets:
            self.targets.append(self.source)
            return
        for target in targets:
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
        '\n        Perform MessageBird Notification\n        '
        if len(self.targets) == 0:
            self.logger.warning('There were no MessageBird targets to notify.')
            return False
        has_error = False
        headers = {'User-Agent': self.app_id, 'Content-Type': 'application/x-www-form-urlencoded', 'Authorization': 'AccessKey {}'.format(self.apikey)}
        payload = {'originator': '+{}'.format(self.source), 'recipients': None, 'body': body}
        targets = list(self.targets)
        while len(targets):
            target = targets.pop(0)
            payload['recipients'] = '+{}'.format(target)
            self.logger.debug('MessageBird POST URL: {} (cert_verify={})'.format(self.notify_url, self.verify_certificate))
            self.logger.debug('MessageBird Payload: {}'.format(payload))
            self.throttle()
            try:
                r = requests.post(self.notify_url, data=payload, headers=headers, verify=self.verify_certificate, timeout=self.request_timeout)
                if r.status_code not in (requests.codes.ok, requests.codes.created):
                    status_str = NotifyMessageBird.http_response_code_lookup(r.status_code)
                    self.logger.warning('Failed to send MessageBird notification to {}: {}{}error={}.'.format(','.join(target), status_str, ', ' if status_str else '', r.status_code))
                    self.logger.debug('Response Details:\r\n{}'.format(r.content))
                    has_error = True
                    continue
                else:
                    self.logger.info('Sent MessageBird notification to {}.'.format(target))
            except requests.RequestException as e:
                self.logger.warning('A Connection error occurred sending MessageBird:%s ' % target + 'notification.')
                self.logger.debug('Socket Exception: %s' % str(e))
                has_error = True
                continue
        return not has_error

    def url(self, privacy=False, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns the URL built dynamically based on specified arguments.\n        '
        params = self.url_parameters(*args, privacy=privacy, **kwargs)
        return '{schema}://{apikey}/{source}/{targets}/?{params}'.format(schema=self.secure_protocol, apikey=self.pprint(self.apikey, privacy, safe=''), source=self.source, targets='/'.join([NotifyMessageBird.quote(x, safe='') for x in self.targets]), params=NotifyMessageBird.urlencode(params))

    def __len__(self):
        if False:
            i = 10
            return i + 15
        '\n        Returns the number of targets associated with this notification\n        '
        targets = len(self.targets)
        return targets if targets > 0 else 1

    @staticmethod
    def parse_url(url):
        if False:
            return 10
        '\n        Parses the URL and returns enough arguments that can allow\n        us to re-instantiate this object.\n\n        '
        results = NotifyBase.parse_url(url, verify_host=False)
        if not results:
            return results
        results['targets'] = NotifyMessageBird.split_path(results['fullpath'])
        try:
            results['source'] = results['targets'].pop(0)
        except IndexError:
            results['source'] = None
        results['apikey'] = NotifyMessageBird.unquote(results['host'])
        if 'to' in results['qsd'] and len(results['qsd']['to']):
            results['targets'] += NotifyMessageBird.parse_phone_no(results['qsd']['to'])
        if 'from' in results['qsd'] and len(results['qsd']['from']):
            results['source'] = NotifyMessageBird.unquote(results['qsd']['from'])
        return results