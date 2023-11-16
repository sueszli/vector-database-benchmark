import re
import requests
from json import dumps
from .NotifyBase import NotifyBase
from ..common import NotifyType
from ..utils import is_phone_no
from ..utils import parse_phone_no, parse_bool
from ..utils import validate_regex
from ..AppriseLocale import gettext_lazy as _

class MSG91PayloadField:
    """
    Identifies the fields available in the JSON Payload
    """
    BODY = 'body'
    MESSAGETYPE = 'type'
RESERVED_KEYWORDS = ('mobiles',)

class NotifyMSG91(NotifyBase):
    """
    A wrapper for MSG91 Notifications
    """
    service_name = 'MSG91'
    service_url = 'https://msg91.com'
    secure_protocol = 'msg91'
    setup_url = 'https://github.com/caronc/apprise/wiki/Notify_msg91'
    notify_url = 'https://control.msg91.com/api/v5/flow/'
    body_maxlen = 160
    title_maxlen = 0
    component_key_re = re.compile('(?P<key>((?P<id>[a-z0-9_-])?|(?P<map>body|type)))', re.IGNORECASE)
    templates = ('{schema}://{template}@{authkey}/{targets}',)
    template_tokens = dict(NotifyBase.template_tokens, **{'template': {'name': _('Template ID'), 'type': 'string', 'required': True, 'private': True, 'regex': ('^[a-z0-9 _-]+$', 'i')}, 'authkey': {'name': _('Authentication Key'), 'type': 'string', 'required': True, 'private': True, 'regex': ('^[a-z0-9]+$', 'i')}, 'target_phone': {'name': _('Target Phone No'), 'type': 'string', 'prefix': '+', 'regex': ('^[0-9\\s)(+-]+$', 'i'), 'map_to': 'targets'}, 'targets': {'name': _('Targets'), 'type': 'list:string', 'required': True}})
    template_args = dict(NotifyBase.template_args, **{'to': {'alias_of': 'targets'}, 'short_url': {'name': _('Short URL'), 'type': 'bool', 'default': False}})
    template_kwargs = {'template_mapping': {'name': _('Template Mapping'), 'prefix': ':'}}

    def __init__(self, template, authkey, targets=None, short_url=None, template_mapping=None, **kwargs):
        if False:
            return 10
        '\n        Initialize MSG91 Object\n        '
        super().__init__(**kwargs)
        self.authkey = validate_regex(authkey, *self.template_tokens['authkey']['regex'])
        if not self.authkey:
            msg = 'An invalid MSG91 Authentication Key ({}) was specified.'.format(authkey)
            self.logger.warning(msg)
            raise TypeError(msg)
        self.template = validate_regex(template, *self.template_tokens['template']['regex'])
        if not self.template:
            msg = 'An invalid MSG91 Template ID ({}) was specified.'.format(template)
            self.logger.warning(msg)
            raise TypeError(msg)
        if short_url is None:
            self.short_url = self.template_args['short_url']['default']
        else:
            self.short_url = parse_bool(short_url)
        self.targets = list()
        for target in parse_phone_no(targets):
            result = is_phone_no(target)
            if not result:
                self.logger.warning('Dropped invalid phone # ({}) specified.'.format(target))
                continue
            self.targets.append(result['full'])
        self.template_mapping = {}
        if template_mapping:
            self.template_mapping.update(template_mapping)
        return

    def send(self, body, title='', notify_type=NotifyType.INFO, **kwargs):
        if False:
            return 10
        '\n        Perform MSG91 Notification\n        '
        if len(self.targets) == 0:
            self.logger.warning('There were no MSG91 targets to notify.')
            return False
        headers = {'User-Agent': self.app_id, 'Content-Type': 'application/json', 'authkey': self.authkey}
        recipient_payload = {'mobiles': None, MSG91PayloadField.BODY: body, MSG91PayloadField.MESSAGETYPE: notify_type}
        for (key, value) in self.template_mapping.items():
            if key in RESERVED_KEYWORDS:
                self.logger.warning('Ignoring MSG91 custom payload entry %s', key)
                continue
            if key in recipient_payload:
                if not value:
                    del recipient_payload[key]
                else:
                    recipient_payload[value] = recipient_payload[key]
                    del recipient_payload[key]
            else:
                recipient_payload[key] = value
        recipients = []
        for target in self.targets:
            recipient = recipient_payload.copy()
            recipient['mobiles'] = target
            recipients.append(recipient)
        payload = {'template_id': self.template, 'short_url': 1 if self.short_url else 0, 'recipients': recipients}
        self.logger.debug('MSG91 POST URL: {} (cert_verify={})'.format(self.notify_url, self.verify_certificate))
        self.logger.debug('MSG91 Payload: {}'.format(payload))
        self.throttle()
        try:
            r = requests.post(self.notify_url, data=dumps(payload), headers=headers, verify=self.verify_certificate, timeout=self.request_timeout)
            if r.status_code != requests.codes.ok:
                status_str = NotifyMSG91.http_response_code_lookup(r.status_code)
                self.logger.warning('Failed to send MSG91 notification to {}: {}{}error={}.'.format(','.join(self.targets), status_str, ', ' if status_str else '', r.status_code))
                self.logger.debug('Response Details:\r\n{}'.format(r.content))
                return False
            else:
                self.logger.info('Sent MSG91 notification to %s.' % ','.join(self.targets))
        except requests.RequestException as e:
            self.logger.warning('A Connection error occurred sending MSG91:%s notification.' % ','.join(self.targets))
            self.logger.debug('Socket Exception: %s' % str(e))
            return False
        return True

    def url(self, privacy=False, *args, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Returns the URL built dynamically based on specified arguments.\n        '
        params = {'short_url': str(self.short_url)}
        params.update(self.url_parameters(*args, privacy=privacy, **kwargs))
        params.update({':{}'.format(k): v for (k, v) in self.template_mapping.items()})
        return '{schema}://{template}@{authkey}/{targets}/?{params}'.format(schema=self.secure_protocol, template=self.pprint(self.template, privacy, safe=''), authkey=self.pprint(self.authkey, privacy, safe=''), targets='/'.join([NotifyMSG91.quote(x, safe='') for x in self.targets]), params=NotifyMSG91.urlencode(params))

    def __len__(self):
        if False:
            print('Hello World!')
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
        results['targets'] = NotifyMSG91.split_path(results['fullpath'])
        results['authkey'] = NotifyMSG91.unquote(results['host'])
        results['template'] = NotifyMSG91.unquote(results['user'])
        if 'short_url' in results['qsd'] and len(results['qsd']['short_url']):
            results['short_url'] = parse_bool(results['qsd']['short_url'])
        if 'to' in results['qsd'] and len(results['qsd']['to']):
            results['targets'] += NotifyMSG91.parse_phone_no(results['qsd']['to'])
        results['template_mapping'] = {NotifyMSG91.unquote(x): NotifyMSG91.unquote(y) for (x, y) in results['qsd:'].items()}
        return results