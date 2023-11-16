import requests
from .NotifyBase import NotifyBase
from ..URLBase import PrivacyMode
from ..common import NotifyType
from ..utils import is_phone_no
from ..utils import parse_phone_no
from ..utils import parse_bool
from ..utils import validate_regex
from ..AppriseLocale import gettext_lazy as _

class BurstSMSCountryCode:
    AU = 'au'
    NZ = 'nz'
    UK = 'gb'
    US = 'us'
BURST_SMS_COUNTRY_CODES = (BurstSMSCountryCode.AU, BurstSMSCountryCode.NZ, BurstSMSCountryCode.UK, BurstSMSCountryCode.US)

class NotifyBurstSMS(NotifyBase):
    """
    A wrapper for Burst SMS Notifications
    """
    service_name = 'Burst SMS'
    service_url = 'https://burstsms.com/'
    secure_protocol = 'burstsms'
    default_batch_size = 500
    setup_url = 'https://github.com/caronc/apprise/wiki/Notify_burst_sms'
    notify_url = 'https://api.transmitsms.com/send-sms.json'
    body_maxlen = 160
    title_maxlen = 0
    templates = ('{schema}://{apikey}:{secret}@{sender_id}/{targets}',)
    template_tokens = dict(NotifyBase.template_tokens, **{'apikey': {'name': _('API Key'), 'type': 'string', 'required': True, 'regex': ('^[a-z0-9]+$', 'i'), 'private': True}, 'secret': {'name': _('API Secret'), 'type': 'string', 'private': True, 'required': True, 'regex': ('^[a-z0-9]+$', 'i')}, 'sender_id': {'name': _('Sender ID'), 'type': 'string', 'required': True, 'map_to': 'source'}, 'target_phone': {'name': _('Target Phone No'), 'type': 'string', 'prefix': '+', 'regex': ('^[0-9\\s)(+-]+$', 'i'), 'map_to': 'targets'}, 'targets': {'name': _('Targets'), 'type': 'list:string', 'required': True}})
    template_args = dict(NotifyBase.template_args, **{'to': {'alias_of': 'targets'}, 'from': {'alias_of': 'sender_id'}, 'key': {'alias_of': 'apikey'}, 'secret': {'alias_of': 'secret'}, 'country': {'name': _('Country'), 'type': 'choice:string', 'values': BURST_SMS_COUNTRY_CODES, 'default': BurstSMSCountryCode.US}, 'validity': {'name': _('validity'), 'type': 'int', 'default': 0}, 'batch': {'name': _('Batch Mode'), 'type': 'bool', 'default': False}})

    def __init__(self, apikey, secret, source, targets=None, country=None, validity=None, batch=None, **kwargs):
        if False:
            return 10
        '\n        Initialize Burst SMS Object\n        '
        super().__init__(**kwargs)
        self.apikey = validate_regex(apikey, *self.template_tokens['apikey']['regex'])
        if not self.apikey:
            msg = 'An invalid Burst SMS API Key ({}) was specified.'.format(apikey)
            self.logger.warning(msg)
            raise TypeError(msg)
        self.secret = validate_regex(secret, *self.template_tokens['secret']['regex'])
        if not self.secret:
            msg = 'An invalid Burst SMS API Secret ({}) was specified.'.format(secret)
            self.logger.warning(msg)
            raise TypeError(msg)
        if not country:
            self.country = self.template_args['country']['default']
        else:
            self.country = country.lower().strip()
            if country not in BURST_SMS_COUNTRY_CODES:
                msg = 'An invalid Burst SMS country ({}) was specified.'.format(country)
                self.logger.warning(msg)
                raise TypeError(msg)
        self.validity = self.template_args['validity']['default']
        if validity:
            try:
                self.validity = int(validity)
            except (ValueError, TypeError):
                msg = 'The Burst SMS Validity specified ({}) is invalid.'.format(validity)
                self.logger.warning(msg)
                raise TypeError(msg)
        self.batch = self.template_args['batch']['default'] if batch is None else batch
        self.source = validate_regex(source)
        if not self.source:
            msg = 'The Account Sender ID specified ({}) is invalid.'.format(source)
            self.logger.warning(msg)
            raise TypeError(msg)
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
            print('Hello World!')
        '\n        Perform Burst SMS Notification\n        '
        if not self.targets:
            self.logger.warning('There are no valid Burst SMS targets to notify.')
            return False
        has_error = False
        headers = {'User-Agent': self.app_id, 'Accept': 'application/json'}
        auth = (self.apikey, self.secret)
        payload = {'countrycode': self.country, 'message': body, 'from': self.source, 'to': None}
        batch_size = 1 if not self.batch else self.default_batch_size
        targets = list(self.targets)
        for index in range(0, len(targets), batch_size):
            payload['to'] = ','.join(self.targets[index:index + batch_size])
            self.logger.debug('Burst SMS POST URL: {} (cert_verify={})'.format(self.notify_url, self.verify_certificate))
            self.logger.debug('Burst SMS Payload: {}'.format(payload))
            self.throttle()
            try:
                r = requests.post(self.notify_url, data=payload, headers=headers, auth=auth, verify=self.verify_certificate, timeout=self.request_timeout)
                if r.status_code != requests.codes.ok:
                    status_str = NotifyBurstSMS.http_response_code_lookup(r.status_code)
                    self.logger.warning('Failed to send Burst SMS notification to {} target(s): {}{}error={}.'.format(len(self.targets[index:index + batch_size]), status_str, ', ' if status_str else '', r.status_code))
                    self.logger.debug('Response Details:\r\n{}'.format(r.content))
                    has_error = True
                    continue
                else:
                    self.logger.info('Sent Burst SMS notification to %d target(s).' % len(self.targets[index:index + batch_size]))
            except requests.RequestException as e:
                self.logger.warning('A Connection error occurred sending Burst SMS notification to %d target(s).' % len(self.targets[index:index + batch_size]))
                self.logger.debug('Socket Exception: %s' % str(e))
                has_error = True
                continue
        return not has_error

    def url(self, privacy=False, *args, **kwargs):
        if False:
            return 10
        '\n        Returns the URL built dynamically based on specified arguments.\n        '
        params = {'country': self.country, 'batch': 'yes' if self.batch else 'no'}
        if self.validity:
            params['validity'] = str(self.validity)
        params.update(self.url_parameters(*args, privacy=privacy, **kwargs))
        return '{schema}://{key}:{secret}@{source}/{targets}/?{params}'.format(schema=self.secure_protocol, key=self.pprint(self.apikey, privacy, safe=''), secret=self.pprint(self.secret, privacy, mode=PrivacyMode.Secret, safe=''), source=NotifyBurstSMS.quote(self.source, safe=''), targets='/'.join([NotifyBurstSMS.quote(x, safe='') for x in self.targets]), params=NotifyBurstSMS.urlencode(params))

    def __len__(self):
        if False:
            print('Hello World!')
        '\n        Returns the number of targets associated with this notification\n        '
        batch_size = 1 if not self.batch else self.default_batch_size
        targets = len(self.targets)
        if batch_size > 1:
            targets = int(targets / batch_size) + (1 if targets % batch_size else 0)
        return targets if targets > 0 else 1

    @staticmethod
    def parse_url(url):
        if False:
            print('Hello World!')
        '\n        Parses the URL and returns enough arguments that can allow\n        us to re-instantiate this object.\n\n        '
        results = NotifyBase.parse_url(url, verify_host=False)
        if not results:
            return results
        results['source'] = NotifyBurstSMS.unquote(results['host'])
        results['targets'] = NotifyBurstSMS.split_path(results['fullpath'])
        results['apikey'] = NotifyBurstSMS.unquote(results['user'])
        results['secret'] = NotifyBurstSMS.unquote(results['password'])
        if 'key' in results['qsd'] and len(results['qsd']['key']):
            results['apikey'] = NotifyBurstSMS.unquote(results['qsd']['key'])
        if 'secret' in results['qsd'] and len(results['qsd']['secret']):
            results['secret'] = NotifyBurstSMS.unquote(results['qsd']['secret'])
        if 'from' in results['qsd'] and len(results['qsd']['from']):
            results['source'] = NotifyBurstSMS.unquote(results['qsd']['from'])
        if 'source' in results['qsd'] and len(results['qsd']['source']):
            results['source'] = NotifyBurstSMS.unquote(results['qsd']['source'])
        if 'country' in results['qsd'] and len(results['qsd']['country']):
            results['country'] = NotifyBurstSMS.unquote(results['qsd']['country'])
        if 'validity' in results['qsd'] and len(results['qsd']['validity']):
            results['validity'] = NotifyBurstSMS.unquote(results['qsd']['validity'])
        if 'batch' in results['qsd'] and len(results['qsd']['batch']):
            results['batch'] = parse_bool(results['qsd']['batch'])
        if 'to' in results['qsd'] and len(results['qsd']['to']):
            results['targets'] += NotifyBurstSMS.parse_phone_no(results['qsd']['to'])
        return results