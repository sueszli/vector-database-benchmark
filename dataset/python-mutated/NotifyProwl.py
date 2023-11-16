import requests
from .NotifyBase import NotifyBase
from ..common import NotifyType
from ..utils import validate_regex
from ..AppriseLocale import gettext_lazy as _

class ProwlPriority:
    LOW = -2
    MODERATE = -1
    NORMAL = 0
    HIGH = 1
    EMERGENCY = 2
PROWL_PRIORITIES = {ProwlPriority.LOW: 'low', ProwlPriority.MODERATE: 'moderate', ProwlPriority.NORMAL: 'normal', ProwlPriority.HIGH: 'high', ProwlPriority.EMERGENCY: 'emergency'}
PROWL_PRIORITY_MAP = {'l': ProwlPriority.LOW, 'm': ProwlPriority.MODERATE, 'n': ProwlPriority.NORMAL, 'h': ProwlPriority.HIGH, 'e': ProwlPriority.EMERGENCY, '-2': ProwlPriority.LOW, '-1': ProwlPriority.MODERATE, '0': ProwlPriority.NORMAL, '1': ProwlPriority.HIGH, '2': ProwlPriority.EMERGENCY}
PROWL_HTTP_ERROR_MAP = {406: 'IP address has exceeded API limit', 409: 'Request not aproved.'}

class NotifyProwl(NotifyBase):
    """
    A wrapper for Prowl Notifications
    """
    service_name = 'Prowl'
    service_url = 'https://www.prowlapp.com/'
    secure_protocol = 'prowl'
    setup_url = 'https://github.com/caronc/apprise/wiki/Notify_prowl'
    notify_url = 'https://api.prowlapp.com/publicapi/add'
    request_rate_per_sec = 0
    body_maxlen = 10000
    title_maxlen = 1024
    templates = ('{schema}://{apikey}', '{schema}://{apikey}/{providerkey}')
    template_tokens = dict(NotifyBase.template_tokens, **{'apikey': {'name': _('API Key'), 'type': 'string', 'private': True, 'required': True, 'regex': ('^[A-Za-z0-9]{40}$', 'i')}, 'providerkey': {'name': _('Provider Key'), 'type': 'string', 'private': True, 'regex': ('^[A-Za-z0-9]{40}$', 'i')}})
    template_args = dict(NotifyBase.template_args, **{'priority': {'name': _('Priority'), 'type': 'choice:int', 'values': PROWL_PRIORITIES, 'default': ProwlPriority.NORMAL}})

    def __init__(self, apikey, providerkey=None, priority=None, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Initialize Prowl Object\n        '
        super().__init__(**kwargs)
        self.priority = NotifyProwl.template_args['priority']['default'] if not priority else next((v for (k, v) in PROWL_PRIORITY_MAP.items() if str(priority).lower().startswith(k)), NotifyProwl.template_args['priority']['default'])
        self.apikey = validate_regex(apikey, *self.template_tokens['apikey']['regex'])
        if not self.apikey:
            msg = 'An invalid Prowl API Key ({}) was specified.'.format(apikey)
            self.logger.warning(msg)
            raise TypeError(msg)
        if providerkey:
            self.providerkey = validate_regex(providerkey, *self.template_tokens['providerkey']['regex'])
            if not self.providerkey:
                msg = 'An invalid Prowl Provider Key ({}) was specified.'.format(providerkey)
                self.logger.warning(msg)
                raise TypeError(msg)
        else:
            self.providerkey = None
        return

    def send(self, body, title='', notify_type=NotifyType.INFO, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Perform Prowl Notification\n        '
        headers = {'User-Agent': self.app_id, 'Content-type': 'application/x-www-form-urlencoded'}
        payload = {'apikey': self.apikey, 'application': self.app_id, 'event': title, 'description': body, 'priority': self.priority}
        if self.providerkey:
            payload['providerkey'] = self.providerkey
        self.logger.debug('Prowl POST URL: %s (cert_verify=%r)' % (self.notify_url, self.verify_certificate))
        self.logger.debug('Prowl Payload: %s' % str(payload))
        self.throttle()
        try:
            r = requests.post(self.notify_url, data=payload, headers=headers, verify=self.verify_certificate, timeout=self.request_timeout)
            if r.status_code != requests.codes.ok:
                status_str = NotifyBase.http_response_code_lookup(r.status_code, PROWL_HTTP_ERROR_MAP)
                self.logger.warning('Failed to send Prowl notification:{}{}error={}.'.format(status_str, ', ' if status_str else '', r.status_code))
                self.logger.debug('Response Details:\r\n{}'.format(r.content))
                return False
            else:
                self.logger.info('Sent Prowl notification.')
        except requests.RequestException as e:
            self.logger.warning('A Connection error occurred sending Prowl notification.')
            self.logger.debug('Socket Exception: %s' % str(e))
            return False
        return True

    def url(self, privacy=False, *args, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Returns the URL built dynamically based on specified arguments.\n        '
        params = {'priority': PROWL_PRIORITIES[self.template_args['priority']['default']] if self.priority not in PROWL_PRIORITIES else PROWL_PRIORITIES[self.priority]}
        params.update(self.url_parameters(*args, privacy=privacy, **kwargs))
        return '{schema}://{apikey}/{providerkey}/?{params}'.format(schema=self.secure_protocol, apikey=self.pprint(self.apikey, privacy, safe=''), providerkey=self.pprint(self.providerkey, privacy, safe=''), params=NotifyProwl.urlencode(params))

    @staticmethod
    def parse_url(url):
        if False:
            for i in range(10):
                print('nop')
        '\n        Parses the URL and returns enough arguments that can allow\n        us to re-instantiate this object.\n\n        '
        results = NotifyBase.parse_url(url, verify_host=False)
        if not results:
            return results
        results['apikey'] = NotifyProwl.unquote(results['host'])
        try:
            results['providerkey'] = NotifyProwl.split_path(results['fullpath'])[0]
        except IndexError:
            pass
        if 'priority' in results['qsd'] and len(results['qsd']['priority']):
            results['priority'] = NotifyProwl.unquote(results['qsd']['priority'])
        return results