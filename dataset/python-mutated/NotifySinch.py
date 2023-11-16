import requests
import json
from .NotifyBase import NotifyBase
from ..URLBase import PrivacyMode
from ..common import NotifyType
from ..utils import is_phone_no
from ..utils import parse_phone_no
from ..utils import validate_regex
from ..AppriseLocale import gettext_lazy as _

class SinchRegion:
    """
    Defines the Sinch Server Regions
    """
    USA = 'us'
    EUROPE = 'eu'
SINCH_REGIONS = (SinchRegion.USA, SinchRegion.EUROPE)

class NotifySinch(NotifyBase):
    """
    A wrapper for Sinch Notifications
    """
    service_name = 'Sinch'
    service_url = 'https://sinch.com/'
    secure_protocol = 'sinch'
    request_rate_per_sec = 0.2
    validity_period = 14400
    setup_url = 'https://github.com/caronc/apprise/wiki/Notify_sinch'
    notify_url = 'https://{region}.sms.api.sinch.com/xms/v1/{spi}/batches'
    body_maxlen = 160
    title_maxlen = 0
    templates = ('{schema}://{service_plan_id}:{api_token}@{from_phone}', '{schema}://{service_plan_id}:{api_token}@{from_phone}/{targets}')
    template_tokens = dict(NotifyBase.template_tokens, **{'service_plan_id': {'name': _('Account SID'), 'type': 'string', 'private': True, 'required': True, 'regex': ('^[a-f0-9]+$', 'i')}, 'api_token': {'name': _('Auth Token'), 'type': 'string', 'private': True, 'required': True, 'regex': ('^[a-f0-9]+$', 'i')}, 'from_phone': {'name': _('From Phone No'), 'type': 'string', 'required': True, 'regex': ('^\\+?[0-9\\s)(+-]+$', 'i'), 'map_to': 'source'}, 'target_phone': {'name': _('Target Phone No'), 'type': 'string', 'prefix': '+', 'regex': ('^[0-9\\s)(+-]+$', 'i'), 'map_to': 'targets'}, 'short_code': {'name': _('Target Short Code'), 'type': 'string', 'regex': ('^[0-9]{5,6}$', 'i'), 'map_to': 'targets'}, 'targets': {'name': _('Targets'), 'type': 'list:string'}})
    template_args = dict(NotifyBase.template_args, **{'to': {'alias_of': 'targets'}, 'from': {'alias_of': 'from_phone'}, 'spi': {'alias_of': 'service_plan_id'}, 'region': {'name': _('Region'), 'type': 'string', 'regex': ('^[a-z]{2}$', 'i'), 'default': SinchRegion.USA}, 'token': {'alias_of': 'api_token'}})

    def __init__(self, service_plan_id, api_token, source, targets=None, region=None, **kwargs):
        if False:
            print('Hello World!')
        '\n        Initialize Sinch Object\n        '
        super().__init__(**kwargs)
        self.service_plan_id = validate_regex(service_plan_id, *self.template_tokens['service_plan_id']['regex'])
        if not self.service_plan_id:
            msg = 'An invalid Sinch Account SID ({}) was specified.'.format(service_plan_id)
            self.logger.warning(msg)
            raise TypeError(msg)
        self.api_token = validate_regex(api_token, *self.template_tokens['api_token']['regex'])
        if not self.api_token:
            msg = 'An invalid Sinch Authentication Token ({}) was specified.'.format(api_token)
            self.logger.warning(msg)
            raise TypeError(msg)
        self.region = self.template_args['region']['default'] if not isinstance(region, str) else region.lower()
        if self.region and self.region not in SINCH_REGIONS:
            msg = 'The region specified ({}) is invalid.'.format(region)
            self.logger.warning(msg)
            raise TypeError(msg)
        result = is_phone_no(source, min_len=5)
        if not result:
            msg = 'The Account (From) Phone # or Short-code specified ({}) is invalid.'.format(source)
            self.logger.warning(msg)
            raise TypeError(msg)
        self.source = result['full']
        if len(self.source) < 11 or len(self.source) > 14:
            if len(self.source) not in (5, 6):
                msg = 'The Account (From) Phone # specified ({}) is invalid.'.format(source)
                self.logger.warning(msg)
                raise TypeError(msg)
        else:
            self.source = '+{}'.format(self.source)
        self.targets = list()
        for target in parse_phone_no(targets):
            result = is_phone_no(target)
            if not result:
                self.logger.warning('Dropped invalid phone # ({}) specified.'.format(target))
                continue
            self.targets.append('+{}'.format(result['full']))
        return

    def send(self, body, title='', notify_type=NotifyType.INFO, **kwargs):
        if False:
            return 10
        '\n        Perform Sinch Notification\n        '
        if not self.targets:
            if len(self.source) in (5, 6):
                self.logger.warning('There are no valid Sinch targets to notify.')
                return False
        has_error = False
        headers = {'User-Agent': self.app_id, 'Authorization': 'Bearer {}'.format(self.api_token), 'Content-Type': 'application/json'}
        payload = {'body': body, 'from': self.source, 'to': None}
        url = self.notify_url.format(region=self.region, spi=self.service_plan_id)
        targets = list(self.targets)
        if len(targets) == 0:
            targets.append(self.source)
        while len(targets):
            target = targets.pop(0)
            payload['to'] = [target]
            self.logger.debug('Sinch POST URL: {} (cert_verify={})'.format(url, self.verify_certificate))
            self.logger.debug('Sinch Payload: {}'.format(payload))
            self.throttle()
            try:
                r = requests.post(url, data=json.dumps(payload), headers=headers, verify=self.verify_certificate, timeout=self.request_timeout)
                if r.status_code not in (requests.codes.created, requests.codes.ok):
                    status_str = NotifyBase.http_response_code_lookup(r.status_code)
                    status_code = r.status_code
                    try:
                        json_response = json.loads(r.content)
                        status_code = json_response.get('code', status_code)
                        status_str = json_response.get('message', status_str)
                    except (AttributeError, TypeError, ValueError):
                        pass
                    self.logger.warning('Failed to send Sinch notification to {}: {}{}error={}.'.format(target, status_str, ', ' if status_str else '', status_code))
                    self.logger.debug('Response Details:\r\n{}'.format(r.content))
                    has_error = True
                    continue
                else:
                    self.logger.info('Sent Sinch notification to {}.'.format(target))
            except requests.RequestException as e:
                self.logger.warning('A Connection error occurred sending Sinch:%s ' % target + 'notification.')
                self.logger.debug('Socket Exception: %s' % str(e))
                has_error = True
                continue
        return not has_error

    def url(self, privacy=False, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Returns the URL built dynamically based on specified arguments.\n        '
        params = {'region': self.region}
        params.update(self.url_parameters(*args, privacy=privacy, **kwargs))
        return '{schema}://{spi}:{token}@{source}/{targets}/?{params}'.format(schema=self.secure_protocol, spi=self.pprint(self.service_plan_id, privacy, mode=PrivacyMode.Tail, safe=''), token=self.pprint(self.api_token, privacy, safe=''), source=NotifySinch.quote(self.source, safe=''), targets='/'.join([NotifySinch.quote(x, safe='') for x in self.targets]), params=NotifySinch.urlencode(params))

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
            while True:
                i = 10
        '\n        Parses the URL and returns enough arguments that can allow\n        us to re-instantiate this object.\n\n        '
        results = NotifyBase.parse_url(url, verify_host=False)
        if not results:
            return results
        results['targets'] = NotifySinch.split_path(results['fullpath'])
        results['source'] = NotifySinch.unquote(results['host'])
        results['service_plan_id'] = NotifySinch.unquote(results['user'])
        results['api_token'] = NotifySinch.unquote(results['password'])
        if 'token' in results['qsd'] and len(results['qsd']['token']):
            results['api_token'] = NotifySinch.unquote(results['qsd']['token'])
        if 'spi' in results['qsd'] and len(results['qsd']['spi']):
            results['service_plan_id'] = NotifySinch.unquote(results['qsd']['spi'])
        if 'from' in results['qsd'] and len(results['qsd']['from']):
            results['source'] = NotifySinch.unquote(results['qsd']['from'])
        if 'source' in results['qsd'] and len(results['qsd']['source']):
            results['source'] = NotifySinch.unquote(results['qsd']['source'])
        if 'region' in results['qsd'] and len(results['qsd']['region']):
            results['region'] = NotifySinch.unquote(results['qsd']['region'])
        if 'to' in results['qsd'] and len(results['qsd']['to']):
            results['targets'] += NotifySinch.parse_phone_no(results['qsd']['to'])
        return results