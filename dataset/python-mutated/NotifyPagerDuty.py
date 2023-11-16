import requests
from json import dumps
from .NotifyBase import NotifyBase
from ..URLBase import PrivacyMode
from ..common import NotifyType
from ..common import NotifyImageSize
from ..utils import validate_regex
from ..utils import parse_bool
from ..AppriseLocale import gettext_lazy as _

class PagerDutySeverity:
    """
    Defines the Pager Duty Severity Levels
    """
    INFO = 'info'
    WARNING = 'warning'
    ERROR = 'error'
    CRITICAL = 'critical'
PAGERDUTY_SEVERITY_MAP = {NotifyType.INFO: PagerDutySeverity.INFO, NotifyType.SUCCESS: PagerDutySeverity.INFO, NotifyType.WARNING: PagerDutySeverity.WARNING, NotifyType.FAILURE: PagerDutySeverity.CRITICAL}
PAGERDUTY_SEVERITIES = (PagerDutySeverity.INFO, PagerDutySeverity.WARNING, PagerDutySeverity.CRITICAL, PagerDutySeverity.ERROR)

class PagerDutyRegion:
    US = 'us'
    EU = 'eu'
PAGERDUTY_API_LOOKUP = {PagerDutyRegion.US: 'https://events.pagerduty.com/v2/enqueue', PagerDutyRegion.EU: 'https://events.eu.pagerduty.com/v2/enqueue'}
PAGERDUTY_REGIONS = (PagerDutyRegion.US, PagerDutyRegion.EU)

class NotifyPagerDuty(NotifyBase):
    """
    A wrapper for Pager Duty Notifications
    """
    service_name = 'Pager Duty'
    service_url = 'https://pagerduty.com/'
    secure_protocol = 'pagerduty'
    setup_url = 'https://github.com/caronc/apprise/wiki/Notify_pagerduty'
    title_maxlen = 0
    image_size = NotifyImageSize.XY_128
    event_action = 'trigger'
    default_region = PagerDutyRegion.US
    templates = ('{schema}://{integrationkey}@{apikey}', '{schema}://{integrationkey}@{apikey}/{source}', '{schema}://{integrationkey}@{apikey}/{source}/{component}')
    template_tokens = dict(NotifyBase.template_tokens, **{'apikey': {'name': _('API Key'), 'type': 'string', 'private': True, 'required': True}, 'integrationkey': {'name': _('Integration Key'), 'type': 'string', 'private': True, 'required': True}, 'source': {'name': _('Source'), 'type': 'string', 'default': 'Apprise'}, 'component': {'name': _('Component'), 'type': 'string', 'default': 'Notification'}})
    template_args = dict(NotifyBase.template_args, **{'group': {'name': _('Group'), 'type': 'string'}, 'class': {'name': _('Class'), 'type': 'string', 'map_to': 'class_id'}, 'click': {'name': _('Click'), 'type': 'string'}, 'region': {'name': _('Region Name'), 'type': 'choice:string', 'values': PAGERDUTY_REGIONS, 'default': PagerDutyRegion.US, 'map_to': 'region_name'}, 'severity': {'name': _('Severity'), 'type': 'choice:string', 'values': PAGERDUTY_SEVERITIES, 'map_to': 'severity'}, 'image': {'name': _('Include Image'), 'type': 'bool', 'default': True, 'map_to': 'include_image'}})
    template_kwargs = {'details': {'name': _('Custom Details'), 'prefix': '+'}}

    def __init__(self, apikey, integrationkey=None, source=None, component=None, group=None, class_id=None, include_image=True, click=None, details=None, region_name=None, severity=None, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Initialize Pager Duty Object\n        '
        super().__init__(**kwargs)
        self.apikey = validate_regex(apikey)
        if not self.apikey:
            msg = 'An invalid Pager Duty API Key ({}) was specified.'.format(apikey)
            self.logger.warning(msg)
            raise TypeError(msg)
        self.integration_key = validate_regex(integrationkey)
        if not self.integration_key:
            msg = 'An invalid Pager Duty Routing Key ({}) was specified.'.format(integrationkey)
            self.logger.warning(msg)
            raise TypeError(msg)
        self.source = self.template_tokens['source']['default']
        if source:
            self.source = validate_regex(source)
            if not self.source:
                msg = 'An invalid Pager Duty Notification Source ({}) was specified.'.format(source)
                self.logger.warning(msg)
                raise TypeError(msg)
        else:
            self.component = self.template_tokens['source']['default']
        self.component = self.template_tokens['component']['default']
        if component:
            self.component = validate_regex(component)
            if not self.component:
                msg = 'An invalid Pager Duty Notification Component ({}) was specified.'.format(component)
                self.logger.warning(msg)
                raise TypeError(msg)
        else:
            self.component = self.template_tokens['component']['default']
        try:
            self.region_name = self.default_region if region_name is None else region_name.lower()
            if self.region_name not in PAGERDUTY_REGIONS:
                raise
        except:
            msg = 'The PagerDuty region specified ({}) is invalid.'.format(region_name)
            self.logger.warning(msg)
            raise TypeError(msg)
        self.severity = None if severity is None else next((s for s in PAGERDUTY_SEVERITIES if str(s).lower().startswith(severity)), False)
        if self.severity is False:
            msg = 'The PagerDuty severity specified ({}) is invalid.'.format(severity)
            self.logger.warning(msg)
            raise TypeError(msg)
        self.click = click
        self.class_id = class_id
        self.group = group
        self.details = {}
        if details:
            self.details.update(details)
        self.include_image = include_image
        return

    def send(self, body, title='', notify_type=NotifyType.INFO, **kwargs):
        if False:
            return 10
        '\n        Send our PagerDuty Notification\n        '
        headers = {'User-Agent': self.app_id, 'Content-Type': 'application/json', 'Authorization': 'Token token={}'.format(self.apikey)}
        payload = {'routing_key': self.integration_key, 'payload': {'summary': body, 'severity': PAGERDUTY_SEVERITY_MAP[notify_type] if not self.severity else self.severity, 'source': self.source, 'component': self.component}, 'client': self.app_id, 'event_action': self.event_action}
        if self.group:
            payload['payload']['group'] = self.group
        if self.class_id:
            payload['payload']['class'] = self.class_id
        if self.click:
            payload['links'] = [{'href': self.click}]
        image_url = None if not self.include_image else self.image_url(notify_type)
        if image_url:
            payload['images'] = [{'src': image_url, 'alt': notify_type}]
        if self.details:
            payload['payload']['custom_details'] = {}
            for (k, v) in self.details.items():
                payload['payload']['custom_details'][k] = v
        notify_url = PAGERDUTY_API_LOOKUP[self.region_name]
        self.logger.debug('Pager Duty POST URL: %s (cert_verify=%r)' % (notify_url, self.verify_certificate))
        self.logger.debug('Pager Duty Payload: %s' % str(payload))
        self.throttle()
        try:
            r = requests.post(notify_url, data=dumps(payload), headers=headers, verify=self.verify_certificate, timeout=self.request_timeout)
            if r.status_code not in (requests.codes.ok, requests.codes.created, requests.codes.accepted):
                status_str = NotifyPagerDuty.http_response_code_lookup(r.status_code)
                self.logger.warning('Failed to send Pager Duty notification: {}{}error={}.'.format(status_str, ', ' if status_str else '', r.status_code))
                self.logger.debug('Response Details:\r\n{}'.format(r.content))
                return False
            else:
                self.logger.info('Sent Pager Duty notification.')
        except requests.RequestException as e:
            self.logger.warning('A Connection error occurred sending Pager Duty notification to %s.' % self.host)
            self.logger.debug('Socket Exception: %s' % str(e))
            return False
        return True

    def url(self, privacy=False, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns the URL built dynamically based on specified arguments.\n        '
        params = {'region': self.region_name, 'image': 'yes' if self.include_image else 'no'}
        if self.class_id:
            params['class'] = self.class_id
        if self.group:
            params['group'] = self.group
        if self.click is not None:
            params['click'] = self.click
        if self.severity:
            params['severity'] = self.severity
        params.update({'+{}'.format(k): v for (k, v) in self.details.items()})
        params.update(self.url_parameters(*args, privacy=privacy, **kwargs))
        url = '{schema}://{integration_key}@{apikey}/{source}/{component}?{params}'
        return url.format(schema=self.secure_protocol, integration_key=self.pprint(self.integration_key, privacy, mode=PrivacyMode.Secret, safe=''), apikey=self.pprint(self.apikey, privacy, mode=PrivacyMode.Secret, safe=''), source=self.pprint(self.source, privacy, safe=''), component=self.pprint(self.component, privacy, safe=''), params=NotifyPagerDuty.urlencode(params))

    @staticmethod
    def parse_url(url):
        if False:
            for i in range(10):
                print('nop')
        '\n        Parses the URL and returns enough arguments that can allow\n        us to re-instantiate this object.\n\n        '
        results = NotifyBase.parse_url(url, verify_host=False)
        if not results:
            return results
        if 'apikey' in results['qsd'] and len(results['qsd']['apikey']):
            results['apikey'] = NotifyPagerDuty.unquote(results['qsd']['apikey'])
        else:
            results['apikey'] = NotifyPagerDuty.unquote(results['host'])
        if 'integrationkey' in results['qsd'] and len(results['qsd']['integrationkey']):
            results['integrationkey'] = NotifyPagerDuty.unquote(results['qsd']['integrationkey'])
        else:
            results['integrationkey'] = NotifyPagerDuty.unquote(results['user'])
        if 'click' in results['qsd'] and len(results['qsd']['click']):
            results['click'] = NotifyPagerDuty.unquote(results['qsd']['click'])
        if 'group' in results['qsd'] and len(results['qsd']['group']):
            results['group'] = NotifyPagerDuty.unquote(results['qsd']['group'])
        if 'class' in results['qsd'] and len(results['qsd']['class']):
            results['class_id'] = NotifyPagerDuty.unquote(results['qsd']['class'])
        if 'severity' in results['qsd'] and len(results['qsd']['severity']):
            results['severity'] = NotifyPagerDuty.unquote(results['qsd']['severity'])
        fullpath = NotifyPagerDuty.split_path(results['fullpath'])
        if 'source' in results['qsd'] and len(results['qsd']['source']):
            results['source'] = NotifyPagerDuty.unquote(results['qsd']['source'])
        else:
            results['source'] = fullpath.pop(0) if fullpath else None
        if 'component' in results['qsd'] and len(results['qsd']['component']):
            results['component'] = NotifyPagerDuty.unquote(results['qsd']['component'])
        else:
            results['component'] = fullpath.pop(0) if fullpath else None
        results['details'] = {NotifyPagerDuty.unquote(x): NotifyPagerDuty.unquote(y) for (x, y) in results['qsd+'].items()}
        if 'region' in results['qsd'] and len(results['qsd']['region']):
            results['region_name'] = NotifyPagerDuty.unquote(results['qsd']['region'])
        results['include_image'] = parse_bool(results['qsd'].get('image', True))
        return results