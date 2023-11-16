import re
import requests
import json
from itertools import chain
from .NotifyBase import NotifyBase
from ..URLBase import PrivacyMode
from ..common import NotifyType
from ..utils import is_phone_no
from ..utils import parse_phone_no
from ..utils import parse_bool
from ..AppriseLocale import gettext_lazy as _
IS_GROUP_RE = re.compile('^(@?(?P<group>[A-Z0-9_-]+))$', re.IGNORECASE)

class BulkSMSRoutingGroup(object):
    """
    The different categories of routing
    """
    ECONOMY = 'ECONOMY'
    STANDARD = 'STANDARD'
    PREMIUM = 'PREMIUM'
BULKSMS_ROUTING_GROUPS = (BulkSMSRoutingGroup.ECONOMY, BulkSMSRoutingGroup.STANDARD, BulkSMSRoutingGroup.PREMIUM)

class BulkSMSEncoding(object):
    """
    The different categories of routing
    """
    TEXT = 'TEXT'
    UNICODE = 'UNICODE'
    BINARY = 'BINARY'

class NotifyBulkSMS(NotifyBase):
    """
    A wrapper for BulkSMS Notifications
    """
    service_name = 'BulkSMS'
    service_url = 'https://bulksms.com/'
    secure_protocol = 'bulksms'
    setup_url = 'https://github.com/caronc/apprise/wiki/Notify_bulksms'
    notify_url = 'https://api.bulksms.com/v1/messages'
    body_maxlen = 160
    default_batch_size = 4000
    title_maxlen = 0
    templates = ('{schema}://{user}:{password}@{targets}',)
    template_tokens = dict(NotifyBase.template_tokens, **{'user': {'name': _('User Name'), 'type': 'string', 'required': True}, 'password': {'name': _('Password'), 'type': 'string', 'private': True, 'required': True}, 'target_phone': {'name': _('Target Phone No'), 'type': 'string', 'prefix': '+', 'regex': ('^[0-9\\s)(+-]+$', 'i'), 'map_to': 'targets'}, 'target_group': {'name': _('Target Group'), 'type': 'string', 'prefix': '+', 'regex': ('^[A-Z0-9 _-]+$', 'i'), 'map_to': 'targets'}, 'targets': {'name': _('Targets'), 'type': 'list:string', 'required': True}})
    template_args = dict(NotifyBase.template_args, **{'to': {'alias_of': 'targets'}, 'from': {'name': _('From Phone No'), 'type': 'string', 'regex': ('^\\+?[0-9\\s)(+-]+$', 'i'), 'map_to': 'source'}, 'route': {'name': _('Route Group'), 'type': 'choice:string', 'values': BULKSMS_ROUTING_GROUPS, 'default': BulkSMSRoutingGroup.STANDARD}, 'unicode': {'name': _('Unicode Characters'), 'type': 'bool', 'default': True}, 'batch': {'name': _('Batch Mode'), 'type': 'bool', 'default': False}})

    def __init__(self, source=None, targets=None, unicode=None, batch=None, route=None, **kwargs):
        if False:
            return 10
        '\n        Initialize BulkSMS Object\n        '
        super(NotifyBulkSMS, self).__init__(**kwargs)
        self.source = None
        if source:
            result = is_phone_no(source)
            if not result:
                msg = 'The Account (From) Phone # specified ({}) is invalid.'.format(source)
                self.logger.warning(msg)
                raise TypeError(msg)
            self.source = '+{}'.format(result['full'])
        self.route = self.template_args['route']['default'] if not isinstance(route, str) else route.upper()
        if self.route not in BULKSMS_ROUTING_GROUPS:
            msg = 'The route specified ({}) is invalid.'.format(route)
            self.logger.warning(msg)
            raise TypeError(msg)
        self.unicode = self.template_args['unicode']['default'] if unicode is None else bool(unicode)
        self.batch = self.template_args['batch']['default'] if batch is None else bool(batch)
        self.targets = list()
        self.groups = list()
        for target in parse_phone_no(targets):
            result = is_phone_no(target)
            if result:
                self.targets.append('+{}'.format(result['full']))
                continue
            group_re = IS_GROUP_RE.match(target)
            if group_re and (not target.isdigit()):
                self.groups.append(group_re.group('group'))
                continue
            self.logger.warning('Dropped invalid phone # and/or Group ({}) specified.'.format(target))
        return

    def send(self, body, title='', notify_type=NotifyType.INFO, **kwargs):
        if False:
            print('Hello World!')
        '\n        Perform BulkSMS Notification\n        '
        if not (self.password and self.user):
            self.logger.warning('There were no valid login credentials provided')
            return False
        if not (self.targets or self.groups):
            self.logger.warning('There are no Twist targets to notify')
            return False
        batch_size = 1 if not self.batch else self.default_batch_size
        has_error = False
        headers = {'User-Agent': self.app_id, 'Content-Type': 'application/json'}
        payload = {'to': None, 'body': body, 'routingGroup': self.route, 'encoding': BulkSMSEncoding.UNICODE if self.unicode else BulkSMSEncoding.TEXT, 'deliveryReports': 'ERRORS'}
        if self.source:
            payload.update({'from': self.source})
        auth = (self.user, self.password)
        targets = list(self.targets) if batch_size == 1 else [self.targets[index:index + batch_size] for index in range(0, len(self.targets), batch_size)]
        targets += [{'type': 'GROUP', 'name': g} for g in self.groups]
        while len(targets):
            target = targets.pop(0)
            payload['to'] = target
            if isinstance(target, dict):
                p_target = target['name']
            elif isinstance(target, list):
                p_target = '{} targets'.format(len(target))
            else:
                p_target = target
            self.logger.debug('BulkSMS POST URL: {} (cert_verify={})'.format(self.notify_url, self.verify_certificate))
            self.logger.debug('BulkSMS Payload: {}'.format(payload))
            self.throttle()
            try:
                r = requests.post(self.notify_url, data=json.dumps(payload), headers=headers, auth=auth, verify=self.verify_certificate, timeout=self.request_timeout)
                if r.status_code not in (requests.codes.created, requests.codes.ok):
                    status_str = NotifyBase.http_response_code_lookup(r.status_code)
                    status_code = r.status_code
                    self.logger.warning('Failed to send BulkSMS notification to {}: {}{}error={}.'.format(p_target, status_str, ', ' if status_str else '', status_code))
                    self.logger.debug('Response Details:\r\n{}'.format(r.content))
                    has_error = True
                    continue
                else:
                    self.logger.info('Sent BulkSMS notification to {}.'.format(p_target))
            except requests.RequestException as e:
                self.logger.warning('A Connection error occurred sending BulkSMS: to %s ', p_target)
                self.logger.debug('Socket Exception: %s' % str(e))
                has_error = True
                continue
        return not has_error

    def url(self, privacy=False, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Returns the URL built dynamically based on specified arguments.\n        '
        params = {'unicode': 'yes' if self.unicode else 'no', 'batch': 'yes' if self.batch else 'no', 'route': self.route}
        if self.source:
            params['from'] = self.source
        params.update(self.url_parameters(*args, privacy=privacy, **kwargs))
        return '{schema}://{user}:{password}@{targets}/?{params}'.format(schema=self.secure_protocol, user=self.pprint(self.user, privacy, safe=''), password=self.pprint(self.password, privacy, mode=PrivacyMode.Secret, safe=''), targets='/'.join(chain([NotifyBulkSMS.quote('{}'.format(x), safe='+') for x in self.targets], [NotifyBulkSMS.quote('@{}'.format(x), safe='@') for x in self.groups])), params=NotifyBulkSMS.urlencode(params))

    def __len__(self):
        if False:
            print('Hello World!')
        '\n        Returns the number of targets associated with this notification\n        '
        batch_size = 1 if not self.batch else self.default_batch_size
        targets = len(self.targets)
        if batch_size > 1:
            targets = int(targets / batch_size) + (1 if targets % batch_size else 0)
        return targets + len(self.groups)

    @staticmethod
    def parse_url(url):
        if False:
            return 10
        '\n        Parses the URL and returns enough arguments that can allow\n        us to re-instantiate this object.\n\n        '
        results = NotifyBase.parse_url(url, verify_host=False)
        if not results:
            return results
        results['targets'] = [NotifyBulkSMS.unquote(results['host']), *NotifyBulkSMS.split_path(results['fullpath'])]
        if 'from' in results['qsd'] and len(results['qsd']['from']):
            results['source'] = NotifyBulkSMS.unquote(results['qsd']['from'])
        if 'to' in results['qsd'] and len(results['qsd']['to']):
            results['targets'] += NotifyBulkSMS.parse_phone_no(results['qsd']['to'])
        results['unicode'] = parse_bool(results['qsd'].get('unicode', NotifyBulkSMS.template_args['unicode']['default']))
        results['batch'] = parse_bool(results['qsd'].get('batch', NotifyBulkSMS.template_args['batch']['default']))
        if 'route' in results['qsd'] and len(results['qsd']['route']):
            results['route'] = NotifyBulkSMS.unquote(results['qsd']['route'])
        return results