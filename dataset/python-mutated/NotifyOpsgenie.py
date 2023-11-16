import requests
from json import dumps
from .NotifyBase import NotifyBase
from ..common import NotifyType
from ..utils import validate_regex
from ..utils import is_uuid
from ..utils import parse_list
from ..utils import parse_bool
from ..AppriseLocale import gettext_lazy as _

class OpsgenieCategory(NotifyBase):
    """
    We define the different category types that we can notify
    """
    USER = 'user'
    SCHEDULE = 'schedule'
    ESCALATION = 'escalation'
    TEAM = 'team'
OPSGENIE_CATEGORIES = (OpsgenieCategory.USER, OpsgenieCategory.SCHEDULE, OpsgenieCategory.ESCALATION, OpsgenieCategory.TEAM)

class OpsgenieRegion:
    US = 'us'
    EU = 'eu'
OPSGENIE_API_LOOKUP = {OpsgenieRegion.US: 'https://api.opsgenie.com/v2/alerts', OpsgenieRegion.EU: 'https://api.eu.opsgenie.com/v2/alerts'}
OPSGENIE_REGIONS = (OpsgenieRegion.US, OpsgenieRegion.EU)

class OpsgeniePriority:
    LOW = 1
    MODERATE = 2
    NORMAL = 3
    HIGH = 4
    EMERGENCY = 5
OPSGENIE_PRIORITIES = {OpsgeniePriority.LOW: 'low', OpsgeniePriority.MODERATE: 'moderate', OpsgeniePriority.NORMAL: 'normal', OpsgeniePriority.HIGH: 'high', OpsgeniePriority.EMERGENCY: 'emergency'}
OPSGENIE_PRIORITY_MAP = {'l': OpsgeniePriority.LOW, 'm': OpsgeniePriority.MODERATE, 'n': OpsgeniePriority.NORMAL, 'h': OpsgeniePriority.HIGH, 'e': OpsgeniePriority.EMERGENCY, '1': OpsgeniePriority.LOW, '2': OpsgeniePriority.MODERATE, '3': OpsgeniePriority.NORMAL, '4': OpsgeniePriority.HIGH, '5': OpsgeniePriority.EMERGENCY, 'p1': OpsgeniePriority.LOW, 'p2': OpsgeniePriority.MODERATE, 'p3': OpsgeniePriority.NORMAL, 'p4': OpsgeniePriority.HIGH, 'p5': OpsgeniePriority.EMERGENCY}

class NotifyOpsgenie(NotifyBase):
    """
    A wrapper for Opsgenie Notifications
    """
    service_name = 'Opsgenie'
    service_url = 'https://opsgenie.com/'
    secure_protocol = 'opsgenie'
    setup_url = 'https://github.com/caronc/apprise/wiki/Notify_opsgenie'
    body_maxlen = 15000
    opsgenie_body_minlen = 130
    opsgenie_default_region = OpsgenieRegion.US
    default_batch_size = 50
    templates = ('{schema}://{apikey}', '{schema}://{apikey}/{targets}')
    template_tokens = dict(NotifyBase.template_tokens, **{'apikey': {'name': _('API Key'), 'type': 'string', 'private': True, 'required': True}, 'target_escalation': {'name': _('Target Escalation'), 'prefix': '^', 'type': 'string', 'map_to': 'targets'}, 'target_schedule': {'name': _('Target Schedule'), 'type': 'string', 'prefix': '*', 'map_to': 'targets'}, 'target_user': {'name': _('Target User'), 'type': 'string', 'prefix': '@', 'map_to': 'targets'}, 'target_team': {'name': _('Target Team'), 'type': 'string', 'prefix': '#', 'map_to': 'targets'}, 'targets': {'name': _('Targets '), 'type': 'list:string'}})
    template_args = dict(NotifyBase.template_args, **{'region': {'name': _('Region Name'), 'type': 'choice:string', 'values': OPSGENIE_REGIONS, 'default': OpsgenieRegion.US, 'map_to': 'region_name'}, 'batch': {'name': _('Batch Mode'), 'type': 'bool', 'default': False}, 'priority': {'name': _('Priority'), 'type': 'choice:int', 'values': OPSGENIE_PRIORITIES, 'default': OpsgeniePriority.NORMAL}, 'entity': {'name': _('Entity'), 'type': 'string'}, 'alias': {'name': _('Alias'), 'type': 'string'}, 'tags': {'name': _('Tags'), 'type': 'string'}, 'to': {'alias_of': 'targets'}})
    template_kwargs = {'details': {'name': _('Details'), 'prefix': '+'}}

    def __init__(self, apikey, targets, region_name=None, details=None, priority=None, alias=None, entity=None, batch=False, tags=None, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Initialize Opsgenie Object\n        '
        super().__init__(**kwargs)
        self.apikey = validate_regex(apikey)
        if not self.apikey:
            msg = 'An invalid Opsgenie API Key ({}) was specified.'.format(apikey)
            self.logger.warning(msg)
            raise TypeError(msg)
        self.priority = NotifyOpsgenie.template_args['priority']['default'] if not priority else next((v for (k, v) in OPSGENIE_PRIORITY_MAP.items() if str(priority).lower().startswith(k)), NotifyOpsgenie.template_args['priority']['default'])
        try:
            self.region_name = self.opsgenie_default_region if region_name is None else region_name.lower()
            if self.region_name not in OPSGENIE_REGIONS:
                raise
        except:
            msg = 'The Opsgenie region specified ({}) is invalid.'.format(region_name)
            self.logger.warning(msg)
            raise TypeError(msg)
        self.details = {}
        if details:
            self.details.update(details)
        self.batch_size = self.default_batch_size if batch else 1
        self.__tags = parse_list(tags)
        self.entity = entity
        self.alias = alias
        self.targets = []
        for _target in parse_list(targets):
            target = _target.strip()
            if len(target) < 2:
                self.logger.debug('Ignoring Opsgenie Entry: %s' % target)
                continue
            if target.startswith(NotifyOpsgenie.template_tokens['target_team']['prefix']):
                self.targets.append({'type': OpsgenieCategory.TEAM, 'id': target[1:]} if is_uuid(target[1:]) else {'type': OpsgenieCategory.TEAM, 'name': target[1:]})
            elif target.startswith(NotifyOpsgenie.template_tokens['target_schedule']['prefix']):
                self.targets.append({'type': OpsgenieCategory.SCHEDULE, 'id': target[1:]} if is_uuid(target[1:]) else {'type': OpsgenieCategory.SCHEDULE, 'name': target[1:]})
            elif target.startswith(NotifyOpsgenie.template_tokens['target_escalation']['prefix']):
                self.targets.append({'type': OpsgenieCategory.ESCALATION, 'id': target[1:]} if is_uuid(target[1:]) else {'type': OpsgenieCategory.ESCALATION, 'name': target[1:]})
            elif target.startswith(NotifyOpsgenie.template_tokens['target_user']['prefix']):
                self.targets.append({'type': OpsgenieCategory.USER, 'id': target[1:]} if is_uuid(target[1:]) else {'type': OpsgenieCategory.USER, 'username': target[1:]})
            else:
                self.logger.debug('Treating ambigious Opsgenie target %s as a user', target)
                self.targets.append({'type': OpsgenieCategory.USER, 'id': target} if is_uuid(target) else {'type': OpsgenieCategory.USER, 'username': target})

    def send(self, body, title='', notify_type=NotifyType.INFO, **kwargs):
        if False:
            return 10
        '\n        Perform Opsgenie Notification\n        '
        headers = {'User-Agent': self.app_id, 'Content-Type': 'application/json', 'Authorization': 'GenieKey {}'.format(self.apikey)}
        notify_url = OPSGENIE_API_LOOKUP[self.region_name]
        has_error = False
        title_body = body if not title else title
        details = self.details.copy()
        if 'type' not in details:
            details['type'] = notify_type
        payload = {'source': self.app_desc, 'message': title_body, 'description': body, 'details': details, 'priority': 'P{}'.format(self.priority)}
        if len(payload['message']) > self.opsgenie_body_minlen:
            payload['message'] = '{}...'.format(title_body[:self.opsgenie_body_minlen - 3])
        if self.__tags:
            payload['tags'] = self.__tags
        if self.entity:
            payload['entity'] = self.entity
        if self.alias:
            payload['alias'] = self.alias
        length = len(self.targets) if self.targets else 1
        for index in range(0, length, self.batch_size):
            if self.targets:
                payload['responders'] = self.targets[index:index + self.batch_size]
            self.logger.debug('Opsgenie POST URL: {} (cert_verify={})'.format(notify_url, self.verify_certificate))
            self.logger.debug('Opsgenie Payload: {}'.format(payload))
            self.throttle()
            try:
                r = requests.post(notify_url, data=dumps(payload), headers=headers, verify=self.verify_certificate, timeout=self.request_timeout)
                if r.status_code not in (requests.codes.accepted, requests.codes.ok):
                    status_str = NotifyBase.http_response_code_lookup(r.status_code)
                    self.logger.warning('Failed to send Opsgenie notification:{}{}error={}.'.format(status_str, ', ' if status_str else '', r.status_code))
                    self.logger.debug('Response Details:\r\n{}'.format(r.content))
                    has_error = True
                    continue
                self.logger.info('Sent Opsgenie notification')
                self.logger.debug('Response Details:\r\n{}'.format(r.content))
            except requests.RequestException as e:
                self.logger.warning('A Connection error occurred sending Opsgenie notification.')
                self.logger.debug('Socket Exception: %s' % str(e))
                has_error = True
        return not has_error

    def url(self, privacy=False, *args, **kwargs):
        if False:
            return 10
        '\n        Returns the URL built dynamically based on specified arguments.\n        '
        params = {'region': self.region_name, 'priority': OPSGENIE_PRIORITIES[self.template_args['priority']['default']] if self.priority not in OPSGENIE_PRIORITIES else OPSGENIE_PRIORITIES[self.priority], 'batch': 'yes' if self.batch_size > 1 else 'no'}
        if self.entity:
            params['entity'] = self.entity
        if self.alias:
            params['alias'] = self.alias
        if self.__tags:
            params['tags'] = ','.join(self.__tags)
        params.update({'+{}'.format(k): v for (k, v) in self.details.items()})
        params.update(self.url_parameters(*args, privacy=privacy, **kwargs))
        __map = {OpsgenieCategory.USER: NotifyOpsgenie.template_tokens['target_user']['prefix'], OpsgenieCategory.SCHEDULE: NotifyOpsgenie.template_tokens['target_schedule']['prefix'], OpsgenieCategory.ESCALATION: NotifyOpsgenie.template_tokens['target_escalation']['prefix'], OpsgenieCategory.TEAM: NotifyOpsgenie.template_tokens['target_team']['prefix']}
        return '{schema}://{apikey}/{targets}/?{params}'.format(schema=self.secure_protocol, apikey=self.pprint(self.apikey, privacy, safe=''), targets='/'.join([NotifyOpsgenie.quote('{}{}'.format(__map[x['type']], x.get('id', x.get('name', x.get('username'))))) for x in self.targets]), params=NotifyOpsgenie.urlencode(params))

    def __len__(self):
        if False:
            i = 10
            return i + 15
        '\n        Returns the number of targets associated with this notification\n        '
        targets = len(self.targets)
        if self.batch_size > 1:
            targets = int(targets / self.batch_size) + (1 if targets % self.batch_size else 0)
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
        results['apikey'] = NotifyOpsgenie.unquote(results['host'])
        results['targets'] = NotifyOpsgenie.split_path(results['fullpath'])
        results['details'] = {NotifyBase.unquote(x): NotifyBase.unquote(y) for (x, y) in results['qsd+'].items()}
        if 'priority' in results['qsd'] and len(results['qsd']['priority']):
            results['priority'] = NotifyOpsgenie.unquote(results['qsd']['priority'])
        results['batch'] = parse_bool(results['qsd'].get('batch', NotifyOpsgenie.template_args['batch']['default']))
        if 'apikey' in results['qsd'] and len(results['qsd']['apikey']):
            results['apikey'] = NotifyOpsgenie.unquote(results['qsd']['apikey'])
        if 'tags' in results['qsd'] and len(results['qsd']['tags']):
            results['tags'] = parse_list(NotifyOpsgenie.unquote(results['qsd']['tags']))
        if 'region' in results['qsd'] and len(results['qsd']['region']):
            results['region_name'] = NotifyOpsgenie.unquote(results['qsd']['region'])
        if 'entity' in results['qsd'] and len(results['qsd']['entity']):
            results['entity'] = NotifyOpsgenie.unquote(results['qsd']['entity'])
        if 'alias' in results['qsd'] and len(results['qsd']['alias']):
            results['alias'] = NotifyOpsgenie.unquote(results['qsd']['alias'])
        if 'to' in results['qsd'] and len(results['qsd']['to']):
            results['targets'].append(results['qsd']['to'])
        return results