import requests
from json import dumps
from json import loads
from .NotifyBase import NotifyBase
from ..common import NotifyType
from ..utils import is_phone_no
from ..utils import parse_phone_no
from ..utils import validate_regex
from ..utils import parse_bool
from ..AppriseLocale import gettext_lazy as _
D7NETWORKS_HTTP_ERROR_MAP = {401: 'Invalid Argument(s) Specified.', 403: 'Unauthorized - Authentication Failure.', 412: 'A Routing Error Occured', 500: 'A Serverside Error Occured Handling the Request.'}

class NotifyD7Networks(NotifyBase):
    """
    A wrapper for D7 Networks Notifications
    """
    service_name = 'D7 Networks'
    service_url = 'https://d7networks.com/'
    secure_protocol = 'd7sms'
    request_rate_per_sec = 0.2
    setup_url = 'https://github.com/caronc/apprise/wiki/Notify_d7networks'
    notify_url = 'https://api.d7networks.com/messages/v1/send'
    body_maxlen = 160
    title_maxlen = 0
    templates = ('{schema}://{token}@{targets}',)
    template_tokens = dict(NotifyBase.template_tokens, **{'token': {'name': _('API Access Token'), 'type': 'string', 'required': True, 'private': True}, 'target_phone': {'name': _('Target Phone No'), 'type': 'string', 'prefix': '+', 'regex': ('^[0-9\\s)(+-]+$', 'i'), 'map_to': 'targets'}, 'targets': {'name': _('Targets'), 'type': 'list:string', 'required': True}})
    template_args = dict(NotifyBase.template_args, **{'unicode': {'name': _('Unicode Characters'), 'type': 'bool', 'default': False}, 'batch': {'name': _('Batch Mode'), 'type': 'bool', 'default': False}, 'to': {'alias_of': 'targets'}, 'source': {'name': _('Originating Address'), 'type': 'string', 'map_to': 'source'}, 'from': {'alias_of': 'source'}})

    def __init__(self, token=None, targets=None, source=None, batch=False, unicode=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Initialize D7 Networks Object\n        '
        super().__init__(**kwargs)
        self.batch = batch
        self.source = None if not isinstance(source, str) else source.strip()
        self.unicode = self.template_args['unicode']['default'] if unicode is None else bool(unicode)
        self.token = validate_regex(token)
        if not self.token:
            msg = 'The D7 Networks token specified ({}) is invalid.'.format(token)
            self.logger.warning(msg)
            raise TypeError(msg)
        self.targets = list()
        for target in parse_phone_no(targets):
            result = result = is_phone_no(target)
            if not result:
                self.logger.warning('Dropped invalid phone # ({}) specified.'.format(target))
                continue
            self.targets.append(result['full'])
        return

    def send(self, body, title='', notify_type=NotifyType.INFO, **kwargs):
        if False:
            print('Hello World!')
        '\n        Depending on whether we are set to batch mode or single mode this\n        redirects to the appropriate handling\n        '
        if len(self.targets) == 0:
            self.logger.warning('There were no D7 Networks targets to notify.')
            return False
        has_error = False
        headers = {'User-Agent': self.app_id, 'Content-Type': 'application/json', 'Accept': 'application/json', 'Authorization': f'Bearer {self.token}'}
        payload = {'message_globals': {'channel': 'sms'}, 'messages': [{'recipients': None, 'content': body, 'data_coding': 'auto' if not self.unicode else 'unicode'}]}
        targets = list(self.targets)
        if self.source:
            payload['message_globals']['originator'] = self.source
        target = None
        while len(targets):
            if self.batch:
                payload['messages'][0]['recipients'] = self.targets
                targets = []
            else:
                target = targets.pop(0)
                payload['messages'][0]['recipients'] = [target]
            self.logger.debug('D7 Networks POST URL: {} (cert_verify={})'.format(self.notify_url, self.verify_certificate))
            self.logger.debug('D7 Networks Payload: {}'.format(payload))
            self.throttle()
            try:
                r = requests.post(self.notify_url, data=dumps(payload), headers=headers, verify=self.verify_certificate, timeout=self.request_timeout)
                if r.status_code not in (requests.codes.created, requests.codes.ok):
                    status_str = NotifyBase.http_response_code_lookup(r.status_code, D7NETWORKS_HTTP_ERROR_MAP)
                    try:
                        json_response = loads(r.content)
                        status_str = json_response.get('message', status_str)
                    except (AttributeError, TypeError, ValueError):
                        pass
                    self.logger.warning('Failed to send D7 Networks SMS notification to {}: {}{}error={}.'.format(', '.join(target) if self.batch else target, status_str, ', ' if status_str else '', r.status_code))
                    self.logger.debug('Response Details:\r\n{}'.format(r.content))
                    has_error = True
                    continue
                else:
                    if self.batch:
                        self.logger.info('Sent D7 Networks batch SMS notification to {} target(s).'.format(len(self.targets)))
                    else:
                        self.logger.info('Sent D7 Networks SMS notification to {}.'.format(target))
                    self.logger.debug('Response Details:\r\n{}'.format(r.content))
            except requests.RequestException as e:
                self.logger.warning('A Connection error occurred sending D7 Networks:%s ' % ', '.join(self.targets) + 'notification.')
                self.logger.debug('Socket Exception: %s' % str(e))
                has_error = True
                continue
        return not has_error

    def url(self, privacy=False, *args, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Returns the URL built dynamically based on specified arguments.\n        '
        params = {'batch': 'yes' if self.batch else 'no', 'unicode': 'yes' if self.unicode else 'no'}
        if self.source:
            params['from'] = self.source
        params.update(self.url_parameters(*args, privacy=privacy, **kwargs))
        return '{schema}://{token}@{targets}/?{params}'.format(schema=self.secure_protocol, token=self.pprint(self.token, privacy, safe=''), targets='/'.join([NotifyD7Networks.quote(x, safe='') for x in self.targets]), params=NotifyD7Networks.urlencode(params))

    def __len__(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns the number of targets associated with this notification\n        '
        return len(self.targets) if not self.batch else 1

    @staticmethod
    def parse_url(url):
        if False:
            for i in range(10):
                print('nop')
        '\n        Parses the URL and returns enough arguments that can allow\n        us to re-instantiate this object.\n\n        '
        results = NotifyBase.parse_url(url, verify_host=False)
        if not results:
            return results
        if 'token' in results['qsd'] and len(results['qsd']['token']):
            results['token'] = NotifyD7Networks.unquote(results['qsd']['token'])
        elif results['user']:
            results['token'] = NotifyD7Networks.unquote(results['user'])
            if results['password']:
                results['token'] += ':' + NotifyD7Networks.unquote(results['password'])
        elif results['password']:
            results['token'] = ':' + NotifyD7Networks.unquote(results['password'])
        results['targets'] = list()
        results['targets'].append(NotifyD7Networks.unquote(results['host']))
        results['targets'].extend(NotifyD7Networks.split_path(results['fullpath']))
        results['batch'] = parse_bool(results['qsd'].get('batch', False))
        results['unicode'] = parse_bool(results['qsd'].get('unicode', False))
        if 'to' in results['qsd'] and len(results['qsd']['to']):
            results['targets'] += NotifyD7Networks.parse_phone_no(results['qsd']['to'])
        if 'from' in results['qsd'] and len(results['qsd']['from']):
            results['source'] = NotifyD7Networks.unquote(results['qsd']['from'])
        elif 'source' in results['qsd'] and len(results['qsd']['source']):
            results['source'] = NotifyD7Networks.unquote(results['qsd']['source'])
        return results