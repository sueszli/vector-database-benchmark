import re
import requests
from json import dumps
from .NotifyBase import NotifyBase
from ..common import NotifyType
from ..utils import parse_list
from ..utils import validate_regex
from ..AppriseLocale import gettext_lazy as _

class NotifyIFTTT(NotifyBase):
    """
    A wrapper for IFTTT Notifications

    """
    service_name = 'IFTTT'
    service_url = 'https://ifttt.com/'
    secure_protocol = 'ifttt'
    setup_url = 'https://github.com/caronc/apprise/wiki/Notify_ifttt'
    ifttt_default_key_prefix = 'value'
    ifttt_default_title_key = 'value1'
    ifttt_default_body_key = 'value2'
    ifttt_default_type_key = 'value3'
    notify_url = 'https://maker.ifttt.com/trigger/{event}/with/key/{webhook_id}'
    templates = ('{schema}://{webhook_id}/{events}',)
    template_tokens = dict(NotifyBase.template_tokens, **{'webhook_id': {'name': _('Webhook ID'), 'type': 'string', 'private': True, 'required': True}, 'events': {'name': _('Events'), 'type': 'list:string', 'required': True}})
    template_args = dict(NotifyBase.template_args, **{'to': {'alias_of': 'events'}})
    template_kwargs = {'add_tokens': {'name': _('Add Tokens'), 'prefix': '+'}, 'del_tokens': {'name': _('Remove Tokens'), 'prefix': '-'}}

    def __init__(self, webhook_id, events, add_tokens=None, del_tokens=None, **kwargs):
        if False:
            return 10
        "\n        Initialize IFTTT Object\n\n        add_tokens can optionally be a dictionary of key/value pairs\n        that you want to include in the IFTTT post to the server.\n\n        del_tokens can optionally be a list/tuple/set of tokens\n        that you want to eliminate from the IFTTT post.  There isn't\n        much real functionality to this one unless you want to remove\n        reference to Value1, Value2, and/or Value3\n\n        "
        super().__init__(**kwargs)
        self.webhook_id = validate_regex(webhook_id)
        if not self.webhook_id:
            msg = 'An invalid IFTTT Webhook ID ({}) was specified.'.format(webhook_id)
            self.logger.warning(msg)
            raise TypeError(msg)
        self.events = parse_list(events)
        if not self.events:
            msg = 'You must specify at least one event you wish to trigger on.'
            self.logger.warning(msg)
            raise TypeError(msg)
        self.add_tokens = {}
        if add_tokens:
            self.add_tokens.update(add_tokens)
        self.del_tokens = []
        if del_tokens is not None:
            if isinstance(del_tokens, (list, tuple, set)):
                self.del_tokens = del_tokens
            elif isinstance(del_tokens, dict):
                self.del_tokens = set(del_tokens.keys())
            else:
                msg = 'del_token must be a list; {} was provided'.format(str(type(del_tokens)))
                self.logger.warning(msg)
                raise TypeError(msg)

    def send(self, body, title='', notify_type=NotifyType.INFO, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Perform IFTTT Notification\n        '
        headers = {'User-Agent': self.app_id, 'Content-Type': 'application/json'}
        payload = {self.ifttt_default_title_key: title, self.ifttt_default_body_key: body, self.ifttt_default_type_key: notify_type}
        payload.update(self.add_tokens)
        payload = {x.lower(): y for (x, y) in payload.items() if x not in self.del_tokens}
        has_error = False
        events = list(self.events)
        while len(events):
            event = events.pop(0)
            url = self.notify_url.format(webhook_id=self.webhook_id, event=event)
            self.logger.debug('IFTTT POST URL: %s (cert_verify=%r)' % (url, self.verify_certificate))
            self.logger.debug('IFTTT Payload: %s' % str(payload))
            self.throttle()
            try:
                r = requests.post(url, data=dumps(payload), headers=headers, verify=self.verify_certificate, timeout=self.request_timeout)
                self.logger.debug(u'IFTTT HTTP response headers: %r' % r.headers)
                self.logger.debug(u'IFTTT HTTP response body: %r' % r.content)
                if r.status_code != requests.codes.ok:
                    status_str = NotifyIFTTT.http_response_code_lookup(r.status_code)
                    self.logger.warning('Failed to send IFTTT notification to {}: {}{}error={}.'.format(event, status_str, ', ' if status_str else '', r.status_code))
                    self.logger.debug('Response Details:\r\n{}'.format(r.content))
                    has_error = True
                    continue
                else:
                    self.logger.info('Sent IFTTT notification to %s.' % event)
            except requests.RequestException as e:
                self.logger.warning('A Connection error occurred sending IFTTT:%s ' % event + 'notification.')
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
        params.update({'+{}'.format(k): v for (k, v) in self.add_tokens})
        params.update({'-{}'.format(k): '' for k in self.del_tokens})
        return '{schema}://{webhook_id}@{events}/?{params}'.format(schema=self.secure_protocol, webhook_id=self.pprint(self.webhook_id, privacy, safe=''), events='/'.join([NotifyIFTTT.quote(x, safe='') for x in self.events]), params=NotifyIFTTT.urlencode(params))

    def __len__(self):
        if False:
            while True:
                i = 10
        '\n        Returns the number of targets associated with this notification\n        '
        return len(self.events)

    @staticmethod
    def parse_url(url):
        if False:
            i = 10
            return i + 15
        '\n        Parses the URL and returns enough arguments that can allow\n        us to re-instantiate this object.\n\n        '
        results = NotifyBase.parse_url(url, verify_host=False)
        if not results:
            return results
        results['webhook_id'] = results['user'] if results['user'] else results['host']
        results['webhook_id'] = NotifyIFTTT.unquote(results['webhook_id'])
        results['add_token'] = results['qsd+']
        results['del_token'] = results['qsd-']
        results['events'] = list()
        if results['user']:
            results['events'].append(NotifyIFTTT.unquote(results['host']))
        results['events'].extend(NotifyIFTTT.split_path(results['fullpath']))
        if 'to' in results['qsd'] and len(results['qsd']['to']):
            results['events'] += NotifyIFTTT.parse_list(results['qsd']['to'])
        return results

    @staticmethod
    def parse_native_url(url):
        if False:
            return 10
        '\n        Support https://maker.ifttt.com/use/WEBHOOK_ID/EVENT_ID\n        '
        result = re.match('^https?://maker\\.ifttt\\.com/use/(?P<webhook_id>[A-Z0-9_-]+)((?P<events>(/[A-Z0-9_-]+)+))?/?(?P<params>\\?.+)?$', url, re.I)
        if result:
            return NotifyIFTTT.parse_url('{schema}://{webhook_id}{events}{params}'.format(schema=NotifyIFTTT.secure_protocol, webhook_id=result.group('webhook_id'), events='' if not result.group('events') else '@{}'.format(result.group('events')), params='' if not result.group('params') else result.group('params')))
        return None