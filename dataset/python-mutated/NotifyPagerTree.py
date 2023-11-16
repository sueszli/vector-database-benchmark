import requests
from json import dumps
from uuid import uuid4
from .NotifyBase import NotifyBase
from ..common import NotifyType
from ..utils import parse_list
from ..utils import validate_regex
from ..AppriseLocale import gettext_lazy as _

class PagerTreeAction:
    CREATE = 'create'
    ACKNOWLEDGE = 'acknowledge'
    RESOLVE = 'resolve'

class PagerTreeUrgency:
    SILENT = 'silent'
    LOW = 'low'
    MEDIUM = 'medium'
    HIGH = 'high'
    CRITICAL = 'critical'
PAGERTREE_ACTIONS = {PagerTreeAction.CREATE: 'create', PagerTreeAction.ACKNOWLEDGE: 'acknowledge', PagerTreeAction.RESOLVE: 'resolve'}
PAGERTREE_URGENCIES = {PagerTreeUrgency.SILENT: 'silent', PagerTreeUrgency.LOW: 'low', PagerTreeUrgency.MEDIUM: 'medium', PagerTreeUrgency.HIGH: 'high', PagerTreeUrgency.CRITICAL: 'critical'}
PAGERTREE_HTTP_ERROR_MAP = {402: 'Payment Required - Please subscribe or upgrade', 403: 'Forbidden - Blocked', 404: 'Not Found - Invalid Integration ID', 405: 'Method Not Allowed - Integration Disabled', 429: 'Too Many Requests - Rate Limit Exceeded'}

class NotifyPagerTree(NotifyBase):
    """
    A wrapper for PagerTree Notifications
    """
    service_name = 'PagerTree'
    service_url = 'https://pagertree.com/'
    secure_protocol = 'pagertree'
    setup_url = 'https://github.com/caronc/apprise/wiki/Notify_pagertree'
    notify_url = 'https://api.pagertree.com/integration/{}'
    templates = ('{schema}://{integration}',)
    template_tokens = dict(NotifyBase.template_tokens, **{'integration': {'name': _('Integration ID'), 'type': 'string', 'private': True, 'required': True}})
    template_args = dict(NotifyBase.template_args, **{'action': {'name': _('Action'), 'type': 'choice:string', 'values': PAGERTREE_ACTIONS, 'default': PagerTreeAction.CREATE}, 'thirdparty': {'name': _('Third Party ID'), 'type': 'string'}, 'urgency': {'name': _('Urgency'), 'type': 'choice:string', 'values': PAGERTREE_URGENCIES}, 'tags': {'name': _('Tags'), 'type': 'string'}})
    template_kwargs = {'headers': {'name': _('HTTP Header'), 'prefix': '+'}, 'payload_extras': {'name': _('Payload Extras'), 'prefix': ':'}, 'meta_extras': {'name': _('Meta Extras'), 'prefix': '-'}}

    def __init__(self, integration, action=None, thirdparty=None, urgency=None, tags=None, headers=None, payload_extras=None, meta_extras=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Initialize PagerTree Object\n        '
        super().__init__(**kwargs)
        self.integration = validate_regex(integration, '^int_[a-zA-Z0-9\\-_]{7,14}$')
        if not self.integration:
            msg = 'An invalid PagerTree Integration ID ({}) was specified.'.format(integration)
            self.logger.warning(msg)
            raise TypeError(msg)
        self.thirdparty = None
        if thirdparty:
            self.thirdparty = validate_regex(thirdparty)
            if not self.thirdparty:
                msg = 'An invalid PagerTree third party ID ({}) was specified.'.format(thirdparty)
                self.logger.warning(msg)
                raise TypeError(msg)
        self.headers = {}
        if headers:
            self.headers.update(headers)
        self.payload_extras = {}
        if payload_extras:
            self.payload_extras.update(payload_extras)
        self.meta_extras = {}
        if meta_extras:
            self.meta_extras.update(meta_extras)
        self.action = NotifyPagerTree.template_args['action']['default'] if action not in PAGERTREE_ACTIONS else PAGERTREE_ACTIONS[action]
        self.urgency = None if urgency not in PAGERTREE_URGENCIES else PAGERTREE_URGENCIES[urgency]
        self.__tags = parse_list(tags)
        return

    def send(self, body, title='', notify_type=NotifyType.INFO, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Perform PagerTree Notification\n        '
        headers = {'User-Agent': self.app_id, 'Content-Type': 'application/json'}
        headers.update(self.headers)
        payload = {'id': self.thirdparty if self.thirdparty else str(uuid4()), 'event_type': self.action}
        if self.action == PagerTreeAction.CREATE:
            payload['title'] = title if title else self.app_desc
            payload['description'] = body
            payload['meta'] = self.meta_extras
            payload['tags'] = self.__tags
            if self.urgency is not None:
                payload['urgency'] = self.urgency
        payload.update(self.payload_extras)
        notify_url = self.notify_url.format(self.integration)
        self.logger.debug('PagerTree POST URL: %s (cert_verify=%r)' % (notify_url, self.verify_certificate))
        self.logger.debug('PagerTree Payload: %s' % str(payload))
        self.throttle()
        try:
            r = requests.post(notify_url, data=dumps(payload), headers=headers, verify=self.verify_certificate, timeout=self.request_timeout)
            if r.status_code not in (requests.codes.ok, requests.codes.created, requests.codes.accepted):
                status_str = NotifyPagerTree.http_response_code_lookup(r.status_code)
                self.logger.warning('Failed to send PagerTree notification: {}{}error={}.'.format(status_str, ', ' if status_str else '', r.status_code))
                self.logger.debug('Response Details:\r\n{}'.format(r.content))
                return False
            else:
                self.logger.info('Sent PagerTree notification.')
        except requests.RequestException as e:
            self.logger.warning('A Connection error occurred sending PagerTree notification to %s.' % self.host)
            self.logger.debug('Socket Exception: %s' % str(e))
            return False
        return True

    def url(self, privacy=False, *args, **kwargs):
        if False:
            return 10
        '\n        Returns the URL built dynamically based on specified arguments.\n        '
        params = {'action': self.action}
        if self.thirdparty:
            params['tid'] = self.thirdparty
        if self.urgency:
            params['urgency'] = self.urgency
        if self.__tags:
            params['tags'] = ','.join([x for x in self.__tags])
        params.update(self.url_parameters(*args, privacy=privacy, **kwargs))
        params.update({'+{}'.format(k): v for (k, v) in self.headers.items()})
        params.update({'-{}'.format(k): v for (k, v) in self.meta_extras.items()})
        params.update({':{}'.format(k): v for (k, v) in self.payload_extras.items()})
        return '{schema}://{integration}?{params}'.format(schema=self.secure_protocol, integration=self.pprint(self.integration, privacy, safe=''), params=NotifyPagerTree.urlencode(params))

    @staticmethod
    def parse_url(url):
        if False:
            print('Hello World!')
        '\n        Parses the URL and returns enough arguments that can allow\n        us to re-instantiate this object.\n\n        '
        results = NotifyBase.parse_url(url, verify_host=False)
        if not results:
            return results
        results['headers'] = {NotifyPagerTree.unquote(x): NotifyPagerTree.unquote(y) for (x, y) in results['qsd+'].items()}
        results['payload_extras'] = {NotifyPagerTree.unquote(x): NotifyPagerTree.unquote(y) for (x, y) in results['qsd:'].items()}
        results['meta_extras'] = {NotifyPagerTree.unquote(x): NotifyPagerTree.unquote(y) for (x, y) in results['qsd-'].items()}
        if 'id' in results['qsd'] and len(results['qsd']['id']):
            results['integration'] = NotifyPagerTree.unquote(results['qsd']['id'])
        elif 'integration' in results['qsd'] and len(results['qsd']['integration']):
            results['integration'] = NotifyPagerTree.unquote(results['qsd']['integration'])
        else:
            results['integration'] = NotifyPagerTree.unquote(results['host'])
        if 'tid' in results['qsd'] and len(results['qsd']['tid']):
            results['thirdparty'] = NotifyPagerTree.unquote(results['qsd']['tid'])
        elif 'thirdparty' in results['qsd'] and len(results['qsd']['thirdparty']):
            results['thirdparty'] = NotifyPagerTree.unquote(results['qsd']['thirdparty'])
        if 'action' in results['qsd'] and len(results['qsd']['action']):
            results['action'] = NotifyPagerTree.unquote(results['qsd']['action'])
        if 'urgency' in results['qsd'] and len(results['qsd']['urgency']):
            results['urgency'] = NotifyPagerTree.unquote(results['qsd']['urgency'])
        if 'tags' in results['qsd'] and len(results['qsd']['tags']):
            results['tags'] = parse_list(NotifyPagerTree.unquote(results['qsd']['tags']))
        return results