import re
import requests
from json import loads, dumps
from .NotifyBase import NotifyBase
from ..common import NotifyType
from ..utils import is_phone_no
from ..utils import parse_phone_no
from ..utils import validate_regex
from ..AppriseLocale import gettext_lazy as _

class NotifyWhatsApp(NotifyBase):
    """
    A wrapper for WhatsApp Notifications
    """
    service_name = 'WhatsApp'
    service_url = 'https://developers.facebook.com/docs/whatsapp/cloud-api/get-started'
    secure_protocol = 'whatsapp'
    request_rate_per_sec = 0.2
    fb_graph_version = 'v17.0'
    setup_url = 'https://github.com/caronc/apprise/wiki/Notify_whatsapp'
    notify_url = 'https://graph.facebook.com/{fb_ver}/{phone_id}/messages'
    body_maxlen = 1024
    title_maxlen = 0
    templates = ('{schema}://{token}@{from_phone_id}/{targets}', '{schema}://{template}:{token}@{from_phone_id}/{targets}')
    template_tokens = dict(NotifyBase.template_tokens, **{'token': {'name': _('Access Token'), 'type': 'string', 'private': True, 'required': True, 'regex': ('^[a-z0-9]+$', 'i')}, 'template': {'name': _('Template Name'), 'type': 'string', 'required': False, 'regex': ('^[^\\s]+$', 'i')}, 'from_phone_id': {'name': _('From Phone ID'), 'type': 'string', 'private': True, 'required': True, 'regex': ('^[0-9]+$', 'i')}, 'target_phone': {'name': _('Target Phone No'), 'type': 'string', 'prefix': '+', 'regex': ('^[0-9\\s)(+-]+$', 'i'), 'map_to': 'targets'}, 'targets': {'name': _('Targets'), 'type': 'list:string'}, 'language': {'name': _('Language'), 'type': 'string', 'default': 'en_US', 'regex': ('^[^0-9\\s]+$', 'i')}})
    template_args = dict(NotifyBase.template_args, **{'to': {'alias_of': 'targets'}, 'from': {'alias_of': 'from_phone_id'}, 'token': {'alias_of': 'token'}, 'template': {'alias_of': 'template'}, 'lang': {'alias_of': 'language'}})
    component_key_re = re.compile('(?P<key>((?P<id>[1-9][0-9]*)|(?P<map>body|type)))', re.IGNORECASE)
    template_kwargs = {'template_mapping': {'name': _('Template Mapping'), 'prefix': ':'}}

    def __init__(self, token, from_phone_id, template=None, targets=None, language=None, template_mapping=None, **kwargs):
        if False:
            return 10
        '\n        Initialize WhatsApp Object\n        '
        super().__init__(**kwargs)
        self.token = validate_regex(token, *self.template_tokens['token']['regex'])
        if not self.token:
            msg = 'An invalid WhatsApp Access Token ({}) was specified.'.format(token)
            self.logger.warning(msg)
            raise TypeError(msg)
        self.from_phone_id = validate_regex(from_phone_id, *self.template_tokens['from_phone_id']['regex'])
        if not self.from_phone_id:
            msg = 'An invalid WhatsApp From Phone ID ({}) was specified.'.format(from_phone_id)
            self.logger.warning(msg)
            raise TypeError(msg)
        if template:
            self.template = validate_regex(template, *self.template_tokens['template']['regex'])
            if not self.template:
                msg = 'An invalid WhatsApp Template Name ({}) was specified.'.format(template)
                self.logger.warning(msg)
                raise TypeError(msg)
            if language:
                self.language = validate_regex(language, *self.template_tokens['language']['regex'])
                if not self.language:
                    msg = 'An invalid WhatsApp Template Language Code ({}) was specified.'.format(language)
                    self.logger.warning(msg)
                    raise TypeError(msg)
            else:
                self.language = self.template_tokens['language']['default']
        else:
            self.template = None
        self.targets = list()
        for target in parse_phone_no(targets):
            result = is_phone_no(target)
            if not result:
                self.logger.warning('Dropped invalid phone # ({}) specified.'.format(target))
                continue
            self.targets.append('+{}'.format(result['full']))
        self.template_mapping = {}
        if template_mapping:
            self.template_mapping.update(template_mapping)
        self.components = dict()
        self.component_keys = list()
        for (key, val) in self.template_mapping.items():
            matched = self.component_key_re.match(key)
            if not matched:
                msg = 'An invalid Template Component ID ({}) was specified.'.format(key)
                self.logger.warning(msg)
                raise TypeError(msg)
            if matched.group('id'):
                index = matched.group('id')
                map_to = {'type': 'text', 'text': val}
            else:
                map_to = matched.group('map').lower()
                matched = self.component_key_re.match(val)
                if not (matched and matched.group('id')):
                    msg = 'An invalid Template Component Mapping (:{}={}) was specified.'.format(key, val)
                    self.logger.warning(msg)
                    raise TypeError(msg)
                index = matched.group('id')
            if index in self.components:
                msg = 'The Template Component index ({}) was already assigned.'.format(key)
                self.logger.warning(msg)
                raise TypeError(msg)
            self.components[index] = map_to
            self.component_keys = self.components.keys()
            sorted(self.component_keys)
        return

    def send(self, body, title='', notify_type=NotifyType.INFO, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Perform WhatsApp Notification\n        '
        if not self.targets:
            self.logger.warning('There are no valid WhatsApp targets to notify.')
            return False
        has_error = False
        url = self.notify_url.format(fb_ver=self.fb_graph_version, phone_id=self.from_phone_id)
        headers = {'User-Agent': self.app_id, 'Accept': 'application/json', 'Content-Type': 'application/json', 'Authorization': f'Bearer {self.token}'}
        payload = {'messaging_product': 'whatsapp', 'to': None}
        if not self.template:
            payload.update({'recipient_type': 'individual', 'type': 'text', 'text': {'body': body}})
        else:
            payload.update({'type': 'template', 'template': {'name': self.template, 'language': {'code': self.language}}})
            if self.components:
                payload['template']['components'] = [{'type': 'body', 'parameters': []}]
                for key in self.component_keys:
                    if isinstance(self.components[key], dict):
                        payload['template']['components'][0]['parameters'].append(self.components[key])
                        continue
                    payload['template']['components'][0]['parameters'].append({'type': 'text', 'text': body if self.components[key] == 'body' else notify_type})
        targets = list(self.targets)
        while len(targets):
            target = targets.pop(0)
            payload['to'] = target
            self.logger.debug('WhatsApp POST URL: {} (cert_verify={})'.format(url, self.verify_certificate))
            self.logger.debug('WhatsApp Payload: {}'.format(payload))
            self.throttle()
            try:
                r = requests.post(url, data=dumps(payload), headers=headers, verify=self.verify_certificate, timeout=self.request_timeout)
                if r.status_code not in (requests.codes.created, requests.codes.ok):
                    status_str = NotifyBase.http_response_code_lookup(r.status_code)
                    status_code = r.status_code
                    try:
                        json_response = loads(r.content)
                        status_code = json_response['error'].get('code', status_code)
                        status_str = json_response['error'].get('message', status_str)
                    except (AttributeError, TypeError, ValueError, KeyError):
                        pass
                    self.logger.warning('Failed to send WhatsApp notification to {}: {}{}error={}.'.format(target, status_str, ', ' if status_str else '', status_code))
                    self.logger.debug('Response Details:\r\n{}'.format(r.content))
                    has_error = True
                    continue
                else:
                    self.logger.info('Sent WhatsApp notification to {}.'.format(target))
            except requests.RequestException as e:
                self.logger.warning('A Connection error occurred sending WhatsApp:%s ' % target + 'notification.')
                self.logger.debug('Socket Exception: %s' % str(e))
                has_error = True
                continue
        return not has_error

    def url(self, privacy=False, *args, **kwargs):
        if False:
            return 10
        '\n        Returns the URL built dynamically based on specified arguments.\n        '
        params = {}
        if self.template:
            params['lang'] = self.language
        params.update(self.url_parameters(*args, privacy=privacy, **kwargs))
        params.update({':{}'.format(k): v for (k, v) in self.template_mapping.items()})
        return '{schema}://{template}{token}@{from_id}/{targets}/?{params}'.format(schema=self.secure_protocol, from_id=self.pprint(self.from_phone_id, privacy, safe=''), token=self.pprint(self.token, privacy, safe=''), template='' if not self.template else '{}:'.format(NotifyWhatsApp.quote(self.template, safe='')), targets='/'.join([NotifyWhatsApp.quote(x, safe='') for x in self.targets]), params=NotifyWhatsApp.urlencode(params))

    def __len__(self):
        if False:
            return 10
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
        results['targets'] = NotifyWhatsApp.split_path(results['fullpath'])
        results['from_phone_id'] = NotifyWhatsApp.unquote(results['host'])
        if results['password']:
            results['template'] = NotifyWhatsApp.unquote(results['user'])
            results['token'] = NotifyWhatsApp.unquote(results['password'])
        else:
            results['token'] = NotifyWhatsApp.unquote(results['user'])
        if 'token' in results['qsd'] and len(results['qsd']['token']):
            results['token'] = NotifyWhatsApp.unquote(results['qsd']['token'])
        if 'template' in results['qsd'] and len(results['qsd']['template']):
            results['template'] = results['qsd']['template']
        if 'lang' in results['qsd'] and len(results['qsd']['lang']):
            results['language'] = results['qsd']['lang']
        if 'from' in results['qsd'] and len(results['qsd']['from']):
            results['from_phone_id'] = NotifyWhatsApp.unquote(results['qsd']['from'])
        if 'source' in results['qsd'] and len(results['qsd']['source']):
            results['from_phone_id'] = NotifyWhatsApp.unquote(results['qsd']['source'])
        if 'to' in results['qsd'] and len(results['qsd']['to']):
            results['targets'] += NotifyWhatsApp.parse_phone_no(results['qsd']['to'])
        results['template_mapping'] = {NotifyWhatsApp.unquote(x): NotifyWhatsApp.unquote(y) for (x, y) in results['qsd:'].items()}
        return results