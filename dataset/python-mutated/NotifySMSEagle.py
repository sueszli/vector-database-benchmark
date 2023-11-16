import re
import requests
from json import dumps, loads
import base64
from itertools import chain
from .NotifyBase import NotifyBase
from ..common import NotifyType
from ..utils import validate_regex
from ..utils import is_phone_no
from ..utils import parse_phone_no
from ..utils import parse_bool
from ..URLBase import PrivacyMode
from ..AppriseLocale import gettext_lazy as _
GROUP_REGEX = re.compile('^\\s*(\\#|\\%35)(?P<group>[a-z0-9_-]+)', re.I)
CONTACT_REGEX = re.compile('^\\s*(\\@|\\%40)?(?P<contact>[a-z0-9_-]+)', re.I)

class SMSEaglePriority:
    NORMAL = 0
    HIGH = 1
SMSEAGLE_PRIORITIES = (SMSEaglePriority.NORMAL, SMSEaglePriority.HIGH)
SMSEAGLE_PRIORITY_MAP = {'normal': SMSEaglePriority.NORMAL, '+': SMSEaglePriority.HIGH, 'high': SMSEaglePriority.HIGH}

class SMSEagleCategory:
    """
    We define the different category types that we can notify via SMS Eagle
    """
    PHONE = 'phone'
    GROUP = 'group'
    CONTACT = 'contact'
SMSEAGLE_CATEGORIES = (SMSEagleCategory.PHONE, SMSEagleCategory.GROUP, SMSEagleCategory.CONTACT)

class NotifySMSEagle(NotifyBase):
    """
    A wrapper for SMSEagle Notifications
    """
    service_name = 'SMS Eagle'
    service_url = 'https://smseagle.eu'
    protocol = 'smseagle'
    secure_protocol = 'smseagles'
    setup_url = 'https://github.com/caronc/apprise/wiki/Notify_smseagle'
    notify_path = '/jsonrpc/sms'
    attachment_support = True
    body_maxlen = 1200
    default_batch_size = 10
    title_maxlen = 0
    templates = ('{schema}://{token}@{host}/{targets}', '{schema}://{token}@{host}:{port}/{targets}')
    template_tokens = dict(NotifyBase.template_tokens, **{'host': {'name': _('Hostname'), 'type': 'string', 'required': True}, 'port': {'name': _('Port'), 'type': 'int', 'min': 1, 'max': 65535}, 'token': {'name': _('Access Token'), 'type': 'string', 'required': True}, 'target_phone': {'name': _('Target Phone No'), 'type': 'string', 'prefix': '+', 'regex': ('^[0-9\\s)(+-]+$', 'i'), 'map_to': 'targets'}, 'target_group': {'name': _('Target Group ID'), 'type': 'string', 'prefix': '#', 'regex': ('^[a-z0-9_-]+$', 'i'), 'map_to': 'targets'}, 'target_contact': {'name': _('Target Contact'), 'type': 'string', 'prefix': '@', 'regex': ('^[a-z0-9_-]+$', 'i'), 'map_to': 'targets'}, 'targets': {'name': _('Targets'), 'type': 'list:string', 'required': True}})
    template_args = dict(NotifyBase.template_args, **{'to': {'alias_of': 'targets'}, 'token': {'alias_of': 'token'}, 'batch': {'name': _('Batch Mode'), 'type': 'bool', 'default': False}, 'status': {'name': _('Show Status'), 'type': 'bool', 'default': False}, 'test': {'name': _('Test Only'), 'type': 'bool', 'default': False}, 'flash': {'name': _('Flash'), 'type': 'bool', 'default': False}, 'priority': {'name': _('Priority'), 'type': 'choice:int', 'values': SMSEAGLE_PRIORITIES, 'default': SMSEaglePriority.NORMAL}})

    def __init__(self, token=None, targets=None, priority=None, batch=False, status=False, flash=False, test=False, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Initialize SMSEagle Object\n        '
        super().__init__(**kwargs)
        self.flash = flash
        self.test = test
        self.batch = batch
        self.status = status
        self.target_phones = list()
        self.target_groups = list()
        self.target_contacts = list()
        self.invalid_targets = list()
        self.token = validate_regex(self.user if not token else token)
        if not self.token:
            msg = 'An invalid SMSEagle Access Token ({}) was specified.'.format(self.user if not token else token)
            self.logger.warning(msg)
            raise TypeError(msg)
        try:
            self.priority = int(priority)
        except TypeError:
            self.priority = self.template_args['priority']['default']
        except ValueError:
            priority = priority.lower().strip()
            result = next((key for key in SMSEAGLE_PRIORITY_MAP.keys() if key.startswith(priority)), None) if priority else None
            if not result:
                msg = 'An invalid SMSEagle priority ({}) was specified.'.format(priority)
                self.logger.warning(msg)
                raise TypeError(msg)
            self.priority = SMSEAGLE_PRIORITY_MAP[result]
        if self.priority is not None and self.priority not in SMSEAGLE_PRIORITY_MAP.values():
            msg = 'An invalid SMSEagle priority ({}) was specified.'.format(priority)
            self.logger.warning(msg)
            raise TypeError(msg)
        for target in parse_phone_no(targets):
            result = is_phone_no(target, min_len=9)
            if result:
                self.target_phones.append('{}{}'.format('' if target[0] != '+' else '+', result['full']))
                continue
            result = GROUP_REGEX.match(target)
            if result:
                self.target_groups.append(result.group('group'))
                continue
            result = CONTACT_REGEX.match(target)
            if result:
                self.target_contacts.append(result.group('contact'))
                continue
            self.logger.warning('Dropped invalid phone/group/contact ({}) specified.'.format(target))
            self.invalid_targets.append(target)
            continue
        return

    def send(self, body, title='', notify_type=NotifyType.INFO, attach=None, **kwargs):
        if False:
            return 10
        '\n        Perform SMSEagle Notification\n        '
        if not self.target_groups and (not self.target_phones) and (not self.target_contacts):
            self.logger.warning('There were no SMSEagle targets to notify.')
            return False
        has_error = False
        attachments = []
        if attach and self.attachment_support:
            for attachment in attach:
                if not attachment:
                    self.logger.error('Could not access attachment {}.'.format(attachment.url(privacy=True)))
                    return False
                if not re.match('^image/.*', attachment.mimetype, re.I):
                    self.logger.warning('Ignoring unsupported SMSEagle attachment {}.'.format(attachment.url(privacy=True)))
                    continue
                try:
                    with open(attachment.path, 'rb') as f:
                        attachments.append({'content_type': attachment.mimetype, 'content': base64.b64encode(f.read()).decode('utf-8')})
                except (OSError, IOError) as e:
                    self.logger.warning('An I/O error occurred while reading {}.'.format(attachment.name if attachment else 'attachment'))
                    self.logger.debug('I/O Exception: %s' % str(e))
                    return False
        headers = {'User-Agent': self.app_id, 'Content-Type': 'application/json'}
        params_template = {'access_token': self.token, 'message': None, 'highpriority': self.priority, 'unicode': 1, 'message_type': 'sms', 'responsetype': 'extended', 'flash': 1 if self.flash else 0, 'test': 1 if self.test else 0}
        schema = 'https' if self.secure else 'http'
        notify_url = '%s://%s' % (schema, self.host)
        if isinstance(self.port, int):
            notify_url += ':%d' % self.port
        notify_url += self.notify_path
        batch_size = 1 if not self.batch else self.default_batch_size
        notify_by = {SMSEagleCategory.PHONE: {'method': 'sms.send_sms', 'target': 'to'}, SMSEagleCategory.GROUP: {'method': 'sms.send_togroup', 'target': 'groupname'}, SMSEagleCategory.CONTACT: {'method': 'sms.send_tocontact', 'target': 'contactname'}}
        for category in SMSEAGLE_CATEGORIES:
            payload = {'method': notify_by[category]['method'], 'params': {notify_by[category]['target']: None}}
            payload['params'].update(params_template)
            payload['params']['message'] = '{}{}'.format('' if not self.status else '{} '.format(self.asset.ascii(notify_type)), body)
            if attachments:
                payload['params']['message_type'] = 'mms'
                payload['params']['attachments'] = attachments
            targets = getattr(self, 'target_{}s'.format(category))
            for index in range(0, len(targets), batch_size):
                payload['params'][notify_by[category]['target']] = ','.join(targets[index:index + batch_size])
                self.logger.debug('SMSEagle POST URL: %s (cert_verify=%r)' % (notify_url, self.verify_certificate))
                self.logger.debug('SMSEagle Payload: %s' % str(payload))
                self.throttle()
                try:
                    r = requests.post(notify_url, data=dumps(payload), headers=headers, verify=self.verify_certificate, timeout=self.request_timeout)
                    try:
                        content = loads(r.content)
                        status_str = str(content['result'])
                    except (AttributeError, TypeError, ValueError, KeyError):
                        content = {}
                    if r.status_code not in (requests.codes.ok, requests.codes.created) or not isinstance(content.get('result'), (dict, list)) or (isinstance(content.get('result'), dict) and content['result'].get('status') != 'ok') or (isinstance(content.get('result'), list) and next((True for entry in content.get('result') if isinstance(entry, dict) and entry.get('status') != 'ok'), False)):
                        status_str = content.get('result') if content.get('result') else NotifySMSEagle.http_response_code_lookup(r.status_code)
                        self.logger.warning('Failed to send {} {} SMSEagle {} notification: {}{}error={}.'.format(len(targets[index:index + batch_size]), 'to {}'.format(targets[index]) if batch_size == 1 else '(s)', category, status_str, ', ' if status_str else '', r.status_code))
                        self.logger.debug('Response {} Details:\r\n{}'.format(category.upper(), r.content))
                        has_error = True
                        continue
                    else:
                        self.logger.info('Sent {} SMSEagle {} notification{}.'.format(len(targets[index:index + batch_size]), category, ' to {}'.format(targets[index]) if batch_size == 1 else '(s)'))
                except requests.RequestException as e:
                    self.logger.warning('A Connection error occured sending {} SMSEagle {} notification(s).'.format(len(targets[index:index + batch_size]), category))
                    self.logger.debug('Socket Exception: %s' % str(e))
                    has_error = True
                    continue
        return not has_error

    def url(self, privacy=False, *args, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Returns the URL built dynamically based on specified arguments.\n        '
        params = {'batch': 'yes' if self.batch else 'no', 'status': 'yes' if self.status else 'no', 'flash': 'yes' if self.flash else 'no', 'test': 'yes' if self.test else 'no'}
        params.update(self.url_parameters(*args, privacy=privacy, **kwargs))
        default_priority = self.template_args['priority']['default']
        if self.priority is not None:
            params['priority'] = next((key for (key, value) in SMSEAGLE_PRIORITY_MAP.items() if value == self.priority), default_priority)
        default_port = 443 if self.secure else 80
        return '{schema}://{token}@{hostname}{port}/{targets}?{params}'.format(schema=self.secure_protocol if self.secure else self.protocol, token=self.pprint(self.token, privacy, mode=PrivacyMode.Secret, safe=''), hostname=self.host, port='' if self.port is None or self.port == default_port else ':{}'.format(self.port), targets='/'.join([NotifySMSEagle.quote(x, safe='#@') for x in chain(self.target_phones, ['@{}'.format(x) for x in self.target_contacts], ['#{}'.format(x) for x in self.target_groups], self.invalid_targets)]), params=NotifySMSEagle.urlencode(params))

    def __len__(self):
        if False:
            i = 10
            return i + 15
        '\n        Returns the number of targets associated with this notification\n        '
        batch_size = 1 if not self.batch else self.default_batch_size
        if batch_size > 1:
            total_targets = 0
            for c in SMSEAGLE_CATEGORIES:
                targets = len(getattr(self, f'target_{c}s'))
                total_targets += int(targets / batch_size) + (1 if targets % batch_size else 0)
            return total_targets
        return len(self.target_phones) + len(self.target_contacts) + len(self.target_groups)

    @staticmethod
    def parse_url(url):
        if False:
            i = 10
            return i + 15
        '\n        Parses the URL and returns enough arguments that can allow\n        us to re-instantiate this object.\n\n        '
        results = NotifyBase.parse_url(url, verify_host=False)
        if not results:
            return results
        results['targets'] = NotifySMSEagle.split_path(results['fullpath'])
        if 'token' in results['qsd'] and len(results['qsd']['token']):
            results['token'] = NotifySMSEagle.unquote(results['qsd']['token'])
        elif not results['password'] and results['user']:
            results['token'] = NotifySMSEagle.unquote(results['user'])
        if 'to' in results['qsd'] and len(results['qsd']['to']):
            results['targets'] += NotifySMSEagle.parse_phone_no(results['qsd']['to'])
        results['batch'] = parse_bool(results['qsd'].get('batch', False))
        results['flash'] = parse_bool(results['qsd'].get('flash', False))
        results['test'] = parse_bool(results['qsd'].get('test', False))
        results['status'] = parse_bool(results['qsd'].get('status', False))
        if 'priority' in results['qsd'] and len(results['qsd']['priority']):
            results['priority'] = NotifySMSEagle.unquote(results['qsd']['priority'])
        return results