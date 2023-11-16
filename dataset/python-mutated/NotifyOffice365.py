import requests
from datetime import datetime
from datetime import timedelta
from json import loads
from json import dumps
from .NotifyBase import NotifyBase
from ..URLBase import PrivacyMode
from ..common import NotifyFormat
from ..common import NotifyType
from ..utils import is_email
from ..utils import parse_emails
from ..utils import validate_regex
from ..AppriseLocale import gettext_lazy as _

class NotifyOffice365(NotifyBase):
    """
    A wrapper for Office 365 Notifications
    """
    service_name = 'Office 365'
    service_url = 'https://office.com/'
    secure_protocol = 'o365'
    request_rate_per_sec = 0.2
    setup_url = 'https://github.com/caronc/apprise/wiki/Notify_office365'
    graph_url = 'https://graph.microsoft.com'
    auth_url = 'https://login.microsoftonline.com/{tenant}/oauth2/v2.0/token'
    scope = '.default'
    notify_format = NotifyFormat.HTML
    templates = ('{schema}://{tenant}:{email}/{client_id}/{secret}', '{schema}://{tenant}:{email}/{client_id}/{secret}/{targets}')
    template_tokens = dict(NotifyBase.template_tokens, **{'tenant': {'name': _('Tenant Domain'), 'type': 'string', 'required': True, 'private': True, 'regex': ('^[a-z0-9-]+$', 'i')}, 'email': {'name': _('Account Email'), 'type': 'string', 'required': True}, 'client_id': {'name': _('Client ID'), 'type': 'string', 'required': True, 'private': True, 'regex': ('^[a-z0-9-]+$', 'i')}, 'secret': {'name': _('Client Secret'), 'type': 'string', 'private': True, 'required': True}, 'target_email': {'name': _('Target Email'), 'type': 'string', 'map_to': 'targets'}, 'targets': {'name': _('Targets'), 'type': 'list:string'}})
    template_args = dict(NotifyBase.template_args, **{'to': {'alias_of': 'targets'}, 'cc': {'name': _('Carbon Copy'), 'type': 'list:string'}, 'bcc': {'name': _('Blind Carbon Copy'), 'type': 'list:string'}, 'oauth_id': {'alias_of': 'client_id'}, 'oauth_secret': {'alias_of': 'secret'}})

    def __init__(self, tenant, email, client_id, secret, targets=None, cc=None, bcc=None, **kwargs):
        if False:
            print('Hello World!')
        '\n        Initialize Office 365 Object\n        '
        super().__init__(**kwargs)
        self.tenant = validate_regex(tenant, *self.template_tokens['tenant']['regex'])
        if not self.tenant:
            msg = 'An invalid Office 365 Tenant({}) was specified.'.format(tenant)
            self.logger.warning(msg)
            raise TypeError(msg)
        result = is_email(email)
        if not result:
            msg = 'An invalid Office 365 Email Account ID({}) was specified.'.format(email)
            self.logger.warning(msg)
            raise TypeError(msg)
        self.email = result['full_email']
        self.client_id = validate_regex(client_id, *self.template_tokens['client_id']['regex'])
        if not self.client_id:
            msg = 'An invalid Office 365 Client OAuth2 ID ({}) was specified.'.format(client_id)
            self.logger.warning(msg)
            raise TypeError(msg)
        self.secret = validate_regex(secret)
        if not self.secret:
            msg = 'An invalid Office 365 Client OAuth2 Secret ({}) was specified.'.format(secret)
            self.logger.warning(msg)
            raise TypeError(msg)
        self.names = {}
        self.cc = set()
        self.bcc = set()
        self.targets = list()
        if targets:
            for recipient in parse_emails(targets):
                result = is_email(recipient)
                if result:
                    self.targets.append((result['name'] if result['name'] else False, result['full_email']))
                    continue
                self.logger.warning('Dropped invalid To email ({}) specified.'.format(recipient))
        else:
            self.targets.append((False, self.email))
        for recipient in parse_emails(cc):
            email = is_email(recipient)
            if email:
                self.cc.add(email['full_email'])
                self.names[email['full_email']] = email['name'] if email['name'] else False
                continue
            self.logger.warning('Dropped invalid Carbon Copy email ({}) specified.'.format(recipient))
        for recipient in parse_emails(bcc):
            email = is_email(recipient)
            if email:
                self.bcc.add(email['full_email'])
                self.names[email['full_email']] = email['name'] if email['name'] else False
                continue
            self.logger.warning('Dropped invalid Blind Carbon Copy email ({}) specified.'.format(recipient))
        self.token = None
        self.token_expiry = datetime.now()
        return

    def send(self, body, title='', notify_type=NotifyType.INFO, **kwargs):
        if False:
            return 10
        '\n        Perform Office 365 Notification\n        '
        has_error = False
        if not self.targets:
            self.logger.warning('There are no Email recipients to notify')
            return False
        content_type = 'HTML' if self.notify_format == NotifyFormat.HTML else 'Text'
        payload = {'Message': {'Subject': title, 'Body': {'ContentType': content_type, 'Content': body}}, 'SaveToSentItems': 'false'}
        emails = list(self.targets)
        url = '{graph_url}/v1.0/users/{email}/sendmail'.format(email=self.email, graph_url=self.graph_url)
        while len(emails):
            if not self.authenticate():
                return False
            (to_name, to_addr) = emails.pop(0)
            cc = self.cc - self.bcc - set([to_addr])
            bcc = self.bcc - set([to_addr])
            payload['Message']['ToRecipients'] = [{'EmailAddress': {'Address': to_addr}}]
            if to_name:
                payload['Message']['ToRecipients'][0]['EmailAddress']['Name'] = to_name
            self.logger.debug('Email To: {}'.format(to_addr))
            if cc:
                payload['Message']['CcRecipients'] = []
                for addr in cc:
                    _payload = {'Address': addr}
                    if self.names.get(addr):
                        _payload['Name'] = self.names[addr]
                    payload['Message']['CcRecipients'].append({'EmailAddress': _payload})
                self.logger.debug('Email Cc: {}'.format(', '.join(['{}{}'.format('' if self.names.get(e) else '{}: '.format(self.names[e]), e) for e in cc])))
            if bcc:
                payload['Message']['BccRecipients'] = []
                for addr in bcc:
                    _payload = {'Address': addr}
                    if self.names.get(addr):
                        _payload['Name'] = self.names[addr]
                    payload['Message']['BccRecipients'].append({'EmailAddress': _payload})
                self.logger.debug('Email Bcc: {}'.format(', '.join(['{}{}'.format('' if self.names.get(e) else '{}: '.format(self.names[e]), e) for e in bcc])))
            (postokay, response) = self._fetch(url=url, payload=dumps(payload), content_type='application/json')
            if not postokay:
                has_error = True
        return not has_error

    def authenticate(self):
        if False:
            i = 10
            return i + 15
        '\n        Logs into and acquires us an authentication token to work with\n        '
        if self.token and self.token_expiry > datetime.now():
            self.logger.debug('Already authenticate with token {}'.format(self.token))
            return True
        payload = {'client_id': self.client_id, 'client_secret': self.secret, 'scope': '{graph_url}/{scope}'.format(graph_url=self.graph_url, scope=self.scope), 'grant_type': 'client_credentials'}
        url = self.auth_url.format(tenant=self.tenant)
        (postokay, response) = self._fetch(url=url, payload=payload)
        if not postokay:
            return False
        self.token = None
        try:
            self.token_expiry = datetime.now() + timedelta(seconds=int(response.get('expires_in')) - 10)
        except (ValueError, AttributeError, TypeError):
            return False
        self.token = response.get('access_token')
        return True if self.token else False

    def _fetch(self, url, payload, content_type='application/x-www-form-urlencoded'):
        if False:
            i = 10
            return i + 15
        '\n        Wrapper to request object\n\n        '
        headers = {'User-Agent': self.app_id, 'Content-Type': content_type}
        if self.token:
            headers['Authorization'] = 'Bearer ' + self.token
        content = {}
        self.logger.debug('Office 365 POST URL: {} (cert_verify={})'.format(url, self.verify_certificate))
        self.logger.debug('Office 365 Payload: {}'.format(payload))
        self.throttle()
        try:
            r = requests.post(url, data=payload, headers=headers, verify=self.verify_certificate, timeout=self.request_timeout)
            if r.status_code not in (requests.codes.ok, requests.codes.accepted):
                status_str = NotifyOffice365.http_response_code_lookup(r.status_code)
                self.logger.warning('Failed to send Office 365 POST to {}: {}error={}.'.format(url, ', ' if status_str else '', r.status_code))
                self.logger.debug('Response Details:\r\n{}'.format(r.content))
                return (False, content)
            try:
                content = loads(r.content)
            except (AttributeError, TypeError, ValueError):
                content = {}
        except requests.RequestException as e:
            self.logger.warning('Exception received when sending Office 365 POST to {}: '.format(url))
            self.logger.debug('Socket Exception: %s' % str(e))
            return (False, content)
        return (True, content)

    def url(self, privacy=False, *args, **kwargs):
        if False:
            return 10
        '\n        Returns the URL built dynamically based on specified arguments.\n        '
        params = self.url_parameters(*args, privacy=privacy, **kwargs)
        if self.cc:
            params['cc'] = ','.join(['{}{}'.format('' if not self.names.get(e) else '{}:'.format(self.names[e]), e) for e in self.cc])
        if self.bcc:
            params['bcc'] = ','.join(['{}{}'.format('' if not self.names.get(e) else '{}:'.format(self.names[e]), e) for e in self.bcc])
        return '{schema}://{tenant}:{email}/{client_id}/{secret}/{targets}/?{params}'.format(schema=self.secure_protocol, tenant=self.pprint(self.tenant, privacy, safe=''), email=self.email, client_id=self.pprint(self.client_id, privacy, safe=''), secret=self.pprint(self.secret, privacy, mode=PrivacyMode.Secret, safe=''), targets='/'.join([NotifyOffice365.quote('{}{}'.format('' if not e[0] else '{}:'.format(e[0]), e[1]), safe='') for e in self.targets]), params=NotifyOffice365.urlencode(params))

    def __len__(self):
        if False:
            print('Hello World!')
        '\n        Returns the number of targets associated with this notification\n        '
        return len(self.targets)

    @staticmethod
    def parse_url(url):
        if False:
            for i in range(10):
                print('nop')
        '\n        Parses the URL and returns enough arguments that can allow\n        us to re-instantiate this object.\n\n        '
        results = NotifyBase.parse_url(url, verify_host=False)
        if not results:
            return results
        entries = NotifyOffice365.split_path(results['fullpath'])
        try:
            results['client_id'] = NotifyOffice365.unquote(entries.pop(0))
        except IndexError:
            pass
        results['targets'] = list()
        while entries:
            entry = NotifyOffice365.unquote(entries.pop(-1))
            if is_email(entry):
                results['targets'].append(entry)
                continue
            entries.append(NotifyOffice365.quote(entry, safe=''))
            break
        results['tenant'] = None
        results['secret'] = '/'.join([NotifyOffice365.unquote(x) for x in entries])
        if results['password']:
            results['email'] = '{}@{}'.format(NotifyOffice365.unquote(results['password']), NotifyOffice365.unquote(results['host']))
            results['tenant'] = NotifyOffice365.unquote(results['user'])
        else:
            results['email'] = '{}@{}'.format(NotifyOffice365.unquote(results['user']), NotifyOffice365.unquote(results['host']))
        if 'oauth_id' in results['qsd'] and len(results['qsd']['oauth_id']):
            results['client_id'] = NotifyOffice365.unquote(results['qsd']['oauth_id'])
        if 'oauth_secret' in results['qsd'] and len(results['qsd']['oauth_secret']):
            results['secret'] = NotifyOffice365.unquote(results['qsd']['oauth_secret'])
        if 'from' in results['qsd'] and len(results['qsd']['from']):
            results['email'] = NotifyOffice365.unquote(results['qsd']['from'])
        if 'tenant' in results['qsd'] and len(results['qsd']['tenant']):
            results['tenant'] = NotifyOffice365.unquote(results['qsd']['tenant'])
        if 'to' in results['qsd'] and len(results['qsd']['to']):
            results['targets'] += NotifyOffice365.parse_list(results['qsd']['to'])
        if 'cc' in results['qsd'] and len(results['qsd']['cc']):
            results['cc'] = results['qsd']['cc']
        if 'bcc' in results['qsd'] and len(results['qsd']['bcc']):
            results['bcc'] = results['qsd']['bcc']
        return results