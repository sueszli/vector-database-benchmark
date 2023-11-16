import requests
import hashlib
from json import dumps
from json import loads
from .NotifyBase import NotifyBase
from ..URLBase import PrivacyMode
from ..utils import parse_bool
from ..common import NotifyType
from .. import __version__ as VERSION
from ..AppriseLocale import gettext_lazy as _

class NotifyEmby(NotifyBase):
    """
    A wrapper for Emby Notifications
    """
    service_name = 'Emby'
    service_url = 'https://emby.media/'
    protocol = 'emby'
    secure_protocol = 'embys'
    setup_url = 'https://github.com/caronc/apprise/wiki/Notify_emby'
    emby_device_id = '48df9504-6843-49be-9f2d-a685e25a0bc8'
    emby_message_timeout_ms = 60000
    templates = ('{schema}://{host}', '{schema}://{host}:{port}', '{schema}://{user}:{password}@{host}', '{schema}://{user}:{password}@{host}:{port}')
    template_tokens = dict(NotifyBase.template_tokens, **{'host': {'name': _('Hostname'), 'type': 'string', 'required': True}, 'port': {'name': _('Port'), 'type': 'int', 'min': 1, 'max': 65535, 'default': 8096}, 'user': {'name': _('Username'), 'type': 'string'}, 'password': {'name': _('Password'), 'type': 'string', 'private': True}})
    template_args = dict(NotifyBase.template_args, **{'modal': {'name': _('Modal'), 'type': 'bool', 'default': False}})

    def __init__(self, modal=False, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Initialize Emby Object\n\n        '
        super().__init__(**kwargs)
        if self.secure:
            self.schema = 'https'
        else:
            self.schema = 'http'
        self.access_token = None
        self.user_id = None
        self.modal = modal
        if not self.port:
            self.port = self.template_tokens['port']['default']
        if not self.user:
            msg = 'No Emby username was specified.'
            self.logger.warning(msg)
            raise TypeError(msg)
        return

    def login(self, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Creates our authentication token and prepares our header\n\n        '
        if self.is_authenticated:
            self.logout()
        url = '%s://%s' % (self.schema, self.host)
        if self.port:
            url += ':%d' % self.port
        url += '/Users/AuthenticateByName'
        payload = {'Username': self.user}
        headers = {'User-Agent': self.app_id, 'Content-Type': 'application/json', 'X-Emby-Authorization': self.emby_auth_header}
        if self.password:
            payload['pw'] = self.password
            password_md5 = hashlib.md5()
            password_md5.update(self.password.encode('utf-8'))
            payload['passwordMd5'] = password_md5.hexdigest()
            password_sha1 = hashlib.sha1()
            password_sha1.update(self.password.encode('utf-8'))
            payload['password'] = password_sha1.hexdigest()
        else:
            payload['password'] = ''
            payload['passwordMd5'] = ''
            payload['pw'] = ''
        self.logger.debug('Emby login() POST URL: %s (cert_verify=%r)' % (url, self.verify_certificate))
        try:
            r = requests.post(url, headers=headers, data=dumps(payload), verify=self.verify_certificate, timeout=self.request_timeout)
            if r.status_code != requests.codes.ok:
                status_str = NotifyEmby.http_response_code_lookup(r.status_code)
                self.logger.warning('Failed to authenticate Emby user {} details: {}{}error={}.'.format(self.user, status_str, ', ' if status_str else '', r.status_code))
                self.logger.debug('Response Details:\r\n{}'.format(r.content))
                return False
        except requests.RequestException as e:
            self.logger.warning('A Connection error occurred authenticating a user with Emby at %s.' % self.host)
            self.logger.debug('Socket Exception: %s' % str(e))
            return False
        try:
            results = loads(r.content)
        except (AttributeError, TypeError, ValueError):
            return False
        self.access_token = results.get('AccessToken')
        self.user_id = results.get('Id')
        if not self.user_id:
            if 'User' in results:
                self.user_id = results['User'].get('Id')
        return self.is_authenticated

    def sessions(self, user_controlled=True):
        if False:
            while True:
                i = 10
        '\n        Acquire our Session Identifiers and store them in a dictionary\n        indexed by the session id itself.\n\n        '
        sessions = dict()
        if not self.is_authenticated and (not self.login()):
            return sessions
        url = '%s://%s' % (self.schema, self.host)
        if self.port:
            url += ':%d' % self.port
        url += '/Sessions'
        if user_controlled is True:
            url += '?ControllableByUserId=%s' % self.user_id
        headers = {'User-Agent': self.app_id, 'Content-Type': 'application/json', 'X-Emby-Authorization': self.emby_auth_header, 'X-MediaBrowser-Token': self.access_token}
        self.logger.debug('Emby session() GET URL: %s (cert_verify=%r)' % (url, self.verify_certificate))
        try:
            r = requests.get(url, headers=headers, verify=self.verify_certificate, timeout=self.request_timeout)
            if r.status_code != requests.codes.ok:
                status_str = NotifyEmby.http_response_code_lookup(r.status_code)
                self.logger.warning('Failed to acquire Emby session for user {}: {}{}error={}.'.format(self.user, status_str, ', ' if status_str else '', r.status_code))
                self.logger.debug('Response Details:\r\n{}'.format(r.content))
                return sessions
        except requests.RequestException as e:
            self.logger.warning('A Connection error occurred querying Emby for session information at %s.' % self.host)
            self.logger.debug('Socket Exception: %s' % str(e))
            return sessions
        try:
            results = loads(r.content)
        except (AttributeError, TypeError, ValueError):
            return sessions
        for entry in results:
            session = entry.get('Id')
            if session:
                sessions[session] = entry
        return sessions

    def logout(self, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Logs out of an already-authenticated session\n\n        '
        if not self.is_authenticated:
            return True
        url = '%s://%s' % (self.schema, self.host)
        if self.port:
            url += ':%d' % self.port
        url += '/Sessions/Logout'
        headers = {'User-Agent': self.app_id, 'Content-Type': 'application/json', 'X-Emby-Authorization': self.emby_auth_header, 'X-MediaBrowser-Token': self.access_token}
        self.logger.debug('Emby logout() POST URL: %s (cert_verify=%r)' % (url, self.verify_certificate))
        try:
            r = requests.post(url, headers=headers, verify=self.verify_certificate, timeout=self.request_timeout)
            if r.status_code not in (requests.codes.unauthorized, requests.codes.ok, requests.codes.no_content):
                status_str = NotifyEmby.http_response_code_lookup(r.status_code)
                self.logger.warning('Failed to logoff Emby user {}: {}{}error={}.'.format(self.user, status_str, ', ' if status_str else '', r.status_code))
                self.logger.debug('Response Details:\r\n{}'.format(r.content))
                return False
        except requests.RequestException as e:
            self.logger.warning('A Connection error occurred querying Emby to logoff user %s at %s.' % (self.user, self.host))
            self.logger.debug('Socket Exception: %s' % str(e))
            return False
        self.access_token = None
        self.user_id = None
        return True

    def send(self, body, title='', notify_type=NotifyType.INFO, **kwargs):
        if False:
            return 10
        '\n        Perform Emby Notification\n        '
        if not self.is_authenticated and (not self.login()):
            return False
        sessions = self.sessions().keys()
        if not sessions:
            self.logger.warning('There were no Emby sessions to notify.')
            return True
        url = '%s://%s' % (self.schema, self.host)
        if self.port:
            url += ':%d' % self.port
        url += '/Sessions/%s/Message'
        payload = {'Header': title, 'Text': body}
        if not self.modal:
            payload['TimeoutMs'] = self.emby_message_timeout_ms
        headers = {'User-Agent': self.app_id, 'Content-Type': 'application/json', 'X-Emby-Authorization': self.emby_auth_header, 'X-MediaBrowser-Token': self.access_token}
        has_error = False
        for session in sessions:
            session_url = url % session
            self.logger.debug('Emby POST URL: %s (cert_verify=%r)' % (session_url, self.verify_certificate))
            self.logger.debug('Emby Payload: %s' % str(payload))
            self.throttle()
            try:
                r = requests.post(session_url, data=dumps(payload), headers=headers, verify=self.verify_certificate, timeout=self.request_timeout)
                if r.status_code not in (requests.codes.ok, requests.codes.no_content):
                    status_str = NotifyEmby.http_response_code_lookup(r.status_code)
                    self.logger.warning('Failed to send Emby notification: {}{}error={}.'.format(status_str, ', ' if status_str else '', r.status_code))
                    self.logger.debug('Response Details:\r\n{}'.format(r.content))
                    has_error = True
                    continue
                else:
                    self.logger.info('Sent Emby notification.')
            except requests.RequestException as e:
                self.logger.warning('A Connection error occurred sending Emby notification to %s.' % self.host)
                self.logger.debug('Socket Exception: %s' % str(e))
                has_error = True
                continue
        return not has_error

    def url(self, privacy=False, *args, **kwargs):
        if False:
            return 10
        '\n        Returns the URL built dynamically based on specified arguments.\n        '
        params = {'modal': 'yes' if self.modal else 'no'}
        params.update(self.url_parameters(*args, privacy=privacy, **kwargs))
        auth = ''
        if self.user and self.password:
            auth = '{user}:{password}@'.format(user=NotifyEmby.quote(self.user, safe=''), password=self.pprint(self.password, privacy, mode=PrivacyMode.Secret, safe=''))
        else:
            auth = '{user}@'.format(user=NotifyEmby.quote(self.user, safe=''))
        return '{schema}://{auth}{hostname}{port}/?{params}'.format(schema=self.secure_protocol if self.secure else self.protocol, auth=auth, hostname=self.host, port='' if self.port is None or self.port == self.template_tokens['port']['default'] else ':{}'.format(self.port), params=NotifyEmby.urlencode(params))

    @property
    def is_authenticated(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Returns True if we're authenticated and False if not.\n\n        "
        return True if self.access_token and self.user_id else False

    @property
    def emby_auth_header(self):
        if False:
            print('Hello World!')
        "\n        Generates the X-Emby-Authorization header response based on whether\n        we're authenticated or not.\n\n        "
        header_args = [('MediaBrowser Client', self.app_id), ('Device', self.app_id), ('DeviceId', self.emby_device_id), ('Version', str(VERSION))]
        if self.user_id:
            header_args.append(('UserId', self.user))
        return ', '.join(['%s="%s"' % (k, v) for (k, v) in header_args])

    @staticmethod
    def parse_url(url):
        if False:
            for i in range(10):
                print('nop')
        '\n        Parses the URL and returns enough arguments that can allow\n        us to re-instantiate this object.\n\n        '
        results = NotifyBase.parse_url(url)
        if not results:
            return results
        results['modal'] = parse_bool(results['qsd'].get('modal', False))
        return results

    def __del__(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Destructor\n        '
        try:
            self.logout()
        except LookupError:
            pass
        except ImportError:
            pass