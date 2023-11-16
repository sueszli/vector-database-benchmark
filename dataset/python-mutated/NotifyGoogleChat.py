import re
import requests
from json import dumps
from .NotifyBase import NotifyBase
from ..common import NotifyFormat
from ..common import NotifyType
from ..utils import validate_regex
from ..AppriseLocale import gettext_lazy as _

class NotifyGoogleChat(NotifyBase):
    """
    A wrapper to Google Chat Notifications

    """
    service_name = 'Google Chat'
    service_url = 'https://chat.google.com/'
    secure_protocol = 'gchat'
    setup_url = 'https://github.com/caronc/apprise/wiki/Notify_googlechat'
    notify_url = 'https://chat.googleapis.com/v1/spaces/{workspace}/messages'
    notify_format = NotifyFormat.MARKDOWN
    title_maxlen = 0
    body_maxlen = 4000
    templates = ('{schema}://{workspace}/{webhook_key}/{webhook_token}', '{schema}://{workspace}/{webhook_key}/{webhook_token}/{thread_key}')
    template_tokens = dict(NotifyBase.template_tokens, **{'workspace': {'name': _('Workspace'), 'type': 'string', 'private': True, 'required': True}, 'webhook_key': {'name': _('Webhook Key'), 'type': 'string', 'private': True, 'required': True}, 'webhook_token': {'name': _('Webhook Token'), 'type': 'string', 'private': True, 'required': True}, 'thread_key': {'name': _('Thread Key'), 'type': 'string', 'private': True}})
    template_args = dict(NotifyBase.template_args, **{'workspace': {'alias_of': 'workspace'}, 'key': {'alias_of': 'webhook_key'}, 'token': {'alias_of': 'webhook_token'}, 'thread': {'alias_of': 'thread_key'}})

    def __init__(self, workspace, webhook_key, webhook_token, thread_key=None, **kwargs):
        if False:
            return 10
        '\n        Initialize Google Chat Object\n\n        '
        super().__init__(**kwargs)
        self.workspace = validate_regex(workspace)
        if not self.workspace:
            msg = 'An invalid Google Chat Workspace ({}) was specified.'.format(workspace)
            self.logger.warning(msg)
            raise TypeError(msg)
        self.webhook_key = validate_regex(webhook_key)
        if not self.webhook_key:
            msg = 'An invalid Google Chat Webhook Key ({}) was specified.'.format(webhook_key)
            self.logger.warning(msg)
            raise TypeError(msg)
        self.webhook_token = validate_regex(webhook_token)
        if not self.webhook_token:
            msg = 'An invalid Google Chat Webhook Token ({}) was specified.'.format(webhook_token)
            self.logger.warning(msg)
            raise TypeError(msg)
        if thread_key:
            self.thread_key = validate_regex(thread_key)
            if not self.thread_key:
                msg = 'An invalid Google Chat Thread Key ({}) was specified.'.format(thread_key)
                self.logger.warning(msg)
                raise TypeError(msg)
        else:
            self.thread_key = None
        return

    def send(self, body, title='', notify_type=NotifyType.INFO, **kwargs):
        if False:
            print('Hello World!')
        '\n        Perform Google Chat Notification\n        '
        headers = {'User-Agent': self.app_id, 'Content-Type': 'application/json; charset=utf-8'}
        payload = {'text': body}
        notify_url = self.notify_url.format(workspace=self.workspace)
        params = {'token': self.webhook_token, 'key': self.webhook_key}
        if self.thread_key:
            params['threadKey'] = self.thread_key
        self.logger.debug('Google Chat POST URL: %s (cert_verify=%r)' % (notify_url, self.verify_certificate))
        self.logger.debug('Google Chat Parameters: %s' % str(params))
        self.logger.debug('Google Chat Payload: %s' % str(payload))
        self.throttle()
        try:
            r = requests.post(notify_url, params=params, data=dumps(payload), headers=headers, verify=self.verify_certificate, timeout=self.request_timeout)
            if r.status_code not in (requests.codes.ok, requests.codes.no_content):
                status_str = NotifyBase.http_response_code_lookup(r.status_code)
                self.logger.warning('Failed to send Google Chat notification: {}{}error={}.'.format(status_str, ', ' if status_str else '', r.status_code))
                self.logger.debug('Response Details:\r\n{}'.format(r.content))
                return False
            else:
                self.logger.info('Sent Google Chat notification.')
        except requests.RequestException as e:
            self.logger.warning('A Connection error occurred postingto Google Chat.')
            self.logger.debug('Socket Exception: %s' % str(e))
            return False
        return True

    def url(self, privacy=False, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Returns the URL built dynamically based on specified arguments.\n        '
        params = self.url_parameters(*args, privacy=privacy, **kwargs)
        return '{schema}://{workspace}/{key}/{token}/{thread}?{params}'.format(schema=self.secure_protocol, workspace=self.pprint(self.workspace, privacy, safe=''), key=self.pprint(self.webhook_key, privacy, safe=''), token=self.pprint(self.webhook_token, privacy, safe=''), thread='' if not self.thread_key else self.pprint(self.thread_key, privacy, safe=''), params=NotifyGoogleChat.urlencode(params))

    @staticmethod
    def parse_url(url):
        if False:
            print('Hello World!')
        '\n        Parses the URL and returns enough arguments that can allow\n        us to re-instantiate this object.\n\n        Syntax:\n          gchat://workspace/webhook_key/webhook_token\n          gchat://workspace/webhook_key/webhook_token/thread_key\n\n        '
        results = NotifyBase.parse_url(url, verify_host=False)
        if not results:
            return results
        results['workspace'] = NotifyGoogleChat.unquote(results['host'])
        tokens = NotifyGoogleChat.split_path(results['fullpath'])
        results['webhook_key'] = tokens.pop(0) if tokens else None
        results['webhook_token'] = tokens.pop(0) if tokens else None
        results['thread_key'] = tokens.pop(0) if tokens else None
        if 'workspace' in results['qsd']:
            results['workspace'] = NotifyGoogleChat.unquote(results['qsd']['workspace'])
        if 'key' in results['qsd']:
            results['webhook_key'] = NotifyGoogleChat.unquote(results['qsd']['key'])
        if 'token' in results['qsd']:
            results['webhook_token'] = NotifyGoogleChat.unquote(results['qsd']['token'])
        if 'thread' in results['qsd']:
            results['thread_key'] = NotifyGoogleChat.unquote(results['qsd']['thread'])
        elif 'threadkey' in results['qsd']:
            results['thread_key'] = NotifyGoogleChat.unquote(results['qsd']['threadkey'])
        return results

    @staticmethod
    def parse_native_url(url):
        if False:
            for i in range(10):
                print('nop')
        "\n        Support\n           https://chat.googleapis.com/v1/spaces/{workspace}/messages\n                 '?key={key}&token={token}\n           https://chat.googleapis.com/v1/spaces/{workspace}/messages\n                 '?key={key}&token={token}&threadKey={thread}\n        "
        result = re.match('^https://chat\\.googleapis\\.com/v1/spaces/(?P<workspace>[A-Z0-9_-]+)/messages/*(?P<params>.+)$', url, re.I)
        if result:
            return NotifyGoogleChat.parse_url('{schema}://{workspace}/{params}'.format(schema=NotifyGoogleChat.secure_protocol, workspace=result.group('workspace'), params=result.group('params')))
        return None