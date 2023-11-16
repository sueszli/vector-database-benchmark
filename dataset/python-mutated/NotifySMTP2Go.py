import base64
import requests
from json import dumps
from email.utils import formataddr
from .NotifyBase import NotifyBase
from ..common import NotifyType
from ..common import NotifyFormat
from ..utils import parse_emails
from ..utils import parse_bool
from ..utils import is_email
from ..utils import validate_regex
from ..AppriseLocale import gettext_lazy as _
SMTP2GO_HTTP_ERROR_MAP = {429: 'To many requests.'}

class NotifySMTP2Go(NotifyBase):
    """
    A wrapper for SMTP2Go Notifications
    """
    service_name = 'SMTP2Go'
    service_url = 'https://www.smtp2go.com/'
    secure_protocol = 'smtp2go'
    request_rate_per_sec = 0.2
    setup_url = 'https://github.com/caronc/apprise/wiki/Notify_smtp2go'
    notify_url = 'https://api.smtp2go.com/v3/email/send'
    attachment_support = True
    notify_format = NotifyFormat.HTML
    default_batch_size = 100
    templates = ('{schema}://{user}@{host}:{apikey}/', '{schema}://{user}@{host}:{apikey}/{targets}')
    template_tokens = dict(NotifyBase.template_tokens, **{'user': {'name': _('User Name'), 'type': 'string', 'required': True}, 'host': {'name': _('Domain'), 'type': 'string', 'required': True}, 'apikey': {'name': _('API Key'), 'type': 'string', 'private': True, 'required': True}, 'targets': {'name': _('Target Emails'), 'type': 'list:string'}})
    template_args = dict(NotifyBase.template_args, **{'name': {'name': _('From Name'), 'type': 'string', 'map_to': 'from_name'}, 'to': {'alias_of': 'targets'}, 'cc': {'name': _('Carbon Copy'), 'type': 'list:string'}, 'bcc': {'name': _('Blind Carbon Copy'), 'type': 'list:string'}, 'batch': {'name': _('Batch Mode'), 'type': 'bool', 'default': False}})
    template_kwargs = {'headers': {'name': _('Email Header'), 'prefix': '+'}}

    def __init__(self, apikey, targets, cc=None, bcc=None, from_name=None, headers=None, batch=False, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Initialize SMTP2Go Object\n        '
        super().__init__(**kwargs)
        self.apikey = validate_regex(apikey)
        if not self.apikey:
            msg = 'An invalid SMTP2Go API Key ({}) was specified.'.format(apikey)
            self.logger.warning(msg)
            raise TypeError(msg)
        if not self.user:
            msg = 'No SMTP2Go username was specified.'
            self.logger.warning(msg)
            raise TypeError(msg)
        self.targets = list()
        self.cc = set()
        self.bcc = set()
        self.names = {}
        self.headers = {}
        if headers:
            self.headers.update(headers)
        self.batch = batch
        self.from_name = from_name
        self.from_addr = '{user}@{host}'.format(user=self.user, host=self.host)
        if not is_email(self.from_addr):
            msg = 'Invalid ~From~ email format: {}'.format(self.from_addr)
            self.logger.warning(msg)
            raise TypeError(msg)
        if targets:
            for recipient in parse_emails(targets):
                result = is_email(recipient)
                if result:
                    self.targets.append((result['name'] if result['name'] else False, result['full_email']))
                    continue
                self.logger.warning('Dropped invalid To email ({}) specified.'.format(recipient))
        else:
            self.targets.append((self.from_name if self.from_name else False, self.from_addr))
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

    def send(self, body, title='', notify_type=NotifyType.INFO, attach=None, **kwargs):
        if False:
            return 10
        '\n        Perform SMTP2Go Notification\n        '
        if not self.targets:
            self.logger.warning('There are no Email recipients to notify')
            return False
        has_error = False
        batch_size = 1 if not self.batch else self.default_batch_size
        headers = {'User-Agent': self.app_id, 'Accept': 'application/json', 'Content-Type': 'application/json'}
        attachments = []
        if attach and self.attachment_support:
            for attachment in attach:
                if not attachment:
                    self.logger.error('Could not access attachment {}.'.format(attachment.url(privacy=True)))
                    return False
                try:
                    with open(attachment.path, 'rb') as f:
                        attachments.append({'filename': attachment.name, 'fileblob': base64.b64encode(f.read()).decode('utf-8'), 'mimetype': attachment.mimetype})
                except (OSError, IOError) as e:
                    self.logger.warning('An I/O error occurred while reading {}.'.format(attachment.name if attachment else 'attachment'))
                    self.logger.debug('I/O Exception: %s' % str(e))
                    return False
        sender = formataddr((self.from_name if self.from_name else False, self.from_addr), charset='utf-8')
        payload = {'api_key': self.apikey, 'sender': sender, 'subject': title, 'to': []}
        if attachments:
            payload['attachments'] = attachments
        if self.notify_format == NotifyFormat.HTML:
            payload['html_body'] = body
        else:
            payload['text_body'] = body
        emails = list(self.targets)
        for index in range(0, len(emails), batch_size):
            cc = self.cc - self.bcc
            bcc = set(self.bcc)
            to = list()
            for to_addr in self.targets[index:index + batch_size]:
                cc = cc - set([to_addr[1]])
                bcc = bcc - set([to_addr[1]])
                to.append(formataddr(to_addr, charset='utf-8'))
            payload['to'] = to
            if cc:
                payload['cc'] = [formataddr((self.names.get(addr, False), addr), charset='utf-8') for addr in cc]
            if bcc:
                payload['bcc'] = list(bcc)
            if self.headers:
                payload['custom_headers'] = [{'header': k, 'value': v} for (k, v) in self.headers.items()]
            self.logger.debug('SMTP2Go POST URL: {} (cert_verify={})'.format(self.notify_url, self.verify_certificate))
            self.logger.debug('SMTP2Go Payload: {}'.format(payload))
            verbose_dest = ', '.join([x[1] for x in self.targets[index:index + batch_size]]) if len(self.targets[index:index + batch_size]) <= 3 else '{} recipients'.format(len(self.targets[index:index + batch_size]))
            self.throttle()
            try:
                r = requests.post(self.notify_url, data=dumps(payload), headers=headers, verify=self.verify_certificate, timeout=self.request_timeout)
                if r.status_code != requests.codes.ok:
                    status_str = NotifyBase.http_response_code_lookup(r.status_code, SMTP2GO_HTTP_ERROR_MAP)
                    self.logger.warning('Failed to send SMTP2Go notification to {}: {}{}error={}.'.format(verbose_dest, status_str, ', ' if status_str else '', r.status_code))
                    self.logger.debug('Response Details:\r\n{}'.format(r.content))
                    has_error = True
                    continue
                else:
                    self.logger.info('Sent SMTP2Go notification to {}.'.format(verbose_dest))
            except requests.RequestException as e:
                self.logger.warning('A Connection error occurred sending SMTP2Go:%s ' % verbose_dest + 'notification.')
                self.logger.debug('Socket Exception: %s' % str(e))
                has_error = True
                continue
            except (OSError, IOError) as e:
                self.logger.warning('An I/O error occurred while reading attachments')
                self.logger.debug('I/O Exception: %s' % str(e))
                has_error = True
                continue
        return not has_error

    def url(self, privacy=False, *args, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Returns the URL built dynamically based on specified arguments.\n        '
        params = {'batch': 'yes' if self.batch else 'no'}
        params.update({'+{}'.format(k): v for (k, v) in self.headers.items()})
        params.update(self.url_parameters(*args, privacy=privacy, **kwargs))
        if self.from_name is not None:
            params['name'] = self.from_name
        if self.cc:
            params['cc'] = ','.join(['{}{}'.format('' if not e not in self.names else '{}:'.format(self.names[e]), e) for e in self.cc])
        if self.bcc:
            params['bcc'] = ','.join(self.bcc)
        has_targets = not (len(self.targets) == 1 and self.targets[0][1] == self.from_addr)
        return '{schema}://{user}@{host}/{apikey}/{targets}?{params}'.format(schema=self.secure_protocol, host=self.host, user=NotifySMTP2Go.quote(self.user, safe=''), apikey=self.pprint(self.apikey, privacy, safe=''), targets='' if not has_targets else '/'.join([NotifySMTP2Go.quote('{}{}'.format('' if not e[0] else '{}:'.format(e[0]), e[1]), safe='') for e in self.targets]), params=NotifySMTP2Go.urlencode(params))

    def __len__(self):
        if False:
            return 10
        '\n        Returns the number of targets associated with this notification\n        '
        batch_size = 1 if not self.batch else self.default_batch_size
        targets = len(self.targets)
        if batch_size > 1:
            targets = int(targets / batch_size) + (1 if targets % batch_size else 0)
        return targets if targets > 0 else 1

    @staticmethod
    def parse_url(url):
        if False:
            return 10
        '\n        Parses the URL and returns enough arguments that can allow\n        us to re-instantiate this object.\n\n        '
        results = NotifyBase.parse_url(url, verify_host=False)
        if not results:
            return results
        results['targets'] = NotifySMTP2Go.split_path(results['fullpath'])
        try:
            results['apikey'] = results['targets'].pop(0)
        except IndexError:
            results['apikey'] = None
        if 'name' in results['qsd'] and len(results['qsd']['name']):
            results['from_name'] = NotifySMTP2Go.unquote(results['qsd']['name'])
        if 'to' in results['qsd'] and len(results['qsd']['to']):
            results['targets'].append(results['qsd']['to'])
        if 'cc' in results['qsd'] and len(results['qsd']['cc']):
            results['cc'] = results['qsd']['cc']
        if 'bcc' in results['qsd'] and len(results['qsd']['bcc']):
            results['bcc'] = results['qsd']['bcc']
        results['headers'] = {NotifyBase.unquote(x): NotifyBase.unquote(y) for (x, y) in results['qsd+'].items()}
        results['batch'] = parse_bool(results['qsd'].get('batch', NotifySMTP2Go.template_args['batch']['default']))
        return results