import dataclasses
import re
import smtplib
import typing as t
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.utils import formataddr, make_msgid
from email.header import Header
from email import charset
from socket import error as SocketError
from datetime import datetime
from datetime import timezone
from .NotifyBase import NotifyBase
from ..URLBase import PrivacyMode
from ..common import NotifyFormat, NotifyType
from ..conversion import convert_between
from ..utils import is_email, parse_emails
from ..AppriseLocale import gettext_lazy as _
from ..logger import logger
charset.add_charset('utf-8', charset.QP, charset.QP, 'utf-8')

class WebBaseLogin:
    """
    This class is just used in conjunction of the default emailers
    to best formulate a login to it using the data detected
    """
    EMAIL = 'Email'
    USERID = 'UserID'

class SecureMailMode:
    INSECURE = 'insecure'
    SSL = 'ssl'
    STARTTLS = 'starttls'
SECURE_MODES = {SecureMailMode.STARTTLS: {'default_port': 587}, SecureMailMode.SSL: {'default_port': 465}, SecureMailMode.INSECURE: {'default_port': 25}}
EMAIL_TEMPLATES = (('Google Mail', re.compile('^((?P<label>[^+]+)\\+)?(?P<id>[^@]+)@(?P<domain>gmail\\.com)$', re.I), {'port': 587, 'smtp_host': 'smtp.gmail.com', 'secure': True, 'secure_mode': SecureMailMode.STARTTLS, 'login_type': (WebBaseLogin.EMAIL,)}), ('Yandex', re.compile('^((?P<label>[^+]+)\\+)?(?P<id>[^@]+)@(?P<domain>yandex\\.(com|ru|ua|by|kz|uz|tr|fr))$', re.I), {'port': 465, 'smtp_host': 'smtp.yandex.ru', 'secure': True, 'secure_mode': SecureMailMode.SSL, 'login_type': (WebBaseLogin.USERID,)}), ('Microsoft Hotmail', re.compile('^((?P<label>[^+]+)\\+)?(?P<id>[^@]+)@(?P<domain>(hotmail|live)\\.com(\\.au)?)$', re.I), {'port': 587, 'smtp_host': 'smtp-mail.outlook.com', 'secure': True, 'secure_mode': SecureMailMode.STARTTLS, 'login_type': (WebBaseLogin.EMAIL,)}), ('Microsoft Outlook', re.compile('^((?P<label>[^+]+)\\+)?(?P<id>[^@]+)@(?P<domain>(smtp\\.)?outlook\\.com(\\.au)?)$', re.I), {'port': 587, 'smtp_host': 'smtp.outlook.com', 'secure': True, 'secure_mode': SecureMailMode.STARTTLS, 'login_type': (WebBaseLogin.EMAIL,)}), ('Microsoft Office 365', re.compile('^[^@]+@(?P<domain>(smtp\\.)?office365\\.com)$', re.I), {'port': 587, 'smtp_host': 'smtp.office365.com', 'secure': True, 'secure_mode': SecureMailMode.STARTTLS}), ('Yahoo Mail', re.compile('^((?P<label>[^+]+)\\+)?(?P<id>[^@]+)@(?P<domain>yahoo\\.(ca|com))$', re.I), {'port': 465, 'smtp_host': 'smtp.mail.yahoo.com', 'secure': True, 'secure_mode': SecureMailMode.STARTTLS, 'login_type': (WebBaseLogin.EMAIL,)}), ('Fast Mail', re.compile('^((?P<label>[^+]+)\\+)?(?P<id>[^@]+)@(?P<domain>fastmail\\.(com|cn|co\\.uk|com\\.au|de|es|fm|fr|im|in|jp|mx|net|nl|org|se|to|tw|uk|us))$', re.I), {'port': 465, 'smtp_host': 'smtp.fastmail.com', 'secure': True, 'secure_mode': SecureMailMode.SSL, 'login_type': (WebBaseLogin.EMAIL,)}), ('Fast Mail Extended Addresses', re.compile('^((?P<label>[^+]+)\\+)?(?P<id>[^@]+)@(?P<domain>123mail\\.org|airpost\\.net|eml\\.cc|fmail\\.co\\.uk|fmgirl\\.com|fmguy\\.com|mailbolt\\.com|mailcan\\.com|mailhaven\\.com|mailmight\\.com|ml1\\.net|mm\\.st|myfastmail\\.com|proinbox\\.com|promessage\\.com|rushpost\\.com|sent\\.(as|at|com)|speedymail\\.org|warpmail\\.net|xsmail\\.com|150mail\\.com|150ml\\.com|16mail\\.com|2-mail\\.com|4email\\.net|50mail\\.com|allmail\\.net|bestmail\\.us|cluemail\\.com|elitemail\\.org|emailcorner\\.net|emailengine\\.(net|org)|emailgroups\\.net|emailplus\\.org|emailuser\\.net|f-m\\.fm|fast-email\\.com|fast-mail\\.org|fastem\\.com|fastemail\\.us|fastemailer\\.com|fastest\\.cc|fastimap\\.com|fastmailbox\\.net|fastmessaging\\.com|fea\\.st|fmailbox\\.com|ftml\\.net|h-mail\\.us|hailmail\\.net|imap-mail\\.com|imap\\.cc|imapmail\\.org|inoutbox\\.com|internet-e-mail\\.com|internet-mail\\.org|internetemails\\.net|internetmailing\\.net|jetemail\\.net|justemail\\.net|letterboxes\\.org|mail-central\\.com|mail-page\\.com|mailandftp\\.com|mailas\\.com|mailc\\.net|mailforce\\.net|mailftp\\.com|mailingaddress\\.org|mailite\\.com|mailnew\\.com|mailsent\\.net|mailservice\\.ms|mailup\\.net|mailworks\\.org|mymacmail\\.com|nospammail\\.net|ownmail\\.net|petml\\.com|postinbox\\.com|postpro\\.net|realemail\\.net|reallyfast\\.biz|reallyfast\\.info|speedpost\\.net|ssl-mail\\.com|swift-mail\\.com|the-fastest\\.net|the-quickest\\.com|theinternetemail\\.com|veryfast\\.biz|veryspeedy\\.net|yepmail\\.net)$', re.I), {'port': 465, 'smtp_host': 'smtp.fastmail.com', 'secure': True, 'secure_mode': SecureMailMode.SSL, 'login_type': (WebBaseLogin.EMAIL,)}), ('Zoho Mail', re.compile('^((?P<label>[^+]+)\\+)?(?P<id>[^@]+)@(?P<domain>zoho(mail)?\\.com)$', re.I), {'port': 587, 'smtp_host': 'smtp.zoho.com', 'secure': True, 'secure_mode': SecureMailMode.STARTTLS, 'login_type': (WebBaseLogin.EMAIL,)}), ('SendGrid', re.compile('^((?P<label>[^+]+)\\+)?(?P<id>[^@]+)@(?P<domain>(\\.smtp)?sendgrid\\.(com|net))$', re.I), {'port': 465, 'smtp_host': 'smtp.sendgrid.net', 'secure': True, 'secure_mode': SecureMailMode.SSL, 'login_type': (WebBaseLogin.USERID,)}), ('163.com', re.compile('^((?P<label>[^+]+)\\+)?(?P<id>[^@]+)@(?P<domain>163\\.com)$', re.I), {'port': 465, 'smtp_host': 'smtp.163.com', 'secure': True, 'secure_mode': SecureMailMode.SSL, 'login_type': (WebBaseLogin.EMAIL,)}), ('Foxmail.com', re.compile('^((?P<label>[^+]+)\\+)?(?P<id>[^@]+)@(?P<domain>(foxmail|qq)\\.com)$', re.I), {'port': 587, 'smtp_host': 'smtp.qq.com', 'secure': True, 'secure_mode': SecureMailMode.STARTTLS, 'login_type': (WebBaseLogin.EMAIL,)}), ('Custom', re.compile('^((?P<label>[^+]+)\\+)?(?P<id>[^@]+)@(?P<domain>.+)$', re.I), {'smtp_host': None}))

@dataclasses.dataclass
class EmailMessage:
    recipient: str
    to_addrs: t.List[str]
    body: str

class NotifyEmail(NotifyBase):
    """
    A wrapper to Email Notifications

    """
    service_name = 'E-Mail'
    protocol = 'mailto'
    secure_protocol = 'mailtos'
    setup_url = 'https://github.com/caronc/apprise/wiki/Notify_email'
    attachment_support = True
    notify_format = NotifyFormat.HTML
    socket_connect_timeout = 15
    templates = ('{schema}://{host}', '{schema}://{host}:{port}', '{schema}://{host}/{targets}', '{schema}://{host}:{port}/{targets}', '{schema}://{user}@{host}', '{schema}://{user}@{host}:{port}', '{schema}://{user}@{host}/{targets}', '{schema}://{user}@{host}:{port}/{targets}', '{schema}://{user}:{password}@{host}', '{schema}://{user}:{password}@{host}:{port}', '{schema}://{user}:{password}@{host}/{targets}', '{schema}://{user}:{password}@{host}:{port}/{targets}')
    template_tokens = dict(NotifyBase.template_tokens, **{'user': {'name': _('User Name'), 'type': 'string'}, 'password': {'name': _('Password'), 'type': 'string', 'private': True}, 'host': {'name': _('Domain'), 'type': 'string', 'required': True}, 'port': {'name': _('Port'), 'type': 'int', 'min': 1, 'max': 65535}, 'target_email': {'name': _('Target Email'), 'type': 'string', 'map_to': 'targets'}, 'targets': {'name': _('Targets'), 'type': 'list:string'}})
    template_args = dict(NotifyBase.template_args, **{'to': {'name': _('To Email'), 'type': 'string', 'map_to': 'targets'}, 'from': {'name': _('From Email'), 'type': 'string', 'map_to': 'from_addr'}, 'name': {'name': _('From Name'), 'type': 'string', 'map_to': 'from_addr'}, 'cc': {'name': _('Carbon Copy'), 'type': 'list:string'}, 'bcc': {'name': _('Blind Carbon Copy'), 'type': 'list:string'}, 'smtp': {'name': _('SMTP Server'), 'type': 'string', 'map_to': 'smtp_host'}, 'mode': {'name': _('Secure Mode'), 'type': 'choice:string', 'values': SECURE_MODES, 'default': SecureMailMode.STARTTLS, 'map_to': 'secure_mode'}, 'reply': {'name': _('Reply To'), 'type': 'list:string', 'map_to': 'reply_to'}})
    template_kwargs = {'headers': {'name': _('Email Header'), 'prefix': '+'}}

    def __init__(self, smtp_host=None, from_addr=None, secure_mode=None, targets=None, cc=None, bcc=None, reply_to=None, headers=None, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Initialize Email Object\n\n        The smtp_host and secure_mode can be automatically detected depending\n        on how the URL was built\n        '
        super().__init__(**kwargs)
        self.targets = list()
        self.cc = set()
        self.bcc = set()
        self.reply_to = set()
        self.names = {}
        self.headers = {}
        if headers:
            self.headers.update(headers)
        self.from_addr = [False, '']
        if self.user and self.host:
            self.from_addr = [self.app_id, '{}@{}'.format(re.split('[\\s@]+', self.user)[0], self.host)]
        if from_addr:
            result = is_email(from_addr)
            if result:
                self.from_addr = (result['name'] if result['name'] else False, result['full_email'])
            else:
                self.from_addr[0] = from_addr
        result = is_email(self.from_addr[1])
        if not result:
            msg = 'Invalid ~From~ email specified: {}'.format('{} <{}>'.format(self.from_addr[0], self.from_addr[1]) if self.from_addr[0] else '{}'.format(self.from_addr[1]))
            self.logger.warning(msg)
            raise TypeError(msg)
        self.names[self.from_addr[1]] = self.from_addr[0]
        self.smtp_host = smtp_host if isinstance(smtp_host, str) else ''
        if secure_mode:
            self.secure_mode = None if not isinstance(secure_mode, str) else secure_mode.lower()
        else:
            self.secure_mode = SecureMailMode.INSECURE if not self.secure else self.template_args['mode']['default']
        if self.secure_mode not in SECURE_MODES:
            msg = 'The secure mode specified ({}) is invalid.'.format(secure_mode)
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
            self.targets.append((False, self.from_addr[1]))
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
        for recipient in parse_emails(reply_to):
            email = is_email(recipient)
            if email:
                self.reply_to.add(email['full_email'])
                self.names[email['full_email']] = email['name'] if email['name'] else False
                continue
            self.logger.warning('Dropped invalid Reply To email ({}) specified.'.format(recipient))
        self.NotifyEmailDefaults(secure_mode=secure_mode, **kwargs)
        if not self.secure and self.secure_mode != SecureMailMode.INSECURE:
            self.secure = True
        if not self.port:
            self.port = SECURE_MODES[self.secure_mode]['default_port']
        if not self.smtp_host:
            self.smtp_host = self.host
        return

    def NotifyEmailDefaults(self, secure_mode=None, port=None, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        A function that prefills defaults based on the email\n        it was provided.\n        '
        if self.smtp_host or not self.user:
            return
        from_addr = '{}@{}'.format(re.split('[\\s@]+', self.user)[0], self.host)
        for i in range(len(EMAIL_TEMPLATES)):
            self.logger.trace('Scanning %s against %s' % (from_addr, EMAIL_TEMPLATES[i][0]))
            match = EMAIL_TEMPLATES[i][1].match(from_addr)
            if match:
                self.logger.info('Applying %s Defaults' % EMAIL_TEMPLATES[i][0])
                self.secure = EMAIL_TEMPLATES[i][2].get('secure', self.secure)
                self.smtp_host = EMAIL_TEMPLATES[i][2].get('smtp_host', self.smtp_host)
                if not port:
                    self.port = EMAIL_TEMPLATES[i][2].get('port', self.port)
                if not secure_mode:
                    self.secure_mode = EMAIL_TEMPLATES[i][2].get('secure_mode', self.secure_mode)
                login_type = EMAIL_TEMPLATES[i][2].get('login_type', [])
                if login_type:
                    if is_email(self.user):
                        if WebBaseLogin.EMAIL not in login_type:
                            self.user = match.group('id')
                    elif WebBaseLogin.USERID not in login_type:
                        self.user = '{}@{}'.format(self.user, self.host)
                break

    def _get_charset(self, input_string):
        if False:
            i = 10
            return i + 15
        '\n        Get utf-8 charset if non ascii string only\n\n        Encode an ascii string to utf-8 is bad for email deliverability\n        because some anti-spam gives a bad score for that\n        like SUBJ_EXCESS_QP flag on Rspamd\n        '
        if not input_string:
            return None
        return 'utf-8' if not all((ord(c) < 128 for c in input_string)) else None

    def send(self, body, title='', notify_type=NotifyType.INFO, attach=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Perform Email Notification\n        '
        if not self.targets:
            self.logger.warning('There are no Email recipients to notify')
            return False
        messages: t.List[EmailMessage] = []
        emails = list(self.targets)
        while len(emails):
            (to_name, to_addr) = emails.pop(0)
            cc = self.cc - self.bcc - set([to_addr])
            bcc = self.bcc - set([to_addr])
            reply_to = self.reply_to - set([to_addr])
            cc = [formataddr((self.names.get(addr, False), addr), charset='utf-8') for addr in cc]
            bcc = [formataddr((self.names.get(addr, False), addr), charset='utf-8') for addr in bcc]
            if reply_to:
                reply_to = [formataddr((self.names.get(addr, False), addr), charset='utf-8') for addr in reply_to]
            self.logger.debug('Email From: {}'.format(formataddr(self.from_addr, charset='utf-8')))
            self.logger.debug('Email To: {}'.format(to_addr))
            if cc:
                self.logger.debug('Email Cc: {}'.format(', '.join(cc)))
            if bcc:
                self.logger.debug('Email Bcc: {}'.format(', '.join(bcc)))
            if reply_to:
                self.logger.debug('Email Reply-To: {}'.format(', '.join(reply_to)))
            self.logger.debug('Login ID: {}'.format(self.user))
            self.logger.debug('Delivery: {}:{}'.format(self.smtp_host, self.port))
            if self.notify_format == NotifyFormat.HTML:
                base = MIMEMultipart('alternative')
                base.attach(MIMEText(convert_between(NotifyFormat.HTML, NotifyFormat.TEXT, body), 'plain', 'utf-8'))
                base.attach(MIMEText(body, 'html', 'utf-8'))
            else:
                base = MIMEText(body, 'plain', 'utf-8')
            if attach and self.attachment_support:
                mixed = MIMEMultipart('mixed')
                mixed.attach(base)
                for attachment in attach:
                    if not attachment:
                        self.logger.error('Could not access attachment {}.'.format(attachment.url(privacy=True)))
                        return False
                    self.logger.debug('Preparing Email attachment {}'.format(attachment.url(privacy=True)))
                    with open(attachment.path, 'rb') as abody:
                        app = MIMEApplication(abody.read())
                        app.set_type(attachment.mimetype)
                        app.add_header('Content-Disposition', 'attachment; filename="{}"'.format(Header(attachment.name, 'utf-8')))
                        mixed.attach(app)
                base = mixed
            for (k, v) in self.headers.items():
                base[k] = Header(v, self._get_charset(v))
            base['Subject'] = Header(title, self._get_charset(title))
            base['From'] = formataddr(self.from_addr, charset='utf-8')
            base['To'] = formataddr((to_name, to_addr), charset='utf-8')
            base['Message-ID'] = make_msgid(domain=self.smtp_host)
            base['Date'] = datetime.now(timezone.utc).strftime('%a, %d %b %Y %H:%M:%S +0000')
            base['X-Application'] = self.app_id
            if cc:
                base['Cc'] = ','.join(cc)
            if reply_to:
                base['Reply-To'] = ','.join(reply_to)
            message = EmailMessage(recipient=to_addr, to_addrs=[to_addr] + list(cc) + list(bcc), body=base.as_string())
            messages.append(message)
        return self.submit(messages)

    def submit(self, messages: t.List[EmailMessage]):
        if False:
            while True:
                i = 10
        has_error = False
        socket = None
        self.throttle()
        try:
            self.logger.debug('Connecting to remote SMTP server...')
            socket_func = smtplib.SMTP
            if self.secure_mode == SecureMailMode.SSL:
                self.logger.debug('Securing connection with SSL...')
                socket_func = smtplib.SMTP_SSL
            socket = socket_func(self.smtp_host, self.port, None, timeout=self.socket_connect_timeout)
            if self.secure_mode == SecureMailMode.STARTTLS:
                self.logger.debug('Securing connection with STARTTLS...')
                socket.starttls()
            if self.user and self.password:
                self.logger.debug('Applying user credentials...')
                socket.login(self.user, self.password)
            for message in messages:
                try:
                    socket.sendmail(self.from_addr[1], message.to_addrs, message.body)
                    self.logger.info(f'Sent Email notification to "{message.recipient}".')
                except (SocketError, smtplib.SMTPException, RuntimeError) as e:
                    self.logger.warning(f'Sending email to "{message.recipient}" failed. Reason: {e}')
                    has_error = True
        except (SocketError, smtplib.SMTPException, RuntimeError) as e:
            self.logger.warning(f'Connection error while submitting email to {self.smtp_host}. Reason: {e}')
            has_error = True
        finally:
            if socket is not None:
                socket.quit()
        return not has_error

    def url(self, privacy=False, *args, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Returns the URL built dynamically based on specified arguments.\n        '
        params = {}
        params.update({'+{}'.format(k): v for (k, v) in self.headers.items()})
        params.update(self.url_parameters(*args, privacy=privacy, **kwargs))
        from_addr = None
        if len(self.targets) == 1 and self.targets[0][1] != self.from_addr[1]:
            from_addr = self.from_addr[1]
        if self.smtp_host != self.host:
            params['smtp'] = self.smtp_host
        if self.secure:
            params['mode'] = self.secure_mode
        if self.from_addr[0] and self.from_addr[0] != self.app_id:
            params['from'] = self.from_addr[0] if not from_addr else formataddr((self.from_addr[0], from_addr), charset='utf-8')
        elif from_addr:
            params['from'] = formataddr((False, from_addr), charset='utf-8')
        elif not self.user:
            params['from'] = formataddr((False, self.from_addr[1]), charset='utf-8')
        if len(self.cc) > 0:
            params['cc'] = ','.join([formataddr((self.names[e] if e in self.names else False, e), charset='utf-8').replace(',', '%2C') for e in self.cc])
        if len(self.bcc) > 0:
            params['bcc'] = ','.join([formataddr((self.names[e] if e in self.names else False, e), charset='utf-8').replace(',', '%2C') for e in self.bcc])
        if self.reply_to:
            params['reply'] = ','.join([formataddr((self.names[e] if e in self.names else False, e), charset='utf-8').replace(',', '%2C') for e in self.reply_to])
        user = None if not self.user else self.user.split('@')[0]
        auth = ''
        if self.user and self.password:
            auth = '{user}:{password}@'.format(user=NotifyEmail.quote(user, safe=''), password=self.pprint(self.password, privacy, mode=PrivacyMode.Secret, safe=''))
        elif user:
            auth = '{user}@'.format(user=NotifyEmail.quote(user, safe=''))
        default_port = SECURE_MODES[self.secure_mode]['default_port']
        has_targets = not (len(self.targets) == 1 and self.targets[0][1] == self.from_addr[1])
        return '{schema}://{auth}{hostname}{port}/{targets}?{params}'.format(schema=self.secure_protocol if self.secure else self.protocol, auth=auth, hostname=self.host, port='' if self.port is None or self.port == default_port else ':{}'.format(self.port), targets='' if not has_targets else '/'.join([NotifyEmail.quote('{}{}'.format('' if not e[0] else '{}:'.format(e[0]), e[1]), safe='') for e in self.targets]), params=NotifyEmail.urlencode(params))

    def __len__(self):
        if False:
            print('Hello World!')
        '\n        Returns the number of targets associated with this notification\n        '
        targets = len(self.targets)
        return targets if targets > 0 else 1

    @staticmethod
    def parse_url(url):
        if False:
            while True:
                i = 10
        '\n        Parses the URL and returns enough arguments that can allow\n        us to re-instantiate this object.\n\n        '
        results = NotifyBase.parse_url(url)
        if not results:
            return results
        from_addr = ''
        smtp_host = ''
        results['targets'] = NotifyEmail.split_path(results['fullpath'])
        if 'to' in results['qsd'] and len(results['qsd']['to']):
            results['targets'].append(results['qsd']['to'])
        if 'from' in results['qsd'] and len(results['qsd']['from']):
            from_addr = NotifyEmail.unquote(results['qsd']['from'])
            if 'name' in results['qsd'] and len(results['qsd']['name']):
                from_addr = formataddr((NotifyEmail.unquote(results['qsd']['name']), from_addr), charset='utf-8')
                logger.warning('Email name= and from= are synonymous; use one or the other.')
        elif 'name' in results['qsd'] and len(results['qsd']['name']):
            from_addr = NotifyEmail.unquote(results['qsd']['name'])
        if 'smtp' in results['qsd'] and len(results['qsd']['smtp']):
            smtp_host = NotifyEmail.unquote(results['qsd']['smtp'])
        if 'mode' in results['qsd'] and len(results['qsd']['mode']):
            results['secure_mode'] = results['qsd']['mode'].lower()
        if 'cc' in results['qsd'] and len(results['qsd']['cc']):
            results['cc'] = results['qsd']['cc']
        if 'bcc' in results['qsd'] and len(results['qsd']['bcc']):
            results['bcc'] = results['qsd']['bcc']
        if 'reply' in results['qsd'] and len(results['qsd']['reply']):
            results['reply_to'] = results['qsd']['reply']
        results['from_addr'] = from_addr
        results['smtp_host'] = smtp_host
        results['headers'] = {NotifyBase.unquote(x): NotifyBase.unquote(y) for (x, y) in results['qsd+'].items()}
        return results