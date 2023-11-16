from email import Encoders
from email.charset import Charset
from email.header import Header
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import COMMASPACE, formataddr, formatdate, getaddresses, make_msgid
import logging
import re
import smtplib
import threading
from odoo import api, fields, models, tools, _
from odoo.exceptions import except_orm, UserError
from odoo.tools import html2text, ustr
_logger = logging.getLogger(__name__)
_test_logger = logging.getLogger('odoo.tests')

class MailDeliveryException(except_orm):
    """Specific exception subclass for mail delivery errors"""

    def __init__(self, name, value):
        if False:
            for i in range(10):
                print('nop')
        super(MailDeliveryException, self).__init__(name, value)

class WriteToLogger(object):
    """debugging helper: behave as a fd and pipe to logger at the given level"""

    def __init__(self, logger, level=logging.DEBUG):
        if False:
            while True:
                i = 10
        self.logger = logger
        self.level = level

    def write(self, s):
        if False:
            i = 10
            return i + 15
        self.logger.log(self.level, s)

def try_coerce_ascii(string_utf8):
    if False:
        i = 10
        return i + 15
    'Attempts to decode the given utf8-encoded string\n       as ASCII after coercing it to UTF-8, then return\n       the confirmed 7-bit ASCII string.\n\n       If the process fails (because the string\n       contains non-ASCII characters) returns ``None``.\n    '
    try:
        string_utf8.decode('ascii')
    except UnicodeDecodeError:
        return
    return string_utf8

def encode_header(header_text):
    if False:
        for i in range(10):
            print('nop')
    'Returns an appropriate representation of the given header value,\n       suitable for direct assignment as a header value in an\n       email.message.Message. RFC2822 assumes that headers contain\n       only 7-bit characters, so we ensure it is the case, using\n       RFC2047 encoding when needed.\n\n       :param header_text: unicode or utf-8 encoded string with header value\n       :rtype: string | email.header.Header\n       :return: if ``header_text`` represents a plain ASCII string,\n                return the same 7-bit string, otherwise returns an email.header.Header\n                that will perform the appropriate RFC2047 encoding of\n                non-ASCII values.\n    '
    if not header_text:
        return ''
    header_text_utf8 = ustr(header_text).encode('utf-8')
    header_text_ascii = try_coerce_ascii(header_text_utf8)
    return header_text_ascii or Header(header_text_utf8, 'utf-8')

def encode_header_param(param_text):
    if False:
        return 10
    'Returns an appropriate RFC2047 encoded representation of the given\n       header parameter value, suitable for direct assignation as the\n       param value (e.g. via Message.set_param() or Message.add_header())\n       RFC2822 assumes that headers contain only 7-bit characters,\n       so we ensure it is the case, using RFC2047 encoding when needed.\n\n       :param param_text: unicode or utf-8 encoded string with header value\n       :rtype: string\n       :return: if ``param_text`` represents a plain ASCII string,\n                return the same 7-bit string, otherwise returns an\n                ASCII string containing the RFC2047 encoded text.\n    '
    if not param_text:
        return ''
    param_text_utf8 = ustr(param_text).encode('utf-8')
    param_text_ascii = try_coerce_ascii(param_text_utf8)
    return param_text_ascii or Charset('utf8').header_encode(param_text_utf8)
address_pattern = re.compile('([^ ,<@]+@[^> ,]+)')

def extract_rfc2822_addresses(text):
    if False:
        for i in range(10):
            print('nop')
    'Returns a list of valid RFC2822 addresses\n       that can be found in ``source``, ignoring\n       malformed ones and non-ASCII ones.\n    '
    if not text:
        return []
    candidates = address_pattern.findall(ustr(text).encode('utf-8'))
    return filter(try_coerce_ascii, candidates)

def encode_rfc2822_address_header(header_text):
    if False:
        print('Hello World!')
    'If ``header_text`` contains non-ASCII characters,\n       attempts to locate patterns of the form\n       ``"Name" <address@domain>`` and replace the\n       ``"Name"`` portion by the RFC2047-encoded\n       version, preserving the address part untouched.\n    '

    def encode_addr(addr):
        if False:
            while True:
                i = 10
        (name, email) = addr
        if not try_coerce_ascii(name):
            name = str(Header(name, 'utf-8'))
        return formataddr((name, email))
    addresses = getaddresses([ustr(header_text).encode('utf-8')])
    return COMMASPACE.join(map(encode_addr, addresses))

class IrMailServer(models.Model):
    """Represents an SMTP server, able to send outgoing emails, with SSL and TLS capabilities."""
    _name = 'ir.mail_server'
    NO_VALID_RECIPIENT = 'At least one valid recipient address should be specified for outgoing emails (To/Cc/Bcc)'
    name = fields.Char(string='Description', required=True, index=True)
    smtp_host = fields.Char(string='SMTP Server', required=True, help='Hostname or IP of SMTP server')
    smtp_port = fields.Integer(string='SMTP Port', size=5, required=True, default=25, help='SMTP Port. Usually 465 for SSL, and 25 or 587 for other cases.')
    smtp_user = fields.Char(string='Username', size=64, help='Optional username for SMTP authentication')
    smtp_pass = fields.Char(string='Password', size=64, help='Optional password for SMTP authentication')
    smtp_encryption = fields.Selection([('none', 'None'), ('starttls', 'TLS (STARTTLS)'), ('ssl', 'SSL/TLS')], string='Connection Security', required=True, default='none', help='Choose the connection encryption scheme:\n- None: SMTP sessions are done in cleartext.\n- TLS (STARTTLS): TLS encryption is requested at start of SMTP session (Recommended)\n- SSL/TLS: SMTP sessions are encrypted with SSL/TLS through a dedicated port (default: 465)')
    smtp_debug = fields.Boolean(string='Debugging', help='If enabled, the full output of SMTP sessions will be written to the server log at DEBUG level(this is very verbose and may include confidential info!)')
    sequence = fields.Integer(string='Priority', default=10, help='When no specific mail server is requested for a mail, the highest priority one is used. Default priority is 10 (smaller number = higher priority)')
    active = fields.Boolean(default=True)

    def __init__(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        if not isinstance(smtplib.stderr, WriteToLogger):
            logpiper = WriteToLogger(_logger)
            smtplib.stderr = logpiper
            smtplib.stdout = logpiper
        super(IrMailServer, self).__init__(*args, **kwargs)

    @api.multi
    def name_get(self):
        if False:
            for i in range(10):
                print('nop')
        return [(server.id, '(%s)' % server.name) for server in self]

    @api.multi
    def test_smtp_connection(self):
        if False:
            while True:
                i = 10
        for server in self:
            smtp = False
            try:
                smtp = self.connect(server.smtp_host, server.smtp_port, user=server.smtp_user, password=server.smtp_pass, encryption=server.smtp_encryption, smtp_debug=server.smtp_debug)
            except Exception as e:
                raise UserError(_('Connection Test Failed! Here is what we got instead:\n %s') % ustr(e))
            finally:
                try:
                    if smtp:
                        smtp.quit()
                except Exception:
                    pass
        raise UserError(_('Connection Test Succeeded! Everything seems properly set up!'))

    def connect(self, host, port, user=None, password=None, encryption=False, smtp_debug=False):
        if False:
            i = 10
            return i + 15
        "Returns a new SMTP connection to the give SMTP server, authenticated\n           with ``user`` and ``password`` if provided, and encrypted as requested\n           by the ``encryption`` parameter.\n        \n           :param host: host or IP of SMTP server to connect to\n           :param int port: SMTP port to connect to\n           :param user: optional username to authenticate with\n           :param password: optional password to authenticate with\n           :param string encryption: optional, ``'ssl'`` | ``'starttls'``\n           :param bool smtp_debug: toggle debugging of SMTP sessions (all i/o\n                              will be output in logs)\n        "
        if encryption == 'ssl':
            if not 'SMTP_SSL' in smtplib.__all__:
                raise UserError(_('Your OpenERP Server does not support SMTP-over-SSL. You could use STARTTLS instead.If SSL is needed, an upgrade to Python 2.6 on the server-side should do the trick.'))
            connection = smtplib.SMTP_SSL(host, port)
        else:
            connection = smtplib.SMTP(host, port)
        connection.set_debuglevel(smtp_debug)
        if encryption == 'starttls':
            connection.starttls()
        if user:
            user = ustr(user).encode('utf-8')
            password = ustr(password).encode('utf-8')
            connection.login(user, password)
        return connection

    def build_email(self, email_from, email_to, subject, body, email_cc=None, email_bcc=None, reply_to=False, attachments=None, message_id=None, references=None, object_id=False, subtype='plain', headers=None, body_alternative=None, subtype_alternative='plain'):
        if False:
            print('Hello World!')
        'Constructs an RFC2822 email.message.Message object based on the keyword arguments passed, and returns it.\n\n           :param string email_from: sender email address\n           :param list email_to: list of recipient addresses (to be joined with commas) \n           :param string subject: email subject (no pre-encoding/quoting necessary)\n           :param string body: email body, of the type ``subtype`` (by default, plaintext).\n                               If html subtype is used, the message will be automatically converted\n                               to plaintext and wrapped in multipart/alternative, unless an explicit\n                               ``body_alternative`` version is passed.\n           :param string body_alternative: optional alternative body, of the type specified in ``subtype_alternative``\n           :param string reply_to: optional value of Reply-To header\n           :param string object_id: optional tracking identifier, to be included in the message-id for\n                                    recognizing replies. Suggested format for object-id is "res_id-model",\n                                    e.g. "12345-crm.lead".\n           :param string subtype: optional mime subtype for the text body (usually \'plain\' or \'html\'),\n                                  must match the format of the ``body`` parameter. Default is \'plain\',\n                                  making the content part of the mail "text/plain".\n           :param string subtype_alternative: optional mime subtype of ``body_alternative`` (usually \'plain\'\n                                              or \'html\'). Default is \'plain\'.\n           :param list attachments: list of (filename, filecontents) pairs, where filecontents is a string\n                                    containing the bytes of the attachment\n           :param list email_cc: optional list of string values for CC header (to be joined with commas)\n           :param list email_bcc: optional list of string values for BCC header (to be joined with commas)\n           :param dict headers: optional map of headers to set on the outgoing mail (may override the\n                                other headers, including Subject, Reply-To, Message-Id, etc.)\n           :rtype: email.message.Message (usually MIMEMultipart)\n           :return: the new RFC2822 email message\n        '
        email_from = email_from or tools.config.get('email_from')
        assert email_from, 'You must either provide a sender address explicitly or configure a global sender address in the server configuration or with the --email-from startup parameter.'
        headers = headers or {}
        email_cc = email_cc or []
        email_bcc = email_bcc or []
        body = body or u''
        email_body_utf8 = ustr(body).encode('utf-8')
        email_text_part = MIMEText(email_body_utf8, _subtype=subtype, _charset='utf-8')
        msg = MIMEMultipart()
        if not message_id:
            if object_id:
                message_id = tools.generate_tracking_message_id(object_id)
            else:
                message_id = make_msgid()
        msg['Message-Id'] = encode_header(message_id)
        if references:
            msg['references'] = encode_header(references)
        msg['Subject'] = encode_header(subject)
        msg['From'] = encode_rfc2822_address_header(email_from)
        del msg['Reply-To']
        if reply_to:
            msg['Reply-To'] = encode_rfc2822_address_header(reply_to)
        else:
            msg['Reply-To'] = msg['From']
        msg['To'] = encode_rfc2822_address_header(COMMASPACE.join(email_to))
        if email_cc:
            msg['Cc'] = encode_rfc2822_address_header(COMMASPACE.join(email_cc))
        if email_bcc:
            msg['Bcc'] = encode_rfc2822_address_header(COMMASPACE.join(email_bcc))
        msg['Date'] = formatdate()
        for (key, value) in headers.iteritems():
            msg[ustr(key).encode('utf-8')] = encode_header(value)
        if subtype == 'html' and (not body_alternative) and html2text:
            text_utf8 = tools.html2text(email_body_utf8.decode('utf-8')).encode('utf-8')
            alternative_part = MIMEMultipart(_subtype='alternative')
            alternative_part.attach(MIMEText(text_utf8, _charset='utf-8', _subtype='plain'))
            alternative_part.attach(email_text_part)
            msg.attach(alternative_part)
        elif body_alternative:
            alternative_part = MIMEMultipart(_subtype='alternative')
            body_alternative_utf8 = ustr(body_alternative).encode('utf-8')
            alternative_body_part = MIMEText(body_alternative_utf8, _subtype=subtype_alternative, _charset='utf-8')
            alternative_part.attach(alternative_body_part)
            alternative_part.attach(email_text_part)
            msg.attach(alternative_part)
        else:
            msg.attach(email_text_part)
        if attachments:
            for (fname, fcontent) in attachments:
                filename_rfc2047 = encode_header_param(fname)
                part = MIMEBase('application', 'octet-stream')
                part.set_param('name', filename_rfc2047)
                part.add_header('Content-Disposition', 'attachment', filename=filename_rfc2047)
                part.set_payload(fcontent)
                Encoders.encode_base64(part)
                msg.attach(part)
        return msg

    @api.model
    def _get_default_bounce_address(self):
        if False:
            i = 10
            return i + 15
        'Compute the default bounce address.\n\n        The default bounce address is used to set the envelop address if no\n        envelop address is provided in the message.  It is formed by properly\n        joining the parameters "mail.catchall.alias" and\n        "mail.catchall.domain".\n\n        If "mail.catchall.alias" is not set it defaults to "postmaster-odoo".\n\n        If "mail.catchall.domain" is not set, return None.\n\n        '
        get_param = self.env['ir.config_parameter'].sudo().get_param
        postmaster = get_param('mail.bounce.alias', default='postmaster-odoo')
        domain = get_param('mail.catchall.domain')
        if postmaster and domain:
            return '%s@%s' % (postmaster, domain)

    @api.model
    def send_email(self, message, mail_server_id=None, smtp_server=None, smtp_port=None, smtp_user=None, smtp_password=None, smtp_encryption=None, smtp_debug=False):
        if False:
            for i in range(10):
                print('nop')
        "Sends an email directly (no queuing).\n\n        No retries are done, the caller should handle MailDeliveryException in order to ensure that\n        the mail is never lost.\n\n        If the mail_server_id is provided, sends using this mail server, ignoring other smtp_* arguments.\n        If mail_server_id is None and smtp_server is None, use the default mail server (highest priority).\n        If mail_server_id is None and smtp_server is not None, use the provided smtp_* arguments.\n        If both mail_server_id and smtp_server are None, look for an 'smtp_server' value in server config,\n        and fails if not found.\n\n        :param message: the email.message.Message to send. The envelope sender will be extracted from the\n                        ``Return-Path`` (if present), or will be set to the default bounce address.\n                        The envelope recipients will be extracted from the combined list of ``To``,\n                        ``CC`` and ``BCC`` headers.\n        :param mail_server_id: optional id of ir.mail_server to use for sending. overrides other smtp_* arguments.\n        :param smtp_server: optional hostname of SMTP server to use\n        :param smtp_encryption: optional TLS mode, one of 'none', 'starttls' or 'ssl' (see ir.mail_server fields for explanation)\n        :param smtp_port: optional SMTP port, if mail_server_id is not passed\n        :param smtp_user: optional SMTP user, if mail_server_id is not passed\n        :param smtp_password: optional SMTP password to use, if mail_server_id is not passed\n        :param smtp_debug: optional SMTP debug flag, if mail_server_id is not passed\n        :return: the Message-ID of the message that was just sent, if successfully sent, otherwise raises\n                 MailDeliveryException and logs root cause.\n        "
        smtp_from = message['Return-Path'] or self._get_default_bounce_address() or message['From']
        assert smtp_from, 'The Return-Path or From header is required for any outbound email'
        from_rfc2822 = extract_rfc2822_addresses(smtp_from)
        assert from_rfc2822, "Malformed 'Return-Path' or 'From' address: %r - It should contain one valid plain ASCII email" % smtp_from
        smtp_from = from_rfc2822[-1]
        email_to = message['To']
        email_cc = message['Cc']
        email_bcc = message['Bcc']
        smtp_to_list = filter(None, tools.flatten(map(extract_rfc2822_addresses, [email_to, email_cc, email_bcc])))
        assert smtp_to_list, self.NO_VALID_RECIPIENT
        x_forge_to = message['X-Forge-To']
        if x_forge_to:
            del message['X-Forge-To']
            del message['To']
            message['To'] = x_forge_to
        if getattr(threading.currentThread(), 'testing', False):
            _test_logger.info('skip sending email in test mode')
            return message['Message-Id']
        mail_server = None
        if mail_server_id:
            mail_server = self.sudo().browse(mail_server_id)
        elif not smtp_server:
            mail_server = self.sudo().search([], order='sequence', limit=1)
        if mail_server:
            smtp_server = mail_server.smtp_host
            smtp_user = mail_server.smtp_user
            smtp_password = mail_server.smtp_pass
            smtp_port = mail_server.smtp_port
            smtp_encryption = mail_server.smtp_encryption
            smtp_debug = smtp_debug or mail_server.smtp_debug
        else:
            smtp_server = smtp_server or tools.config.get('smtp_server')
            smtp_port = tools.config.get('smtp_port', 25) if smtp_port is None else smtp_port
            smtp_user = smtp_user or tools.config.get('smtp_user')
            smtp_password = smtp_password or tools.config.get('smtp_password')
            if smtp_encryption is None and tools.config.get('smtp_ssl'):
                smtp_encryption = 'starttls'
        if not smtp_server:
            raise UserError(_('Missing SMTP Server') + '\n' + _('Please define at least one SMTP server, or provide the SMTP parameters explicitly.'))
        try:
            message_id = message['Message-Id']
            if smtp_server.startswith('maildir:/'):
                from mailbox import Maildir
                maildir_path = smtp_server[8:]
                mdir = Maildir(maildir_path, factory=None, create=True)
                mdir.add(message.as_string(True))
                return message_id
            smtp = None
            try:
                smtp = self.connect(smtp_server, smtp_port, smtp_user, smtp_password, smtp_encryption or False, smtp_debug)
                smtp.sendmail(smtp_from, smtp_to_list, message.as_string())
            finally:
                if smtp is not None:
                    smtp.quit()
        except Exception as e:
            params = (ustr(smtp_server), e.__class__.__name__, ustr(e))
            msg = _("Mail delivery failed via SMTP server '%s'.\n%s: %s") % params
            _logger.info(msg)
            raise MailDeliveryException(_('Mail Delivery Failed'), msg)
        return message_id

    @api.onchange('smtp_encryption')
    def _onchange_encryption(self):
        if False:
            return 10
        result = {}
        if self.smtp_encryption == 'ssl':
            self.smtp_port = 465
            if not 'SMTP_SSL' in smtplib.__all__:
                result['warning'] = {'title': _('Warning'), 'message': _('Your server does not seem to support SSL, you may want to try STARTTLS instead')}
        else:
            self.smtp_port = 25
        return result