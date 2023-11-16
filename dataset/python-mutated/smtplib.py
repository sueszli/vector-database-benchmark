"""SMTP/ESMTP client class.

This should follow RFC 821 (SMTP), RFC 1869 (ESMTP), RFC 2554 (SMTP
Authentication) and RFC 2487 (Secure SMTP over TLS).

Notes:

Please remember, when doing ESMTP, that the names of the SMTP service
extensions are NOT the same thing as the option keywords for the RCPT
and MAIL commands!

Example:

  >>> import smtplib
  >>> s=smtplib.SMTP("localhost")
  >>> print s.help()
  This is Sendmail version 8.8.4
  Topics:
      HELO    EHLO    MAIL    RCPT    DATA
      RSET    NOOP    QUIT    HELP    VRFY
      EXPN    VERB    ETRN    DSN
  For more info use "HELP <topic>".
  To report bugs in the implementation send email to
      sendmail-bugs@sendmail.org.
  For local information send email to Postmaster at your site.
  End of HELP info
  >>> s.putcmd("vrfy","someone@here")
  >>> s.getreply()
  (250, "Somebody OverHere <somebody@here.my.org>")
  >>> s.quit()
"""
import socket
import re
import email.utils
import base64
import hmac
from email.base64mime import body_encode as encode_base64
from sys import stderr
from functools import partial
from polyglot.builtins import string_or_bytes
__all__ = ['SMTPException', 'SMTPServerDisconnected', 'SMTPResponseException', 'SMTPSenderRefused', 'SMTPRecipientsRefused', 'SMTPDataError', 'SMTPConnectError', 'SMTPHeloError', 'SMTPAuthenticationError', 'quoteaddr', 'quotedata', 'SMTP']
SMTP_PORT = 25
SMTP_SSL_PORT = 465
CRLF = '\r\n'
_MAXLINE = 8192
OLDSTYLE_AUTH = re.compile('auth=(.*)', re.I)

class SMTPException(Exception):
    """Base class for all exceptions raised by this module."""

class SMTPServerDisconnected(SMTPException):
    """Not connected to any SMTP server.

    This exception is raised when the server unexpectedly disconnects,
    or when an attempt is made to use the SMTP instance before
    connecting it to a server.
    """

class SMTPResponseException(SMTPException):
    """Base class for all exceptions that include an SMTP error code.

    These exceptions are generated in some instances when the SMTP
    server returns an error code.  The error code is stored in the
    `smtp_code' attribute of the error, and the `smtp_error' attribute
    is set to the error message.
    """

    def __init__(self, code, msg):
        if False:
            while True:
                i = 10
        self.smtp_code = code
        self.smtp_error = msg
        self.args = (code, msg)

class SMTPSenderRefused(SMTPResponseException):
    """Sender address refused.

    In addition to the attributes set by on all SMTPResponseException
    exceptions, this sets `sender' to the string that the SMTP refused.
    """

    def __init__(self, code, msg, sender):
        if False:
            return 10
        self.smtp_code = code
        self.smtp_error = msg
        self.sender = sender
        self.args = (code, msg, sender)

class SMTPRecipientsRefused(SMTPException):
    """All recipient addresses refused.

    The errors for each recipient are accessible through the attribute
    'recipients', which is a dictionary of exactly the same sort as
    SMTP.sendmail() returns.
    """

    def __init__(self, recipients):
        if False:
            i = 10
            return i + 15
        self.recipients = recipients
        self.args = (recipients,)

class SMTPDataError(SMTPResponseException):
    """The SMTP server didn't accept the data."""

class SMTPConnectError(SMTPResponseException):
    """Error during connection establishment."""

class SMTPHeloError(SMTPResponseException):
    """The server refused our HELO reply."""

class SMTPAuthenticationError(SMTPResponseException):
    """Authentication error.

    Most probably the server didn't accept the username/password
    combination provided.
    """

def quoteaddr(addr):
    if False:
        i = 10
        return i + 15
    'Quote a subset of the email addresses defined by RFC 821.\n\n    Should be able to handle anything rfc822.parseaddr can handle.\n    '
    m = (None, None)
    try:
        m = email.utils.parseaddr(addr)[1]
    except AttributeError:
        pass
    if m == (None, None):
        return '<%s>' % addr
    elif m is None:
        return '<>'
    else:
        return '<%s>' % m

def _addr_only(addrstring):
    if False:
        return 10
    (displayname, addr) = email.utils.parseaddr(addrstring)
    if (displayname, addr) == ('', ''):
        return addrstring
    return addr

def quotedata(data):
    if False:
        return 10
    "Quote data for email.\n\n    Double leading '.', and change Unix newline '\\n', or Mac '\\r' into\n    Internet CRLF end-of-line.\n    "
    return re.sub('(?m)^\\.', '..', re.sub('(?:\\r\\n|\\n|\\r(?!\\n))', CRLF, data))
try:
    import ssl
except ImportError:
    _have_ssl = False
else:

    class SSLFakeFile:
        """A fake file like object that really wraps a SSLObject.

        It only supports what is needed in smtplib.
        """

        def __init__(self, sslobj):
            if False:
                for i in range(10):
                    print('nop')
            self.sslobj = sslobj

        def readline(self, size=-1):
            if False:
                i = 10
                return i + 15
            if size < 0:
                size = None
            str = ''
            chr = None
            while chr != '\n':
                if size is not None and len(str) >= size:
                    break
                chr = self.sslobj.read(1)
                if not chr:
                    break
                str += chr
            return str

        def close(self):
            if False:
                while True:
                    i = 10
            pass
    _have_ssl = True

class SMTP:
    """This class manages a connection to an SMTP or ESMTP server.
    SMTP Objects:
        SMTP objects have the following attributes:
            helo_resp
                This is the message given by the server in response to the
                most recent HELO command.

            ehlo_resp
                This is the message given by the server in response to the
                most recent EHLO command. This is usually multiline.

            does_esmtp
                This is a True value _after you do an EHLO command_, if the
                server supports ESMTP.

            esmtp_features
                This is a dictionary, which, if the server supports ESMTP,
                will _after you do an EHLO command_, contain the names of the
                SMTP service extensions this server supports, and their
                parameters (if any).

                Note, all extension names are mapped to lower case in the
                dictionary.

        See each method's docstrings for details.  In general, there is a
        method of the same name to perform each SMTP command.  There is also a
        method called 'sendmail' that will do an entire mail transaction.
        """
    debuglevel = 0
    file = None
    helo_resp = None
    ehlo_msg = 'ehlo'
    ehlo_resp = None
    does_esmtp = 0
    default_port = SMTP_PORT

    def __init__(self, host='', port=0, local_hostname=None, timeout=socket._GLOBAL_DEFAULT_TIMEOUT, debug_to=partial(print, file=stderr)):
        if False:
            i = 10
            return i + 15
        "Initialize a new instance.\n\n        If specified, `host' is the name of the remote host to which to\n        connect.  If specified, `port' specifies the port to which to connect.\n        By default, smtplib.SMTP_PORT is used.  An SMTPConnectError is raised\n        if the specified `host' doesn't respond correctly.  If specified,\n        `local_hostname` is used as the FQDN of the local host.  By default,\n        the local hostname is found using socket.getfqdn(). `debug_to`\n        specifies where debug output is written to. By default it is written to\n        sys.stderr. You should pass in a print function of your own to control\n        where debug output is written.\n        "
        self._host = host
        self.timeout = timeout
        self.debug = debug_to
        self.esmtp_features = {}
        if host:
            (code, msg) = self.connect(host, port)
            if code != 220:
                raise SMTPConnectError(code, msg)
        if local_hostname is not None:
            self.local_hostname = local_hostname
        else:
            fqdn = socket.getfqdn()
            if '.' in fqdn:
                self.local_hostname = fqdn
            else:
                addr = '127.0.0.1'
                try:
                    addr = socket.gethostbyname(socket.gethostname())
                except socket.gaierror:
                    pass
                self.local_hostname = '[%s]' % addr

    def set_debuglevel(self, debuglevel):
        if False:
            print('Hello World!')
        'Set the debug output level.\n\n        A value of 0 means no debug logging. A value of 1 means all interaction\n        with the server is logged except that long lines are truncated to 100\n        characters and AUTH messages are censored. A value of 2 or higher means\n        the complete session is logged.\n\n        '
        self.debuglevel = debuglevel

    def _get_socket(self, host, port, timeout):
        if False:
            i = 10
            return i + 15
        if self.debuglevel > 0:
            self.debug('connect:', (host, port))
        return socket.create_connection((host, port), timeout)

    def connect(self, host='localhost', port=0):
        if False:
            for i in range(10):
                print('nop')
        "Connect to a host on a given port.\n\n        If the hostname ends with a colon (`:') followed by a number, and\n        there is no port specified, that suffix will be stripped off and the\n        number interpreted as the port number to use.\n\n        Note: This method is automatically invoked by __init__, if a host is\n        specified during instantiation.\n\n        "
        if not port and host.find(':') == host.rfind(':'):
            i = host.rfind(':')
            if i >= 0:
                (host, port) = (host[:i], host[i + 1:])
                try:
                    port = int(port)
                except ValueError:
                    raise OSError('nonnumeric port')
        if not port:
            port = self.default_port
        if self.debuglevel > 0:
            self.debug('connect:', (host, port))
        self._host = host
        self.sock = self._get_socket(host, port, self.timeout)
        (code, msg) = self.getreply()
        if self.debuglevel > 0:
            self.debug('connect:', msg)
        return (code, msg)

    def send(self, str):
        if False:
            return 10
        "Send `str' to the server."
        if self.debuglevel > 0:
            raw = repr(str)
            self.debug('send:', raw)
        if hasattr(self, 'sock') and self.sock:
            try:
                self.sock.sendall(str)
            except OSError:
                self.close()
                raise SMTPServerDisconnected('Server not connected')
        else:
            raise SMTPServerDisconnected('please run connect() first')

    def putcmd(self, cmd, args=''):
        if False:
            return 10
        'Send a command to the server.'
        if args == '':
            str = f'{cmd}{CRLF}'
        else:
            str = f'{cmd} {args}{CRLF}'
        self.send(str)

    def getreply(self):
        if False:
            i = 10
            return i + 15
        "Get a reply from the server.\n\n        Returns a tuple consisting of:\n\n          - server response code (e.g. '250', or such, if all goes well)\n            Note: returns -1 if it can't read response code.\n\n          - server response string corresponding to response code (multiline\n            responses are converted to a single, multiline string).\n\n        Raises SMTPServerDisconnected if end-of-file is reached.\n        "
        resp = []
        if self.file is None:
            self.file = self.sock.makefile('rb')
        while True:
            try:
                line = self.file.readline(_MAXLINE + 1)
            except OSError as e:
                self.close()
                raise SMTPServerDisconnected('Connection unexpectedly closed: ' + str(e))
            if line == '':
                self.close()
                raise SMTPServerDisconnected('Connection unexpectedly closed')
            if self.debuglevel > 0:
                self.debug('reply:', repr(line))
            if len(line) > _MAXLINE:
                raise SMTPResponseException(500, 'Line too long.')
            resp.append(line[4:].strip())
            code = line[:3]
            try:
                errcode = int(code)
            except ValueError:
                errcode = -1
                break
            if line[3:4] != '-':
                break
        errmsg = '\n'.join(resp)
        if self.debuglevel > 0:
            self.debug(f'reply: retcode ({errcode}); Msg: {errmsg}')
        return (errcode, errmsg)

    def docmd(self, cmd, args=''):
        if False:
            i = 10
            return i + 15
        'Send a command, and return its response code.'
        self.putcmd(cmd, args)
        return self.getreply()

    def helo(self, name=''):
        if False:
            print('Hello World!')
        "SMTP 'helo' command.\n        Hostname to send for this command defaults to the FQDN of the local\n        host.\n        "
        self.putcmd('helo', name or self.local_hostname)
        (code, msg) = self.getreply()
        self.helo_resp = msg
        return (code, msg)

    def ehlo(self, name=''):
        if False:
            return 10
        " SMTP 'ehlo' command.\n        Hostname to send for this command defaults to the FQDN of the local\n        host.\n        "
        self.esmtp_features = {}
        self.putcmd(self.ehlo_msg, name or self.local_hostname)
        (code, msg) = self.getreply()
        if code == -1 and len(msg) == 0:
            self.close()
            raise SMTPServerDisconnected('Server not connected')
        self.ehlo_resp = msg
        if code != 250:
            return (code, msg)
        self.does_esmtp = 1
        resp = self.ehlo_resp.split('\n')
        del resp[0]
        for each in resp:
            auth_match = OLDSTYLE_AUTH.match(each)
            if auth_match:
                self.esmtp_features['auth'] = self.esmtp_features.get('auth', '') + ' ' + auth_match.groups(0)[0]
                continue
            m = re.match('(?P<feature>[A-Za-z0-9][A-Za-z0-9\\-]*) ?', each)
            if m:
                feature = m.group('feature').lower()
                params = m.string[m.end('feature'):].strip()
                if feature == 'auth':
                    self.esmtp_features[feature] = self.esmtp_features.get(feature, '') + ' ' + params
                else:
                    self.esmtp_features[feature] = params
        return (code, msg)

    def has_extn(self, opt):
        if False:
            print('Hello World!')
        'Does the server support a given SMTP service extension?'
        return opt.lower() in self.esmtp_features

    def help(self, args=''):
        if False:
            while True:
                i = 10
        "SMTP 'help' command.\n        Returns help text from server."
        self.putcmd('help', args)
        return self.getreply()[1]

    def rset(self):
        if False:
            i = 10
            return i + 15
        "SMTP 'rset' command -- resets session."
        return self.docmd('rset')

    def noop(self):
        if False:
            i = 10
            return i + 15
        "SMTP 'noop' command -- doesn't do anything :>"
        return self.docmd('noop')

    def mail(self, sender, options=[]):
        if False:
            for i in range(10):
                print('nop')
        "SMTP 'mail' command -- begins mail xfer session."
        optionlist = ''
        if options and self.does_esmtp:
            optionlist = ' ' + ' '.join(options)
        self.putcmd('mail', f'FROM:{quoteaddr(sender)}{optionlist}')
        return self.getreply()

    def rcpt(self, recip, options=[]):
        if False:
            while True:
                i = 10
        "SMTP 'rcpt' command -- indicates 1 recipient for this mail."
        optionlist = ''
        if options and self.does_esmtp:
            optionlist = ' ' + ' '.join(options)
        self.putcmd('rcpt', f'TO:{quoteaddr(recip)}{optionlist}')
        return self.getreply()

    def data(self, msg):
        if False:
            return 10
        "SMTP 'DATA' command -- sends message data to server.\n\n        Automatically quotes lines beginning with a period per rfc821.\n        Raises SMTPDataError if there is an unexpected reply to the\n        DATA command; the return value from this method is the final\n        response code received when the all data is sent.\n        "
        self.putcmd('data')
        (code, repl) = self.getreply()
        if self.debuglevel > 0:
            self.debug('data:', (code, repl))
        if code != 354:
            raise SMTPDataError(code, repl)
        else:
            q = quotedata(msg)
            if q[-2:] != CRLF:
                q = q + CRLF
            q = q + '.' + CRLF
            self.send(q)
            (code, msg) = self.getreply()
            if self.debuglevel > 0:
                self.debug('data:', (code, msg))
            return (code, msg)

    def verify(self, address):
        if False:
            print('Hello World!')
        "SMTP 'verify' command -- checks for address validity."
        self.putcmd('vrfy', _addr_only(address))
        return self.getreply()
    vrfy = verify

    def expn(self, address):
        if False:
            for i in range(10):
                print('nop')
        "SMTP 'expn' command -- expands a mailing list."
        self.putcmd('expn', _addr_only(address))
        return self.getreply()

    def ehlo_or_helo_if_needed(self):
        if False:
            return 10
        "Call self.ehlo() and/or self.helo() if needed.\n\n        If there has been no previous EHLO or HELO command this session, this\n        method tries ESMTP EHLO first.\n\n        This method may raise the following exceptions:\n\n         SMTPHeloError            The server didn't reply properly to\n                                  the helo greeting.\n        "
        if self.helo_resp is None and self.ehlo_resp is None:
            if not 200 <= self.ehlo()[0] <= 299:
                (code, resp) = self.helo()
                if not 200 <= code <= 299:
                    raise SMTPHeloError(code, resp)

    def login(self, user, password):
        if False:
            for i in range(10):
                print('nop')
        "Log in on an SMTP server that requires authentication.\n\n        The arguments are:\n            - user:     The user name to authenticate with.\n            - password: The password for the authentication.\n\n        If there has been no previous EHLO or HELO command this session, this\n        method tries ESMTP EHLO first.\n\n        This method will return normally if the authentication was successful.\n\n        This method may raise the following exceptions:\n\n         SMTPHeloError            The server didn't reply properly to\n                                  the helo greeting.\n         SMTPAuthenticationError  The server didn't accept the username/\n                                  password combination.\n         SMTPException            No suitable authentication method was\n                                  found.\n        "

        def encode_cram_md5(challenge, user, password):
            if False:
                while True:
                    i = 10
            challenge = base64.decodestring(challenge)
            if isinstance(password, str):
                password = password.encode('utf-8')
            response = user + ' ' + hmac.HMAC(password, challenge).hexdigest()
            return encode_base64(response, eol='')

        def encode_plain(user, password):
            if False:
                return 10
            return encode_base64(f'\x00{user}\x00{password}', eol='')
        AUTH_PLAIN = 'PLAIN'
        AUTH_CRAM_MD5 = 'CRAM-MD5'
        AUTH_LOGIN = 'LOGIN'
        self.ehlo_or_helo_if_needed()
        if not self.has_extn('auth'):
            raise SMTPException('SMTP AUTH extension not supported by server.')
        authlist = self.esmtp_features['auth'].split()
        preferred_auths = [AUTH_CRAM_MD5, AUTH_PLAIN, AUTH_LOGIN]
        authmethod = None
        for method in preferred_auths:
            if method in authlist:
                authmethod = method
                break
        if authmethod == AUTH_CRAM_MD5:
            (code, resp) = self.docmd('AUTH', AUTH_CRAM_MD5)
            if code == 503:
                return (code, resp)
            (code, resp) = self.docmd(encode_cram_md5(resp, user, password))
        elif authmethod == AUTH_PLAIN:
            (code, resp) = self.docmd('AUTH', AUTH_PLAIN + ' ' + encode_plain(user, password))
        elif authmethod == AUTH_LOGIN:
            (code, resp) = self.docmd('AUTH', '{} {}'.format(AUTH_LOGIN, encode_base64(user, eol='')))
            if code != 334:
                raise SMTPAuthenticationError(code, resp)
            (code, resp) = self.docmd(encode_base64(password, eol=''))
        elif authmethod is None:
            raise SMTPException('No suitable authentication method found.')
        if code not in (235, 503):
            raise SMTPAuthenticationError(code, resp)
        return (code, resp)

    def starttls(self, context=None):
        if False:
            i = 10
            return i + 15
        "Puts the connection to the SMTP server into TLS mode.\n\n        If there has been no previous EHLO or HELO command this session, this\n        method tries ESMTP EHLO first.\n\n        If the server supports TLS, this will encrypt the rest of the SMTP\n        session. If you provide the keyfile and certfile parameters,\n        the identity of the SMTP server and client can be checked. This,\n        however, depends on whether the socket module really checks the\n        certificates.\n\n        This method may raise the following exceptions:\n\n         SMTPHeloError            The server didn't reply properly to\n                                  the helo greeting.\n        "
        self.ehlo_or_helo_if_needed()
        if not self.has_extn('starttls'):
            raise SMTPException('STARTTLS extension not supported by server.')
        (resp, reply) = self.docmd('STARTTLS')
        if resp == 220:
            if not _have_ssl:
                raise RuntimeError('No SSL support included in this Python')
            if context is None:
                self.sock = ssl.wrap_socket(self.sock)
            else:
                self.sock = context.wrap_socket(self.sock, server_hostname=self._host)
            self.file = SSLFakeFile(self.sock)
            self.helo_resp = None
            self.ehlo_resp = None
            self.esmtp_features = {}
            self.does_esmtp = 0
        else:
            raise SMTPResponseException(resp, reply)
        return (resp, reply)

    def sendmail(self, from_addr, to_addrs, msg, mail_options=[], rcpt_options=[]):
        if False:
            while True:
                i = 10
        'This command performs an entire mail transaction.\n\n        The arguments are:\n            - from_addr    : The address sending this mail.\n            - to_addrs     : A list of addresses to send this mail to.  A bare\n                             string will be treated as a list with 1 address.\n            - msg          : The message to send.\n            - mail_options : List of ESMTP options (such as 8bitmime) for the\n                             mail command.\n            - rcpt_options : List of ESMTP options (such as DSN commands) for\n                             all the rcpt commands.\n\n        If there has been no previous EHLO or HELO command this session, this\n        method tries ESMTP EHLO first.  If the server does ESMTP, message size\n        and each of the specified options will be passed to it.  If EHLO\n        fails, HELO will be tried and ESMTP options suppressed.\n\n        This method will return normally if the mail is accepted for at least\n        one recipient.  It returns a dictionary, with one entry for each\n        recipient that was refused.  Each entry contains a tuple of the SMTP\n        error code and the accompanying error message sent by the server.\n\n        This method may raise the following exceptions:\n\n         SMTPHeloError          The server didn\'t reply properly to\n                                the helo greeting.\n         SMTPRecipientsRefused  The server rejected ALL recipients\n                                (no mail was sent).\n         SMTPSenderRefused      The server didn\'t accept the from_addr.\n         SMTPDataError          The server replied with an unexpected\n                                error code (other than a refusal of\n                                a recipient).\n\n        Note: the connection will be open even after an exception is raised.\n\n        Example:\n\n         >>> import smtplib\n         >>> s=smtplib.SMTP("localhost")\n         >>> tolist=["one@one.org","two@two.org","three@three.org","four@four.org"]\n         >>> msg = \'\'\'\\\n         ... From: Me@my.org\n         ... Subject: testin\'...\n         ...\n         ... This is a test \'\'\'\n         >>> s.sendmail("me@my.org",tolist,msg)\n         { "three@three.org" : ( 550 ,"User unknown" ) }\n         >>> s.quit()\n\n        In the above example, the message was accepted for delivery to three\n        of the four addresses, and one was rejected, with the error code\n        550.  If all addresses are accepted, then the method will return an\n        empty dictionary.\n\n        '
        self.ehlo_or_helo_if_needed()
        esmtp_opts = []
        if self.does_esmtp:
            if self.has_extn('size'):
                esmtp_opts.append('size=%d' % len(msg))
            for option in mail_options:
                esmtp_opts.append(option)
        (code, resp) = self.mail(from_addr, esmtp_opts)
        if code != 250:
            self.rset()
            raise SMTPSenderRefused(code, resp, from_addr)
        senderrs = {}
        if isinstance(to_addrs, string_or_bytes):
            to_addrs = [to_addrs]
        for each in to_addrs:
            (code, resp) = self.rcpt(each, rcpt_options)
            if code != 250 and code != 251:
                senderrs[each] = (code, resp)
        if len(senderrs) == len(to_addrs):
            self.rset()
            raise SMTPRecipientsRefused(senderrs)
        (code, resp) = self.data(msg)
        if code != 250:
            self.rset()
            raise SMTPDataError(code, resp)
        return senderrs

    def close(self):
        if False:
            i = 10
            return i + 15
        'Close the connection to the SMTP server.'
        try:
            file = self.file
            self.file = None
            if file:
                file.close()
        finally:
            sock = self.sock
            self.sock = None
            if sock:
                sock.close()

    def quit(self):
        if False:
            for i in range(10):
                print('nop')
        'Terminate the SMTP session.'
        res = self.docmd('quit')
        self.ehlo_resp = self.helo_resp = None
        self.esmtp_features = {}
        self.does_esmtp = False
        self.close()
        return res
if _have_ssl:

    class SMTP_SSL(SMTP):
        """ This is a subclass derived from SMTP that connects over an SSL encrypted
        socket (to use this class you need a socket module that was compiled with SSL
        support). If host is not specified, '' (the local host) is used. If port is
        omitted, the standard SMTP-over-SSL port (465) is used. keyfile and certfile
        are also optional - they can contain a PEM formatted private key and
        certificate chain file for the SSL connection.
        """
        default_port = SMTP_SSL_PORT

        def __init__(self, host='', port=0, local_hostname=None, keyfile=None, certfile=None, timeout=socket._GLOBAL_DEFAULT_TIMEOUT, debug_to=partial(print, file=stderr)):
            if False:
                return 10
            self.keyfile = keyfile
            self.certfile = certfile
            SMTP.__init__(self, host, port, local_hostname, timeout, debug_to=debug_to)

        def _get_socket(self, host, port, timeout):
            if False:
                i = 10
                return i + 15
            if self.debuglevel > 0:
                self.debug('connect:', (host, port))
            new_socket = socket.create_connection((host, port), timeout)
            new_socket = ssl.wrap_socket(new_socket, self.keyfile, self.certfile)
            self.file = SSLFakeFile(new_socket)
            return new_socket
    __all__.append('SMTP_SSL')
LMTP_PORT = 2003

class LMTP(SMTP):
    """LMTP - Local Mail Transfer Protocol

    The LMTP protocol, which is very similar to ESMTP, is heavily based
    on the standard SMTP client. It's common to use Unix sockets for LMTP,
    so our connect() method must support that as well as a regular
    host:port server. To specify a Unix socket, you must use an absolute
    path as the host, starting with a '/'.

    Authentication is supported, using the regular SMTP mechanism. When
    using a Unix socket, LMTP generally don't support or require any
    authentication, but your mileage might vary."""
    ehlo_msg = 'lhlo'

    def __init__(self, host='', port=LMTP_PORT, local_hostname=None):
        if False:
            for i in range(10):
                print('nop')
        'Initialize a new instance.'
        SMTP.__init__(self, host, port, local_hostname)

    def connect(self, host='localhost', port=0):
        if False:
            i = 10
            return i + 15
        'Connect to the LMTP daemon, on either a Unix or a TCP socket.'
        if host[0] != '/':
            return SMTP.connect(self, host, port)
        try:
            self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self.sock.connect(host)
        except OSError:
            if self.debuglevel > 0:
                self.debug('connect fail:', host)
            if self.sock:
                self.sock.close()
            self.sock = None
            raise
        (code, msg) = self.getreply()
        if self.debuglevel > 0:
            self.debug('connect:', msg)
        return (code, msg)
if __name__ == '__main__':
    import sys

    def prompt(prompt):
        if False:
            return 10
        sys.stdout.write(prompt + ': ')
        return sys.stdin.readline().strip()
    fromaddr = prompt('From')
    toaddrs = prompt('To').split(',')
    print('Enter message, end with ^D:')
    msg = ''
    while 1:
        line = sys.stdin.readline()
        if not line:
            break
        msg = msg + line
    print('Message length is %d' % len(msg))
    server = SMTP('localhost')
    server.set_debuglevel(1)
    server.sendmail(fromaddr, toaddrs, msg)
    server.quit()