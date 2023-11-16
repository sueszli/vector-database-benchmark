import re
import smtplib
import socket
try:
    import DNS
    ServerError = DNS.ServerError
except:
    DNS = None

    class ServerError(Exception):
        pass
WSP = '[ \\t]'
CRLF = '(?:\\r\\n)'
NO_WS_CTL = '\\x01-\\x08\\x0b\\x0c\\x0f-\\x1f\\x7f'
QUOTED_PAIR = '(?:\\\\.)'
FWS = '(?:(?:' + WSP + '*' + CRLF + ')?' + WSP + '+)'
CTEXT = '[' + NO_WS_CTL + '\\x21-\\x27\\x2a-\\x5b\\x5d-\\x7e]'
CCONTENT = '(?:' + CTEXT + '|' + QUOTED_PAIR + ')'
COMMENT = '\\((?:' + FWS + '?' + CCONTENT + ')*' + FWS + '?\\)'
CFWS = '(?:' + FWS + '?' + COMMENT + ')*(?:' + FWS + '?' + COMMENT + '|' + FWS + ')'
ATEXT = "[\\w!#$%&\\'\\*\\+\\-/=\\?\\^`\\{\\|\\}~]"
ATOM = CFWS + '?' + ATEXT + '+' + CFWS + '?'
DOT_ATOM_TEXT = ATEXT + '+(?:\\.' + ATEXT + '+)*'
DOT_ATOM = CFWS + '?' + DOT_ATOM_TEXT + CFWS + '?'
QTEXT = '[' + NO_WS_CTL + '\\x21\\x23-\\x5b\\x5d-\\x7e]'
QCONTENT = '(?:' + QTEXT + '|' + QUOTED_PAIR + ')'
QUOTED_STRING = CFWS + '?' + '"(?:' + FWS + '?' + QCONTENT + ')*' + FWS + '?' + '"' + CFWS + '?'
LOCAL_PART = '(?:' + DOT_ATOM + '|' + QUOTED_STRING + ')'
DTEXT = '[' + NO_WS_CTL + '\\x21-\\x5a\\x5e-\\x7e]'
DCONTENT = '(?:' + DTEXT + '|' + QUOTED_PAIR + ')'
DOMAIN_LITERAL = CFWS + '?' + '\\[' + '(?:' + FWS + '?' + DCONTENT + ')*' + FWS + '?\\]' + CFWS + '?'
DOMAIN = '(?:' + DOT_ATOM + '|' + DOMAIN_LITERAL + ')'
ADDR_SPEC = LOCAL_PART + '@' + DOMAIN
VALID_ADDRESS_REGEXP = '^' + ADDR_SPEC + '$'

def validate_email(email, check_mx=False, verify=False):
    if False:
        for i in range(10):
            print('nop')
    "Indicate whether the given string is a valid email address\n    according to the 'addr-spec' portion of RFC 2822 (see section\n    3.4.1).  Parts of the spec that are marked obsolete are *not*\n    included in this test, and certain arcane constructions that\n    depend on circular definitions in the spec may not pass, but in\n    general this should correctly identify any email address likely\n    to be in use as of 2011."
    try:
        assert re.match(VALID_ADDRESS_REGEXP, email) is not None
        check_mx |= verify
        if check_mx:
            if not DNS:
                raise Exception('For check the mx records or check if the email exists you must have installed pyDNS python package')
            DNS.DiscoverNameServers()
            hostname = email[email.find('@') + 1:]
            mx_hosts = DNS.mxlookup(hostname)
            for mx in mx_hosts:
                try:
                    smtp = smtplib.SMTP()
                    smtp.connect(mx[1])
                    if not verify:
                        return True
                    (status, _) = smtp.helo()
                    if status != 250:
                        continue
                    smtp.mail('')
                    (status, _) = smtp.rcpt(email)
                    if status != 250:
                        return False
                    break
                except smtplib.SMTPServerDisconnected:
                    break
                except smtplib.SMTPConnectError:
                    continue
    except (AssertionError, ServerError):
        return False
    return True