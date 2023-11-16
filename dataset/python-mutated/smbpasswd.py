import sys
import logging
from getpass import getpass
from argparse import ArgumentParser
from impacket import version
from impacket.examples import logger
from impacket.examples.utils import parse_target
from impacket.dcerpc.v5 import transport, samr

class SMBPasswd:

    def __init__(self, address, domain='', username='', oldPassword='', newPassword='', oldPwdHashLM='', oldPwdHashNT='', newPwdHashLM='', newPwdHashNT=''):
        if False:
            return 10
        self.address = address
        self.domain = domain
        self.username = username
        self.oldPassword = oldPassword
        self.newPassword = newPassword
        self.oldPwdHashLM = oldPwdHashLM
        self.oldPwdHashNT = oldPwdHashNT
        self.newPwdHashLM = newPwdHashLM
        self.newPwdHashNT = newPwdHashNT
        self.dce = None

    def connect(self, domain='', username='', password='', nthash='', anonymous=False):
        if False:
            for i in range(10):
                print('nop')
        rpctransport = transport.SMBTransport(self.address, filename='\\samr')
        if anonymous:
            rpctransport.set_credentials(username='', password='', domain='', lmhash='', nthash='', aesKey='')
        elif username != '':
            lmhash = ''
            rpctransport.set_credentials(username, password, domain, lmhash, nthash, aesKey='')
        else:
            rpctransport.set_credentials(self.username, self.oldPassword, self.domain, self.oldPwdHashLM, self.oldPwdHashNT, aesKey='')
        self.dce = rpctransport.get_dce_rpc()
        self.dce.connect()
        self.dce.bind(samr.MSRPC_UUID_SAMR)

    def hSamrUnicodeChangePasswordUser2(self):
        if False:
            for i in range(10):
                print('nop')
        try:
            resp = samr.hSamrUnicodeChangePasswordUser2(self.dce, '\x00', self.username, self.oldPassword, self.newPassword, self.oldPwdHashLM, self.oldPwdHashNT)
        except Exception as e:
            if 'STATUS_PASSWORD_RESTRICTION' in str(e):
                logging.critical('Some password update rule has been violated. For example, the password may not meet length criteria.')
            else:
                raise e
        else:
            if resp['ErrorCode'] == 0:
                logging.info('Password was changed successfully.')
            else:
                logging.error('Non-zero return code, something weird happened.')
                resp.dump()

    def hSamrChangePasswordUser(self):
        if False:
            print('Hello World!')
        try:
            serverHandle = samr.hSamrConnect(self.dce, self.address + '\x00')['ServerHandle']
            domainSID = samr.hSamrLookupDomainInSamServer(self.dce, serverHandle, self.domain)['DomainId']
            domainHandle = samr.hSamrOpenDomain(self.dce, serverHandle, domainId=domainSID)['DomainHandle']
            userRID = samr.hSamrLookupNamesInDomain(self.dce, domainHandle, (self.username,))['RelativeIds']['Element'][0]
            userHandle = samr.hSamrOpenUser(self.dce, domainHandle, userId=userRID)['UserHandle']
        except Exception as e:
            if 'STATUS_NO_SUCH_DOMAIN' in str(e):
                logging.critical('Wrong realm. Try to set the domain name for the target user account explicitly in format DOMAIN/username.')
                return
            else:
                raise e
        try:
            resp = samr.hSamrChangePasswordUser(self.dce, userHandle, self.oldPassword, newPassword='', oldPwdHashNT=self.oldPwdHashNT, newPwdHashLM=self.newPwdHashLM, newPwdHashNT=self.newPwdHashNT)
        except Exception as e:
            if 'STATUS_PASSWORD_RESTRICTION' in str(e):
                logging.critical('Some password update rule has been violated. For example, the password history policy may prohibit the use of recent passwords.')
            else:
                raise e
        else:
            if resp['ErrorCode'] == 0:
                logging.info('NTLM hashes were changed successfully.')
            else:
                logging.error('Non-zero return code, something weird happened.')
                resp.dump()

    def hSamrSetInformationUser(self):
        if False:
            return 10
        try:
            serverHandle = samr.hSamrConnect(self.dce, self.address + '\x00')['ServerHandle']
            domainSID = samr.hSamrLookupDomainInSamServer(self.dce, serverHandle, self.domain)['DomainId']
            domainHandle = samr.hSamrOpenDomain(self.dce, serverHandle, domainId=domainSID)['DomainHandle']
            userRID = samr.hSamrLookupNamesInDomain(self.dce, domainHandle, (self.username,))['RelativeIds']['Element'][0]
            userHandle = samr.hSamrOpenUser(self.dce, domainHandle, userId=userRID)['UserHandle']
        except Exception as e:
            if 'STATUS_NO_SUCH_DOMAIN' in str(e):
                logging.critical('Wrong realm. Try to set the domain name for the target user account explicitly in format DOMAIN/username.')
                return
            else:
                raise e
        try:
            resp = samr.hSamrSetNTInternal1(self.dce, userHandle, self.newPassword, self.newPwdHashNT)
        except Exception as e:
            raise e
        else:
            if resp['ErrorCode'] == 0:
                logging.info('Credentials were injected into SAM successfully.')
            else:
                logging.error('Non-zero return code, something weird happened.')
                resp.dump()

def init_logger(options):
    if False:
        for i in range(10):
            print('nop')
    logger.init(options.ts)
    if options.debug is True:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.debug(version.getInstallationPath())
    else:
        logging.getLogger().setLevel(logging.INFO)

def parse_args():
    if False:
        i = 10
        return i + 15
    parser = ArgumentParser(description='Change password over SMB.')
    parser.add_argument('target', action='store', help='[[domain/]username[:password]@]<targetName or address>')
    parser.add_argument('-ts', action='store_true', help='adds timestamp to every logging output')
    parser.add_argument('-debug', action='store_true', help='turn DEBUG output ON')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-newpass', action='store', default=None, help='new SMB password')
    group.add_argument('-newhashes', action='store', default=None, metavar='LMHASH:NTHASH', help='new NTLM hashes, format is LMHASH:NTHASH (the user will be asked to change their password at next logon)')
    group = parser.add_argument_group('authentication')
    group.add_argument('-hashes', action='store', default=None, metavar='LMHASH:NTHASH', help='NTLM hashes, format is LMHASH:NTHASH')
    group = parser.add_argument_group('RPC authentication')
    group.add_argument('-altuser', action='store', default=None, help='alternative username')
    group.add_argument('-altpass', action='store', default=None, help='alternative password')
    group.add_argument('-althash', action='store', default=None, help='alternative NT hash')
    group = parser.add_argument_group('set credentials method')
    group.add_argument('-admin', action='store_true', help="injects credentials into SAM (requires admin's priveleges on a machine, but can bypass password history policy)")
    return parser.parse_args()
if __name__ == '__main__':
    print(version.BANNER)
    print(version.WARNING_BANNER)
    options = parse_args()
    init_logger(options)
    (domain, username, oldPassword, address) = parse_target(options.target)
    if domain is None:
        domain = 'Builtin'
    if options.hashes is not None:
        try:
            (oldPwdHashLM, oldPwdHashNT) = options.hashes.split(':')
        except ValueError:
            logging.critical('Wrong hashes string format. For more information run with --help option.')
            sys.exit(1)
    else:
        oldPwdHashLM = ''
        oldPwdHashNT = ''
    if oldPassword == '' and oldPwdHashNT == '' and (not options.admin):
        oldPassword = getpass('Current SMB password: ')
    if options.newhashes is not None:
        try:
            (newPwdHashLM, newPwdHashNT) = options.newhashes.split(':')
        except ValueError:
            logging.critical('Wrong new hashes string format. For more information run with --help option.')
            sys.exit(1)
        newPassword = ''
    else:
        newPwdHashLM = ''
        newPwdHashNT = ''
        if options.newpass is None:
            newPassword = getpass('New SMB password: ')
            if newPassword != getpass('Retype new SMB password: '):
                logging.critical('Passwords do not match, try again.')
                sys.exit(1)
        else:
            newPassword = options.newpass
    smbpasswd = SMBPasswd(address, domain, username, oldPassword, newPassword, oldPwdHashLM, oldPwdHashNT, newPwdHashLM, newPwdHashNT)
    if options.altuser is not None:
        try:
            (altDomain, altUsername) = options.altuser.split('/')
        except ValueError:
            altDomain = domain
            altUsername = options.altuser
        if options.altpass is not None and options.althash is None:
            altPassword = options.altpass
            altNTHash = ''
        elif options.altpass is None and options.althash is not None:
            altPassword = ''
            altNTHash = options.althash
        elif options.altpass is None and options.althash is None:
            logging.critical('Please, provide either alternative password or NT hash for RPC authentication.')
            sys.exit(1)
        else:
            logging.critical('Argument -altpass not allowed with argument -althash.')
            sys.exit(1)
    else:
        altUsername = ''
    try:
        if altUsername == '':
            smbpasswd.connect()
        else:
            logging.debug('Using {}\\{} credentials to connect to RPC.'.format(altDomain, altUsername))
            smbpasswd.connect(altDomain, altUsername, altPassword, altNTHash)
    except Exception as e:
        if any((msg in str(e) for msg in ['STATUS_PASSWORD_MUST_CHANGE', 'STATUS_PASSWORD_EXPIRED'])):
            if newPassword:
                logging.warning('Password is expired, trying to bind with a null session.')
                smbpasswd.connect(anonymous=True)
            else:
                logging.critical('Cannot set new NTLM hashes when current password is expired. Provide a plaintext value for the new password.')
                sys.exit(1)
        elif 'STATUS_LOGON_FAILURE' in str(e):
            logging.critical('Authentication failure.')
            sys.exit(1)
        else:
            raise e
    if options.admin:
        smbpasswd.hSamrSetInformationUser()
    elif newPassword:
        smbpasswd.hSamrUnicodeChangePasswordUser2()
    else:
        smbpasswd.hSamrChangePasswordUser()