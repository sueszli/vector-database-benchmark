import argparse
import logging
import sys
from getpass import getpass
from impacket import version
from impacket.dcerpc.v5 import transport, samr, epm
from impacket.krb5 import kpasswd
from impacket.ldap import ldap, ldapasn1
from impacket.examples import logger
from impacket.examples.utils import parse_target
EMPTY_LM_HASH = 'aad3b435b51404eeaad3b435b51404ee'

class PasswordHandler:
    """Generic interface for all the password protocols supported by this script"""

    def __init__(self, address, domain='', authUsername='', authPassword='', authPwdHashLM='', authPwdHashNT='', doKerberos=False, aesKey='', kdcHost=None):
        if False:
            return 10
        '\n        Instantiate password change or reset with the credentials of the account making the changes.\n        It can be the target user, or a privileged account.\n\n        :param string address:  IP address or hostname of the server or domain controller where the password will be changed\n        :param string domain:   AD domain where the password will be changed\n        :param string username: account that will attempt the password change or reset on the target(s)\n        :param string password: password of the account that will attempt the password change\n        :param string pwdHashLM: LM hash of the account that will attempt the password change\n        :param string pwdHashNT: NT hash of the account that will attempt the password change\n        :param bool doKerberos: use Kerberos authentication instead of NTLM\n        :param string aesKey:   AES key for Kerberos authentication\n        :param string kdcHost:  KDC host\n        '
        self.address = address
        self.domain = domain
        self.username = authUsername
        self.password = authPassword
        self.pwdHashLM = authPwdHashLM
        self.pwdHashNT = authPwdHashNT
        self.doKerberos = doKerberos
        self.aesKey = aesKey
        self.kdcHost = kdcHost

    def _changePassword(self, targetUsername, targetDomain, oldPassword, newPassword, oldPwdHashLM, oldPwdHashNT, newPwdHashLM, newPwdHashNT):
        if False:
            for i in range(10):
                print('nop')
        'Implementation of a password change'
        raise NotImplementedError

    def changePassword(self, targetUsername=None, targetDomain=None, oldPassword=None, newPassword='', oldPwdHashLM=None, oldPwdHashNT=None, newPwdHashLM='', newPwdHashNT=''):
        if False:
            while True:
                i = 10
        '\n        Change the password of a target account, knowing the previous password.\n\n        :param string targetUsername: account whose password will be changed, if different from the user performing the change\n        :param string targetDomain:   domain of the account\n        :param string oldPassword:    current password\n        :param string newPassword:    new password\n        :param string oldPwdHashLM:   current password, as LM hash\n        :param string oldPwdHashMT:   current password, as NT hash\n        :param string newPwdHashLM:   new password, as LM hash\n        :param string newPwdHashMT:   new password, as NT hash\n\n        :return bool success\n        '
        if targetUsername is None:
            targetUsername = self.username
            if targetDomain is None:
                targetDomain = self.domain
            if oldPassword is None:
                oldPassword = self.password
            if oldPwdHashLM is None:
                oldPwdHashLM = self.pwdHashLM
            if oldPwdHashNT is None:
                oldPwdHashNT = self.pwdHashNT
        logging.info(f'Changing the password of {targetDomain}\\{targetUsername}')
        return self._changePassword(targetUsername, targetDomain, oldPassword, newPassword, oldPwdHashLM, oldPwdHashNT, newPwdHashLM, newPwdHashNT)

    def _setPassword(self, targetUsername, targetDomain, newPassword, newPwdHashLM, newPwdHashNT):
        if False:
            for i in range(10):
                print('nop')
        'Implementation of a password set'
        raise NotImplementedError

    def setPassword(self, targetUsername, targetDomain=None, newPassword='', newPwdHashLM='', newPwdHashNT=''):
        if False:
            print('Hello World!')
        '\n        Set or Reset the password of a target account, with privileges.\n\n        :param string targetUsername:   account whose password will be changed\n        :param string targetDomain:     domain of the account\n        :param string newPassword:      new password\n        :param string newPwdHashLM:     new password, as LM hash\n        :param string newPwdHashMT:     new password, as NT hash\n\n        :return bool success\n        '
        if targetDomain is None:
            targetDomain = self.domain
        logging.info(f'Setting the password of {targetDomain}\\{targetUsername} as {self.domain}\\{self.username}')
        return self._setPassword(targetUsername, targetDomain, newPassword, newPwdHashLM, newPwdHashNT)

class KPassword(PasswordHandler):
    """Use Kerberos Change-Password or Set-Password protocols (rfc3244) to change passwords"""

    def _changePassword(self, targetUsername, targetDomain, oldPassword, newPassword, oldPwdHashLM, oldPwdHashNT, newPwdHashLM, newPwdHashNT):
        if False:
            return 10
        if targetUsername != self.username:
            logging.critical('KPassword does not support changing the password of another user (try setPassword instead)')
            return False
        if not newPassword:
            logging.critical('KPassword requires the new password as plaintext')
            return False
        try:
            logging.debug((targetUsername, targetDomain, newPassword, oldPassword, oldPwdHashLM, oldPwdHashNT, self.aesKey, self.kdcHost))
            kpasswd.changePassword(targetUsername, targetDomain, newPassword, oldPassword, oldPwdHashLM, oldPwdHashNT, aesKey=self.aesKey, kdcHost=self.kdcHost)
        except kpasswd.KPasswdError as e:
            logging.error(f'Password not changed: {e}')
            return False
        else:
            logging.info('Password was changed successfully.')
            return True

    def _setPassword(self, targetUsername, targetDomain, newPassword, newPwdHashLM, newPwdHashNT):
        if False:
            i = 10
            return i + 15
        if not newPassword:
            logging.critical('KPassword requires the new password as plaintext')
            return False
        try:
            kpasswd.setPassword(self.username, self.domain, targetUsername, targetDomain, newPassword, self.password, self.pwdHashLM, self.pwdHashNT, aesKey=self.aesKey, kdcHost=self.kdcHost)
        except kpasswd.KPasswdError as e:
            logging.error(f'Password not changed for {targetDomain}\\{targetUsername}: {e}')
        else:
            logging.info(f'Password was set successfully for {targetDomain}\\{targetUsername}.')

class SamrPassword(PasswordHandler):
    """Use MS-SAMR protocol to change or reset the password of a user"""
    dce = None
    anonymous = False

    def rpctransport(self):
        if False:
            while True:
                i = 10
        '\n        Return a new transport for our RPC/DCE.\n\n        :return rpc: RPC transport instance\n        '
        raise NotImplementedError

    def authenticate(self, anonymous=False):
        if False:
            print('Hello World!')
        '\n        Instantiate a new transport and try to authenticate\n\n        :param bool anonymous: Attempt a null binding\n        :return dce: DCE/RPC, bound to SAMR\n        '
        rpctransport = self.rpctransport()
        if hasattr(rpctransport, 'set_credentials'):
            if anonymous:
                rpctransport.set_credentials(username='', password='', domain='', lmhash='', nthash='', aesKey='')
            else:
                rpctransport.set_credentials(self.username, self.password, self.domain, self.pwdHashLM, self.pwdHashNT, aesKey=self.aesKey)
        if anonymous:
            self.anonymous = True
            rpctransport.set_kerberos(False, None)
        else:
            self.anonymous = False
            rpctransport.set_kerberos(self.doKerberos, self.kdcHost)
        as_user = 'null session' if anonymous else f'{self.domain}\\{self.username}'
        logging.info(f'Connecting to DCE/RPC as {as_user}')
        dce = rpctransport.get_dce_rpc()
        dce.connect()
        dce.bind(samr.MSRPC_UUID_SAMR)
        logging.debug('Successfully bound to SAMR')
        return dce

    def connect(self, retry_if_expired=False):
        if False:
            for i in range(10):
                print('nop')
        '\n        Connect to SAMR using our transport protocol.\n\n        This method must instantiate self.dce\n\n        :param bool retry_if_expired: Retry as null binding if our password is expired\n        :return bool: success\n        '
        if self.dce:
            return True
        try:
            self.dce = self.authenticate(anonymous=False)
        except Exception as e:
            if any((msg in str(e) for msg in ('STATUS_PASSWORD_MUST_CHANGE', 'STATUS_PASSWORD_EXPIRED'))):
                if retry_if_expired:
                    logging.warning('Password is expired or must be changed, trying to bind with a null session.')
                    self.dce = self.authenticate(anonymous=True)
                else:
                    logging.critical('Cannot set new NTLM hashes when current password is expired. Provide a plaintext value for the new password.')
                    logging.debug(str(e))
                    return False
            elif 'STATUS_LOGON_FAILURE' in str(e):
                logging.critical('Authentication failure when connecting to RPC: wrong credentials?')
                logging.debug(str(e))
                return False
            elif 'STATUS_ACCOUNT_RESTRICTION' in str(e):
                logging.critical("Account restriction: username and credentials are valid, but some other restriction preventsauthentication, like 'Protected Users' group or time-of-day restriction")
                logging.debug(str(e))
                return False
            else:
                raise e
        return True

    def hSamrOpenUser(self, username):
        if False:
            for i in range(10):
                print('nop')
        'Open an handle on the target user'
        try:
            serverHandle = samr.hSamrConnect(self.dce, self.address + '\x00')['ServerHandle']
            domainSID = samr.hSamrLookupDomainInSamServer(self.dce, serverHandle, self.domain)['DomainId']
            domainHandle = samr.hSamrOpenDomain(self.dce, serverHandle, domainId=domainSID)['DomainHandle']
            userRID = samr.hSamrLookupNamesInDomain(self.dce, domainHandle, (username,))['RelativeIds']['Element'][0]
            userHandle = samr.hSamrOpenUser(self.dce, domainHandle, userId=userRID)['UserHandle']
        except Exception as e:
            if 'STATUS_NO_SUCH_DOMAIN' in str(e):
                logging.critical('Wrong realm. Try to set the domain name for the target user account explicitly in format DOMAIN/username.')
                logging.debug(str(e))
                return False
            elif self.anonymous and 'STATUS_ACCESS_DENIED' in str(e):
                logging.critical('Our anonymous session cannot get a handle to the target user. Retry with a user whose password is not expired.')
                logging.debug(str(e))
                return False
            else:
                raise e
        return userHandle

    def _SamrWrapper(self, samrProcedure, *args, _change=True, **kwargs):
        if False:
            print('Hello World!')
        '\n        Handles common errors when changing/resetting the password, regardless of the procedure\n\n        :param callable samrProcedure: Function that will send the SAMR call\n                                args and kwargs are passed verbatim\n        :param bool _change:    Used for more precise error reporting,\n                                True if it is a password change, False if it is a reset\n        '
        logging.debug(f'Sending SAMR call {samrProcedure.__name__}')
        try:
            resp = samrProcedure(self.dce, *args, **kwargs)
        except Exception as e:
            if 'STATUS_PASSWORD_RESTRICTION' in str(e):
                logging.critical('Some password update rule has been violated. For example, the password history policy may prohibit the use of recent passwords or the password may not meet length criteria.')
                logging.debug(str(e))
                return False
            elif 'STATUS_ACCESS_DENIED' in str(e):
                if _change:
                    logging.critical('Target user is not allowed to change their own password')
                else:
                    logging.critical(f'{self.domain}\\{self.username} user is not allowed to set the password of the target')
                logging.debug(str(e))
                return False
            else:
                raise e
        if resp['ErrorCode'] == 0:
            logging.info('Password was changed successfully.')
            return True
        logging.error('Non-zero return code, something weird happened.')
        resp.dump()
        return False

    def hSamrUnicodeChangePasswordUser2(self, username, oldPassword, newPassword, oldPwdHashLM, oldPwdHashNT, newPwdHashLM, newPwdHashNT):
        if False:
            while True:
                i = 10
        return self._SamrWrapper(samr.hSamrUnicodeChangePasswordUser2, '\x00', username, oldPassword, newPassword, oldPwdHashLM, oldPwdHashNT, _change=True)

    def hSamrChangePasswordUser(self, username, oldPassword, newPassword, oldPwdHashLM, oldPwdHashNT, newPwdHashLM, newPwdHashNT):
        if False:
            while True:
                i = 10
        userHandle = self.hSamrOpenUser(username)
        if not userHandle:
            return False
        return self._SamrWrapper(samr.hSamrChangePasswordUser, userHandle, oldPassword=oldPassword, newPassword=newPassword, oldPwdHashNT=oldPwdHashNT, newPwdHashLM=newPwdHashLM, newPwdHashNT=newPwdHashNT, _change=True)

    def hSamrSetInformationUser(self, username, newPassword, newPwdHashLM, newPwdHashNT):
        if False:
            i = 10
            return i + 15
        userHandle = self.hSamrOpenUser(username)
        if not userHandle:
            return False
        return self._SamrWrapper(samr.hSamrSetNTInternal1, userHandle, newPassword, newPwdHashNT, _change=False)

    def _changePassword(self, targetUsername, targetDomain, oldPassword, newPassword, oldPwdHashLM, oldPwdHashNT, newPwdHashLM, newPwdHashNT):
        if False:
            print('Hello World!')
        if not self.connect(retry_if_expired=True):
            return False
        if newPassword:
            return self.hSamrUnicodeChangePasswordUser2(targetUsername, oldPassword, newPassword, oldPwdHashLM, oldPwdHashNT, '', '')
        else:
            res = self.hSamrChangePasswordUser(targetUsername, oldPassword, '', oldPwdHashLM, oldPwdHashNT, newPwdHashLM, newPwdHashNT)
            if res:
                logging.warning('User will need to change their password on next logging because we are using hashes.')
            return res

    def _setPassword(self, targetUsername, targetDomain, newPassword, newPwdHashLM, newPwdHashNT):
        if False:
            return 10
        if not self.connect(retry_if_expired=False):
            return False
        res = self.hSamrSetInformationUser(targetUsername, newPassword, newPwdHashLM, newPwdHashNT)
        if res:
            logging.warning('User no longer has valid AES keys for Kerberos, until they change their password again.')
        return res

class RpcPassword(SamrPassword):

    def rpctransport(self):
        if False:
            i = 10
            return i + 15
        stringBinding = epm.hept_map(self.address, samr.MSRPC_UUID_SAMR, protocol='ncacn_ip_tcp')
        rpctransport = transport.DCERPCTransportFactory(stringBinding)
        rpctransport.setRemoteHost(self.address)
        return rpctransport

    def _changePassword(self, targetUsername, targetDomain, oldPassword, newPassword, oldPwdHashLM, oldPwdHashNT, newPwdHashLM, newPwdHashNT):
        if False:
            while True:
                i = 10
        if not newPassword:
            logging.warning('MS-RPC transport requires new password in plaintext in default Active Directory configuration. Trying anyway.')
        super()._changePassword(targetUsername, targetDomain, oldPassword, newPassword, oldPwdHashLM, oldPwdHashNT, newPwdHashLM, newPwdHashNT)

    def _setPassword(self, targetUsername, targetDomain, newPassword, newPwdHashLM, newPwdHashNT):
        if False:
            while True:
                i = 10
        logging.warning('MS-RPC transport does not allow password reset in default Active Directory configuration. Trying anyway.')
        super()._setPassword(targetUsername, targetDomain, newPassword, newPwdHashLM, newPwdHashNT)

class SmbPassword(SamrPassword):

    def rpctransport(self):
        if False:
            print('Hello World!')
        return transport.SMBTransport(self.address, filename='\\samr')

class LdapPassword(PasswordHandler):
    """Use LDAP to change or reset a user's password"""
    ldapConnection = None
    baseDN = None

    def connect(self, targetDomain):
        if False:
            while True:
                i = 10
        'Connect to LDAPS with the credentials provided in __init__'
        if self.ldapConnection:
            return True
        ldapURI = 'ldaps://' + self.address
        self.baseDN = 'DC=' + ',DC='.join(targetDomain.split('.'))
        logging.debug(f'Connecting to {ldapURI} as {self.domain}\\{self.username}')
        try:
            ldapConnection = ldap.LDAPConnection(ldapURI, self.baseDN, self.address)
            if not self.doKerberos:
                ldapConnection.login(self.username, self.password, self.domain, self.pwdHashLM, self.pwdHashNT)
            else:
                ldapConnection.kerberosLogin(self.username, self.password, self.domain, self.pwdHashLM, self.pwdHashNT, self.aesKey, kdcHost=self.kdcHost)
        except ldap.LDAPSessionError as e:
            logging.error(f'Cannot connect to {ldapURI} as {self.domain}\\{self.username}: {e}')
            return False
        self.ldapConnection = ldapConnection
        return True

    def encodeLdapPassword(self, password):
        if False:
            while True:
                i = 10
        "\n        Encode the password according to Microsoft's specifications\n\n        Password must be surrounded by quotes and UTF-16 encoded\n        "
        return f'"{password}"'.encode('utf-16-le')

    def findTargetDN(self, targetUsername, targetDomain):
        if False:
            return 10
        'Find the DN of the targeted user'
        answers = self.ldapConnection.search(searchFilter=f'(sAMAccountName={targetUsername})', searchBase=self.baseDN, attributes=('distinguishedName',))
        for item in answers:
            if not isinstance(item, ldapasn1.SearchResultEntry):
                continue
            return str(item['objectName'])

    def _modifyPassword(self, change, targetUsername, targetDomain, oldPasswordEncoded, newPasswordEncoded):
        if False:
            while True:
                i = 10
        if not self.connect(targetDomain):
            return False
        targetDN = self.findTargetDN(targetUsername, targetDomain)
        if not targetDN:
            logging.critical('Could not find the target user in LDAP')
            return False
        logging.debug(f'Found target distinguishedName: {targetDN}')
        request = ldapasn1.ModifyRequest()
        request['object'] = targetDN
        if change:
            request['changes'][0]['operation'] = ldapasn1.Operation('delete')
            request['changes'][0]['modification']['type'] = 'unicodePwd'
            request['changes'][0]['modification']['vals'][0] = oldPasswordEncoded
            request['changes'][1]['operation'] = ldapasn1.Operation('add')
            request['changes'][1]['modification']['type'] = 'unicodePwd'
            request['changes'][1]['modification']['vals'][0] = newPasswordEncoded
        else:
            request['changes'][0]['operation'] = ldapasn1.Operation('replace')
            request['changes'][0]['modification']['type'] = 'unicodePwd'
            request['changes'][0]['modification']['vals'][0] = newPasswordEncoded
        logging.debug(f'Sending: {str(request)}')
        response = self.ldapConnection.sendReceive(request)[0]
        logging.debug(f'Receiving: {str(response)}')
        resultCode = int(response['protocolOp']['modifyResponse']['resultCode'])
        result = str(ldapasn1.ResultCode(resultCode))
        diagMessage = str(response['protocolOp']['modifyResponse']['diagnosticMessage'])
        if result == 'success':
            logging.info(f'Password was changed successfully for {targetDN}')
            return True
        if result == 'constraintViolation':
            logging.error(f'Could not change the password of {targetDN}, possibly due to the password policy or an invalid oldPassword.')
        elif result == 'insufficientAccessRights':
            logging.error(f'Could not set the password of {targetDN}, {self.domain}\\{self.username} has insufficient rights')
        else:
            logging.error(f'Could not change the password of {targetDN}. {result}: {diagMessage}')
        return False

    def _changePassword(self, targetUsername, targetDomain, oldPassword, newPassword, oldPwdHashLM, oldPwdHashNT, newPwdHashLM, newPwdHashNT):
        if False:
            for i in range(10):
                print('nop')
        '\n        Change the password of a user.\n\n        Must send a delete operation with the oldPassword and an add\n        operation with the newPassword in the same modify request.\n        '
        if not oldPassword or not newPassword:
            logging.critical('LDAP requires the old and new passwords in plaintext')
            return False
        oldPasswordEncoded = self.encodeLdapPassword(oldPassword)
        newPasswordEncoded = self.encodeLdapPassword(newPassword)
        return self._modifyPassword(True, targetUsername, targetDomain, oldPasswordEncoded, newPasswordEncoded)

    def _setPassword(self, targetUsername, targetDomain, newPassword, newPwdHashLM, newPwdHashNT):
        if False:
            while True:
                i = 10
        '\n        Set the password of a user.\n\n        Must send a modify operation with the newPassword (must have privileges).\n        '
        if not newPassword:
            logging.critical('LDAP requires the new password in plaintext')
            return False
        newPasswordEncoded = self.encodeLdapPassword(newPassword)
        return self._modifyPassword(False, targetUsername, targetDomain, None, newPasswordEncoded)

def init_logger(options):
    if False:
        print('Hello World!')
    logger.init(options.ts)
    if options.debug is True:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.debug(version.getInstallationPath())
    else:
        logging.getLogger().setLevel(logging.INFO)

def parse_args():
    if False:
        for i in range(10):
            print('nop')
    parser = argparse.ArgumentParser(description='Change or reset passwords over different protocols.')
    parser.add_argument('target', action='store', help='[[domain/]username[:password]@]<hostname or address>')
    parser.add_argument('-ts', action='store_true', help='adds timestamp to every logging output')
    parser.add_argument('-debug', action='store_true', help='turn DEBUG output ON')
    group = parser.add_argument_group('New credentials for target')
    exgroup = group.add_mutually_exclusive_group()
    exgroup.add_argument('-newpass', action='store', default=None, help='new password')
    exgroup.add_argument('-newhashes', action='store', default=None, metavar='LMHASH:NTHASH', help='new NTLM hashes, format is NTHASH or LMHASH:NTHASH')
    group = parser.add_argument_group('Authentication (target user whose password is changed)')
    group.add_argument('-hashes', action='store', default=None, metavar='LMHASH:NTHASH', help='NTLM hashes, format is NTHASH or LMHASH:NTHASH')
    group.add_argument('-no-pass', action='store_true', help="Don't ask for password (useful for Kerberos, -k)")
    group = parser.add_argument_group('Authentication (optional, privileged user performing the change)')
    group.add_argument('-altuser', action='store', default=None, help='Alternative username')
    exgroup = group.add_mutually_exclusive_group()
    exgroup.add_argument('-altpass', action='store', default=None, help='Alternative password')
    exgroup.add_argument('-althash', '-althashes', action='store', default=None, help='Alternative NT hash, format is NTHASH or LMHASH:NTHASH')
    group = parser.add_argument_group('Method of operations')
    group.add_argument('-protocol', '-p', action='store', help='Protocol to use for password change/reset', default='smb-samr', choices=('smb-samr', 'rpc-samr', 'kpasswd', 'ldap'))
    group.add_argument('-reset', '-admin', action='store_true', help='Try to reset the password with privileges (may bypass some password policies)')
    group = parser.add_argument_group('Kerberos authentication', description='Applicable to the authenticating user (-altuser if defined, else target)')
    group.add_argument('-k', action='store_true', help='Use Kerberos authentication. Grabs credentials from ccache file (KRB5CCNAME) based on target parameters. If valid credentials cannot be found, it will use the ones specified in the command line')
    group.add_argument('-aesKey', action='store', metavar='hex key', help='AES key to use for Kerberos Authentication (128 or 256 bits)')
    group.add_argument('-dc-ip', action='store', metavar='ip address', help='IP Address of the domain controller, for Kerberos. If omitted it will use the domain part (FQDN) specified in the target parameter')
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()
if __name__ == '__main__':
    print(version.BANNER)
    options = parse_args()
    init_logger(options)
    handlers = {'kpasswd': KPassword, 'rpc-samr': RpcPassword, 'smb-samr': SmbPassword, 'ldap': LdapPassword}
    try:
        PasswordProtocol = handlers[options.protocol]
    except KeyError:
        logging.critical(f'Unsupported password protocol {options.protocol}')
        sys.exit(1)
    (targetDomain, targetUsername, oldPassword, address) = parse_target(options.target)
    if not targetDomain:
        if options.protocol in ('rpc-samr', 'smb-samr'):
            targetDomain = 'Builtin'
        else:
            targetDomain = address
    if options.hashes is not None:
        try:
            (oldPwdHashLM, oldPwdHashNT) = options.hashes.split(':')
        except ValueError:
            oldPwdHashLM = EMPTY_LM_HASH
            oldPwdHashNT = options.hashes
    else:
        oldPwdHashLM = ''
        oldPwdHashNT = ''
    if oldPassword == '' and oldPwdHashNT == '':
        if options.reset:
            pass
        elif options.no_pass:
            logging.info('Current password not given: will use KRB5CCNAME')
        else:
            oldPassword = getpass('Current password: ')
    if options.newhashes is not None:
        newPassword = ''
        try:
            (newPwdHashLM, newPwdHashNT) = options.newhashes.split(':')
            if not newPwdHashLM:
                newPwdHashLM = EMPTY_LM_HASH
        except ValueError:
            newPwdHashLM = EMPTY_LM_HASH
            newPwdHashNT = options.newhashes
    else:
        newPwdHashLM = ''
        newPwdHashNT = ''
        if options.newpass is None:
            newPassword = getpass('New password: ')
            if newPassword != getpass('Retype new password: '):
                logging.critical('Passwords do not match, try again.')
                sys.exit(1)
        else:
            newPassword = options.newpass
    if options.altuser is not None:
        try:
            (authDomain, authUsername) = options.altuser.split('/')
        except ValueError:
            authDomain = targetDomain
            authUsername = options.altuser
        if options.althash is not None:
            try:
                (authPwdHashLM, authPwdHashNT) = options.althash.split(':')
            except ValueError:
                authPwdHashLM = ''
                authPwdHashNT = options.althash
        else:
            authPwdHashLM = ''
            authPwdHashNT = ''
        authPassword = ''
        if options.altpass is not None:
            authPassword = options.altpass
        if options.altpass is None and options.althash is None and (not options.no_pass):
            logging.critical('Please, provide either alternative password (-altpass) or NT hash (-althash) for authentication, or specify -no-pass if you rely on Kerberos only')
            sys.exit(1)
    else:
        authDomain = targetDomain
        authUsername = targetUsername
        authPassword = oldPassword
        authPwdHashLM = oldPwdHashLM
        authPwdHashNT = oldPwdHashNT
    doKerberos = options.k
    if options.protocol == 'kpasswd' and (not doKerberos):
        logging.debug('Using the KPassword protocol implies Kerberos authentication (-k)')
        doKerberos = True
    handler = PasswordProtocol(address, authDomain, authUsername, authPassword, authPwdHashLM, authPwdHashNT, doKerberos, options.aesKey, kdcHost=options.dc_ip)
    if options.reset:
        handler.setPassword(targetUsername, targetDomain, newPassword, newPwdHashLM, newPwdHashNT)
    else:
        if (authDomain, authUsername) != (targetDomain, targetUsername):
            logging.warning(f"Attempting to *change* the password of {targetDomain}/{targetUsername} as {authDomain}/{authUsername}. You may want to use '-reset' to *reset* the password of the target.")
        handler.changePassword(targetUsername, targetDomain, oldPassword, newPassword, oldPwdHashLM, oldPwdHashNT, newPwdHashLM, newPwdHashNT)