import sys
import logging
import argparse
from impacket.examples import logger
from impacket.examples.utils import parse_target
from impacket import version
from impacket.uuid import bin_to_string
from impacket.dcerpc.v5 import transport, dssp

class MachineRole:
    MACHINE_ROLES = {dssp.DSROLE_MACHINE_ROLE.DsRole_RoleStandaloneWorkstation: 'Standalone Workstation', dssp.DSROLE_MACHINE_ROLE.DsRole_RoleMemberWorkstation: 'Domain-joined Workstation', dssp.DSROLE_MACHINE_ROLE.DsRole_RoleStandaloneServer: 'Standalone Server', dssp.DSROLE_MACHINE_ROLE.DsRole_RoleMemberServer: 'Domain-joined Server', dssp.DSROLE_MACHINE_ROLE.DsRole_RoleBackupDomainController: 'Backup Domain Controller', dssp.DSROLE_MACHINE_ROLE.DsRole_RolePrimaryDomainController: 'Primary Domain Controller'}

    def __init__(self, username='', password='', domain='', hashes=None, aesKey=None, doKerberos=False, kdcHost=None, port=445):
        if False:
            for i in range(10):
                print('nop')
        self.__username = username
        self.__password = password
        self.__domain = domain
        self.__lmhash = ''
        self.__nthash = ''
        self.__aesKey = aesKey
        self.__doKerberos = doKerberos
        self.__kdcHost = kdcHost
        self.__port = port
        if hashes is not None:
            (self.__lmhash, self.__nthash) = hashes.split(':')

    def print_info(self, remoteName, remoteHost):
        if False:
            i = 10
            return i + 15
        try:
            dce = self.__authenticate(remoteName, remoteHost)
        except Exception as e:
            self.__log_and_exit(str(e))
        try:
            output = self.__fetch(dce)
        except Exception as e:
            self.__log_and_exit(str(e))
        for (key, value) in output.items():
            print('%s: %s' % (key, value))
        dce.disconnect()

    def __authenticate(self, remoteName, remoteHost):
        if False:
            i = 10
            return i + 15
        dce = self.__get_transport(remoteName, remoteHost)
        dce.connect()
        dce.bind(dssp.MSRPC_UUID_DSSP)
        return dce

    def __get_transport(self, remoteName, remoteHost):
        if False:
            for i in range(10):
                print('nop')
        stringbinding = 'ncacn_np:%s[\\pipe\\lsarpc]' % remoteName
        logging.debug('StringBinding %s' % stringbinding)
        rpctransport = transport.DCERPCTransportFactory(stringbinding)
        rpctransport.set_dport(self.__port)
        rpctransport.setRemoteHost(remoteHost)
        if hasattr(rpctransport, 'set_credentials'):
            rpctransport.set_credentials(self.__username, self.__password, self.__domain, self.__lmhash, self.__nthash, self.__aesKey)
        rpctransport.set_kerberos(self.__doKerberos, self.__kdcHost)
        return rpctransport.get_dce_rpc()

    def __fetch(self, dce):
        if False:
            return 10
        output = {}
        domain_info = dssp.hDsRolerGetPrimaryDomainInformation(dce, 1)
        output['Machine Role'] = self.MACHINE_ROLES[domain_info['DomainInfo']['DomainInfoBasic']['MachineRole']]
        output['NetBIOS Domain Name'] = domain_info['DomainInfo']['DomainInfoBasic']['DomainNameFlat']
        output['Domain Name'] = domain_info['DomainInfo']['DomainInfoBasic']['DomainNameDns']
        output['Forest Name'] = domain_info['DomainInfo']['DomainInfoBasic']['DomainForestName']
        output['Domain GUID'] = bin_to_string(domain_info['DomainInfo']['DomainInfoBasic']['DomainGuid'])
        return output

    def __log_and_exit(self, error):
        if False:
            while True:
                i = 10
        logging.critical('Error while enumerating host: %s' % error)
        sys.exit(1)
if __name__ == '__main__':
    print(version.BANNER)
    parser = argparse.ArgumentParser(add_help=True, description="Retrieve a host's role along with its primary domain details.")
    parser.add_argument('target', action='store', help='[[domain/]username[:password]@]<targetName or address>')
    parser.add_argument('-ts', action='store_true', help='Adds timestamp to every logging output')
    parser.add_argument('-debug', action='store_true', help='Turn DEBUG output ON')
    group = parser.add_argument_group('connection')
    group.add_argument('-dc-ip', action='store', metavar='ip address', help='IP Address of the domain controller. If ommited it use the domain part (FQDN) specified in the target parameter')
    group.add_argument('-target-ip', action='store', metavar='ip address', help='IP Address of the target machine. If ommited it will use whatever was specified as target. This is useful when target is the NetBIOS name and you cannot resolve it')
    group.add_argument('-port', choices=['139', '445'], nargs='?', default='445', metavar='destination port', help='Destination port to connect to SMB Server')
    group = parser.add_argument_group('authentication')
    group.add_argument('-hashes', action='store', metavar='LMHASH:NTHASH', help='NTLM hashes, format is LMHASH:NTHASH')
    group.add_argument('-no-pass', action='store_true', help="don't ask for password (useful for -k)")
    group.add_argument('-k', action='store_true', help='Use Kerberos authentication. Grabs credentials from ccache file (KRB5CCNAME) based on target parameters. If valid credentials cannot be found, it will use the ones specified in the command line')
    group.add_argument('-aesKey', action='store', metavar='hex key', help='AES key to use for Kerberos Authentication (128 or 256 bits)')
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    options = parser.parse_args()
    logger.init(options.ts)
    if options.debug is True:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.debug(version.getInstallationPath())
    else:
        logging.getLogger().setLevel(logging.INFO)
    (domain, username, password, remoteName) = parse_target(options.target)
    if domain is None:
        domain = ''
    if options.target_ip is None:
        options.target_ip = remoteName
    if options.aesKey is not None:
        options.k = True
    if password == '' and username != '' and (options.hashes is None) and (options.no_pass is False) and (options.aesKey is None):
        from getpass import getpass
        password = getpass('Password:')
    machine_role = MachineRole(username, password, domain, options.hashes, options.aesKey, options.k, options.dc_ip, int(options.port))
    machine_role.print_info(remoteName, options.target_ip)