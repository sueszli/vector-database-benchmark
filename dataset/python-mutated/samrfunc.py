import logging
from impacket.dcerpc.v5 import transport, lsat, lsad, samr
from impacket.dcerpc.v5.dtypes import MAXIMUM_ALLOWED
from impacket.dcerpc.v5.rpcrt import RPC_C_AUTHN_GSS_NEGOTIATE
from impacket.nmb import NetBIOSError
from impacket.smbconnection import SessionError
from cme.logger import cme_logger

class SamrFunc:

    def __init__(self, connection):
        if False:
            print('Hello World!')
        self.logger = connection.logger
        self.addr = connection.host if not connection.kerberos else connection.hostname + '.' + connection.domain
        self.protocol = connection.args.port
        self.username = connection.username
        self.password = connection.password
        self.domain = connection.domain
        self.hash = connection.hash
        self.lmhash = ''
        self.nthash = ''
        self.aesKey = connection.aesKey
        self.doKerberos = connection.kerberos
        if self.hash is not None:
            if self.hash.find(':') != -1:
                (self.lmhash, self.nthash) = self.hash.split(':')
            else:
                self.nthash = self.hash
        if self.password is None:
            self.password = ''
        self.samr_query = SAMRQuery(username=self.username, password=self.password, domain=self.domain, remote_name=self.addr, remote_host=self.addr, kerberos=self.doKerberos, aesKey=self.aesKey)
        self.lsa_query = LSAQuery(username=self.username, password=self.password, domain=self.domain, remote_name=self.addr, remote_host=self.addr, kerberos=self.doKerberos, aesKey=self.aesKey, logger=self.logger)

    def get_builtin_groups(self):
        if False:
            return 10
        domains = self.samr_query.get_domains()
        if 'Builtin' not in domains:
            logging.error(f'No Builtin group to query locally on')
            return
        domain_handle = self.samr_query.get_domain_handle('Builtin')
        groups = self.samr_query.get_domain_aliases(domain_handle)
        return groups

    def get_custom_groups(self):
        if False:
            i = 10
            return i + 15
        domains = self.samr_query.get_domains()
        custom_groups = {}
        for domain in domains:
            if domain == 'Builtin':
                continue
            domain_handle = self.samr_query.get_domain_handle(domain)
            custom_groups.update(self.samr_query.get_domain_aliases(domain_handle))
        return custom_groups

    def get_local_groups(self):
        if False:
            for i in range(10):
                print('nop')
        builtin_groups = self.get_builtin_groups()
        custom_groups = self.get_custom_groups()
        return {**builtin_groups, **custom_groups}

    def get_local_users(self):
        if False:
            i = 10
            return i + 15
        pass

    def get_local_administrators(self):
        if False:
            for i in range(10):
                print('nop')
        self.get_builtin_groups()
        if 'Administrators' in self.groups:
            self.logger.success(f"Found Local Administrators group: RID {self.groups['Administrators']}")
        domain_handle = self.samr_query.get_domain_handle('Builtin')
        self.logger.debug(f'Querying group members')
        member_sids = self.samr_query.get_alias_members(domain_handle, self.groups['Administrators'])
        member_names = self.lsa_query.lookup_sids(member_sids)
        for (sid, name) in zip(member_sids, member_names):
            print(f'{name} - {sid}')

class SAMRQuery:

    def __init__(self, username='', password='', domain='', port=445, remote_name='', remote_host='', kerberos=None, aesKey=''):
        if False:
            while True:
                i = 10
        self.__username = username
        self.__password = password
        self.__domain = domain
        self.__lmhash = ''
        self.__nthash = ''
        self.__aesKey = aesKey
        self.__port = port
        self.__remote_name = remote_name
        self.__remote_host = remote_host
        self.__kerberos = kerberos
        self.dce = self.get_dce()
        self.server_handle = self.get_server_handle()

    def get_transport(self):
        if False:
            return 10
        string_binding = f'ncacn_np:{self.__port}[\\pipe\\samr]'
        cme_logger.debug(f'Binding to {string_binding}')
        rpc_transport = transport.SMBTransport(self.__remote_host, self.__port, '\\samr', self.__username, self.__password, self.__domain, self.__lmhash, self.__nthash, self.__aesKey, doKerberos=self.__kerberos)
        return rpc_transport

    def get_dce(self):
        if False:
            while True:
                i = 10
        rpc_transport = self.get_transport()
        try:
            dce = rpc_transport.get_dce_rpc()
            dce.connect()
            dce.bind(samr.MSRPC_UUID_SAMR)
        except NetBIOSError as e:
            logging.error(f'NetBIOSError on Connection: {e}')
            return
        except SessionError as e:
            logging.error(f'SessionError on Connection: {e}')
            return
        return dce

    def get_server_handle(self):
        if False:
            while True:
                i = 10
        if self.dce:
            try:
                resp = samr.hSamrConnect(self.dce)
            except samr.DCERPCException as e:
                cme_logger.debug(f'Error while connecting with Samr: {e}')
                return None
            return resp['ServerHandle']
        else:
            cme_logger.debug(f'Error creating Samr handle')
            return

    def get_domains(self):
        if False:
            i = 10
            return i + 15
        resp = samr.hSamrEnumerateDomainsInSamServer(self.dce, self.server_handle)
        domains = resp['Buffer']['Buffer']
        domain_names = []
        for domain in domains:
            domain_names.append(domain['Name'])
        return domain_names

    def get_domain_handle(self, domain_name):
        if False:
            while True:
                i = 10
        resp = samr.hSamrLookupDomainInSamServer(self.dce, self.server_handle, domain_name)
        resp = samr.hSamrOpenDomain(self.dce, serverHandle=self.server_handle, domainId=resp['DomainId'])
        return resp['DomainHandle']

    def get_domain_aliases(self, domain_handle):
        if False:
            print('Hello World!')
        resp = samr.hSamrEnumerateAliasesInDomain(self.dce, domain_handle)
        aliases = {}
        for alias in resp['Buffer']['Buffer']:
            aliases[alias['Name']] = alias['RelativeId']
        return aliases

    def get_alias_handle(self, domain_handle, alias_id):
        if False:
            return 10
        resp = samr.hSamrOpenAlias(self.dce, domain_handle, desiredAccess=MAXIMUM_ALLOWED, aliasId=alias_id)
        return resp['AliasHandle']

    def get_alias_members(self, domain_handle, alias_id):
        if False:
            print('Hello World!')
        alias_handle = self.get_alias_handle(domain_handle, alias_id)
        resp = samr.hSamrGetMembersInAlias(self.dce, alias_handle)
        member_sids = []
        for member in resp['Members']['Sids']:
            member_sids.append(member['SidPointer'].formatCanonical())
        return member_sids

class LSAQuery:

    def __init__(self, username='', password='', domain='', port=445, remote_name='', remote_host='', aesKey='', kerberos=None, logger=None):
        if False:
            i = 10
            return i + 15
        self.__username = username
        self.__password = password
        self.__domain = domain
        self.__lmhash = ''
        self.__nthash = ''
        self.__aesKey = aesKey
        self.__port = port
        self.__remote_name = remote_name
        self.__remote_host = remote_host
        self.__kerberos = kerberos
        self.dce = self.get_dce()
        self.policy_handle = self.get_policy_handle()
        self.logger = logger

    def get_transport(self):
        if False:
            return 10
        string_binding = f'ncacn_np:{self.__remote_name}[\\pipe\\lsarpc]'
        rpc_transport = transport.DCERPCTransportFactory(string_binding)
        rpc_transport.set_dport(self.__port)
        rpc_transport.setRemoteHost(self.__remote_host)
        if self.__kerberos:
            rpc_transport.set_kerberos(True, None)
        if hasattr(rpc_transport, 'set_credentials'):
            rpc_transport.set_credentials(self.__username, self.__password, self.__domain, self.__lmhash, self.__nthash, self.__aesKey)
        return rpc_transport

    def get_dce(self):
        if False:
            print('Hello World!')
        rpc_transport = self.get_transport()
        try:
            dce = rpc_transport.get_dce_rpc()
            if self.__kerberos:
                dce.set_auth_type(RPC_C_AUTHN_GSS_NEGOTIATE)
            dce.connect()
            dce.bind(lsat.MSRPC_UUID_LSAT)
        except NetBIOSError as e:
            self.logger.fail(f'NetBIOSError on Connection: {e}')
            return None
        return dce

    def get_policy_handle(self):
        if False:
            print('Hello World!')
        resp = lsad.hLsarOpenPolicy2(self.dce, MAXIMUM_ALLOWED | lsat.POLICY_LOOKUP_NAMES)
        return resp['PolicyHandle']

    def lookup_sids(self, sids):
        if False:
            print('Hello World!')
        resp = lsat.hLsarLookupSids(self.dce, self.policy_handle, sids, lsat.LSAP_LOOKUP_LEVEL.LsapLookupWksta)
        names = []
        for translated_names in resp['TranslatedNames']['Names']:
            names.append(translated_names['Name'])
        return names