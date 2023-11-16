import pytest
import unittest
from struct import unpack
from tests.dcerpc import DCERPCTests
from impacket.dcerpc.v5 import nrpc
from impacket.dcerpc.v5.rpcrt import DCERPCException
from impacket.dcerpc.v5.dtypes import NULL
from impacket import ntlm

class NRPCTests(DCERPCTests):
    iface_uuid = nrpc.MSRPC_UUID_NRPC
    authn = True
    machine_account = True

    def authenticate(self, dce):
        if False:
            return 10
        resp = nrpc.hNetrServerReqChallenge(dce, self.serverName, self.machine_user, b'12345678')
        resp.dump()
        serverChallenge = resp['ServerChallenge']
        bnthash = self.machine_user_bnthash or None
        self.sessionKey = nrpc.ComputeSessionKeyStrongKey('', b'12345678', serverChallenge, bnthash)
        self.clientStoredCredential = nrpc.ComputeNetlogonCredential(b'12345678', self.sessionKey)
        try:
            resp = nrpc.hNetrServerAuthenticate3(dce, self.serverName, self.machine_user + '\x00', nrpc.NETLOGON_SECURE_CHANNEL_TYPE.WorkstationSecureChannel, self.machine_user, self.clientStoredCredential, 1611661311)
            resp.dump()
        except nrpc.DCERPCSessionError as e:
            if str(e).find('STATUS_DOWNGRADE_DETECTED') < 0:
                raise

    def update_authenticator(self):
        if False:
            return 10
        return nrpc.ComputeNetlogonAuthenticator(self.clientStoredCredential, self.sessionKey)

    def test_DsrGetDcNameEx2(self):
        if False:
            return 10
        (dce, rpctransport) = self.connect()
        request = nrpc.DsrGetDcNameEx2()
        request['ComputerName'] = self.serverName + '\x00'
        request['AccountName'] = 'Administrator\x00'
        request['AllowableAccountControlBits'] = 1 << 9
        request['DomainName'] = NULL
        request['DomainGuid'] = NULL
        request['SiteName'] = NULL
        request['Flags'] = 0
        resp = dce.request(request)
        resp.dump()

    def test_hDsrGetDcNameEx2(self):
        if False:
            return 10
        (dce, rpctransport) = self.connect()
        resp = nrpc.hDsrGetDcNameEx2(dce, NULL, 'Administrator\x00', 1 << 9, NULL, NULL, NULL, 0)
        resp.dump()

    def test_DsrGetDcNameEx(self):
        if False:
            return 10
        (dce, rpctransport) = self.connect()
        request = nrpc.DsrGetDcNameEx()
        request['ComputerName'] = NULL
        request['DomainName'] = NULL
        request['DomainGuid'] = NULL
        request['SiteName'] = NULL
        request['Flags'] = 0
        resp = dce.request(request)
        resp.dump()

    def test_hDsrGetDcNameEx(self):
        if False:
            i = 10
            return i + 15
        (dce, rpctransport) = self.connect()
        resp = nrpc.hDsrGetDcNameEx(dce, NULL, NULL, NULL, NULL, 0)
        resp.dump()

    def test_DsrGetDcName(self):
        if False:
            return 10
        (dce, rpctransport) = self.connect()
        request = nrpc.DsrGetDcName()
        request['ComputerName'] = NULL
        request['DomainName'] = NULL
        request['DomainGuid'] = NULL
        request['SiteGuid'] = NULL
        request['Flags'] = 0
        resp = dce.request(request)
        resp.dump()

    def test_hDsrGetDcName(self):
        if False:
            for i in range(10):
                print('nop')
        (dce, rpctransport) = self.connect()
        resp = nrpc.hDsrGetDcName(dce, self.serverName, NULL, NULL, NULL, 0)
        resp.dump()

    def test_NetrGetDCName(self):
        if False:
            print('Hello World!')
        (dce, rpctransport) = self.connect()
        request = nrpc.NetrGetDCName()
        request['ServerName'] = '\x00' * 20
        request['DomainName'] = self.domain.split('.')[0] + '\x00'
        resp = dce.request(request)
        resp.dump()

    def test_hNetrGetDCName(self):
        if False:
            i = 10
            return i + 15
        (dce, rpctransport) = self.connect()
        resp = nrpc.hNetrGetDCName(dce, '\x00' * 20, self.domain.split('.')[0])
        resp.dump()

    def test_NetrGetAnyDCName(self):
        if False:
            while True:
                i = 10
        (dce, rpctransport) = self.connect()
        request = nrpc.NetrGetAnyDCName()
        request['ServerName'] = self.serverName + '\x00'
        request['DomainName'] = self.domain + '\x00'
        try:
            dce.request(request)
        except DCERPCException as e:
            if str(e).find('ERROR_NO_SUCH_DOMAIN') < 0:
                raise

    def test_hNetrGetAnyDCName(self):
        if False:
            for i in range(10):
                print('nop')
        (dce, rpctransport) = self.connect()
        try:
            nrpc.hNetrGetAnyDCName(dce, NULL, self.domain)
        except DCERPCException as e:
            if str(e).find('ERROR_NO_SUCH_DOMAIN') < 0:
                raise

    def test_DsrGetSiteName(self):
        if False:
            for i in range(10):
                print('nop')
        (dce, rpctransport) = self.connect()
        request = nrpc.DsrGetSiteName()
        request['ComputerName'] = NULL
        resp = dce.request(request)
        resp.dump()

    def test_hDsrGetSiteName(self):
        if False:
            while True:
                i = 10
        (dce, rpctransport) = self.connect()
        resp = nrpc.hDsrGetSiteName(dce, NULL)
        resp.dump()

    def test_DsrGetDcSiteCoverageW(self):
        if False:
            for i in range(10):
                print('nop')
        (dce, rpctransport) = self.connect()
        request = nrpc.DsrGetDcSiteCoverageW()
        request['ServerName'] = self.serverName + '\x00'
        resp = dce.request(request)
        resp.dump()

    def test_hDsrGetDcSiteCoverageW(self):
        if False:
            for i in range(10):
                print('nop')
        (dce, rpctransport) = self.connect()
        resp = nrpc.hDsrGetDcSiteCoverageW(dce, self.serverName)
        resp.dump()

    def test_DsrAddressToSiteNamesW(self):
        if False:
            for i in range(10):
                print('nop')
        (dce, rpctransport) = self.connect()
        request = nrpc.DsrAddressToSiteNamesW()
        request['ComputerName'] = NULL
        request['EntryCount'] = 1
        addr = nrpc.IPv4Address()
        import socket
        addr['AddressFamily'] = socket.AF_INET
        addr['Port'] = 0
        addr['Address'] = unpack('>L', socket.inet_aton(self.machine))[0]
        socketAddress = nrpc.NL_SOCKET_ADDRESS()
        socketAddress['lpSockaddr'] = list(addr.getData())
        socketAddress['iSockaddrLength'] = len(addr.getData())
        request['SocketAddresses'].append(socketAddress)
        resp = dce.request(request)
        resp.dump()

    def test_hDsrAddressToSiteNamesW(self):
        if False:
            while True:
                i = 10
        (dce, rpctransport) = self.connect()
        request = nrpc.DsrAddressToSiteNamesW()
        request['ComputerName'] = NULL
        request['EntryCount'] = 1
        addr = nrpc.IPv4Address()
        import socket
        addr['AddressFamily'] = socket.AF_INET
        addr['Port'] = 0
        addr['Address'] = unpack('>L', socket.inet_aton(self.machine))[0]
        socketAddress = nrpc.NL_SOCKET_ADDRESS()
        socketAddress['lpSockaddr'] = list(addr.getData())
        socketAddress['iSockaddrLength'] = len(addr.getData())
        request['SocketAddresses'].append(socketAddress)
        resp = dce.request(request)
        resp.dump()

    def test_DsrAddressToSiteNamesExW(self):
        if False:
            i = 10
            return i + 15
        (dce, rpctransport) = self.connect()
        request = nrpc.DsrAddressToSiteNamesExW()
        request['ComputerName'] = NULL
        request['EntryCount'] = 1
        addr = nrpc.IPv4Address()
        import socket
        addr['AddressFamily'] = socket.AF_INET
        addr['Port'] = 0
        addr['Address'] = unpack('>L', socket.inet_aton(self.machine))[0]
        socketAddress = nrpc.NL_SOCKET_ADDRESS()
        socketAddress['lpSockaddr'] = list(addr.getData())
        socketAddress['iSockaddrLength'] = len(addr.getData())
        request['SocketAddresses'].append(socketAddress)
        resp = dce.request(request)
        resp.dump()

    def test_DsrDeregisterDnsHostRecords(self):
        if False:
            while True:
                i = 10
        (dce, rpctransport) = self.connect()
        request = nrpc.DsrDeregisterDnsHostRecords()
        request['ServerName'] = NULL
        request['DnsDomainName'] = 'BETUS\x00'
        request['DomainGuid'] = NULL
        request['DsaGuid'] = NULL
        request['DnsHostName'] = 'BETUS\x00'
        try:
            dce.request(request)
        except DCERPCException as e:
            if str(e).find('rpc_s_access_denied') < 0:
                raise

    def test_NetrServerReqChallenge_NetrServerAuthenticate3(self):
        if False:
            return 10
        (dce, rpctransport) = self.connect()
        request = nrpc.NetrServerReqChallenge()
        request['PrimaryName'] = self.serverName + '\x00'
        request['ComputerName'] = self.machine_user + '\x00'
        request['ClientChallenge'] = b'12345678'
        resp = dce.request(request)
        resp.dump()
        serverChallenge = resp['ServerChallenge']
        bnthash = self.machine_user_bnthash or None
        sessionKey = nrpc.ComputeSessionKeyStrongKey(self.password, b'12345678', serverChallenge, bnthash)
        ppp = nrpc.ComputeNetlogonCredential(b'12345678', sessionKey)
        request = nrpc.NetrServerAuthenticate3()
        request['PrimaryName'] = self.serverName + '\x00'
        request['AccountName'] = self.machine_user + '\x00'
        request['SecureChannelType'] = nrpc.NETLOGON_SECURE_CHANNEL_TYPE.WorkstationSecureChannel
        request['ComputerName'] = self.machine_user + '\x00'
        request['ClientCredential'] = ppp
        request['NegotiateFlags'] = 1611661311
        resp = dce.request(request)
        resp.dump()

    def test_hNetrServerReqChallenge_hNetrServerAuthenticate3(self):
        if False:
            while True:
                i = 10
        (dce, rpctransport) = self.connect()
        resp = nrpc.hNetrServerReqChallenge(dce, self.serverName, self.machine_user, b'12345678')
        resp.dump()
        serverChallenge = resp['ServerChallenge']
        bnthash = self.machine_user_bnthash or None
        sessionKey = nrpc.ComputeSessionKeyStrongKey(self.password, b'12345678', serverChallenge, bnthash)
        ppp = nrpc.ComputeNetlogonCredential(b'12345678', sessionKey)
        resp = nrpc.hNetrServerAuthenticate3(dce, self.serverName, self.machine_user, nrpc.NETLOGON_SECURE_CHANNEL_TYPE.WorkstationSecureChannel, self.machine_user, ppp, 1611661311)
        resp.dump()

    def test_NetrServerReqChallenge_hNetrServerAuthenticate2(self):
        if False:
            print('Hello World!')
        (dce, rpctransport) = self.connect()
        request = nrpc.NetrServerReqChallenge()
        request['PrimaryName'] = self.serverName + '\x00'
        request['ComputerName'] = self.machine_user + '\x00'
        request['ClientChallenge'] = b'12345678'
        resp = dce.request(request)
        resp.dump()
        serverChallenge = resp['ServerChallenge']
        bnthash = self.machine_user_bnthash or None
        sessionKey = nrpc.ComputeSessionKeyStrongKey(self.password, b'12345678', serverChallenge, bnthash)
        ppp = nrpc.ComputeNetlogonCredential(b'12345678', sessionKey)
        resp = nrpc.hNetrServerAuthenticate2(dce, self.serverName, self.machine_user, nrpc.NETLOGON_SECURE_CHANNEL_TYPE.WorkstationSecureChannel, self.machine_user, ppp, 1611661311)
        resp.dump()

    def test_hNetrServerReqChallenge_NetrServerAuthenticate2(self):
        if False:
            i = 10
            return i + 15
        (dce, rpctransport) = self.connect()
        resp = nrpc.hNetrServerReqChallenge(dce, self.serverName, self.machine_user, b'12345678')
        resp.dump()
        serverChallenge = resp['ServerChallenge']
        bnthash = self.machine_user_bnthash or None
        sessionKey = nrpc.ComputeSessionKeyStrongKey(self.password, b'12345678', serverChallenge, bnthash)
        ppp = nrpc.ComputeNetlogonCredential(b'12345678', sessionKey)
        request = nrpc.NetrServerAuthenticate2()
        request['PrimaryName'] = self.serverName + '\x00'
        request['AccountName'] = self.machine_user + '\x00'
        request['SecureChannelType'] = nrpc.NETLOGON_SECURE_CHANNEL_TYPE.WorkstationSecureChannel
        request['ComputerName'] = self.machine_user + '\x00'
        request['ClientCredential'] = ppp
        request['NegotiateFlags'] = 1611661311
        resp = dce.request(request)
        resp.dump()

    def test_NetrServerReqChallenge_NetrServerAuthenticate(self):
        if False:
            i = 10
            return i + 15
        (dce, rpctransport) = self.connect()
        request = nrpc.NetrServerReqChallenge()
        request['PrimaryName'] = self.serverName + '\x00'
        request['ComputerName'] = self.machine_user + '\x00'
        request['ClientChallenge'] = b'12345678'
        resp = dce.request(request)
        resp.dump()
        serverChallenge = resp['ServerChallenge']
        bnthash = self.machine_user_bnthash or None
        sessionKey = nrpc.ComputeSessionKeyStrongKey(self.password, b'12345678', serverChallenge, bnthash)
        ppp = nrpc.ComputeNetlogonCredential(b'12345678', sessionKey)
        request = nrpc.NetrServerAuthenticate()
        request['PrimaryName'] = self.serverName + '\x00'
        request['AccountName'] = self.machine_user + '\x00'
        request['SecureChannelType'] = nrpc.NETLOGON_SECURE_CHANNEL_TYPE.WorkstationSecureChannel
        request['ComputerName'] = self.machine_user + '\x00'
        request['ClientCredential'] = ppp
        try:
            dce.request(request)
        except DCERPCException as e:
            if str(e).find('STATUS_DOWNGRADE_DETECTED') < 0:
                raise

    def test_hNetrServerReqChallenge_hNetrServerAuthenticate(self):
        if False:
            return 10
        (dce, rpctransport) = self.connect()
        resp = nrpc.hNetrServerReqChallenge(dce, self.serverName, self.machine_user, b'12345678')
        resp.dump()
        serverChallenge = resp['ServerChallenge']
        bnthash = self.machine_user_bnthash or None
        sessionKey = nrpc.ComputeSessionKeyStrongKey(self.password, b'12345678', serverChallenge, bnthash)
        ppp = nrpc.ComputeNetlogonCredential(b'12345678', sessionKey)
        resp.dump()
        try:
            nrpc.hNetrServerAuthenticate(dce, self.serverName, self.machine_user, nrpc.NETLOGON_SECURE_CHANNEL_TYPE.WorkstationSecureChannel, self.serverName, ppp)
        except DCERPCException as e:
            if str(e).find('STATUS_DOWNGRADE_DETECTED') < 0:
                raise

    @pytest.mark.xfail
    def test_NetrServerPasswordGet(self):
        if False:
            for i in range(10):
                print('nop')
        (dce, rpctransport) = self.connect()
        self.authenticate(dce)
        request = nrpc.NetrServerPasswordGet()
        request['PrimaryName'] = self.serverName + '\x00'
        request['AccountName'] = self.machine_user + '\x00'
        request['AccountType'] = nrpc.NETLOGON_SECURE_CHANNEL_TYPE.WorkstationSecureChannel
        request['ComputerName'] = self.machine_user + '\x00'
        request['Authenticator'] = self.update_authenticator()
        try:
            dce.request(request)
        except DCERPCException as e:
            if str(e).find('STATUS_ACCESS_DENIED') < 0:
                raise

    @pytest.mark.xfail
    def test_hNetrServerPasswordGet(self):
        if False:
            i = 10
            return i + 15
        (dce, rpctransport) = self.connect()
        self.authenticate(dce)
        try:
            nrpc.hNetrServerPasswordGet(dce, self.serverName, self.machine_user, nrpc.NETLOGON_SECURE_CHANNEL_TYPE.WorkstationSecureChannel, self.machine_user, self.update_authenticator())
        except DCERPCException as e:
            if str(e).find('STATUS_ACCESS_DENIED') < 0:
                raise

    @pytest.mark.xfail
    def test_NetrServerTrustPasswordsGet(self):
        if False:
            print('Hello World!')
        (dce, rpctransport) = self.connect()
        self.authenticate(dce)
        request = nrpc.NetrServerTrustPasswordsGet()
        request['TrustedDcName'] = self.serverName + '\x00'
        request['AccountName'] = self.machine_user + '\x00'
        request['SecureChannelType'] = nrpc.NETLOGON_SECURE_CHANNEL_TYPE.WorkstationSecureChannel
        request['ComputerName'] = self.machine_user + '\x00'
        request['Authenticator'] = self.update_authenticator()
        resp = dce.request(request)
        resp.dump()

    @pytest.mark.xfail
    def test_hNetrServerTrustPasswordsGet(self):
        if False:
            print('Hello World!')
        (dce, rpctransport) = self.connect()
        self.authenticate(dce)
        resp = nrpc.hNetrServerTrustPasswordsGet(dce, self.serverName, self.machine_user, nrpc.NETLOGON_SECURE_CHANNEL_TYPE.WorkstationSecureChannel, self.machine_user, self.update_authenticator())
        resp.dump()

    @pytest.mark.xfail
    def test_NetrServerPasswordSet2(self):
        if False:
            print('Hello World!')
        (dce, rpctransport) = self.connect()
        self.authenticate(dce)
        request = nrpc.NetrServerPasswordSet2()
        request['PrimaryName'] = self.serverName + '\x00'
        request['AccountName'] = self.machine_user + '\x00'
        request['SecureChannelType'] = nrpc.NETLOGON_SECURE_CHANNEL_TYPE.WorkstationSecureChannel
        request['ComputerName'] = self.machine_user + '\x00'
        request['Authenticator'] = self.update_authenticator()
        cnp = nrpc.NL_TRUST_PASSWORD()
        cnp['Buffer'] = b'\x00' * 512
        cnp['Length'] = 8
        request['ClearNewPassword'] = cnp.getData()
        request.dump()
        try:
            dce.request(request)
        except DCERPCException as e:
            if str(e).find('STATUS_ACCESS_DENIED') < 0:
                raise

    @pytest.mark.xfail
    def test_hNetrServerPasswordSet2(self):
        if False:
            while True:
                i = 10
        (dce, rpctransport) = self.connect()
        self.authenticate(dce)
        cnp = nrpc.NL_TRUST_PASSWORD()
        cnp['Buffer'] = b'\x00' * 512
        cnp['Length'] = 8
        try:
            nrpc.hNetrServerPasswordSet2(dce, self.serverName, self.machine_user, nrpc.NETLOGON_SECURE_CHANNEL_TYPE.WorkstationSecureChannel, self.machine_user, self.update_authenticator(), cnp.getData())
        except DCERPCException as e:
            if str(e).find('STATUS_ACCESS_DENIED') < 0:
                raise

    @pytest.mark.xfail
    def test_NetrLogonGetDomainInfo(self):
        if False:
            print('Hello World!')
        (dce, rpctransport) = self.connect()
        self.authenticate(dce)
        request = nrpc.NetrLogonGetDomainInfo()
        request['ServerName'] = self.serverName + '\x00'
        request['ComputerName'] = self.machine_user + '\x00'
        request['Authenticator'] = self.update_authenticator()
        request['ReturnAuthenticator']['Credential'] = b'\x00' * 8
        request['ReturnAuthenticator']['Timestamp'] = 0
        request['Level'] = 1
        request['WkstaBuffer']['tag'] = 1
        request['WkstaBuffer']['WorkstationInfo']['DnsHostName'] = NULL
        request['WkstaBuffer']['WorkstationInfo']['SiteName'] = NULL
        request['WkstaBuffer']['WorkstationInfo']['OsName'] = ''
        request['WkstaBuffer']['WorkstationInfo']['Dummy1'] = NULL
        request['WkstaBuffer']['WorkstationInfo']['Dummy2'] = NULL
        request['WkstaBuffer']['WorkstationInfo']['Dummy3'] = NULL
        request['WkstaBuffer']['WorkstationInfo']['Dummy4'] = NULL
        resp = dce.request(request)
        resp.dump()

    @pytest.mark.xfail
    def test_hNetrLogonGetDomainInfo(self):
        if False:
            i = 10
            return i + 15
        (dce, rpctransport) = self.connect()
        self.authenticate(dce)
        resp = nrpc.hNetrLogonGetDomainInfo(dce, self.serverName, self.machine_user, self.update_authenticator(), 0, 1)
        resp.dump()

    @pytest.mark.xfail
    def test_NetrLogonGetCapabilities(self):
        if False:
            for i in range(10):
                print('nop')
        (dce, rpctransport) = self.connect()
        self.authenticate(dce)
        request = nrpc.NetrLogonGetCapabilities()
        request['ServerName'] = self.serverName + '\x00'
        request['ComputerName'] = self.machine_user + '\x00'
        request['Authenticator'] = self.update_authenticator()
        request['ReturnAuthenticator']['Credential'] = b'\x00' * 8
        request['ReturnAuthenticator']['Timestamp'] = 0
        request['QueryLevel'] = 1
        resp = dce.request(request)
        resp.dump()

    @pytest.mark.xfail
    def test_hNetrLogonGetCapabilities(self):
        if False:
            i = 10
            return i + 15
        (dce, rpctransport) = self.connect()
        self.authenticate(dce)
        resp = nrpc.hNetrLogonGetCapabilities(dce, self.serverName, self.machine_user, self.update_authenticator(), 0)
        resp.dump()

    def test_NetrLogonSamLogonEx(self):
        if False:
            print('Hello World!')
        (dce, rpctransport) = self.connect()
        self.authenticate(dce)
        request = nrpc.NetrLogonSamLogonEx()
        request['LogonServer'] = self.serverName + '\x00'
        request['ComputerName'] = self.machine_user + '\x00'
        request['LogonLevel'] = nrpc.NETLOGON_LOGON_INFO_CLASS.NetlogonInteractiveInformation
        request['LogonInformation']['tag'] = nrpc.NETLOGON_LOGON_INFO_CLASS.NetlogonInteractiveInformation
        request['LogonInformation']['LogonInteractive']['Identity']['LogonDomainName'] = self.domain
        request['LogonInformation']['LogonInteractive']['Identity']['ParameterControl'] = 2 + 2 ** 14 + 2 ** 7 + 2 ** 9 + 2 ** 5 + 2 ** 11
        request['LogonInformation']['LogonInteractive']['Identity']['UserName'] = self.username
        request['LogonInformation']['LogonInteractive']['Identity']['Workstation'] = ''
        if len(self.hashes):
            blmhash = self.blmhash
            bnthash = self.bnthash
        else:
            blmhash = ntlm.LMOWFv1(self.password)
            bnthash = ntlm.NTOWFv1(self.password)
        try:
            from Cryptodome.Cipher import ARC4
        except Exception:
            print("Warning: You don't have any crypto installed. You need pycryptodomex")
            print('See https://pypi.org/project/pycryptodomex/')
        rc4 = ARC4.new(self.sessionKey)
        blmhash = rc4.encrypt(blmhash)
        rc4 = ARC4.new(self.sessionKey)
        bnthash = rc4.encrypt(bnthash)
        request['LogonInformation']['LogonInteractive']['LmOwfPassword'] = blmhash
        request['LogonInformation']['LogonInteractive']['NtOwfPassword'] = bnthash
        request['ValidationLevel'] = nrpc.NETLOGON_VALIDATION_INFO_CLASS.NetlogonValidationSamInfo4
        request['ExtraFlags'] = 1
        try:
            dce.request(request)
        except DCERPCException as e:
            if str(e).find('STATUS_INTERNAL_ERROR') < 0:
                raise

    @pytest.mark.xfail
    def test_NetrLogonSamLogonWithFlags(self):
        if False:
            for i in range(10):
                print('nop')
        (dce, rpctransport) = self.connect()
        self.authenticate(dce)
        request = nrpc.NetrLogonSamLogonWithFlags()
        request['LogonServer'] = self.serverName + '\x00'
        request['ComputerName'] = self.machine_user + '\x00'
        request['LogonLevel'] = nrpc.NETLOGON_LOGON_INFO_CLASS.NetlogonInteractiveInformation
        request['LogonInformation']['tag'] = nrpc.NETLOGON_LOGON_INFO_CLASS.NetlogonInteractiveInformation
        request['LogonInformation']['LogonInteractive']['Identity']['LogonDomainName'] = self.domain
        request['LogonInformation']['LogonInteractive']['Identity']['ParameterControl'] = 2 + 2 ** 14 + 2 ** 7 + 2 ** 9 + 2 ** 5 + 2 ** 11
        request['LogonInformation']['LogonInteractive']['Identity']['UserName'] = self.username
        request['LogonInformation']['LogonInteractive']['Identity']['Workstation'] = ''
        if len(self.hashes):
            blmhash = self.blmhash
            bnthash = self.bnthash
        else:
            blmhash = ntlm.LMOWFv1(self.password)
            bnthash = ntlm.NTOWFv1(self.password)
        try:
            from Cryptodome.Cipher import ARC4
        except Exception:
            print("Warning: You don't have any crypto installed. You need pycryptodomex")
            print('See https://pypi.org/project/pycryptodomex/')
        rc4 = ARC4.new(self.sessionKey)
        blmhash = rc4.encrypt(blmhash)
        rc4 = ARC4.new(self.sessionKey)
        bnthash = rc4.encrypt(bnthash)
        request['LogonInformation']['LogonInteractive']['LmOwfPassword'] = blmhash
        request['LogonInformation']['LogonInteractive']['NtOwfPassword'] = bnthash
        request['ValidationLevel'] = nrpc.NETLOGON_VALIDATION_INFO_CLASS.NetlogonValidationSamInfo4
        request['Authenticator'] = self.update_authenticator()
        request['ReturnAuthenticator']['Credential'] = b'\x00' * 8
        request['ReturnAuthenticator']['Timestamp'] = 0
        request['ExtraFlags'] = 0
        try:
            dce.request(request)
        except DCERPCException as e:
            if str(e).find('STATUS_NO_SUCH_USER') < 0:
                raise

    @pytest.mark.xfail
    def test_NetrLogonSamLogon(self):
        if False:
            while True:
                i = 10
        (dce, rpctransport) = self.connect()
        self.authenticate(dce)
        request = nrpc.NetrLogonSamLogon()
        request['LogonServer'] = self.serverName + '\x00'
        request['ComputerName'] = self.machine_user + '\x00'
        request['LogonLevel'] = nrpc.NETLOGON_LOGON_INFO_CLASS.NetlogonInteractiveInformation
        request['LogonInformation']['tag'] = nrpc.NETLOGON_LOGON_INFO_CLASS.NetlogonInteractiveInformation
        request['LogonInformation']['LogonInteractive']['Identity']['LogonDomainName'] = self.domain
        request['LogonInformation']['LogonInteractive']['Identity']['ParameterControl'] = 2
        request['LogonInformation']['LogonInteractive']['Identity']['UserName'] = self.username
        request['LogonInformation']['LogonInteractive']['Identity']['Workstation'] = ''
        if len(self.hashes):
            blmhash = self.blmhash
            bnthash = self.bnthash
        else:
            blmhash = ntlm.LMOWFv1(self.password)
            bnthash = ntlm.NTOWFv1(self.password)
        try:
            from Cryptodome.Cipher import ARC4
        except Exception:
            print("Warning: You don't have any crypto installed. You need PyCrypto")
            print('See http://www.pycrypto.org/')
        rc4 = ARC4.new(self.sessionKey)
        blmhash = rc4.encrypt(blmhash)
        rc4 = ARC4.new(self.sessionKey)
        bnthash = rc4.encrypt(bnthash)
        request['LogonInformation']['LogonInteractive']['LmOwfPassword'] = blmhash
        request['LogonInformation']['LogonInteractive']['NtOwfPassword'] = bnthash
        request['ValidationLevel'] = nrpc.NETLOGON_VALIDATION_INFO_CLASS.NetlogonValidationSamInfo2
        request['Authenticator'] = self.update_authenticator()
        request['ReturnAuthenticator']['Credential'] = b'\x00' * 8
        request['ReturnAuthenticator']['Timestamp'] = 0
        try:
            dce.request(request)
        except DCERPCException as e:
            if str(e).find('STATUS_NO_SUCH_USER') < 0:
                raise

    @pytest.mark.xfail
    def test_NetrDatabaseDeltas(self):
        if False:
            print('Hello World!')
        (dce, rpctransport) = self.connect()
        self.authenticate(dce)
        request = nrpc.NetrDatabaseDeltas()
        request['PrimaryName'] = self.serverName + '\x00'
        request['ComputerName'] = self.machine_user + '\x00'
        request['Authenticator'] = self.update_authenticator()
        request['ReturnAuthenticator']['Credential'] = b'\x00' * 8
        request['ReturnAuthenticator']['Timestamp'] = 0
        request['DatabaseID'] = 0
        request['PreferredMaximumLength'] = 4294967295
        try:
            dce.request(request)
        except DCERPCException as e:
            if str(e).find('STATUS_NOT_SUPPORTED') < 0:
                raise

    @pytest.mark.xfail
    def test_NetrDatabaseSync2(self):
        if False:
            i = 10
            return i + 15
        (dce, rpctransport) = self.connect()
        self.authenticate(dce)
        request = nrpc.NetrDatabaseSync2()
        request['PrimaryName'] = self.serverName + '\x00'
        request['ComputerName'] = self.machine_user + '\x00'
        request['Authenticator'] = self.update_authenticator()
        request['ReturnAuthenticator']['Credential'] = b'\x00' * 8
        request['ReturnAuthenticator']['Timestamp'] = 0
        request['DatabaseID'] = 0
        request['RestartState'] = nrpc.SYNC_STATE.NormalState
        request['SyncContext'] = 0
        request['PreferredMaximumLength'] = 4294967295
        try:
            dce.request(request)
        except DCERPCException as e:
            if str(e).find('STATUS_NOT_SUPPORTED') < 0:
                raise

    @pytest.mark.xfail
    def test_NetrDatabaseSync(self):
        if False:
            while True:
                i = 10
        (dce, rpctransport) = self.connect()
        self.authenticate(dce)
        request = nrpc.NetrDatabaseSync()
        request['PrimaryName'] = self.serverName + '\x00'
        request['ComputerName'] = self.machine_user + '\x00'
        request['Authenticator'] = self.update_authenticator()
        request['ReturnAuthenticator']['Credential'] = b'\x00' * 8
        request['ReturnAuthenticator']['Timestamp'] = 0
        request['DatabaseID'] = 0
        request['SyncContext'] = 0
        request['PreferredMaximumLength'] = 4294967295
        try:
            dce.request(request)
        except DCERPCException as e:
            if str(e).find('STATUS_NOT_SUPPORTED') < 0:
                raise

    @pytest.mark.xfail
    def test_NetrDatabaseRedo(self):
        if False:
            while True:
                i = 10
        (dce, rpctransport) = self.connect()
        self.authenticate(dce)
        request = nrpc.NetrDatabaseRedo()
        request['PrimaryName'] = self.serverName + '\x00'
        request['ComputerName'] = self.machine_user + '\x00'
        request['Authenticator'] = self.update_authenticator()
        request['ReturnAuthenticator']['Credential'] = b'\x00' * 8
        request['ReturnAuthenticator']['Timestamp'] = 0
        request['ChangeLogEntry'] = NULL
        request['ChangeLogEntrySize'] = 0
        try:
            dce.request(request)
        except DCERPCException as e:
            if str(e).find('STATUS_NOT_SUPPORTED') < 0:
                raise

    def test_DsrEnumerateDomainTrusts(self):
        if False:
            return 10
        (dce, rpctransport) = self.connect()
        request = nrpc.DsrEnumerateDomainTrusts()
        request['ServerName'] = self.serverName + '\x00'
        request['Flags'] = 1
        try:
            dce.request(request)
        except DCERPCException as e:
            if str(e).find('STATUS_NOT_SUPPORTED') < 0:
                raise

    def test_NetrEnumerateTrustedDomainsEx(self):
        if False:
            i = 10
            return i + 15
        (dce, rpctransport) = self.connect()
        request = nrpc.NetrEnumerateTrustedDomainsEx()
        request['ServerName'] = NULL
        try:
            dce.request(request)
        except DCERPCException as e:
            if str(e).find('STATUS_NOT_SUPPORTED') < 0:
                raise

    def test_NetrEnumerateTrustedDomains(self):
        if False:
            for i in range(10):
                print('nop')
        (dce, rpctransport) = self.connect()
        request = nrpc.NetrEnumerateTrustedDomains()
        request['ServerName'] = NULL
        try:
            dce.request(request)
        except DCERPCException as e:
            if str(e).find('STATUS_NOT_SUPPORTED') < 0:
                raise

    @pytest.mark.xfail
    def test_NetrGetForestTrustInformation(self):
        if False:
            i = 10
            return i + 15
        (dce, rpctransport) = self.connect()
        self.authenticate(dce)
        request = nrpc.NetrGetForestTrustInformation()
        request['ServerName'] = self.serverName + '\x00'
        request['ComputerName'] = self.machine_user + '\x00'
        request['Authenticator'] = self.update_authenticator()
        request['ReturnAuthenticator']['Credential'] = b'\x00' * 8
        request['ReturnAuthenticator']['Timestamp'] = 0
        request['Flags'] = 0
        try:
            dce.request(request)
        except DCERPCException as e:
            if str(e).find('STATUS_NOT_IMPLEMENTED') < 0 and str(e).find('STATUS_ACCESS_DENIED') < 0:
                raise

    def test_DsrGetForestTrustInformation(self):
        if False:
            for i in range(10):
                print('nop')
        (dce, rpctransport) = self.connect()
        request = nrpc.DsrGetForestTrustInformation()
        request['ServerName'] = self.serverName + '\x00'
        request['TrustedDomainName'] = self.domain + '\x00'
        request['Flags'] = 0
        try:
            dce.request(request)
        except DCERPCException as e:
            if str(e).find('ERROR_NO_SUCH_DOMAIN') < 0 and str(e).find('rpc_s_access_denied') < 0:
                raise

    @pytest.mark.xfail
    def test_NetrServerGetTrustInfo(self):
        if False:
            return 10
        (dce, rpctransport) = self.connect()
        self.authenticate(dce)
        request = nrpc.NetrServerGetTrustInfo()
        request['TrustedDcName'] = self.serverName + '\x00'
        request['AccountName'] = self.machine_user + '\x00'
        request['SecureChannelType'] = nrpc.NETLOGON_SECURE_CHANNEL_TYPE.WorkstationSecureChannel
        request['ComputerName'] = self.machine_user + '\x00'
        request['Authenticator'] = self.update_authenticator()
        try:
            dce.request(request)
        except DCERPCException as e:
            if str(e).find('ERROR_NO_SUCH_DOMAIN') < 0:
                raise

    @pytest.mark.xfail
    def test_hNetrServerGetTrustInfo(self):
        if False:
            while True:
                i = 10
        (dce, rpctransport) = self.connect()
        self.authenticate(dce)
        try:
            nrpc.hNetrServerGetTrustInfo(dce, self.serverName, self.machine_user, nrpc.NETLOGON_SECURE_CHANNEL_TYPE.WorkstationSecureChannel, self.machine_user, self.update_authenticator())
        except DCERPCException as e:
            if str(e).find('ERROR_NO_SUCH_DOMAIN') < 0:
                raise

    def test_NetrLogonGetTrustRid(self):
        if False:
            for i in range(10):
                print('nop')
        (dce, rpctransport) = self.connect()
        request = nrpc.NetrLogonGetTrustRid()
        request['ServerName'] = NULL
        request['DomainName'] = self.domain + '\x00'
        try:
            dce.request(request)
        except DCERPCException as e:
            if str(e).find('rpc_s_access_denied') < 0:
                raise

    def test_NetrLogonComputeServerDigest(self):
        if False:
            return 10
        (dce, rpctransport) = self.connect()
        request = nrpc.NetrLogonComputeServerDigest()
        request['ServerName'] = NULL
        request['Rid'] = 1001
        request['Message'] = b'HOLABETOCOMOANDAS\x00'
        request['MessageSize'] = len(b'HOLABETOCOMOANDAS\x00')
        try:
            dce.request(request)
        except DCERPCException as e:
            if str(e).find('rpc_s_access_denied') < 0:
                raise

    def test_NetrLogonComputeClientDigest(self):
        if False:
            print('Hello World!')
        (dce, rpctransport) = self.connect()
        request = nrpc.NetrLogonComputeClientDigest()
        request['ServerName'] = NULL
        request['DomainName'] = self.domain + '\x00'
        request['Message'] = b'HOLABETOCOMOANDAS\x00'
        request['MessageSize'] = len(request['Message'])
        try:
            dce.request(request)
        except DCERPCException as e:
            if str(e).find('rpc_s_access_denied') < 0:
                raise

    @pytest.mark.xfail
    def test_NetrLogonSendToSam(self):
        if False:
            return 10
        (dce, rpctransport) = self.connect()
        self.authenticate(dce)
        request = nrpc.NetrLogonSendToSam()
        request['PrimaryName'] = self.serverName + '\x00'
        request['ComputerName'] = self.machine_user + '\x00'
        request['Authenticator'] = self.update_authenticator()
        request['OpaqueBuffer'] = b'HOLABETOCOMOANDAS\x00'
        request['OpaqueBufferSize'] = len(b'HOLABETOCOMOANDAS\x00')
        try:
            dce.request(request)
        except DCERPCException as e:
            if str(e).find('STATUS_ACCESS_DENIED') < 0:
                raise

    def test_NetrLogonSetServiceBits(self):
        if False:
            while True:
                i = 10
        (dce, rpctransport) = self.connect()
        request = nrpc.NetrLogonSetServiceBits()
        request['ServerName'] = NULL
        request['ServiceBitsOfInterest'] = 1 << 7
        request['ServiceBits'] = 1 << 7
        try:
            dce.request(request)
        except DCERPCException as e:
            if str(e).find('rpc_s_access_denied') < 0:
                raise

    def test_NetrLogonGetTimeServiceParentDomain(self):
        if False:
            return 10
        (dce, rpctransport) = self.connect()
        request = nrpc.NetrLogonGetTimeServiceParentDomain()
        request['ServerName'] = self.serverName + '\x00'
        try:
            dce.request(request)
        except DCERPCException as e:
            if str(e).find('rpc_s_access_denied') < 0 and str(e).find('ERROR_NO_SUCH_DOMAIN') < 0:
                raise

    def test_NetrLogonControl2Ex(self):
        if False:
            return 10
        (dce, rpctransport) = self.connect()
        request = nrpc.NetrLogonControl2Ex()
        request['ServerName'] = NULL
        request['FunctionCode'] = nrpc.NETLOGON_CONTROL_FIND_USER
        request['QueryLevel'] = 4
        request['Data']['tag'] = 8
        request['Data']['UserName'] = self.username + '\x00'
        try:
            dce.request(request)
        except DCERPCException as e:
            if str(e).find('rpc_s_access_denied') < 0 and str(e).find('0x8ad') < 0:
                raise

    def test_NetrLogonControl2(self):
        if False:
            for i in range(10):
                print('nop')
        (dce, rpctransport) = self.connect()
        request = nrpc.NetrLogonControl2()
        request['ServerName'] = NULL
        request['FunctionCode'] = nrpc.NETLOGON_CONTROL_FIND_USER
        request['QueryLevel'] = 4
        request['Data']['tag'] = 8
        request['Data']['UserName'] = self.username + '\x00'
        try:
            dce.request(request)
        except DCERPCException as e:
            if str(e).find('rpc_s_access_denied') < 0 and str(e).find('0x8ad') < 0:
                raise

    def test_NetrLogonControl(self):
        if False:
            while True:
                i = 10
        (dce, rpctransport) = self.connect()
        request = nrpc.NetrLogonControl()
        request['ServerName'] = NULL
        request['FunctionCode'] = nrpc.NETLOGON_CONTROL_QUERY
        request['QueryLevel'] = 4
        request['Data']['tag'] = 65534
        request['Data']['DebugFlag'] = 1
        try:
            dce.request(request)
        except DCERPCException as e:
            if str(e).find('ERROR_INVALID_LEVEL') < 0:
                raise

    def test_NetrLogonUasLogon(self):
        if False:
            i = 10
            return i + 15
        (dce, rpctransport) = self.connect()
        request = nrpc.NetrLogonUasLogon()
        request['ServerName'] = NULL
        request['UserName'] = self.username + '\x00'
        request['Workstation'] = self.serverName + '\x00'
        try:
            dce.request(request)
        except DCERPCException as e:
            if str(e).find('rpc_s_access_denied') < 0:
                raise

    def test_NetrLogonUasLogoff(self):
        if False:
            return 10
        (dce, rpctransport) = self.connect()
        request = nrpc.NetrLogonUasLogoff()
        request['ServerName'] = NULL
        request['UserName'] = self.username + '\x00'
        request['Workstation'] = self.serverName + '\x00'
        try:
            dce.request(request)
        except DCERPCException as e:
            if str(e).find('rpc_s_access_denied') < 0:
                raise

@pytest.mark.remote
class NRPCTestsSMBTransport(NRPCTests, unittest.TestCase):
    string_binding = 'ncacn_np:{0.machine}[\\PIPE\\netlogon]'
    string_binding_formatting = DCERPCTests.STRING_BINDING_FORMATTING

@pytest.mark.remote
class NRPCTestsTCPTransport(NRPCTests, unittest.TestCase):
    protocol = 'ncacn_ip_tcp'
    string_binding_formatting = DCERPCTests.STRING_BINDING_MAPPER
if __name__ == '__main__':
    unittest.main(verbosity=1)