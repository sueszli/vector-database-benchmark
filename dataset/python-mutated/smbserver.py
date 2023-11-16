"""
SMB 1 / 2 Server Automaton
"""
import time
from scapy.automaton import ATMT, Automaton
from scapy.layers.ntlm import NTLM_CHALLENGE, NTLM_Server
from scapy.volatile import RandUUID
from scapy.layers.netbios import NBTSession
from scapy.layers.gssapi import GSSAPI_BLOB, SPNEGO_MechListMIC, SPNEGO_MechType, SPNEGO_Token, SPNEGO_negToken, SPNEGO_negTokenInit, SPNEGO_negTokenResp
from scapy.layers.smb import SMB_Header, SMBNegotiate_Request, SMBNegotiate_Response_Security, SMBNegotiate_Response_Extended_Security, SMBSession_Null, SMBSession_Setup_AndX_Request, SMBSession_Setup_AndX_Request_Extended_Security, SMBSession_Setup_AndX_Response, SMBSession_Setup_AndX_Response_Extended_Security, SMBTree_Connect_AndX
from scapy.layers.smb2 import SMB2_Header, SMB2_IOCTL_Response, SMB2_IOCTL_Validate_Negotiate_Info_Response, SMB2_Negotiate_Protocol_Request, SMB2_Negotiate_Protocol_Response, SMB2_Session_Setup_Request, SMB2_Session_Setup_Response, SMB2_IOCTL_Request, SMB2_Error_Response

class NTLM_SMB_Server(NTLM_Server, Automaton):
    port = 445
    cls = NBTSession

    def __init__(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        self.CLIENT_PROVIDES_NEGOEX = kwargs.pop('CLIENT_PROVIDES_NEGOEX', False)
        self.ECHO = kwargs.pop('ECHO', False)
        self.ANONYMOUS_LOGIN = kwargs.pop('ANONYMOUS_LOGIN', False)
        self.GUEST_LOGIN = kwargs.pop('GUEST_LOGIN', False)
        self.PASS_NEGOEX = kwargs.pop('PASS_NEGOEX', False)
        self.EXTENDED_SECURITY = kwargs.pop('EXTENDED_SECURITY', True)
        self.ALLOW_SMB2 = kwargs.pop('ALLOW_SMB2', True)
        self.REQUIRE_SIGNATURE = kwargs.pop('REQUIRE_SIGNATURE', False)
        self.REAL_HOSTNAME = kwargs.pop('REAL_HOSTNAME', None)
        assert self.ALLOW_SMB2 or self.REAL_HOSTNAME, 'SMB1 requires REAL_HOSTNAME !'
        self.SMB2 = False
        self.Dialect = None
        self.GUID = False
        super(NTLM_SMB_Server, self).__init__(*args, **kwargs)

    def send(self, pkt):
        if False:
            return 10
        if self.Dialect and self.SigningSessionKey:
            if isinstance(pkt.payload, SMB2_Header):
                smb = pkt[SMB2_Header]
                smb.Flags += 'SMB2_FLAGS_SIGNED'
                smb.sign(self.Dialect, self.SigningSessionKey)
        return super(NTLM_SMB_Server, self).send(pkt)

    @ATMT.state(initial=1)
    def BEGIN(self):
        if False:
            while True:
                i = 10
        self.authenticated = False
        assert not self.ECHO or self.cli_atmt, 'Cannot use ECHO without binding to a client !'

    @ATMT.receive_condition(BEGIN)
    def received_negotiate(self, pkt):
        if False:
            print('Hello World!')
        if SMBNegotiate_Request in pkt:
            if self.cli_atmt:
                self.start_client()
            raise self.NEGOTIATED().action_parameters(pkt)

    @ATMT.receive_condition(BEGIN)
    def received_negotiate_smb2_begin(self, pkt):
        if False:
            for i in range(10):
                print('nop')
        if SMB2_Negotiate_Protocol_Request in pkt:
            self.SMB2 = True
            if self.cli_atmt:
                self.start_client(CONTINUE_SMB2=True, SMB2_INIT_PARAMS={'ClientGUID': pkt.ClientGUID})
            raise self.NEGOTIATED().action_parameters(pkt)

    @ATMT.action(received_negotiate_smb2_begin)
    def on_negotiate_smb2_begin(self, pkt):
        if False:
            print('Hello World!')
        self.on_negotiate(pkt)

    @ATMT.action(received_negotiate)
    def on_negotiate(self, pkt):
        if False:
            return 10
        if self.CLIENT_PROVIDES_NEGOEX:
            (negoex_token, _, _, _) = self.get_token(negoex=True)
        else:
            negoex_token = None
        if not self.SMB2 and (not self.get('GUID', 0)):
            self.EXTENDED_SECURITY = False
        DialectIndex = None
        DialectRevision = None
        if SMB2_Negotiate_Protocol_Request in pkt:
            DialectRevisions = pkt[SMB2_Negotiate_Protocol_Request].Dialects
            DialectRevisions.sort()
            DialectRevision = DialectRevisions[0]
            if DialectRevision >= 768:
                raise ValueError('SMB client requires SMB3 which is unimplemented.')
        else:
            DialectIndexes = [x.DialectString for x in pkt[SMBNegotiate_Request].Dialects]
            if self.ALLOW_SMB2:
                for (key, rev) in [(b'SMB 2.???', 767), (b'SMB 2.002', 514)]:
                    try:
                        DialectIndex = DialectIndexes.index(key)
                        DialectRevision = rev
                        self.SMB2 = True
                        break
                    except ValueError:
                        pass
                else:
                    DialectIndex = DialectIndexes.index(b'NT LM 0.12')
            else:
                DialectIndex = DialectIndexes.index(b'NT LM 0.12')
        if DialectRevision and DialectRevision & 255 != 255:
            self.Dialect = DialectRevision
        cls = None
        if self.SMB2:
            cls = SMB2_Negotiate_Protocol_Response
            self.smb_header = NBTSession() / SMB2_Header(CreditsRequested=1, CreditCharge=1)
            if SMB2_Negotiate_Protocol_Request in pkt:
                self.smb_header.MID = pkt.MID
                self.smb_header.TID = pkt.TID
                self.smb_header.AsyncId = pkt.AsyncId
                self.smb_header.SessionId = pkt.SessionId
        else:
            self.smb_header = NBTSession() / SMB_Header(Flags='REPLY+CASE_INSENSITIVE+CANONICALIZED_PATHS', Flags2='LONG_NAMES+EAS+NT_STATUS+SMB_SECURITY_SIGNATURE+UNICODE+EXTENDED_SECURITY', TID=pkt.TID, MID=pkt.MID, UID=pkt.UID, PIDLow=pkt.PIDLow)
            if self.EXTENDED_SECURITY:
                cls = SMBNegotiate_Response_Extended_Security
            else:
                cls = SMBNegotiate_Response_Security
        if self.SMB2:
            resp = self.smb_header.copy() / cls(DialectRevision=DialectRevision, SecurityMode=3 if self.REQUIRE_SIGNATURE else self.get('SecurityMode', bool(self.IDENTITIES)), ServerTime=self.get('ServerTime', time.time() + 11644473600), ServerStartTime=0, MaxTransactionSize=65536, MaxReadSize=65536, MaxWriteSize=65536)
        else:
            resp = self.smb_header.copy() / cls(DialectIndex=DialectIndex, ServerCapabilities='UNICODE+LARGE_FILES+NT_SMBS+RPC_REMOTE_APIS+STATUS32+LEVEL_II_OPLOCKS+LOCK_AND_READ+NT_FIND+LWIO+INFOLEVEL_PASSTHRU+LARGE_READX+LARGE_WRITEX', SecurityMode=3 if self.REQUIRE_SIGNATURE else self.get('SecurityMode', bool(self.IDENTITIES)), ServerTime=self.get('ServerTime'), ServerTimeZone=self.get('ServerTimeZone'))
            if self.EXTENDED_SECURITY:
                resp.ServerCapabilities += 'EXTENDED_SECURITY'
        if self.EXTENDED_SECURITY or self.SMB2:
            resp.SecurityBlob = GSSAPI_BLOB(innerContextToken=SPNEGO_negToken(token=SPNEGO_negTokenInit(mechTypes=[SPNEGO_MechType(oid='1.3.6.1.4.1.311.2.2.10')])))
            self.GUID = resp.GUID = self.get('GUID', RandUUID()._fix())
            if self.PASS_NEGOEX:
                resp.SecurityBlob.innerContextToken.token.mechTypes.insert(0, SPNEGO_MechType(oid='1.3.6.1.4.1.311.2.2.30'))
                resp.SecurityBlob.innerContextToken.token.mechToken = SPNEGO_Token(value=negoex_token)
        else:
            resp.Challenge = self.get('Challenge')
            resp.DomainName = self.get('DomainName')
            resp.ServerName = self.get('ServerName')
            resp.Flags2 -= 'EXTENDED_SECURITY'
        if not self.SMB2:
            resp[SMB_Header].Flags2 = resp[SMB_Header].Flags2 - 'SMB_SECURITY_SIGNATURE' + 'SMB_SECURITY_SIGNATURE_REQUIRED+IS_LONG_NAME'
        self.send(resp)

    @ATMT.state()
    def NEGOTIATED(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    def update_smbheader(self, pkt):
        if False:
            print('Hello World!')
        self.smb_header.TID = pkt.TID
        self.smb_header.MID = pkt.MID
        self.smb_header.PID = pkt.PID

    @ATMT.receive_condition(NEGOTIATED)
    def received_negotiate_smb2(self, pkt):
        if False:
            for i in range(10):
                print('nop')
        if SMB2_Negotiate_Protocol_Request in pkt:
            raise self.NEGOTIATED().action_parameters(pkt)

    @ATMT.action(received_negotiate_smb2)
    def on_negotiate_smb2(self, pkt):
        if False:
            for i in range(10):
                print('nop')
        self.on_negotiate(pkt)

    @ATMT.receive_condition(NEGOTIATED)
    def receive_setup_andx_request(self, pkt):
        if False:
            return 10
        if SMBSession_Setup_AndX_Request_Extended_Security in pkt or SMBSession_Setup_AndX_Request in pkt:
            if SMBSession_Setup_AndX_Request_Extended_Security in pkt:
                ntlm_tuple = self._get_token(pkt.SecurityBlob)
            else:
                self.set_cli('AccountName', pkt.AccountName)
                self.set_cli('PrimaryDomain', pkt.PrimaryDomain)
                self.set_cli('Path', pkt.Path)
                self.set_cli('Service', pkt.Service)
                ntlm_tuple = self._get_token(pkt[SMBSession_Setup_AndX_Request].UnicodePassword)
            self.set_cli('VCNumber', pkt.VCNumber)
            self.set_cli('SecuritySignature', pkt.SecuritySignature)
            self.set_cli('UID', pkt.UID)
            self.set_cli('MID', pkt.MID)
            self.set_cli('TID', pkt.TID)
            self.received_ntlm_token(ntlm_tuple)
            raise self.RECEIVED_SETUP_ANDX_REQUEST().action_parameters(pkt)
        elif SMB2_Session_Setup_Request in pkt:
            ntlm_tuple = self._get_token(pkt.SecurityBlob)
            self.set_cli('SecuritySignature', pkt.SecuritySignature)
            self.set_cli('MID', pkt.MID)
            self.set_cli('TID', pkt.TID)
            self.set_cli('AsyncId', pkt.AsyncId)
            self.set_cli('SessionId', pkt.SessionId)
            self.set_cli('SecurityMode', pkt.SecurityMode)
            self.received_ntlm_token(ntlm_tuple)
            raise self.RECEIVED_SETUP_ANDX_REQUEST().action_parameters(pkt)

    @ATMT.state()
    def RECEIVED_SETUP_ANDX_REQUEST(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    @ATMT.action(receive_setup_andx_request)
    def on_setup_andx_request(self, pkt):
        if False:
            i = 10
            return i + 15
        (ntlm_token, negResult, MIC, rawToken) = ntlm_tuple = self.get_token()
        if SMBSession_Setup_AndX_Request_Extended_Security in pkt or SMBSession_Setup_AndX_Request in pkt or SMB2_Session_Setup_Request in pkt:
            if SMB2_Session_Setup_Request in pkt:
                self.smb_header.MID = self.get('MID', self.smb_header.MID + 1)
                self.smb_header.TID = self.get('TID', self.smb_header.TID)
                if self.smb_header.Flags.SMB2_FLAGS_ASYNC_COMMAND:
                    self.smb_header.AsyncId = self.get('AsyncId', self.smb_header.AsyncId)
                self.smb_header.SessionId = self.get('SessionId', 281474976710677)
            else:
                self.smb_header.UID = self.get('UID')
                self.smb_header.MID = self.get('MID')
                self.smb_header.TID = self.get('TID')
            if ntlm_tuple == (None, None, None, None):
                if SMB2_Session_Setup_Request in pkt:
                    resp = self.smb_header.copy() / SMB2_Session_Setup_Response()
                else:
                    resp = self.smb_header.copy() / SMBSession_Null()
                resp.Status = self.get('Status', 3221225581)
            else:
                if SMBSession_Setup_AndX_Request_Extended_Security in pkt or SMB2_Session_Setup_Request in pkt:
                    if SMB2_Session_Setup_Request in pkt:
                        resp = self.smb_header.copy() / SMB2_Session_Setup_Response()
                        if self.GUEST_LOGIN:
                            resp.SessionFlags = 'IS_GUEST'
                        if self.ANONYMOUS_LOGIN:
                            resp.SessionFlags = 'IS_NULL'
                    else:
                        resp = self.smb_header.copy() / SMBSession_Setup_AndX_Response_Extended_Security(NativeOS=self.get('NativeOS'), NativeLanMan=self.get('NativeLanMan'))
                        if self.GUEST_LOGIN:
                            resp.Action = 'SMB_SETUP_GUEST'
                    if not ntlm_token:
                        resp.SecurityBlob = SPNEGO_negToken(token=SPNEGO_negTokenResp(negResult=negResult))
                        if MIC and (not self.DROP_MIC):
                            resp.SecurityBlob.token.mechListMIC = SPNEGO_MechListMIC(value=MIC)
                        if negResult == 0:
                            self.authenticated = True
                    elif isinstance(ntlm_token, NTLM_CHALLENGE) and (not rawToken):
                        resp.SecurityBlob = SPNEGO_negToken(token=SPNEGO_negTokenResp(negResult=negResult or 1, supportedMech=SPNEGO_MechType(oid='1.3.6.1.4.1.311.2.2.10'), responseToken=SPNEGO_Token(value=ntlm_token)))
                    else:
                        resp.SecurityBlob = ntlm_token
                elif SMBSession_Setup_AndX_Request in pkt:
                    resp = self.smb_header.copy() / SMBSession_Setup_AndX_Response(NativeOS=self.get('NativeOS'), NativeLanMan=self.get('NativeLanMan'))
                resp.Status = self.get('Status', 0 if self.authenticated else 3221225494)
        self.send(resp)

    @ATMT.condition(RECEIVED_SETUP_ANDX_REQUEST)
    def wait_for_next_request(self):
        if False:
            return 10
        if self.authenticated:
            raise self.AUTHENTICATED()
        else:
            raise self.NEGOTIATED()

    @ATMT.state()
    def AUTHENTICATED(self):
        if False:
            print('Hello World!')
        'Dev: overload this'
        pass

    @ATMT.condition(AUTHENTICATED, prio=1)
    def should_end(self):
        if False:
            for i in range(10):
                print('nop')
        if not self.ECHO:
            raise self.END()

    @ATMT.receive_condition(AUTHENTICATED, prio=2)
    def receive_packet_echo(self, pkt):
        if False:
            while True:
                i = 10
        if self.ECHO:
            raise self.AUTHENTICATED().action_parameters(pkt)

    def _ioctl_error(self, Status='STATUS_NOT_SUPPORTED'):
        if False:
            i = 10
            return i + 15
        pkt = self.smb_header.copy() / SMB2_Error_Response(ErrorData=b'\xff')
        pkt.Status = Status
        pkt.Command = 'SMB2_IOCTL'
        self.send(pkt)

    @ATMT.action(receive_packet_echo)
    def pass_packet(self, pkt):
        if False:
            while True:
                i = 10
        pkt.show()
        if not self.SMB2:
            if SMBTree_Connect_AndX in pkt and self.REAL_HOSTNAME:
                pkt.LENGTH = None
                pkt.ByteCount = None
                pkt.Path = '\\\\%s\\' % self.REAL_HOSTNAME + pkt.Path[2:].split('\\', 1)[1]
        else:
            self.smb_header.MID += 1
            if SMB2_IOCTL_Request in pkt and pkt.CtlCode == 1311236:
                if self.SigningSessionKey:
                    pkt = self.smb_header.copy() / SMB2_IOCTL_Response(CtlCode=1311236, FileId=pkt[SMB2_IOCTL_Request].FileId, Buffer=[('Output', SMB2_IOCTL_Validate_Negotiate_Info_Response(GUID=self.GUID, DialectRevision=self.Dialect, SecurityMode=3 if self.REQUIRE_SIGNATURE else self.get('SecurityMode', bool(self.IDENTITIES))))])
                else:
                    self._ioctl_error(Status='STATUS_FILE_CLOSED')
                    return
        self.echo(pkt)

    @ATMT.state(final=1)
    def END(self):
        if False:
            print('Hello World!')
        self.end()