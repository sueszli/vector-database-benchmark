import ntpath
import socket
from impacket import smb, smb3, nmb, nt_errors, LOG
from impacket.ntlm import compute_lmhash, compute_nthash
from impacket.smb3structs import SMB2Packet, SMB2_DIALECT_002, SMB2_DIALECT_21, SMB2_DIALECT_30, GENERIC_ALL, FILE_SHARE_READ, FILE_SHARE_WRITE, FILE_SHARE_DELETE, FILE_NON_DIRECTORY_FILE, FILE_OVERWRITE_IF, FILE_ATTRIBUTE_NORMAL, SMB2_IL_IMPERSONATION, SMB2_OPLOCK_LEVEL_NONE, FILE_READ_DATA, FILE_WRITE_DATA, FILE_OPEN, GENERIC_READ, GENERIC_WRITE, FILE_OPEN_REPARSE_POINT, MOUNT_POINT_REPARSE_DATA_STRUCTURE, FSCTL_SET_REPARSE_POINT, SMB2_0_IOCTL_IS_FSCTL, MOUNT_POINT_REPARSE_GUID_DATA_STRUCTURE, FSCTL_DELETE_REPARSE_POINT, FSCTL_SRV_ENUMERATE_SNAPSHOTS, SRV_SNAPSHOT_ARRAY, FILE_SYNCHRONOUS_IO_NONALERT, FILE_READ_EA, FILE_READ_ATTRIBUTES, READ_CONTROL, SYNCHRONIZE, SMB2_DIALECT_311
SMB_DIALECT = smb.SMB_DIALECT

class SMBConnection:
    """
    SMBConnection class

    :param string remoteName: name of the remote host, can be its NETBIOS name, IP or *\\*SMBSERVER*.  If the later,
           and port is 139, the library will try to get the target's server name.
    :param string remoteHost: target server's remote address (IPv4, IPv6) or FQDN
    :param string/optional myName: client's NETBIOS name
    :param integer/optional sess_port: target port to connect
    :param integer/optional timeout: timeout in seconds when receiving packets
    :param optional preferredDialect: the dialect desired to talk with the target server. If not specified the highest
           one available will be used
    :param optional boolean manualNegotiate: the user manually performs SMB_COM_NEGOTIATE

    :return: a SMBConnection instance, if not raises a SessionError exception
    """

    def __init__(self, remoteName='', remoteHost='', myName=None, sess_port=nmb.SMB_SESSION_PORT, timeout=60, preferredDialect=None, existingConnection=None, manualNegotiate=False):
        if False:
            for i in range(10):
                print('nop')
        self._SMBConnection = 0
        self._dialect = ''
        self._nmbSession = 0
        self._sess_port = sess_port
        self._myName = myName
        self._remoteHost = remoteHost
        self._remoteName = remoteName
        self._timeout = timeout
        self._preferredDialect = preferredDialect
        self._existingConnection = existingConnection
        self._manualNegotiate = manualNegotiate
        self._doKerberos = False
        self._kdcHost = None
        self._useCache = True
        self._ntlmFallback = True
        if existingConnection is not None:
            assert isinstance(existingConnection, smb.SMB) or isinstance(existingConnection, smb3.SMB3)
            self._SMBConnection = existingConnection
            self._preferredDialect = self._SMBConnection.getDialect()
            self._doKerberos = self._SMBConnection.getKerberos()
            return
        if manualNegotiate is False:
            self.negotiateSession(preferredDialect)

    def negotiateSession(self, preferredDialect=None, flags1=smb.SMB.FLAGS1_PATHCASELESS | smb.SMB.FLAGS1_CANONICALIZED_PATHS, flags2=smb.SMB.FLAGS2_EXTENDED_SECURITY | smb.SMB.FLAGS2_NT_STATUS | smb.SMB.FLAGS2_LONG_NAMES, negoData='\x02NT LM 0.12\x00\x02SMB 2.002\x00\x02SMB 2.???\x00'):
        if False:
            i = 10
            return i + 15
        '\n        Perform protocol negotiation\n\n        :param string preferredDialect: the dialect desired to talk with the target server. If None is specified the highest one available will be used\n        :param string flags1: the SMB FLAGS capabilities\n        :param string flags2: the SMB FLAGS2 capabilities\n        :param string negoData: data to be sent as part of the nego handshake\n\n        :return: True\n        :raise SessionError: if error\n        '
        if self._sess_port == nmb.SMB_SESSION_PORT and self._remoteName == '*SMBSERVER':
            self._remoteName = self._remoteHost
        elif self._sess_port == nmb.NETBIOS_SESSION_PORT and self._remoteName == '*SMBSERVER':
            nb = nmb.NetBIOS()
            try:
                res = nb.getnetbiosname(self._remoteHost)
            except:
                pass
            else:
                self._remoteName = res
        if self._sess_port == nmb.NETBIOS_SESSION_PORT:
            negoData = '\x02NT LM 0.12\x00\x02SMB 2.002\x00'
        hostType = nmb.TYPE_SERVER
        if preferredDialect is None:
            packet = self.negotiateSessionWildcard(self._myName, self._remoteName, self._remoteHost, self._sess_port, self._timeout, True, flags1=flags1, flags2=flags2, data=negoData)
            if packet[0:1] == b'\xfe':
                self._SMBConnection = smb3.SMB3(self._remoteName, self._remoteHost, self._myName, hostType, self._sess_port, self._timeout, session=self._nmbSession, negSessionResponse=SMB2Packet(packet))
            else:
                self._SMBConnection = smb.SMB(self._remoteName, self._remoteHost, self._myName, hostType, self._sess_port, self._timeout, session=self._nmbSession, negPacket=packet)
        elif preferredDialect == smb.SMB_DIALECT:
            self._SMBConnection = smb.SMB(self._remoteName, self._remoteHost, self._myName, hostType, self._sess_port, self._timeout)
        elif preferredDialect in [SMB2_DIALECT_002, SMB2_DIALECT_21, SMB2_DIALECT_30, SMB2_DIALECT_311]:
            self._SMBConnection = smb3.SMB3(self._remoteName, self._remoteHost, self._myName, hostType, self._sess_port, self._timeout, preferredDialect=preferredDialect)
        else:
            raise Exception('Unknown dialect %s')
        if isinstance(self._SMBConnection, smb.SMB):
            if self._SMBConnection.get_flags()[1] & smb.SMB.FLAGS2_UNICODE:
                flags2 |= smb.SMB.FLAGS2_UNICODE
            self._SMBConnection.set_flags(flags1=flags1, flags2=flags2)
        return True

    def negotiateSessionWildcard(self, myName, remoteName, remoteHost, sess_port, timeout, extended_security=True, flags1=0, flags2=0, data=None):
        if False:
            for i in range(10):
                print('nop')
        if not myName:
            myName = socket.gethostname()
            i = myName.find('.')
            if i > -1:
                myName = myName[:i]
        tries = 0
        smbp = smb.NewSMBPacket()
        smbp['Flags1'] = flags1
        smbp['Flags2'] = flags2 | smb.SMB.FLAGS2_UNICODE
        resp = None
        while tries < 2:
            self._nmbSession = nmb.NetBIOSTCPSession(myName, remoteName, remoteHost, nmb.TYPE_SERVER, sess_port, timeout)
            negSession = smb.SMBCommand(smb.SMB.SMB_COM_NEGOTIATE)
            if extended_security is True:
                smbp['Flags2'] |= smb.SMB.FLAGS2_EXTENDED_SECURITY
            negSession['Data'] = data
            smbp.addCommand(negSession)
            self._nmbSession.send_packet(smbp.getData())
            try:
                resp = self._nmbSession.recv_packet(timeout)
                break
            except nmb.NetBIOSError:
                smbp['Flags2'] |= smb.SMB.FLAGS2_NT_STATUS | smb.SMB.FLAGS2_LONG_NAMES | smb.SMB.FLAGS2_UNICODE
                smbp['Data'] = []
            tries += 1
        if resp is None:
            raise Exception('No answer!')
        return resp.get_trailer()

    def getNMBServer(self):
        if False:
            return 10
        return self._nmbSession

    def getSMBServer(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        returns the SMB/SMB3 instance being used. Useful for calling low level methods\n        '
        return self._SMBConnection

    def getDialect(self):
        if False:
            return 10
        return self._SMBConnection.getDialect()

    def getServerName(self):
        if False:
            while True:
                i = 10
        return self._SMBConnection.get_server_name()

    def getClientName(self):
        if False:
            while True:
                i = 10
        return self._SMBConnection.get_client_name()

    def getRemoteHost(self):
        if False:
            while True:
                i = 10
        return self._SMBConnection.get_remote_host()

    def getRemoteName(self):
        if False:
            i = 10
            return i + 15
        return self._SMBConnection.get_remote_name()

    def setRemoteName(self, name):
        if False:
            i = 10
            return i + 15
        return self._SMBConnection.set_remote_name(name)

    def getServerDomain(self):
        if False:
            i = 10
            return i + 15
        return self._SMBConnection.get_server_domain()

    def getServerDNSDomainName(self):
        if False:
            i = 10
            return i + 15
        return self._SMBConnection.get_server_dns_domain_name()

    def getServerDNSHostName(self):
        if False:
            i = 10
            return i + 15
        return self._SMBConnection.get_server_dns_host_name()

    def getServerOS(self):
        if False:
            return 10
        return self._SMBConnection.get_server_os()

    def getServerOSMajor(self):
        if False:
            i = 10
            return i + 15
        return self._SMBConnection.get_server_os_major()

    def getServerOSMinor(self):
        if False:
            while True:
                i = 10
        return self._SMBConnection.get_server_os_minor()

    def getServerOSBuild(self):
        if False:
            while True:
                i = 10
        return self._SMBConnection.get_server_os_build()

    def doesSupportNTLMv2(self):
        if False:
            return 10
        return self._SMBConnection.doesSupportNTLMv2()

    def isLoginRequired(self):
        if False:
            print('Hello World!')
        return self._SMBConnection.is_login_required()

    def isSigningRequired(self):
        if False:
            for i in range(10):
                print('nop')
        return self._SMBConnection.is_signing_required()

    def getCredentials(self):
        if False:
            i = 10
            return i + 15
        return self._SMBConnection.getCredentials()

    def getIOCapabilities(self):
        if False:
            return 10
        return self._SMBConnection.getIOCapabilities()

    def login(self, user, password, domain='', lmhash='', nthash='', ntlmFallback=True):
        if False:
            i = 10
            return i + 15
        '\n        logins into the target system\n\n        :param string user: username\n        :param string password: password for the user\n        :param string domain: domain where the account is valid for\n        :param string lmhash: LMHASH used to authenticate using hashes (password is not used)\n        :param string nthash: NTHASH used to authenticate using hashes (password is not used)\n        :param bool ntlmFallback: If True it will try NTLMv1 authentication if NTLMv2 fails. Only available for SMBv1\n\n        :return: None\n        :raise SessionError: if error\n        '
        self._ntlmFallback = ntlmFallback
        try:
            if self.getDialect() == smb.SMB_DIALECT:
                return self._SMBConnection.login(user, password, domain, lmhash, nthash, ntlmFallback)
            else:
                return self._SMBConnection.login(user, password, domain, lmhash, nthash)
        except (smb.SessionError, smb3.SessionError) as e:
            raise SessionError(e.get_error_code(), e.get_error_packet())

    def kerberosLogin(self, user, password, domain='', lmhash='', nthash='', aesKey='', kdcHost=None, TGT=None, TGS=None, useCache=True):
        if False:
            print('Hello World!')
        "\n        logins into the target system explicitly using Kerberos. Hashes are used if RC4_HMAC is supported.\n\n        :param string user: username\n        :param string password: password for the user\n        :param string domain: domain where the account is valid for (required)\n        :param string lmhash: LMHASH used to authenticate using hashes (password is not used)\n        :param string nthash: NTHASH used to authenticate using hashes (password is not used)\n        :param string aesKey: aes256-cts-hmac-sha1-96 or aes128-cts-hmac-sha1-96 used for Kerberos authentication\n        :param string kdcHost: hostname or IP Address for the KDC. If None, the domain will be used (it needs to resolve tho)\n        :param struct TGT: If there's a TGT available, send the structure here and it will be used\n        :param struct TGS: same for TGS. See smb3.py for the format\n        :param bool useCache: whether or not we should use the ccache for credentials lookup. If TGT or TGS are specified this is False\n\n        :return: None\n        :raise SessionError: if error\n        "
        from impacket.krb5.ccache import CCache
        from impacket.krb5.kerberosv5 import KerberosError
        from impacket.krb5 import constants
        self._kdcHost = kdcHost
        self._useCache = useCache
        if TGT is not None or TGS is not None:
            useCache = False
        if useCache:
            (domain, user, TGT, TGS) = CCache.parseFile(domain, user, 'cifs/%s' % self.getRemoteName())
        while True:
            try:
                if self.getDialect() == smb.SMB_DIALECT:
                    return self._SMBConnection.kerberos_login(user, password, domain, lmhash, nthash, aesKey, kdcHost, TGT, TGS)
                return self._SMBConnection.kerberosLogin(user, password, domain, lmhash, nthash, aesKey, kdcHost, TGT, TGS)
            except (smb.SessionError, smb3.SessionError) as e:
                raise SessionError(e.get_error_code(), e.get_error_packet())
            except KerberosError as e:
                if e.getErrorCode() == constants.ErrorCodes.KDC_ERR_ETYPE_NOSUPP.value:
                    if lmhash == '' and nthash == '' and (aesKey == '' or aesKey is None) and (TGT is None) and (TGS is None):
                        lmhash = compute_lmhash(password)
                        nthash = compute_nthash(password)
                    else:
                        raise e
                else:
                    raise e

    def isGuestSession(self):
        if False:
            i = 10
            return i + 15
        try:
            return self._SMBConnection.isGuestSession()
        except (smb.SessionError, smb3.SessionError) as e:
            raise SessionError(e.get_error_code(), e.get_error_packet())

    def logoff(self):
        if False:
            while True:
                i = 10
        try:
            return self._SMBConnection.logoff()
        except (smb.SessionError, smb3.SessionError) as e:
            raise SessionError(e.get_error_code(), e.get_error_packet())

    def connectTree(self, share):
        if False:
            print('Hello World!')
        if self.getDialect() == smb.SMB_DIALECT:
            if ntpath.ismount(share) is False:
                share = ntpath.basename(share)
                share = '\\\\' + self.getRemoteHost() + '\\' + share
        try:
            return self._SMBConnection.connect_tree(share)
        except (smb.SessionError, smb3.SessionError) as e:
            raise SessionError(e.get_error_code(), e.get_error_packet())

    def disconnectTree(self, treeId):
        if False:
            print('Hello World!')
        try:
            return self._SMBConnection.disconnect_tree(treeId)
        except (smb.SessionError, smb3.SessionError) as e:
            raise SessionError(e.get_error_code(), e.get_error_packet())

    def listShares(self):
        if False:
            while True:
                i = 10
        '\n        get a list of available shares at the connected target\n\n        :return: a list containing dict entries for each share\n        :raise SessionError: if error\n        '
        from impacket.dcerpc.v5 import transport, srvs
        rpctransport = transport.SMBTransport(self.getRemoteName(), self.getRemoteHost(), filename='\\srvsvc', smb_connection=self)
        dce = rpctransport.get_dce_rpc()
        dce.connect()
        dce.bind(srvs.MSRPC_UUID_SRVS)
        resp = srvs.hNetrShareEnum(dce, 1)
        return resp['InfoStruct']['ShareInfo']['Level1']['Buffer']

    def listPath(self, shareName, path, password=None):
        if False:
            i = 10
            return i + 15
        '\n        list the files/directories under shareName/path\n\n        :param string shareName: a valid name for the share where the files/directories are going to be searched\n        :param string path: a base path relative to shareName\n        :param string password: the password for the share\n\n        :return: a list containing smb.SharedFile items\n        :raise SessionError: if error\n        '
        try:
            return self._SMBConnection.list_path(shareName, path, password)
        except (smb.SessionError, smb3.SessionError) as e:
            raise SessionError(e.get_error_code(), e.get_error_packet())

    def createFile(self, treeId, pathName, desiredAccess=GENERIC_ALL, shareMode=FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, creationOption=FILE_NON_DIRECTORY_FILE, creationDisposition=FILE_OVERWRITE_IF, fileAttributes=FILE_ATTRIBUTE_NORMAL, impersonationLevel=SMB2_IL_IMPERSONATION, securityFlags=0, oplockLevel=SMB2_OPLOCK_LEVEL_NONE, createContexts=None):
        if False:
            return 10
        '\n        Creates a remote file\n\n        :param HANDLE treeId: a valid handle for the share where the file is to be created\n        :param string pathName: the path name of the file to create\n        :param int desiredAccess: The level of access that is required, as specified in https://msdn.microsoft.com/en-us/library/cc246503.aspx\n        :param int shareMode: Specifies the sharing mode for the open.\n        :param int creationOption: Specifies the options to be applied when creating or opening the file.\n        :param int creationDisposition: Defines the action the server MUST take if the file that is specified in the name\n        field already exists.\n        :param int fileAttributes: This field MUST be a combination of the values specified in [MS-FSCC] section 2.6, and MUST NOT include any values other than those specified in that section.\n        :param int impersonationLevel: This field specifies the impersonation level requested by the application that is issuing the create request.\n        :param int securityFlags: This field MUST NOT be used and MUST be reserved. The client MUST set this to 0, and the server MUST ignore it.\n        :param int oplockLevel: The requested oplock level\n        :param createContexts: A variable-length attribute that is sent with an SMB2 CREATE Request or SMB2 CREATE Response that either gives extra information about how the create will be processed, or returns extra information about how the create was processed.\n\n        :return: a valid file descriptor\n        :raise SessionError: if error\n        '
        if self.getDialect() == smb.SMB_DIALECT:
            (_, flags2) = self._SMBConnection.get_flags()
            pathName = pathName.replace('/', '\\')
            packetPathName = pathName.encode('utf-16le') if flags2 & smb.SMB.FLAGS2_UNICODE else pathName
            ntCreate = smb.SMBCommand(smb.SMB.SMB_COM_NT_CREATE_ANDX)
            ntCreate['Parameters'] = smb.SMBNtCreateAndX_Parameters()
            ntCreate['Data'] = smb.SMBNtCreateAndX_Data(flags=flags2)
            ntCreate['Parameters']['FileNameLength'] = len(packetPathName)
            ntCreate['Parameters']['AccessMask'] = desiredAccess
            ntCreate['Parameters']['FileAttributes'] = fileAttributes
            ntCreate['Parameters']['ShareAccess'] = shareMode
            ntCreate['Parameters']['Disposition'] = creationDisposition
            ntCreate['Parameters']['CreateOptions'] = creationOption
            ntCreate['Parameters']['Impersonation'] = impersonationLevel
            ntCreate['Parameters']['SecurityFlags'] = securityFlags
            ntCreate['Parameters']['CreateFlags'] = 22
            ntCreate['Data']['FileName'] = packetPathName
            if flags2 & smb.SMB.FLAGS2_UNICODE:
                ntCreate['Data']['Pad'] = 0
            if createContexts is not None:
                LOG.error('CreateContexts not supported in SMB1')
            try:
                return self._SMBConnection.nt_create_andx(treeId, pathName, cmd=ntCreate)
            except (smb.SessionError, smb3.SessionError) as e:
                raise SessionError(e.get_error_code(), e.get_error_packet())
        else:
            try:
                return self._SMBConnection.create(treeId, pathName, desiredAccess, shareMode, creationOption, creationDisposition, fileAttributes, impersonationLevel, securityFlags, oplockLevel, createContexts)
            except (smb.SessionError, smb3.SessionError) as e:
                raise SessionError(e.get_error_code(), e.get_error_packet())

    def openFile(self, treeId, pathName, desiredAccess=FILE_READ_DATA | FILE_WRITE_DATA, shareMode=FILE_SHARE_READ, creationOption=FILE_NON_DIRECTORY_FILE, creationDisposition=FILE_OPEN, fileAttributes=FILE_ATTRIBUTE_NORMAL, impersonationLevel=SMB2_IL_IMPERSONATION, securityFlags=0, oplockLevel=SMB2_OPLOCK_LEVEL_NONE, createContexts=None):
        if False:
            print('Hello World!')
        '\n        opens a remote file\n\n        :param HANDLE treeId: a valid handle for the share where the file is to be opened\n        :param string pathName: the path name to open\n        :param int desiredAccess: The level of access that is required, as specified in https://msdn.microsoft.com/en-us/library/cc246503.aspx\n        :param int shareMode: Specifies the sharing mode for the open.\n        :param int creationOption: Specifies the options to be applied when creating or opening the file.\n        :param int creationDisposition: Defines the action the server MUST take if the file that is specified in the name\n        field already exists.\n        :param int fileAttributes: This field MUST be a combination of the values specified in [MS-FSCC] section 2.6, and MUST NOT include any values other than those specified in that section.\n        :param int impersonationLevel: This field specifies the impersonation level requested by the application that is issuing the create request.\n        :param int securityFlags: This field MUST NOT be used and MUST be reserved. The client MUST set this to 0, and the server MUST ignore it.\n        :param int oplockLevel: The requested oplock level\n        :param createContexts: A variable-length attribute that is sent with an SMB2 CREATE Request or SMB2 CREATE Response that either gives extra information about how the create will be processed, or returns extra information about how the create was processed.\n\n        :return: a valid file descriptor\n        :raise SessionError: if error\n        '
        if self.getDialect() == smb.SMB_DIALECT:
            (_, flags2) = self._SMBConnection.get_flags()
            pathName = pathName.replace('/', '\\')
            packetPathName = pathName.encode('utf-16le') if flags2 & smb.SMB.FLAGS2_UNICODE else pathName
            ntCreate = smb.SMBCommand(smb.SMB.SMB_COM_NT_CREATE_ANDX)
            ntCreate['Parameters'] = smb.SMBNtCreateAndX_Parameters()
            ntCreate['Data'] = smb.SMBNtCreateAndX_Data(flags=flags2)
            ntCreate['Parameters']['FileNameLength'] = len(packetPathName)
            ntCreate['Parameters']['AccessMask'] = desiredAccess
            ntCreate['Parameters']['FileAttributes'] = fileAttributes
            ntCreate['Parameters']['ShareAccess'] = shareMode
            ntCreate['Parameters']['Disposition'] = creationDisposition
            ntCreate['Parameters']['CreateOptions'] = creationOption
            ntCreate['Parameters']['Impersonation'] = impersonationLevel
            ntCreate['Parameters']['SecurityFlags'] = securityFlags
            ntCreate['Parameters']['CreateFlags'] = 22
            ntCreate['Data']['FileName'] = packetPathName
            if flags2 & smb.SMB.FLAGS2_UNICODE:
                ntCreate['Data']['Pad'] = 0
            if createContexts is not None:
                LOG.error('CreateContexts not supported in SMB1')
            try:
                return self._SMBConnection.nt_create_andx(treeId, pathName, cmd=ntCreate)
            except (smb.SessionError, smb3.SessionError) as e:
                raise SessionError(e.get_error_code(), e.get_error_packet())
        else:
            try:
                return self._SMBConnection.create(treeId, pathName, desiredAccess, shareMode, creationOption, creationDisposition, fileAttributes, impersonationLevel, securityFlags, oplockLevel, createContexts)
            except (smb.SessionError, smb3.SessionError) as e:
                raise SessionError(e.get_error_code(), e.get_error_packet())

    def writeFile(self, treeId, fileId, data, offset=0):
        if False:
            print('Hello World!')
        '\n        writes data to a file\n\n        :param HANDLE treeId: a valid handle for the share where the file is to be written\n        :param HANDLE fileId: a valid handle for the file\n        :param string data: buffer with the data to write\n        :param integer offset: offset where to start writing the data\n\n        :return: amount of bytes written\n        :raise SessionError: if error\n        '
        try:
            return self._SMBConnection.writeFile(treeId, fileId, data, offset)
        except (smb.SessionError, smb3.SessionError) as e:
            raise SessionError(e.get_error_code(), e.get_error_packet())

    def readFile(self, treeId, fileId, offset=0, bytesToRead=None, singleCall=True):
        if False:
            i = 10
            return i + 15
        "\n        reads data from a file\n\n        :param HANDLE treeId: a valid handle for the share where the file is to be read\n        :param HANDLE fileId: a valid handle for the file to be read\n        :param integer offset: offset where to start reading the data\n        :param integer bytesToRead: amount of bytes to attempt reading. If None, it will attempt to read Dialect['MaxBufferSize'] bytes.\n        :param boolean singleCall: If True it won't attempt to read all bytesToRead. It will only make a single read call\n\n        :return: the data read. Length of data read is not always bytesToRead\n        :raise SessionError: if error\n        "
        finished = False
        data = b''
        maxReadSize = self._SMBConnection.getIOCapabilities()['MaxReadSize']
        if bytesToRead is None:
            bytesToRead = maxReadSize
        remainingBytesToRead = bytesToRead
        while not finished:
            if remainingBytesToRead > maxReadSize:
                toRead = maxReadSize
            else:
                toRead = remainingBytesToRead
            try:
                bytesRead = self._SMBConnection.read_andx(treeId, fileId, offset, toRead)
            except (smb.SessionError, smb3.SessionError) as e:
                if e.get_error_code() == nt_errors.STATUS_END_OF_FILE:
                    toRead = b''
                    break
                else:
                    raise SessionError(e.get_error_code(), e.get_error_packet())
            data += bytesRead
            if len(data) >= bytesToRead:
                finished = True
            elif len(bytesRead) == 0:
                finished = True
            elif singleCall is True:
                finished = True
            else:
                offset += len(bytesRead)
                remainingBytesToRead -= len(bytesRead)
        return data

    def closeFile(self, treeId, fileId):
        if False:
            for i in range(10):
                print('nop')
        '\n        closes a file handle\n\n        :param HANDLE treeId: a valid handle for the share where the file is to be opened\n        :param HANDLE fileId: a valid handle for the file/directory to be closed\n\n        :return: None\n        :raise SessionError: if error\n        '
        try:
            return self._SMBConnection.close(treeId, fileId)
        except (smb.SessionError, smb3.SessionError) as e:
            raise SessionError(e.get_error_code(), e.get_error_packet())

    def deleteFile(self, shareName, pathName):
        if False:
            return 10
        '\n        removes a file\n\n        :param string shareName: a valid name for the share where the file is to be deleted\n        :param string pathName: the path name to remove\n\n        :return: None\n        :raise SessionError: if error\n        '
        try:
            return self._SMBConnection.remove(shareName, pathName)
        except (smb.SessionError, smb3.SessionError) as e:
            raise SessionError(e.get_error_code(), e.get_error_packet())

    def queryInfo(self, treeId, fileId):
        if False:
            return 10
        '\n        queries basic information about an opened file/directory\n\n        :param HANDLE treeId: a valid handle for the share where the file is to be queried\n        :param HANDLE fileId: a valid handle for the file/directory to be queried\n\n        :return: a smb.SMBQueryFileStandardInfo structure.\n        :raise SessionError: if error\n        '
        try:
            if self.getDialect() == smb.SMB_DIALECT:
                res = self._SMBConnection.query_file_info(treeId, fileId)
            else:
                res = self._SMBConnection.queryInfo(treeId, fileId)
            return smb.SMBQueryFileStandardInfo(res)
        except (smb.SessionError, smb3.SessionError) as e:
            raise SessionError(e.get_error_code(), e.get_error_packet())

    def createDirectory(self, shareName, pathName):
        if False:
            for i in range(10):
                print('nop')
        '\n        creates a directory\n\n        :param string shareName: a valid name for the share where the directory is to be created\n        :param string pathName: the path name or the directory to create\n\n        :return: None\n        :raise SessionError: if error\n        '
        try:
            return self._SMBConnection.mkdir(shareName, pathName)
        except (smb.SessionError, smb3.SessionError) as e:
            raise SessionError(e.get_error_code(), e.get_error_packet())

    def deleteDirectory(self, shareName, pathName):
        if False:
            return 10
        '\n        deletes a directory\n\n        :param string shareName: a valid name for the share where directory is to be deleted\n        :param string pathName: the path name or the directory to delete\n\n        :return: None\n        :raise SessionError: if error\n        '
        try:
            return self._SMBConnection.rmdir(shareName, pathName)
        except (smb.SessionError, smb3.SessionError) as e:
            raise SessionError(e.get_error_code(), e.get_error_packet())

    def waitNamedPipe(self, treeId, pipeName, timeout=5):
        if False:
            while True:
                i = 10
        '\n        waits for a named pipe\n\n        :param HANDLE treeId: a valid handle for the share where the pipe is\n        :param string pipeName: the pipe name to check\n        :param integer timeout: time to wait for an answer\n\n        :return: None\n        :raise SessionError: if error\n        '
        try:
            return self._SMBConnection.waitNamedPipe(treeId, pipeName, timeout=timeout)
        except (smb.SessionError, smb3.SessionError) as e:
            raise SessionError(e.get_error_code(), e.get_error_packet())

    def transactNamedPipe(self, treeId, fileId, data, waitAnswer=True):
        if False:
            for i in range(10):
                print('nop')
        '\n        writes to a named pipe using a transaction command\n\n        :param HANDLE treeId: a valid handle for the share where the pipe is\n        :param HANDLE fileId: a valid handle for the pipe\n        :param string data: buffer with the data to write\n        :param boolean waitAnswer: whether or not to wait for an answer\n\n        :return: None\n        :raise SessionError: if error\n        '
        try:
            return self._SMBConnection.TransactNamedPipe(treeId, fileId, data, waitAnswer=waitAnswer)
        except (smb.SessionError, smb3.SessionError) as e:
            raise SessionError(e.get_error_code(), e.get_error_packet())

    def transactNamedPipeRecv(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        reads from a named pipe using a transaction command\n\n        :return: data read\n        :raise SessionError: if error\n        '
        try:
            return self._SMBConnection.TransactNamedPipeRecv()
        except (smb.SessionError, smb3.SessionError) as e:
            raise SessionError(e.get_error_code(), e.get_error_packet())

    def writeNamedPipe(self, treeId, fileId, data, waitAnswer=True):
        if False:
            for i in range(10):
                print('nop')
        '\n        writes to a named pipe\n\n        :param HANDLE treeId: a valid handle for the share where the pipe is\n        :param HANDLE fileId: a valid handle for the pipe\n        :param string data: buffer with the data to write\n        :param boolean waitAnswer: whether or not to wait for an answer\n\n        :return: None\n        :raise SessionError: if error\n        '
        try:
            if self.getDialect() == smb.SMB_DIALECT:
                return self._SMBConnection.write_andx(treeId, fileId, data, wait_answer=waitAnswer, write_pipe_mode=True)
            else:
                return self.writeFile(treeId, fileId, data, 0)
        except (smb.SessionError, smb3.SessionError) as e:
            raise SessionError(e.get_error_code(), e.get_error_packet())

    def readNamedPipe(self, treeId, fileId, bytesToRead=None):
        if False:
            while True:
                i = 10
        '\n        read from a named pipe\n\n        :param HANDLE treeId: a valid handle for the share where the pipe resides\n        :param HANDLE fileId: a valid handle for the pipe\n        :param integer bytesToRead: amount of data to read\n\n        :return: None\n        :raise SessionError: if error\n        '
        try:
            return self.readFile(treeId, fileId, bytesToRead=bytesToRead, singleCall=True)
        except (smb.SessionError, smb3.SessionError) as e:
            raise SessionError(e.get_error_code(), e.get_error_packet())

    def getFile(self, shareName, pathName, callback, shareAccessMode=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        downloads a file\n\n        :param string shareName: name for the share where the file is to be retrieved\n        :param string pathName: the path name to retrieve\n        :param callback callback: function called to write the contents read.\n        :param int shareAccessMode:\n\n        :return: None\n        :raise SessionError: if error\n        '
        try:
            if shareAccessMode is None:
                return self._SMBConnection.retr_file(shareName, pathName, callback)
            else:
                return self._SMBConnection.retr_file(shareName, pathName, callback, shareAccessMode=shareAccessMode)
        except (smb.SessionError, smb3.SessionError) as e:
            raise SessionError(e.get_error_code(), e.get_error_packet())

    def putFile(self, shareName, pathName, callback, shareAccessMode=None):
        if False:
            return 10
        '\n        uploads a file\n\n        :param string shareName: name for the share where the file is to be uploaded\n        :param string pathName: the path name to upload\n        :param callback callback: function called to read the contents to be written.\n        :param int shareAccessMode:\n\n        :return: None\n        :raise SessionError: if error\n        '
        try:
            if shareAccessMode is None:
                return self._SMBConnection.stor_file(shareName, pathName, callback)
            else:
                return self._SMBConnection.stor_file(shareName, pathName, callback, shareAccessMode)
        except (smb.SessionError, smb3.SessionError) as e:
            raise SessionError(e.get_error_code(), e.get_error_packet())

    def listSnapshots(self, tid, path):
        if False:
            print('Hello World!')
        '\n        lists the snapshots for the given directory\n\n        :param int tid: tree id of current connection\n        :param string path: directory to list the snapshots of\n\n        :raise SessionError: if error\n        '
        if self.getDialect() not in [SMB2_DIALECT_002, SMB2_DIALECT_21, SMB2_DIALECT_30]:
            raise SessionError(error=nt_errors.STATUS_NOT_SUPPORTED)
        fid = self.openFile(tid, path, FILE_READ_DATA | FILE_READ_EA | FILE_READ_ATTRIBUTES | READ_CONTROL | SYNCHRONIZE, fileAttributes=None, creationOption=FILE_SYNCHRONOUS_IO_NONALERT, shareMode=FILE_SHARE_READ | FILE_SHARE_WRITE)
        try:
            snapshotData = SRV_SNAPSHOT_ARRAY(self._SMBConnection.ioctl(tid, fid, FSCTL_SRV_ENUMERATE_SNAPSHOTS, flags=SMB2_0_IOCTL_IS_FSCTL, maxOutputResponse=16))
        except (smb.SessionError, smb3.SessionError) as e:
            self.closeFile(tid, fid)
            raise SessionError(e.get_error_code(), e.get_error_packet())
        if snapshotData['SnapShotArraySize'] >= 52:
            try:
                snapshotData = SRV_SNAPSHOT_ARRAY(self._SMBConnection.ioctl(tid, fid, FSCTL_SRV_ENUMERATE_SNAPSHOTS, flags=SMB2_0_IOCTL_IS_FSCTL, maxOutputResponse=snapshotData['SnapShotArraySize'] + 12))
            except (smb.SessionError, smb3.SessionError) as e:
                self.closeFile(tid, fid)
                raise SessionError(e.get_error_code(), e.get_error_packet())
        self.closeFile(tid, fid)
        return list(filter(None, snapshotData['SnapShots'].decode('utf16').split('\x00')))

    def createMountPoint(self, tid, path, target):
        if False:
            return 10
        '\n        creates a mount point at an existing directory\n\n        :param int tid: tree id of current connection\n        :param string path: directory at which to create mount point (must already exist)\n        :param string target: target address of mount point\n\n        :raise SessionError: if error\n        '
        if self.getDialect() not in [SMB2_DIALECT_002, SMB2_DIALECT_21, SMB2_DIALECT_30]:
            raise SessionError(error=nt_errors.STATUS_NOT_SUPPORTED)
        fid = self.openFile(tid, path, GENERIC_READ | GENERIC_WRITE, creationOption=FILE_OPEN_REPARSE_POINT)
        if target.startswith('\\'):
            fixed_name = target.encode('utf-16le')
        else:
            fixed_name = ('\\??\\' + target).encode('utf-16le')
        name = target.encode('utf-16le')
        reparseData = MOUNT_POINT_REPARSE_DATA_STRUCTURE()
        reparseData['PathBuffer'] = fixed_name + b'\x00\x00' + name + b'\x00\x00'
        reparseData['SubstituteNameLength'] = len(fixed_name)
        reparseData['PrintNameOffset'] = len(fixed_name) + 2
        reparseData['PrintNameLength'] = len(name)
        self._SMBConnection.ioctl(tid, fid, FSCTL_SET_REPARSE_POINT, flags=SMB2_0_IOCTL_IS_FSCTL, inputBlob=reparseData)
        self.closeFile(tid, fid)

    def removeMountPoint(self, tid, path):
        if False:
            print('Hello World!')
        '\n        removes a mount point without deleting the underlying directory\n\n        :param int tid: tree id of current connection\n        :param string path: path to mount point to remove\n\n        :raise SessionError: if error\n        '
        if self.getDialect() not in [SMB2_DIALECT_002, SMB2_DIALECT_21, SMB2_DIALECT_30]:
            raise SessionError(error=nt_errors.STATUS_NOT_SUPPORTED)
        fid = self.openFile(tid, path, GENERIC_READ | GENERIC_WRITE, creationOption=FILE_OPEN_REPARSE_POINT)
        reparseData = MOUNT_POINT_REPARSE_GUID_DATA_STRUCTURE()
        reparseData['DataBuffer'] = b''
        try:
            self._SMBConnection.ioctl(tid, fid, FSCTL_DELETE_REPARSE_POINT, flags=SMB2_0_IOCTL_IS_FSCTL, inputBlob=reparseData)
        except (smb.SessionError, smb3.SessionError) as e:
            self.closeFile(tid, fid)
            raise SessionError(e.get_error_code(), e.get_error_packet())
        self.closeFile(tid, fid)

    def rename(self, shareName, oldPath, newPath):
        if False:
            for i in range(10):
                print('nop')
        '\n        renames a file/directory\n\n        :param string shareName: name for the share where the files/directories are\n        :param string oldPath: the old path name or the directory/file to rename\n        :param string newPath: the new path name or the directory/file to rename\n\n        :return: True\n        :raise SessionError: if error\n        '
        try:
            return self._SMBConnection.rename(shareName, oldPath, newPath)
        except (smb.SessionError, smb3.SessionError) as e:
            raise SessionError(e.get_error_code(), e.get_error_packet())

    def reconnect(self):
        if False:
            print('Hello World!')
        '\n        reconnects the SMB object based on the original options and credentials used. Only exception is that\n        manualNegotiate will not be honored.\n        Not only the connection will be created but also a login attempt using the original credentials and\n        method (Kerberos, PtH, etc)\n\n        :return: True\n        :raise SessionError: if error\n        '
        (userName, password, domain, lmhash, nthash, aesKey, TGT, TGS) = self.getCredentials()
        self.negotiateSession(self._preferredDialect)
        if self._doKerberos is True:
            self.kerberosLogin(userName, password, domain, lmhash, nthash, aesKey, self._kdcHost, TGT, TGS, self._useCache)
        else:
            self.login(userName, password, domain, lmhash, nthash, self._ntlmFallback)
        return True

    def setTimeout(self, timeout):
        if False:
            i = 10
            return i + 15
        try:
            return self._SMBConnection.set_timeout(timeout)
        except (smb.SessionError, smb3.SessionError) as e:
            raise SessionError(e.get_error_code(), e.get_error_packet())

    def getSessionKey(self):
        if False:
            return 10
        if self.getDialect() == smb.SMB_DIALECT:
            return self._SMBConnection.get_session_key()
        else:
            return self._SMBConnection.getSessionKey()

    def setSessionKey(self, key):
        if False:
            while True:
                i = 10
        if self.getDialect() == smb.SMB_DIALECT:
            return self._SMBConnection.set_session_key(key)
        else:
            return self._SMBConnection.setSessionKey(key)

    def setHostnameValidation(self, validate, accept_empty, hostname):
        if False:
            i = 10
            return i + 15
        return self._SMBConnection.set_hostname_validation(validate, accept_empty, hostname)

    def close(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        logs off and closes the underlying _NetBIOSSession()\n\n        :return: None\n        '
        try:
            self.logoff()
        except:
            pass
        self._SMBConnection.close_session()

class SessionError(Exception):
    """
    This is the exception every client should catch regardless of the underlying
    SMB version used. We'll take care of that. NETBIOS exceptions are NOT included,
    since all SMB versions share the same NETBIOS instances.
    """

    def __init__(self, error=0, packet=0):
        if False:
            i = 10
            return i + 15
        Exception.__init__(self)
        self.error = error
        self.packet = packet

    def getErrorCode(self):
        if False:
            while True:
                i = 10
        return self.error

    def getErrorPacket(self):
        if False:
            return 10
        return self.packet

    def getErrorString(self):
        if False:
            i = 10
            return i + 15
        return str(self)

    def __str__(self):
        if False:
            i = 10
            return i + 15
        key = self.error
        if key in nt_errors.ERROR_MESSAGES:
            error_msg_short = nt_errors.ERROR_MESSAGES[key][0]
            error_msg_verbose = nt_errors.ERROR_MESSAGES[key][1]
            return 'SMB SessionError: code: 0x%x - %s - %s' % (self.error, error_msg_short, error_msg_verbose)
        else:
            return 'SMB SessionError: unknown error code: 0x%x' % self.error