import random
import string
from impacket.dcerpc.v5 import transport, srvs, scmr
from impacket import smb, smb3, LOG
from impacket.smbconnection import SMBConnection
from impacket.smb3structs import FILE_WRITE_DATA, FILE_DIRECTORY_FILE

class ServiceInstall:

    def __init__(self, SMBObject, exeFile, serviceName='', binary_service_name=None):
        if False:
            i = 10
            return i + 15
        self._rpctransport = 0
        self.__service_name = serviceName if len(serviceName) > 0 else ''.join([random.choice(string.ascii_letters) for i in range(4)])
        if binary_service_name is None:
            self.__binary_service_name = ''.join([random.choice(string.ascii_letters) for i in range(8)]) + '.exe'
        else:
            self.__binary_service_name = binary_service_name
        self.__exeFile = exeFile
        if isinstance(SMBObject, smb.SMB) or isinstance(SMBObject, smb3.SMB3):
            self.connection = SMBConnection(existingConnection=SMBObject)
        else:
            self.connection = SMBObject
        self.share = ''

    @property
    def serviceName(self):
        if False:
            return 10
        return self.__service_name

    @property
    def binaryServiceName(self):
        if False:
            return 10
        return self.__binary_service_name

    def getShare(self):
        if False:
            return 10
        return self.share

    def getShares(self):
        if False:
            print('Hello World!')
        LOG.info('Requesting shares on %s.....' % self.connection.getRemoteHost())
        try:
            self._rpctransport = transport.SMBTransport(self.connection.getRemoteHost(), self.connection.getRemoteHost(), filename='\\srvsvc', smb_connection=self.connection)
            dce_srvs = self._rpctransport.get_dce_rpc()
            dce_srvs.connect()
            dce_srvs.bind(srvs.MSRPC_UUID_SRVS)
            resp = srvs.hNetrShareEnum(dce_srvs, 1)
            return resp['InfoStruct']['ShareInfo']['Level1']
        except:
            LOG.critical('Error requesting shares on %s, aborting.....' % self.connection.getRemoteHost())
            raise

    def createService(self, handle, share, path):
        if False:
            return 10
        LOG.info('Creating service %s on %s.....' % (self.__service_name, self.connection.getRemoteHost()))
        try:
            resp = scmr.hROpenServiceW(self.rpcsvc, handle, self.__service_name + '\x00')
        except Exception as e:
            if str(e).find('ERROR_SERVICE_DOES_NOT_EXIST') >= 0:
                pass
            else:
                raise e
        else:
            scmr.hRDeleteService(self.rpcsvc, resp['lpServiceHandle'])
            scmr.hRCloseServiceHandle(self.rpcsvc, resp['lpServiceHandle'])
        command = '%s\\%s' % (path, self.__binary_service_name)
        try:
            resp = scmr.hRCreateServiceW(self.rpcsvc, handle, self.__service_name + '\x00', self.__service_name + '\x00', lpBinaryPathName=command + '\x00', dwStartType=scmr.SERVICE_DEMAND_START)
        except:
            LOG.critical('Error creating service %s on %s' % (self.__service_name, self.connection.getRemoteHost()))
            raise
        else:
            return resp['lpServiceHandle']

    def openSvcManager(self):
        if False:
            i = 10
            return i + 15
        LOG.info('Opening SVCManager on %s.....' % self.connection.getRemoteHost())
        self._rpctransport = transport.SMBTransport(self.connection.getRemoteHost(), self.connection.getRemoteHost(), filename='\\svcctl', smb_connection=self.connection)
        self.rpcsvc = self._rpctransport.get_dce_rpc()
        self.rpcsvc.connect()
        self.rpcsvc.bind(scmr.MSRPC_UUID_SCMR)
        try:
            resp = scmr.hROpenSCManagerW(self.rpcsvc)
        except:
            LOG.critical('Error opening SVCManager on %s.....' % self.connection.getRemoteHost())
            raise Exception('Unable to open SVCManager')
        else:
            return resp['lpScHandle']

    def copy_file(self, src, tree, dst):
        if False:
            for i in range(10):
                print('nop')
        LOG.info('Uploading file %s' % dst)
        if isinstance(src, str):
            fh = open(src, 'rb')
        else:
            fh = src
        f = dst
        pathname = f.replace('/', '\\')
        try:
            self.connection.putFile(tree, pathname, fh.read)
        except:
            LOG.critical('Error uploading file %s, aborting.....' % dst)
            raise
        fh.close()

    def findWritableShare(self, shares):
        if False:
            while True:
                i = 10
        writeableShare = None
        for i in shares['Buffer']:
            if i['shi1_type'] == srvs.STYPE_DISKTREE or i['shi1_type'] == srvs.STYPE_SPECIAL:
                share = i['shi1_netname'][:-1]
                tid = 0
                try:
                    tid = self.connection.connectTree(share)
                    self.connection.openFile(tid, '\\', FILE_WRITE_DATA, creationOption=FILE_DIRECTORY_FILE)
                except:
                    LOG.debug('Exception', exc_info=True)
                    LOG.critical("share '%s' is not writable." % share)
                    pass
                else:
                    LOG.info('Found writable share %s' % share)
                    writeableShare = str(share)
                    break
                finally:
                    if tid != 0:
                        self.connection.disconnectTree(tid)
        return writeableShare

    def install(self):
        if False:
            return 10
        if self.connection.isGuestSession():
            LOG.critical('Authenticated as Guest. Aborting')
            self.connection.logoff()
            del self.connection
        else:
            fileCopied = False
            serviceCreated = False
            try:
                shares = self.getShares()
                self.share = self.findWritableShare(shares)
                if self.share is None:
                    return False
                self.copy_file(self.__exeFile, self.share, self.__binary_service_name)
                fileCopied = True
                svcManager = self.openSvcManager()
                if svcManager != 0:
                    serverName = self.connection.getServerName()
                    if self.share.lower() == 'admin$':
                        path = '%systemroot%'
                    elif serverName != '':
                        path = '\\\\%s\\%s' % (serverName, self.share)
                    else:
                        path = '\\\\127.0.0.1\\' + self.share
                    service = self.createService(svcManager, self.share, path)
                    serviceCreated = True
                    if service != 0:
                        LOG.info('Starting service %s.....' % self.__service_name)
                        try:
                            scmr.hRStartServiceW(self.rpcsvc, service)
                        except:
                            pass
                        scmr.hRCloseServiceHandle(self.rpcsvc, service)
                    scmr.hRCloseServiceHandle(self.rpcsvc, svcManager)
                    return True
            except Exception as e:
                LOG.critical('Error performing the installation, cleaning up: %s' % e)
                LOG.debug('Exception', exc_info=True)
                try:
                    scmr.hRControlService(self.rpcsvc, service, scmr.SERVICE_CONTROL_STOP)
                except:
                    pass
                if fileCopied is True:
                    try:
                        self.connection.deleteFile(self.share, self.__binary_service_name)
                    except:
                        pass
                if serviceCreated is True:
                    try:
                        scmr.hRDeleteService(self.rpcsvc, service)
                    except:
                        pass
            return False

    def uninstall(self):
        if False:
            for i in range(10):
                print('nop')
        fileCopied = True
        serviceCreated = True
        try:
            svcManager = self.openSvcManager()
            if svcManager != 0:
                resp = scmr.hROpenServiceW(self.rpcsvc, svcManager, self.__service_name + '\x00')
                service = resp['lpServiceHandle']
                LOG.info('Stopping service %s.....' % self.__service_name)
                try:
                    scmr.hRControlService(self.rpcsvc, service, scmr.SERVICE_CONTROL_STOP)
                except:
                    pass
                LOG.info('Removing service %s.....' % self.__service_name)
                scmr.hRDeleteService(self.rpcsvc, service)
                scmr.hRCloseServiceHandle(self.rpcsvc, service)
                scmr.hRCloseServiceHandle(self.rpcsvc, svcManager)
            LOG.info('Removing file %s.....' % self.__binary_service_name)
            self.connection.deleteFile(self.share, self.__binary_service_name)
        except Exception:
            LOG.critical('Error performing the uninstallation, cleaning up')
            try:
                scmr.hRControlService(self.rpcsvc, service, scmr.SERVICE_CONTROL_STOP)
            except:
                pass
            if fileCopied is True:
                try:
                    self.connection.deleteFile(self.share, self.__binary_service_name)
                except:
                    try:
                        self.connection.deleteFile(self.share, self.__binary_service_name)
                    except:
                        pass
                    pass
            if serviceCreated is True:
                try:
                    scmr.hRDeleteService(self.rpcsvc, service)
                except:
                    pass