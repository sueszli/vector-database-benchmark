from binascii import unhexlify
from impacket.krb5.kerberosv5 import getKerberosTGT
from impacket.krb5 import constants
from impacket.krb5.types import Principal

class CMEModule:
    name = 'nopac'
    description = 'Check if the DC is vulnerable to CVE-2021-42278 and CVE-2021-42287 to impersonate DA from standard domain user'
    supported_protocols = ['smb']
    opsec_safe = True
    multiple_hosts = True

    def options(self, context, module_options):
        if False:
            for i in range(10):
                print('nop')
        ' '

    def on_login(self, context, connection):
        if False:
            print('Hello World!')
        user_name = Principal(connection.username, type=constants.PrincipalNameType.NT_PRINCIPAL.value)
        try:
            (tgt_with_pac, cipher, old_session_key, session_key) = getKerberosTGT(user_name, connection.password, connection.domain, unhexlify(connection.lmhash), unhexlify(connection.nthash), connection.aesKey, connection.host, requestPAC=True)
            context.log.highlight('TGT with PAC size ' + str(len(tgt_with_pac)))
            (tgt_no_pac, cipher, old_session_key, session_key) = getKerberosTGT(user_name, connection.password, connection.domain, unhexlify(connection.lmhash), unhexlify(connection.nthash), connection.aesKey, connection.host, requestPAC=False)
            context.log.highlight('TGT without PAC size ' + str(len(tgt_no_pac)))
            if len(tgt_no_pac) < len(tgt_with_pac):
                context.log.highlight('')
                context.log.highlight('VULNERABLE')
                context.log.highlight('Next step: https://github.com/Ridter/noPac')
        except OSError as e:
            context.log.debug(f'Error connecting to Kerberos (port 88) on {connection.host}')