"""
nmap.py - version and date, see below

Source code : https://bitbucket.org/xael/python-nmap

Author :

* Alexandre Norman - norman at xael.org

Contributors:

* Steve 'Ashcrow' Milner - steve at gnulinux.net
* Brian Bustin - brian at bustin.us
* old.schepperhand
* Johan Lundberg
* Thomas D. maaaaz
* Robert Bost
* David Peltier

Licence: GPL v3 or any later version for python-nmap


This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.


**************
IMPORTANT NOTE
**************

The Nmap Security Scanner used by python-nmap is distributed
under it's own licence that you can find at https://svn.nmap.org/nmap/COPYING

Any redistribution of python-nmap along with the Nmap Security Scanner
must conform to the Nmap Security Scanner licence

"""
import csv
import io
import os
import re
import shlex
import subprocess
import sys
from xml.etree import ElementTree as ET
try:
    from multiprocessing import Process
except ImportError:
    from threading import Thread as Process
__author__ = 'Alexandre Norman (norman@xael.org)'
__version__ = '0.6.3'
__last_modification__ = '2018/09/23'

class PortScanner(object):
    """
    PortScanner class allows to use nmap from python

    """

    def __init__(self, nmap_search_path=('nmap', '/usr/bin/nmap', '/usr/local/bin/nmap', '/sw/bin/nmap', '/opt/local/bin/nmap')):
        if False:
            while True:
                i = 10
        '\n        Initialize PortScanner module\n\n        * detects nmap on the system and nmap version\n        * may raise PortScannerError exception if nmap is not found in the path\n\n        :param nmap_search_path: tupple of string where to search for nmap executable. Change this if you want to use a specific version of nmap.\n        :returns: nothing\n\n        '
        self._nmap_path = ''
        self._scan_result = {}
        self._nmap_version_number = 0
        self._nmap_subversion_number = 0
        self._nmap_last_output = ''
        is_nmap_found = False
        self.__process = None
        regex = re.compile('Nmap version [0-9]*\\.[0-9]*[^ ]* \\( http(|s)://.* \\)')
        for nmap_path in nmap_search_path:
            try:
                if sys.platform.startswith('freebsd') or sys.platform.startswith('linux') or sys.platform.startswith('darwin'):
                    p = subprocess.Popen([nmap_path, '-V'], bufsize=10000, stdout=subprocess.PIPE, close_fds=True)
                else:
                    p = subprocess.Popen([nmap_path, '-V'], bufsize=10000, stdout=subprocess.PIPE)
            except OSError:
                pass
            else:
                self._nmap_path = nmap_path
                break
        else:
            raise PortScannerError('nmap program was not found in path. PATH is : {0}'.format(os.getenv('PATH')))
        self._nmap_last_output = bytes.decode(p.communicate()[0])
        for line in self._nmap_last_output.split(os.linesep):
            if regex.match(line) is not None:
                is_nmap_found = True
                regex_version = re.compile('[0-9]+')
                regex_subversion = re.compile('\\.[0-9]+')
                rv = regex_version.search(line)
                rsv = regex_subversion.search(line)
                if rv is not None and rsv is not None:
                    self._nmap_version_number = int(line[rv.start():rv.end()])
                    self._nmap_subversion_number = int(line[rsv.start() + 1:rsv.end()])
                break
        if not is_nmap_found:
            raise PortScannerError('nmap program was not found in path')
        return

    def get_nmap_last_output(self):
        if False:
            return 10
        '\n        Returns the last text output of nmap in raw text\n        this may be used for debugging purpose\n\n        :returns: string containing the last text output of nmap in raw text\n        '
        return self._nmap_last_output

    def nmap_version(self):
        if False:
            while True:
                i = 10
        '\n        returns nmap version if detected (int version, int subversion)\n        or (0, 0) if unknown\n        :returns: (nmap_version_number, nmap_subversion_number)\n        '
        return (self._nmap_version_number, self._nmap_subversion_number)

    def listscan(self, hosts='127.0.0.1'):
        if False:
            print('Hello World!')
        '\n        do not scan but interpret target hosts and return a list a hosts\n        '
        assert type(hosts) is str, 'Wrong type for [hosts], should be a string [was {0}]'.format(type(hosts))
        output = self.scan(hosts, arguments='-sL')
        if 'scaninfo' in output['nmap'] and 'error' in output['nmap']['scaninfo'] and (len(output['nmap']['scaninfo']['error']) > 0) and ('looks like an IPv6 target specification' in output['nmap']['scaninfo']['error'][0]):
            self.scan(hosts, arguments='-sL -6')
        return self.all_hosts()

    def scan(self, hosts='127.0.0.1', ports=None, arguments='-sV', sudo=False):
        if False:
            for i in range(10):
                print('nop')
        "\n        Scan given hosts\n\n        May raise PortScannerError exception if nmap output was not xml\n\n        Test existance of the following key to know\n        if something went wrong : ['nmap']['scaninfo']['error']\n        If not present, everything was ok.\n\n        :param hosts: string for hosts as nmap use it 'scanme.nmap.org' or '198.116.0-255.1-127' or '216.163.128.20/20'\n        :param ports: string for ports as nmap use it '22,53,110,143-4564'\n        :param arguments: string of arguments for nmap '-sU -sX -sC'\n        :param sudo: launch nmap with sudo if True\n\n        :returns: scan_result as dictionnary\n        "
        if sys.version_info[0] == 2:
            assert type(hosts) in (str, unicode), 'Wrong type for [hosts], should be a string [was {0}]'.format(type(hosts))
            assert type(ports) in (str, unicode, type(None)), 'Wrong type for [ports], should be a string [was {0}]'.format(type(ports))
            assert type(arguments) in (str, unicode), 'Wrong type for [arguments], should be a string [was {0}]'.format(type(arguments))
        else:
            assert type(hosts) is str, 'Wrong type for [hosts], should be a string [was {0}]'.format(type(hosts))
            assert type(ports) in (str, type(None)), 'Wrong type for [ports], should be a string [was {0}]'.format(type(ports))
            assert type(arguments) is str, 'Wrong type for [arguments], should be a string [was {0}]'.format(type(arguments))
        for redirecting_output in ['-oX', '-oA']:
            assert redirecting_output not in arguments, "Xml output can't be redirected from command line.\nYou can access it after a scan using:\nnmap.nm.get_nmap_last_output()"
        h_args = shlex.split(hosts)
        f_args = shlex.split(arguments)
        args = [self._nmap_path, '-oX', '-'] + h_args + ['-p', ports] * (ports is not None) + f_args
        if sudo:
            args = ['sudo'] + args
        p = subprocess.Popen(args, bufsize=100000, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        (self._nmap_last_output, nmap_err) = p.communicate()
        self._nmap_last_output = bytes.decode(self._nmap_last_output)
        nmap_err = bytes.decode(nmap_err)
        nmap_err_keep_trace = []
        nmap_warn_keep_trace = []
        if len(nmap_err) > 0:
            regex_warning = re.compile('^Warning: .*', re.IGNORECASE)
            for line in nmap_err.split(os.linesep):
                if len(line) > 0:
                    rgw = regex_warning.search(line)
                    if rgw is not None:
                        nmap_warn_keep_trace.append(line + os.linesep)
                    else:
                        nmap_err_keep_trace.append(nmap_err)
        return self.analyse_nmap_xml_scan(nmap_xml_output=self._nmap_last_output, nmap_err=nmap_err, nmap_err_keep_trace=nmap_err_keep_trace, nmap_warn_keep_trace=nmap_warn_keep_trace)

    def analyse_nmap_xml_scan(self, nmap_xml_output=None, nmap_err='', nmap_err_keep_trace='', nmap_warn_keep_trace=''):
        if False:
            i = 10
            return i + 15
        "\n        Analyses NMAP xml scan ouput\n\n        May raise PortScannerError exception if nmap output was not xml\n\n        Test existance of the following key to know if something went wrong : ['nmap']['scaninfo']['error']\n        If not present, everything was ok.\n\n        :param nmap_xml_output: xml string to analyse\n        :returns: scan_result as dictionnary\n        "
        if nmap_xml_output is not None:
            self._nmap_last_output = nmap_xml_output
        scan_result = {}
        try:
            dom = ET.fromstring(self._nmap_last_output)
        except Exception:
            if len(nmap_err) > 0:
                raise PortScannerError(nmap_err)
            else:
                raise PortScannerError(self._nmap_last_output)
        scan_result['nmap'] = {'command_line': dom.get('args'), 'scaninfo': {}, 'scanstats': {'timestr': dom.find('runstats/finished').get('timestr'), 'elapsed': dom.find('runstats/finished').get('elapsed'), 'uphosts': dom.find('runstats/hosts').get('up'), 'downhosts': dom.find('runstats/hosts').get('down'), 'totalhosts': dom.find('runstats/hosts').get('total')}}
        if len(nmap_err_keep_trace) > 0:
            scan_result['nmap']['scaninfo']['error'] = nmap_err_keep_trace
        if len(nmap_warn_keep_trace) > 0:
            scan_result['nmap']['scaninfo']['warning'] = nmap_warn_keep_trace
        for dsci in dom.findall('scaninfo'):
            scan_result['nmap']['scaninfo'][dsci.get('protocol')] = {'method': dsci.get('type'), 'services': dsci.get('services')}
        scan_result['scan'] = {}
        for dhost in dom.findall('host'):
            host = None
            address_block = {}
            vendor_block = {}
            for address in dhost.findall('address'):
                addtype = address.get('addrtype')
                address_block[addtype] = address.get('addr')
                if addtype == 'ipv4':
                    host = address_block[addtype]
                elif addtype == 'mac' and address.get('vendor') is not None:
                    vendor_block[address_block[addtype]] = address.get('vendor')
            if host is None:
                host = dhost.find('address').get('addr')
            hostnames = []
            if len(dhost.findall('hostnames/hostname')) > 0:
                for dhostname in dhost.findall('hostnames/hostname'):
                    hostnames.append({'name': dhostname.get('name'), 'type': dhostname.get('type')})
            else:
                hostnames.append({'name': '', 'type': ''})
            scan_result['scan'][host] = PortScannerHostDict({'hostnames': hostnames})
            scan_result['scan'][host]['addresses'] = address_block
            scan_result['scan'][host]['vendor'] = vendor_block
            for dstatus in dhost.findall('status'):
                scan_result['scan'][host]['status'] = {'state': dstatus.get('state'), 'reason': dstatus.get('reason')}
            for dstatus in dhost.findall('uptime'):
                scan_result['scan'][host]['uptime'] = {'seconds': dstatus.get('seconds'), 'lastboot': dstatus.get('lastboot')}
            for dport in dhost.findall('ports/port'):
                proto = dport.get('protocol')
                port = int(dport.get('portid'))
                state = dport.find('state').get('state')
                reason = dport.find('state').get('reason')
                name = product = version = extrainfo = conf = cpe = ''
                for dname in dport.findall('service'):
                    name = dname.get('name')
                    if dname.get('product'):
                        product = dname.get('product')
                    if dname.get('version'):
                        version = dname.get('version')
                    if dname.get('extrainfo'):
                        extrainfo = dname.get('extrainfo')
                    if dname.get('conf'):
                        conf = dname.get('conf')
                    for dcpe in dname.findall('cpe'):
                        cpe = dcpe.text
                if proto not in list(scan_result['scan'][host].keys()):
                    scan_result['scan'][host][proto] = {}
                scan_result['scan'][host][proto][port] = {'state': state, 'reason': reason, 'name': name, 'product': product, 'version': version, 'extrainfo': extrainfo, 'conf': conf, 'cpe': cpe}
                script_id = ''
                script_out = ''
                for dscript in dport.findall('script'):
                    script_id = dscript.get('id')
                    script_out = dscript.get('output')
                    if 'script' not in list(scan_result['scan'][host][proto][port].keys()):
                        scan_result['scan'][host][proto][port]['script'] = {}
                    scan_result['scan'][host][proto][port]['script'][script_id] = script_out
            for dhostscript in dhost.findall('hostscript'):
                for dname in dhostscript.findall('script'):
                    hsid = dname.get('id')
                    hsoutput = dname.get('output')
                    if 'hostscript' not in list(scan_result['scan'][host].keys()):
                        scan_result['scan'][host]['hostscript'] = []
                    scan_result['scan'][host]['hostscript'].append({'id': hsid, 'output': hsoutput})
            for dos in dhost.findall('os'):
                osmatch = []
                portused = []
                for dportused in dos.findall('portused'):
                    state = dportused.get('state')
                    proto = dportused.get('proto')
                    portid = dportused.get('portid')
                    portused.append({'state': state, 'proto': proto, 'portid': portid})
                scan_result['scan'][host]['portused'] = portused
                for dosmatch in dos.findall('osmatch'):
                    name = dosmatch.get('name')
                    accuracy = dosmatch.get('accuracy')
                    line = dosmatch.get('line')
                    osclass = []
                    for dosclass in dosmatch.findall('osclass'):
                        ostype = dosclass.get('type')
                        vendor = dosclass.get('vendor')
                        osfamily = dosclass.get('osfamily')
                        osgen = dosclass.get('osgen')
                        accuracy = dosclass.get('accuracy')
                        cpe = []
                        for dcpe in dosclass.findall('cpe'):
                            cpe.append(dcpe.text)
                        osclass.append({'type': ostype, 'vendor': vendor, 'osfamily': osfamily, 'osgen': osgen, 'accuracy': accuracy, 'cpe': cpe})
                    osmatch.append({'name': name, 'accuracy': accuracy, 'line': line, 'osclass': osclass})
                else:
                    scan_result['scan'][host]['osmatch'] = osmatch
            for dport in dhost.findall('osfingerprint'):
                fingerprint = dport.get('fingerprint')
                scan_result['scan'][host]['fingerprint'] = fingerprint
        self._scan_result = scan_result
        return scan_result

    def __getitem__(self, host):
        if False:
            return 10
        '\n        returns a host detail\n        '
        if sys.version_info[0] == 2:
            assert type(host) in (str, unicode), 'Wrong type for [host], should be a string [was {0}]'.format(type(host))
        else:
            assert type(host) is str, 'Wrong type for [host], should be a string [was {0}]'.format(type(host))
        return self._scan_result['scan'][host]

    def all_hosts(self):
        if False:
            while True:
                i = 10
        '\n        returns a sorted list of all hosts\n        '
        if 'scan' not in list(self._scan_result.keys()):
            return []
        listh = list(self._scan_result['scan'].keys())
        listh.sort()
        return listh

    def command_line(self):
        if False:
            return 10
        '\n        returns command line used for the scan\n\n        may raise AssertionError exception if called before scanning\n        '
        assert 'nmap' in self._scan_result, 'Do a scan before trying to get result !'
        assert 'command_line' in self._scan_result['nmap'], 'Do a scan before trying to get result !'
        return self._scan_result['nmap']['command_line']

    def scaninfo(self):
        if False:
            i = 10
            return i + 15
        "\n        returns scaninfo structure\n        {'tcp': {'services': '22', 'method': 'connect'}}\n\n        may raise AssertionError exception if called before scanning\n        "
        assert 'nmap' in self._scan_result, 'Do a scan before trying to get result !'
        assert 'scaninfo' in self._scan_result['nmap'], 'Do a scan before trying to get result !'
        return self._scan_result['nmap']['scaninfo']

    def scanstats(self):
        if False:
            i = 10
            return i + 15
        "\n        returns scanstats structure\n        {'uphosts': '3', 'timestr': 'Thu Jun  3 21:45:07 2010', 'downhosts': '253', 'totalhosts': '256', 'elapsed': '5.79'}\n\n        may raise AssertionError exception if called before scanning\n        "
        assert 'nmap' in self._scan_result, 'Do a scan before trying to get result !'
        assert 'scanstats' in self._scan_result['nmap'], 'Do a scan before trying to get result !'
        return self._scan_result['nmap']['scanstats']

    def has_host(self, host):
        if False:
            while True:
                i = 10
        '\n        returns True if host has result, False otherwise\n        '
        assert type(host) is str, 'Wrong type for [host], should be a string [was {0}]'.format(type(host))
        assert 'scan' in self._scan_result, 'Do a scan before trying to get result !'
        if host in list(self._scan_result['scan'].keys()):
            return True
        return False

    def csv(self):
        if False:
            return 10
        '\n        returns CSV output as text\n\n        Example :\n        host;hostname;hostname_type;protocol;port;name;state;product;extrainfo;reason;version;conf;cpe\n        127.0.0.1;localhost;PTR;tcp;22;ssh;open;OpenSSH;protocol 2.0;syn-ack;5.9p1 Debian 5ubuntu1;10;cpe\n        127.0.0.1;localhost;PTR;tcp;23;telnet;closed;;;conn-refused;;3;\n        127.0.0.1;localhost;PTR;tcp;24;priv-mail;closed;;;conn-refused;;3;\n        '
        assert 'scan' in self._scan_result, 'Do a scan before trying to get result !'
        if sys.version_info < (3, 0):
            fd = io.BytesIO()
        else:
            fd = io.StringIO()
        csv_ouput = csv.writer(fd, delimiter=';')
        csv_header = ['host', 'hostname', 'hostname_type', 'protocol', 'port', 'name', 'state', 'product', 'extrainfo', 'reason', 'version', 'conf', 'cpe']
        csv_ouput.writerow(csv_header)
        for host in self.all_hosts():
            for proto in self[host].all_protocols():
                if proto not in ['tcp', 'udp']:
                    continue
                lport = list(self[host][proto].keys())
                lport.sort()
                for port in lport:
                    hostname = ''
                    for h in self[host]['hostnames']:
                        hostname = h['name']
                        hostname_type = h['type']
                        csv_row = [host, hostname, hostname_type, proto, port, self[host][proto][port]['name'], self[host][proto][port]['state'], self[host][proto][port]['product'], self[host][proto][port]['extrainfo'], self[host][proto][port]['reason'], self[host][proto][port]['version'], self[host][proto][port]['conf'], self[host][proto][port]['cpe']]
                        csv_ouput.writerow(csv_row)
        return fd.getvalue()

def __scan_progressive__(self, hosts, ports, arguments, callback, sudo):
    if False:
        i = 10
        return i + 15
    '\n    Used by PortScannerAsync for callback\n    '
    for host in self._nm.listscan(hosts):
        try:
            scan_data = self._nm.scan(host, ports, arguments, sudo)
        except PortScannerError:
            scan_data = None
        if callback is not None:
            callback(host, scan_data)
    return

class PortScannerAsync(object):
    """
    PortScannerAsync allows to use nmap from python asynchronously
    for each host scanned, callback is called with scan result for the host

    """

    def __init__(self):
        if False:
            return 10
        '\n        Initialize the module\n\n        * detects nmap on the system and nmap version\n        * may raise PortScannerError exception if nmap is not found in the path\n\n        '
        self._process = None
        self._nm = PortScanner()
        return

    def __del__(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Cleanup when deleted\n\n        '
        if self._process is not None:
            try:
                if self._process.is_alive():
                    self._process.terminate()
            except AssertionError:
                pass
        self._process = None
        return

    def scan(self, hosts='127.0.0.1', ports=None, arguments='-sV', callback=None, sudo=False):
        if False:
            i = 10
            return i + 15
        "\n        Scan given hosts in a separate process and return host by host result using callback function\n\n        PortScannerError exception from standard nmap is catched and you won't know about but get None as scan_data\n\n        :param hosts: string for hosts as nmap use it 'scanme.nmap.org' or '198.116.0-255.1-127' or '216.163.128.20/20'\n        :param ports: string for ports as nmap use it '22,53,110,143-4564'\n        :param arguments: string of arguments for nmap '-sU -sX -sC'\n        :param callback: callback function which takes (host, scan_data) as arguments\n        :param sudo: launch nmap with sudo if true\n        "
        if sys.version_info[0] == 2:
            assert type(hosts) in (str, unicode), 'Wrong type for [hosts], should be a string [was {0}]'.format(type(hosts))
            assert type(ports) in (str, unicode, type(None)), 'Wrong type for [ports], should be a string [was {0}]'.format(type(ports))
            assert type(arguments) in (str, unicode), 'Wrong type for [arguments], should be a string [was {0}]'.format(type(arguments))
        else:
            assert type(hosts) is str, 'Wrong type for [hosts], should be a string [was {0}]'.format(type(hosts))
            assert type(ports) in (str, type(None)), 'Wrong type for [ports], should be a string [was {0}]'.format(type(ports))
            assert type(arguments) is str, 'Wrong type for [arguments], should be a string [was {0}]'.format(type(arguments))
        assert callable(callback) or callback is None, 'The [callback] {0} should be callable or None.'.format(str(callback))
        for redirecting_output in ['-oX', '-oA']:
            assert redirecting_output not in arguments, "Xml output can't be redirected from command line.\nYou can access it after a scan using:\nnmap.nm.get_nmap_last_output()"
        self._process = Process(target=__scan_progressive__, args=(self, hosts, ports, arguments, callback, sudo))
        self._process.daemon = True
        self._process.start()
        return

    def stop(self):
        if False:
            while True:
                i = 10
        '\n        Stop the current scan process\n\n        '
        if self._process is not None:
            self._process.terminate()
        return

    def wait(self, timeout=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Wait for the current scan process to finish, or timeout\n\n        :param timeout: default = None, wait timeout seconds\n\n        '
        assert type(timeout) in (int, type(None)), 'Wrong type for [timeout], should be an int or None [was {0}]'.format(type(timeout))
        self._process.join(timeout)
        return

    def still_scanning(self):
        if False:
            while True:
                i = 10
        '\n        :returns: True if a scan is currently running, False otherwise\n\n        '
        try:
            return self._process.is_alive()
        except:
            return False

class PortScannerYield(PortScannerAsync):
    """
    PortScannerYield allows to use nmap from python with a generator
    for each host scanned, yield is called with scan result for the host

    """

    def __init__(self):
        if False:
            print('Hello World!')
        '\n        Initialize the module\n\n        * detects nmap on the system and nmap version\n        * may raise PortScannerError exception if nmap is not found in the path\n\n        '
        PortScannerAsync.__init__(self)
        return

    def scan(self, hosts='127.0.0.1', ports=None, arguments='-sV', sudo=False):
        if False:
            while True:
                i = 10
        "\n        Scan given hosts in a separate process and return host by host result using callback function\n\n        PortScannerError exception from standard nmap is catched and you won't know about it\n\n        :param hosts: string for hosts as nmap use it 'scanme.nmap.org' or '198.116.0-255.1-127' or '216.163.128.20/20'\n        :param ports: string for ports as nmap use it '22,53,110,143-4564'\n        :param arguments: string of arguments for nmap '-sU -sX -sC'\n        :param callback: callback function which takes (host, scan_data) as arguments\n        :param sudo: launch nmap with sudo if true\n\n        "
        assert type(hosts) is str, 'Wrong type for [hosts], should be a string [was {0}]'.format(type(hosts))
        assert type(ports) in (str, type(None)), 'Wrong type for [ports], should be a string [was {0}]'.format(type(ports))
        assert type(arguments) is str, 'Wrong type for [arguments], should be a string [was {0}]'.format(type(arguments))
        for redirecting_output in ['-oX', '-oA']:
            assert redirecting_output not in arguments, "Xml output can't be redirected from command line.\nYou can access it after a scan using:\nnmap.nm.get_nmap_last_output()"
        for host in self._nm.listscan(hosts):
            try:
                scan_data = self._nm.scan(host, ports, arguments, sudo)
            except PortScannerError:
                scan_data = None
            yield (host, scan_data)
        return

    def stop(self):
        if False:
            while True:
                i = 10
        pass

    def wait(self, timeout=None):
        if False:
            return 10
        pass

    def still_scanning(self):
        if False:
            for i in range(10):
                print('nop')
        pass

class PortScannerHostDict(dict):
    """
    Special dictionnary class for storing and accessing host scan result

    """

    def hostnames(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        :returns: list of hostnames\n\n        '
        return self['hostnames']

    def hostname(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        For compatibility purpose...\n        :returns: try to return the user record or the first hostname of the list hostnames\n\n        '
        hostname = ''
        for h in self['hostnames']:
            if h['type'] == 'user':
                return h['name']
        else:
            if len(self['hostnames']) > 0 and 'name' in self['hostnames'][0]:
                return self['hostnames'][0]['name']
            else:
                return ''
        return hostname

    def state(self):
        if False:
            while True:
                i = 10
        '\n        :returns: host state\n\n        '
        return self['status']['state']

    def uptime(self):
        if False:
            while True:
                i = 10
        '\n        :returns: host state\n\n        '
        return self['uptime']

    def all_protocols(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        :returns: a list of all scanned protocols\n\n        '

        def _proto_filter(x):
            if False:
                i = 10
                return i + 15
            return x in ['ip', 'tcp', 'udp', 'sctp']
        lp = list(filter(_proto_filter, list(self.keys())))
        lp.sort()
        return lp

    def all_tcp(self):
        if False:
            print('Hello World!')
        '\n        :returns: list of tcp ports\n\n        '
        if 'tcp' in list(self.keys()):
            ltcp = list(self['tcp'].keys())
            ltcp.sort()
            return ltcp
        return []

    def has_tcp(self, port):
        if False:
            while True:
                i = 10
        '\n        :param port: (int) tcp port\n        :returns: True if tcp port has info, False otherwise\n\n        '
        assert type(port) is int, 'Wrong type for [port], should be an int [was {0}]'.format(type(port))
        if 'tcp' in list(self.keys()) and port in list(self['tcp'].keys()):
            return True
        return False

    def tcp(self, port):
        if False:
            while True:
                i = 10
        '\n        :param port: (int) tcp port\n        :returns: info for tpc port\n\n        '
        assert type(port) is int, 'Wrong type for [port], should be an int [was {0}]'.format(type(port))
        return self['tcp'][port]

    def all_udp(self):
        if False:
            while True:
                i = 10
        '\n        :returns: list of udp ports\n\n        '
        if 'udp' in list(self.keys()):
            ludp = list(self['udp'].keys())
            ludp.sort()
            return ludp
        return []

    def has_udp(self, port):
        if False:
            for i in range(10):
                print('nop')
        '\n        :param port: (int) udp port\n        :returns: True if udp port has info, False otherwise\n\n        '
        assert type(port) is int, 'Wrong type for [port], should be an int [was {0}]'.format(type(port))
        if 'udp' in list(self.keys()) and 'port' in list(self['udp'].keys()):
            return True
        return False

    def udp(self, port):
        if False:
            return 10
        '\n        :param port: (int) udp port\n        :returns: info for udp port\n\n        '
        assert type(port) is int, 'Wrong type for [port], should be an int [was {0}]'.format(type(port))
        return self['udp'][port]

    def all_ip(self):
        if False:
            while True:
                i = 10
        '\n        :returns: list of ip ports\n\n        '
        if 'ip' in list(self.keys()):
            lip = list(self['ip'].keys())
            lip.sort()
            return lip
        return []

    def has_ip(self, port):
        if False:
            return 10
        '\n        :param port: (int) ip port\n        :returns: True if ip port has info, False otherwise\n\n        '
        assert type(port) is int, 'Wrong type for [port], should be an int [was {0}]'.format(type(port))
        if 'ip' in list(self.keys()) and port in list(self['ip'].keys()):
            return True
        return False

    def ip(self, port):
        if False:
            for i in range(10):
                print('nop')
        '\n        :param port: (int) ip port\n        :returns: info for ip port\n\n        '
        assert type(port) is int, 'Wrong type for [port], should be an int [was {0}]'.format(type(port))
        return self['ip'][port]

    def all_sctp(self):
        if False:
            print('Hello World!')
        '\n        :returns: list of sctp ports\n\n        '
        if 'sctp' in list(self.keys()):
            lsctp = list(self['sctp'].keys())
            lsctp.sort()
            return lsctp
        return []

    def has_sctp(self, port):
        if False:
            for i in range(10):
                print('nop')
        '\n        :returns: True if sctp port has info, False otherwise\n\n        '
        assert type(port) is int, 'Wrong type for [port], should be an int [was {0}]'.format(type(port))
        if 'sctp' in list(self.keys()) and port in list(self['sctp'].keys()):
            return True
        return False

    def sctp(self, port):
        if False:
            i = 10
            return i + 15
        '\n        :returns: info for sctp port\n\n        '
        assert type(port) is int, 'Wrong type for [port], should be an int [was {0}]'.format(type(port))
        return self['sctp'][port]

class PortScannerError(Exception):
    """
    Exception error class for PortScanner class

    """

    def __init__(self, value):
        if False:
            print('Hello World!')
        self.value = value

    def __str__(self):
        if False:
            return 10
        return repr(self.value)

    def __repr__(self):
        if False:
            print('Hello World!')
        return 'PortScannerError exception {0}'.format(self.value)

def __get_last_online_version():
    if False:
        while True:
            i = 10
    "\n    Gets last python-nmap published version\n\n    WARNING : it does an http connection to http://xael.org/pages/python-nmap/python-nmap_CURRENT_VERSION.txt\n\n    :returns: a string which indicate last published version (example :'0.4.3')\n\n    "
    import http.client
    conn = http.client.HTTPConnection('xael.org')
    conn.request('GET', '/pages/python-nmap/python-nmap_CURRENT_VERSION.txt')
    online_version = bytes.decode(conn.getresponse().read()).strip()
    return online_version

def convert_nmap_output_to_encoding(value, code='ascii'):
    if False:
        while True:
            i = 10
    '\n    Change encoding for scan_result object from unicode to whatever\n\n    :param value: scan_result as dictionnary\n    :param code: default = "ascii", encoding destination\n\n    :returns: scan_result as dictionnary with new encoding\n    '
    new_value = {}
    for k in value:
        if type(value[k]) in [dict, PortScannerHostDict]:
            new_value[k] = convert_nmap_output_to_encoding(value[k], code)
        elif type(value[k]) is list:
            new_value[k] = [convert_nmap_output_to_encoding(x, code) for x in value[k]]
        else:
            new_value[k] = value[k].encode(code)
    return new_value