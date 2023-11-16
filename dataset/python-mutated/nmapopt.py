"""This sub-module is responsible for generating Nmap options."""
from argparse import ArgumentParser
from shlex import quote
from typing import Dict, Iterable, List, Optional
from ivre import config
ARGPARSER = ArgumentParser(add_help=False)
ARGPARSER.add_argument('--nmap-template', help='Select Nmap scan template', choices=config.NMAP_SCAN_TEMPLATES, default='default')
NMAP_OPT_PORTS: Dict[Optional[str], List[str]] = {None: [], 'fast': ['-F'], 'more': ['--top-ports', '2000'], 'all': ['-p', '-']}

class Scan:

    def __init__(self, nmap: str='nmap', pings: str='SE', scans: str='SV', osdetect: bool=True, traceroute: bool=True, resolve: int=1, verbosity: int=2, ports: Optional[str]=None, top_ports: Optional[int]=None, host_timeout: Optional[str]=None, script_timeout: Optional[str]=None, scripts_categories: Optional[Iterable[str]]=None, scripts_exclude: Optional[Iterable[str]]=None, scripts_force: Optional[Iterable[str]]=None, extra_options: Optional[Iterable[str]]=None) -> None:
        if False:
            i = 10
            return i + 15
        self.nmap = nmap
        self.pings = set(pings)
        self.scans = set(scans)
        self.osdetect = osdetect
        self.traceroute = traceroute
        self.resolve = resolve
        self.verbosity = verbosity
        self.ports = ports
        self.top_ports = top_ports
        self.host_timeout = host_timeout
        self.script_timeout = script_timeout
        if self.ports:
            self.top_ports = None
        if scripts_categories is None:
            self.scripts_categories: Iterable[str] = []
        else:
            self.scripts_categories = scripts_categories
        if scripts_exclude is None:
            self.scripts_exclude: Iterable[str] = []
        else:
            self.scripts_exclude = scripts_exclude
        if scripts_force is None:
            self.scripts_force: Iterable[str] = []
        else:
            self.scripts_force = scripts_force
        self.extra_options = extra_options

    @property
    def options(self) -> List[str]:
        if False:
            while True:
                i = 10
        options = [self.nmap]
        if ('C' in self.scans or self.scripts_categories or self.scripts_exclude or self.scripts_force) and 'V' in self.scans and self.osdetect and self.traceroute:
            options.append('-A')
            self.scans.difference_update('CV')
            self.osdetect = False
            self.traceroute = False
        scripts = ''
        if self.scripts_categories:
            scripts = ' or '.join(self.scripts_categories)
        if self.scripts_exclude:
            if scripts:
                scripts = '(%s) and not (%s)' % (scripts if scripts else '', ' or '.join(self.scripts_exclude))
            else:
                scripts = 'not (%s)' % ' or '.join(self.scripts_exclude)
        if self.scripts_force:
            if scripts:
                scripts = '(%s) or %s' % (scripts if scripts else '', ' or '.join(self.scripts_force))
            else:
                scripts = ' or '.join(self.scripts_force)
        if scripts == 'default':
            scripts = ''
            if '-A' not in options:
                self.scans.add('C')
        elif scripts and 'C' in self.scans:
            self.scans.remove('C')
        options.extend(('-P%s' % x for x in self.pings))
        options.extend(('-s%s' % x for x in self.scans))
        if self.osdetect:
            options.append('-O')
        if self.traceroute:
            options.append('--traceroute')
        if not self.resolve:
            options.append('-n')
        elif self.resolve == 2:
            options.append('-R')
        if self.verbosity:
            options.append('-%s' % ('v' * self.verbosity))
        options.extend(NMAP_OPT_PORTS.get(self.ports, ['-p', self.ports]))
        if self.top_ports is not None:
            options.extend(['--top-ports', str(self.top_ports)])
        if self.host_timeout is not None:
            options.extend(['--host-timeout', self.host_timeout])
        if self.script_timeout is not None:
            options.extend(['--script-timeout', self.script_timeout])
        if scripts:
            options.extend(['--script', scripts])
        if self.extra_options:
            options.extend(self.extra_options)
        return options

def build_nmap_options(template: str='default') -> List[str]:
    if False:
        while True:
            i = 10
    return Scan(**config.NMAP_SCAN_TEMPLATES[template]).options

def build_nmap_commandline(template: str='default') -> str:
    if False:
        for i in range(10):
            print('nop')
    return ' '.join((quote(elt) for elt in build_nmap_options(template=template)))