"""
Private support for parsing /etc/hosts.

"""
from __future__ import absolute_import, division, print_function
import sys
import os
import re
from gevent.resolver._addresses import is_ipv4_addr
from gevent.resolver._addresses import is_ipv6_addr
from gevent._compat import iteritems

class HostsFile(object):
    """
    A class to read the contents of a hosts file (/etc/hosts).
    """
    LINES_RE = re.compile('\n        \\s*  # Leading space\n        ([^\\r\\n#]+?)  # The actual match, non-greedy so as not to include trailing space\n        \\s*  # Trailing space\n        (?:[#][^\\r\\n]+)?  # Comments\n        (?:$|[\\r\\n]+)  # EOF or newline\n    ', re.VERBOSE)

    def __init__(self, fname=None):
        if False:
            print('Hello World!')
        self.v4 = {}
        self.v6 = {}
        self.aliases = {}
        self.reverse = {}
        if fname is None:
            if os.name == 'posix':
                fname = '/etc/hosts'
            elif os.name == 'nt':
                fname = os.path.expandvars('%SystemRoot%\\system32\\drivers\\etc\\hosts')
        self.fname = fname
        assert self.fname
        self._last_load = 0

    def _readlines(self):
        if False:
            return 10
        with open(self.fname, 'rb') as fp:
            fdata = fp.read()
        udata = fdata.decode(errors='ignore') if not isinstance(fdata, str) else fdata
        return self.LINES_RE.findall(udata)

    def load(self):
        if False:
            while True:
                i = 10
        try:
            load_time = os.stat(self.fname).st_mtime
            needs_load = load_time > self._last_load
        except OSError:
            from gevent import get_hub
            get_hub().handle_error(self, *sys.exc_info())
            needs_load = False
        if not needs_load:
            return
        v4 = {}
        v6 = {}
        aliases = {}
        reverse = {}
        for line in self._readlines():
            parts = line.split()
            if len(parts) < 2:
                continue
            ip = parts.pop(0)
            if is_ipv4_addr(ip):
                ipmap = v4
            elif is_ipv6_addr(ip):
                if ip.startswith('fe80'):
                    continue
                ipmap = v6
            else:
                continue
            cname = parts.pop(0).lower()
            ipmap[cname] = ip
            for alias in parts:
                alias = alias.lower()
                ipmap[alias] = ip
                aliases[alias] = cname
            if ipmap is v4:
                ptr = '.'.join(reversed(ip.split('.'))) + '.in-addr.arpa'
            else:
                ptr = ip + '.ip6.arpa.'
            if ptr not in reverse:
                reverse[ptr] = cname
        self._last_load = load_time
        self.v4 = v4
        self.v6 = v6
        self.aliases = aliases
        self.reverse = reverse

    def iter_all_host_addr_pairs(self):
        if False:
            print('Hello World!')
        self.load()
        for (name, addr) in iteritems(self.v4):
            yield (name, addr)
        for (name, addr) in iteritems(self.v6):
            yield (name, addr)