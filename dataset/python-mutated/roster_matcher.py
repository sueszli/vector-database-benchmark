"""
Roster matching by various criteria (glob, pcre, etc)
"""
import copy
import fnmatch
import functools
import logging
import re
HAS_RANGE = False
try:
    import seco.range
    HAS_RANGE = True
except ImportError:
    pass
log = logging.getLogger(__name__)

def targets(conditioned_raw, tgt, tgt_type, ipv='ipv4'):
    if False:
        print('Hello World!')
    rmatcher = RosterMatcher(conditioned_raw, tgt, tgt_type, ipv)
    return rmatcher.targets()

def _tgt_set(tgt):
    if False:
        for i in range(10):
            print('nop')
    '\n    Return the tgt as a set of literal names\n    '
    try:
        return set(tgt.split(','))
    except AttributeError:
        return set(tgt)

class RosterMatcher:
    """
    Matcher for the roster data structure
    """

    def __init__(self, raw, tgt, tgt_type, ipv='ipv4'):
        if False:
            while True:
                i = 10
        self.tgt = tgt
        self.tgt_type = tgt_type
        self.raw = raw
        self.ipv = ipv

    def targets(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Execute the correct tgt_type routine and return\n        '
        try:
            return getattr(self, 'ret_{}_minions'.format(self.tgt_type))()
        except AttributeError:
            return {}

    def _ret_minions(self, filter_):
        if False:
            while True:
                i = 10
        '\n        Filter minions by a generic filter.\n        '
        minions = {}
        for minion in filter_(self.raw):
            data = self.get_data(minion)
            if data:
                minions[minion] = data.copy()
        return minions

    def ret_glob_minions(self):
        if False:
            return 10
        '\n        Return minions that match via glob\n        '
        fnfilter = functools.partial(fnmatch.filter, pat=self.tgt)
        return self._ret_minions(fnfilter)

    def ret_pcre_minions(self):
        if False:
            print('Hello World!')
        '\n        Return minions that match via pcre\n        '
        tgt = re.compile(self.tgt)
        refilter = functools.partial(filter, tgt.match)
        return self._ret_minions(refilter)

    def ret_list_minions(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return minions that match via list\n        '
        tgt = _tgt_set(self.tgt)
        return self._ret_minions(tgt.intersection)

    def ret_nodegroup_minions(self):
        if False:
            i = 10
            return i + 15
        '\n        Return minions which match the special list-only groups defined by\n        ssh_list_nodegroups\n        '
        nodegroup = __opts__.get('ssh_list_nodegroups', {}).get(self.tgt, [])
        nodegroup = _tgt_set(nodegroup)
        return self._ret_minions(nodegroup.intersection)

    def ret_range_minions(self):
        if False:
            while True:
                i = 10
        '\n        Return minions that are returned by a range query\n        '
        if HAS_RANGE is False:
            raise RuntimeError("Python lib 'seco.range' is not available")
        minions = {}
        range_hosts = _convert_range_to_list(self.tgt, __opts__['range_server'])
        return self._ret_minions(range_hosts.__contains__)

    def get_data(self, minion):
        if False:
            return 10
        '\n        Return the configured ip\n        '
        ret = copy.deepcopy(__opts__.get('roster_defaults', {}))
        if isinstance(self.raw[minion], str):
            ret.update({'host': self.raw[minion]})
            return ret
        elif isinstance(self.raw[minion], dict):
            ret.update(self.raw[minion])
            return ret
        return False

def _convert_range_to_list(tgt, range_server):
    if False:
        while True:
            i = 10
    '\n    convert a seco.range range into a list target\n    '
    r = seco.range.Range(range_server)
    try:
        return r.expand(tgt)
    except seco.range.RangeException as err:
        log.error('Range server exception: %s', err)
        return []