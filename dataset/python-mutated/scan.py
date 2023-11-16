"""
@author:       AAron Walters
@license:      GNU General Public License 2.0 
@contact:      awalters@4tphi.net
@organization: Volatility Foundation
"""
import volatility.debug as debug
import volatility.registry as registry
import volatility.addrspace as addrspace
import volatility.constants as constants
import volatility.conf as conf

class BaseScanner(object):
    """ A more thorough scanner which checks every byte """
    checks = []

    def __init__(self, window_size=8):
        if False:
            for i in range(10):
                print('nop')
        self.buffer = addrspace.BufferAddressSpace(conf.DummyConfig(), data='\x00' * 1024)
        self.window_size = window_size
        self.constraints = []
        self.error_count = 0

    def check_addr(self, found):
        if False:
            for i in range(10):
                print('nop')
        ' This calls all our constraints on the offset found and\n        returns the number of contraints that matched.\n\n        We shortcut the loop as soon as its obvious that there will\n        not be sufficient matches to fit the criteria. This allows for\n        an early exit and a speed boost.\n        '
        cnt = 0
        for check in self.constraints:
            try:
                val = check.check(found)
            except Exception:
                debug.b()
                val = False
            if not val:
                cnt = cnt + 1
            if cnt > self.error_count:
                return False
        return True
    overlap = 20

    def scan(self, address_space, offset=0, maxlen=None):
        if False:
            while True:
                i = 10
        self.buffer.profile = address_space.profile
        current_offset = offset
        self.constraints = []
        for (class_name, args) in self.checks:
            check = registry.get_plugin_classes(ScannerCheck)[class_name](self.buffer, **args)
            self.constraints.append(check)
        skippers = [c for c in self.constraints if hasattr(c, 'skip')]
        for (range_start, range_size) in sorted(address_space.get_available_addresses()):
            current_offset = max(range_start, current_offset)
            range_end = range_start + range_size
            if maxlen:
                range_end = min(range_end, offset + maxlen)
            while current_offset < range_end:
                l = min(constants.SCAN_BLOCKSIZE + self.overlap, range_end - current_offset)
                data = address_space.zread(current_offset, l)
                self.buffer.assign_buffer(data, current_offset)
                i = 0
                while i < l:
                    if self.check_addr(i + current_offset):
                        yield (i + current_offset)
                    skip = 1
                    for s in skippers:
                        skip = max(skip, s.skip(data, i))
                    i += skip
                current_offset += min(constants.SCAN_BLOCKSIZE, l)

class DiscontigScanner(BaseScanner):

    def scan(self, address_space, offset=0, maxlen=None):
        if False:
            while True:
                i = 10
        debug.warning('DiscontigScanner has been deprecated, all functionality is now contained in BaseScanner')
        for match in BaseScanner.scan(self, address_space, offset, maxlen):
            yield match

class ScannerCheck(object):
    """ A scanner check is a special class which is invoked on an AS to check for a specific condition.

    The main method is def check(self, offset):
    This will return True if the condition is true or False otherwise.

    This class is the base class for all checks.
    """

    def __init__(self, address_space, **_kwargs):
        if False:
            return 10
        self.address_space = address_space

    def object_offset(self, offset, address_space):
        if False:
            i = 10
            return i + 15
        return offset

    def check(self, _offset):
        if False:
            for i in range(10):
                print('nop')
        return False