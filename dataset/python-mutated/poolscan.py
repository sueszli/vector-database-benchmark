import volatility.scan as scan
import volatility.constants as constants
import volatility.utils as utils
import volatility.obj as obj
import volatility.registry as registry

class MultiPoolScanner(object):
    """An optimized scanner for pool tags"""

    def __init__(self, needles=None):
        if False:
            print('Hello World!')
        self.needles = needles
        self.overlap = 20

    def scan(self, address_space, offset=None, maxlen=None):
        if False:
            while True:
                i = 10
        if offset is None:
            current_offset = 0
        else:
            current_offset = offset
        for (range_start, range_size) in sorted(address_space.get_available_addresses()):
            current_offset = max(range_start, current_offset)
            range_end = range_start + range_size
            if maxlen is not None:
                range_end = min(range_end, current_offset + maxlen)
            while current_offset < range_end:
                l = min(constants.SCAN_BLOCKSIZE + self.overlap, range_end - current_offset)
                data = address_space.zread(current_offset, l)
                for needle in self.needles:
                    for addr in utils.iterfind(data, needle):
                        yield (data[addr:addr + 4], addr + current_offset)
                current_offset += min(constants.SCAN_BLOCKSIZE, l)

class MultiScanInterface(object):
    """An interface into a scanner that can find multiple pool tags
    in a single pass through an address space."""

    def __init__(self, addr_space, scanners=[], scan_virtual=False, show_unalloc=False, use_top_down=False, start_offset=None, max_length=None):
        if False:
            for i in range(10):
                print('nop')
        'An interface into the multiple concurrent pool scanner. \n\n        @param addr_space: a Volatility address space\n        \n        @param scanners: a list of PoolScanner classes to scan for. \n\n        @param scan_virtual: True to scan in virtual/kernel space \n        or False to scan at the physical layer.\n\n        @param show_unalloc: True to skip unallocated objects whose\n        _OBJECT_TYPE structure are 0xbad0b0b0. \n\n        @param use_topdown: True to carve objects out of the pool using\n        the top-down approach or False to use the bottom-up trick.\n\n        @param start_offset: the starting offset to begin scanning. \n\n        @param max_length: the size in bytes to scan from the start. \n        '
        self.scanners = scanners
        self.scan_virtual = scan_virtual
        self.show_unalloc = show_unalloc
        self.use_top_down = use_top_down
        self.start_offset = start_offset
        self.max_length = max_length
        self.address_space = addr_space
        self.pool_alignment = obj.VolMagic(self.address_space).PoolAlignment.v()

    def _check_pool_size(self, check, pool_header):
        if False:
            return 10
        'An alternate to the existing CheckPoolSize class. \n\n        This prevents us from create a second copy of the \n        _POOL_HEADER object which is quite unnecessary. \n        \n        @param check: a dictionary of arguments for the check\n\n        @param pool_header: the target _POOL_HEADER to check\n        '
        condition = check['condition']
        block_size = pool_header.BlockSize.v()
        return condition(block_size * self.pool_alignment)

    def _check_pool_type(self, check, pool_header):
        if False:
            while True:
                i = 10
        'An alternate to the existing CheckPoolType class. \n\n        This prevents us from create a second copy of the \n        _POOL_HEADER object which is quite unnecessary. \n        \n        @param check: a dictionary of arguments for the check\n\n        @param pool_header: the target _POOL_HEADER to check\n        '
        try:
            paged = check['paged']
        except KeyError:
            paged = False
        try:
            non_paged = check['non_paged']
        except KeyError:
            non_paged = False
        try:
            free = check['free']
        except KeyError:
            free = False
        return non_paged and pool_header.NonPagedPool or (free and pool_header.FreePool) or (paged and pool_header.PagedPool)

    def _check_pool_index(self, check, pool_header):
        if False:
            for i in range(10):
                print('nop')
        'An alternate to the existing CheckPoolIndex class. \n\n        This prevents us from create a second copy of the \n        _POOL_HEADER object which is quite unnecessary. \n        \n        @param check: a dictionary of arguments for the check\n\n        @param pool_header: the target _POOL_HEADER to check\n        '
        value = check['value']
        if callable(value):
            return value(pool_header.PoolIndex)
        else:
            return pool_header.PoolIndex == check['value']

    def _run_all_checks(self, checks, pool_header):
        if False:
            for i in range(10):
                print('nop')
        'Execute all constraint checks. \n\n        @param checks: a dictionary with check names as keys and \n        another dictionary of arguments as the values. \n\n        @param pool_header: the target _POOL_HEADER to check\n\n        @returns False if any checks fail, otherwise True. \n        '
        for (check, args) in checks:
            if check == 'CheckPoolSize':
                if not self._check_pool_size(args, pool_header):
                    return False
            elif check == 'CheckPoolType':
                if not self._check_pool_type(args, pool_header):
                    return False
            elif check == 'CheckPoolIndex':
                if not self._check_pool_index(args, pool_header):
                    return False
            else:
                custom_check = registry.get_plugin_classes(scan.ScannerCheck)[check](pool_header.obj_vm, **args)
                return custom_check.check(pool_header.PoolTag.obj_offset)
        return True

    def scan(self):
        if False:
            i = 10
            return i + 15
        meta = self.address_space.profile.metadata
        win10 = (meta.get('major'), meta.get('minor')) == (6, 4)
        if self.scan_virtual or win10:
            space = self.address_space
        else:
            space = self.address_space.physical_space()
        if win10:
            cookie = obj.VolMagic(space).ObHeaderCookie.v()
        scanners = [scanner(space) for scanner in self.scanners]
        needles = dict(((scanner.pooltag, scanner) for scanner in scanners))
        scanner = MultiPoolScanner(needles=[scanner.pooltag for scanner in scanners])
        pool_tag_offset = space.profile.get_obj_offset('_POOL_HEADER', 'PoolTag')
        for (tag, offset) in scanner.scan(address_space=space, offset=self.start_offset, maxlen=self.max_length):
            pool = obj.Object('_POOL_HEADER', offset=offset - pool_tag_offset, vm=space, native_vm=self.address_space)
            scanobj = needles[tag]
            if not self._run_all_checks(checks=scanobj.checks, pool_header=pool):
                continue
            use_top_down = scanobj.use_top_down or self.use_top_down
            skip_type_check = scanobj.skip_type_check or self.show_unalloc
            result = pool.get_object(struct_name=scanobj.struct_name, object_type=scanobj.object_type, use_top_down=use_top_down, skip_type_check=skip_type_check)
            if scanobj.padding > 0:
                result = obj.Object(scanobj.struct_name, offset=result.obj_offset + scanobj.padding, vm=result.obj_vm, native_vm=result.obj_native_vm)
            if result.is_valid():
                yield result

class PoolScanner(object):
    """A generic pool scanner class"""

    def __init__(self, address_space):
        if False:
            print('Hello World!')
        self.address_space = address_space
        self.struct_name = ''
        self.object_type = ''
        self.use_top_down = False
        self.skip_type_check = False
        self.pooltag = None
        self.checks = []
        self.padding = 0

class PoolTagCheck(scan.ScannerCheck):
    """ This scanner checks for the occurance of a pool tag """

    def __init__(self, address_space, tag=None, **kwargs):
        if False:
            while True:
                i = 10
        scan.ScannerCheck.__init__(self, address_space, **kwargs)
        self.tag = tag

    def skip(self, data, offset):
        if False:
            i = 10
            return i + 15
        try:
            nextval = data.index(self.tag, offset + 1)
            return nextval - offset
        except ValueError:
            return len(data) - offset

    def check(self, offset):
        if False:
            for i in range(10):
                print('nop')
        data = self.address_space.read(offset, len(self.tag))
        return data == self.tag

class CheckPoolType(scan.ScannerCheck):
    """ Check the pool type """

    def __init__(self, address_space, paged=False, non_paged=False, free=False, **kwargs):
        if False:
            print('Hello World!')
        scan.ScannerCheck.__init__(self, address_space, **kwargs)
        self.non_paged = non_paged
        self.paged = paged
        self.free = free

    def check(self, offset):
        if False:
            i = 10
            return i + 15
        pool_hdr = obj.Object('_POOL_HEADER', vm=self.address_space, offset=offset - 4)
        return self.non_paged and pool_hdr.NonPagedPool or (self.free and pool_hdr.FreePool) or (self.paged and pool_hdr.PagedPool)

class CheckPoolSize(scan.ScannerCheck):
    """ Check pool block size """

    def __init__(self, address_space, condition=lambda x: x == 8, **kwargs):
        if False:
            print('Hello World!')
        scan.ScannerCheck.__init__(self, address_space, **kwargs)
        self.condition = condition

    def check(self, offset):
        if False:
            return 10
        pool_hdr = obj.Object('_POOL_HEADER', vm=self.address_space, offset=offset - 4)
        block_size = pool_hdr.BlockSize.v()
        pool_alignment = obj.VolMagic(self.address_space).PoolAlignment.v()
        return self.condition(block_size * pool_alignment)

class SinglePoolScanner(scan.BaseScanner):

    def object_offset(self, found, address_space):
        if False:
            return 10
        ' \n        The name of this function "object_offset" can be misleading depending\n        on how its used. Even before removing the preambles (r1324), it may not\n        always return the offset of an object. Here are the rules:\n\n        If you subclass PoolScanner and do not override this function, it \n        will return the offset of _POOL_HEADER. If you do override this function,\n        it should be used to calculate and return the offset of your desired \n        object within the pool. Thus there are two different ways it can be done. \n\n        Example 1. \n\n        For an example of subclassing PoolScanner and not overriding this function, \n        see filescan.PoolScanFile. In this case, the plugin (filescan.FileScan) \n        treats the offset returned by this function as the start of _POOL_HEADER \n        and then works out the object from the bottom up: \n\n            for offset in PoolScanFile().scan(address_space):\n                pool_obj = obj.Object("_POOL_HEADER", vm = address_space,\n                     offset = offset)\n                ##\n                ## Work out objects base here\n                ## \n\n        Example 2. \n\n        For an example of subclassing PoolScanner and overriding this function, \n        see filescan.PoolScanProcess. In this case, the "work" described above is\n        done here (in the sublcassed object_offset). Thus in the plugin (filescan.PSScan)\n        it can directly instantiate _EPROCESS from the offset we return. \n\n            for offset in PoolScanProcess().scan(address_space):\n                eprocess = obj.Object(\'_EPROCESS\', vm = address_space,\n                        native_vm = kernel_as, offset = offset)\n        '
        return found - self.buffer.profile.get_obj_offset('_POOL_HEADER', 'PoolTag')

    def scan(self, address_space, offset=0, maxlen=None):
        if False:
            while True:
                i = 10
        for i in scan.BaseScanner.scan(self, address_space, offset, maxlen):
            yield self.object_offset(i, address_space)