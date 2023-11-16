import volatility.addrspace as addrspace
import volatility.obj as obj

class AbstractPagedMemory(addrspace.AbstractVirtualAddressSpace):
    """ Class to handle all the details of a paged virtual address space
        
    Note: Pages can be of any size
    """
    checkname = 'Intel'

    def __init__(self, base, config, dtb=0, skip_as_check=False, *args, **kwargs):
        if False:
            while True:
                i = 10
        self.as_assert(base, 'No base Address Space')
        addrspace.AbstractVirtualAddressSpace.__init__(self, base, config, *args, **kwargs)
        self.as_assert(not (hasattr(base, 'paging_address_space') and base.paging_address_space), 'Can not stack over another paging address space')
        self.dtb = dtb or self.load_dtb()
        self.as_assert(self.dtb != None, 'No valid DTB found')
        if not skip_as_check:
            volmag = obj.VolMagic(self)
            if hasattr(volmag, self.checkname):
                self.as_assert(getattr(volmag, self.checkname).v(), 'Failed valid Address Space check')
            else:
                self.as_assert(False, 'Profile does not have valid Address Space check')
        self.name = 'Kernel AS'

    def is_user_page(self, entry):
        if False:
            i = 10
            return i + 15
        'True if the page is accessible to ring 3 code'
        raise NotImplementedError

    def is_supervisor_page(self, entry):
        if False:
            for i in range(10):
                print('nop')
        'True if the page is /only/ accessible to ring 0 code'
        raise NotImplementedError

    def is_writeable(self, entry):
        if False:
            while True:
                i = 10
        'True if the page can be written to'
        raise NotImplementedError

    def is_dirty(self, entry):
        if False:
            print('Hello World!')
        'True if the page has been written to'
        raise NotImplementedError

    def is_nx(self, entry):
        if False:
            for i in range(10):
                print('nop')
        'True if the page /cannot/ be executed'
        raise NotImplementedError

    def is_accessed(self, entry):
        if False:
            print('Hello World!')
        'True if the page has been accessed'
        raise NotImplementedError

    def is_copyonwrite(self, entry):
        if False:
            return 10
        'True if the page is copy-on-write'
        raise NotImplementedError

    def is_prototype(self, entry):
        if False:
            print('Hello World!')
        'True if the page is a prototype PTE'
        raise NotImplementedError

    def load_dtb(self):
        if False:
            print('Hello World!')
        'Loads the DTB as quickly as possible from the config, then the base, then searching for it'
        try:
            if self._config.DTB:
                raise AttributeError
            return self.base.dtb
        except AttributeError:
            dtb = obj.VolMagic(self.base).DTB.v()
            if dtb:
                self.base.dtb = dtb
                return dtb

    def __getstate__(self):
        if False:
            i = 10
            return i + 15
        result = addrspace.BaseAddressSpace.__getstate__(self)
        result['dtb'] = self.dtb
        return result

    @staticmethod
    def register_options(config):
        if False:
            while True:
                i = 10
        config.add_option('DTB', type='int', default=0, help='DTB Address')

    def vtop(self, addr):
        if False:
            return 10
        'Abstract function that converts virtual (paged) addresses to physical addresses'
        pass

    def get_available_pages(self):
        if False:
            while True:
                i = 10
        'A generator that returns (addr, size) for each of the virtual addresses present, sorted by offset'
        pass

    def get_available_allocs(self):
        if False:
            while True:
                i = 10
        return self.get_available_pages()

    def get_available_addresses(self):
        if False:
            for i in range(10):
                print('nop')
        'A generator that returns (addr, size) for each valid address block'
        runLength = None
        currentOffset = None
        for (offset, size) in self.get_available_pages():
            if runLength == None:
                runLength = size
                currentOffset = offset
            elif offset <= currentOffset + runLength:
                runLength += currentOffset + runLength - offset + size
            else:
                yield (currentOffset, runLength)
                runLength = size
                currentOffset = offset
        if runLength != None and currentOffset != None:
            yield (currentOffset, runLength)
        raise StopIteration

    def is_valid_address(self, vaddr):
        if False:
            for i in range(10):
                print('nop')
        'Returns whether a virtual address is valid'
        if vaddr == None or vaddr < 0:
            return False
        try:
            paddr = self.vtop(vaddr)
        except BaseException:
            return False
        if paddr == None:
            return False
        return self.base.is_valid_address(paddr)

class AbstractWritablePagedMemory(AbstractPagedMemory):
    """
    Mixin class that can be used to add write functionality
    to any standard address space that supports write() and
    vtop().
    """

    def write(self, vaddr, buf):
        if False:
            return 10
        'Writes the data from buf to the vaddr specified\n        \n           Note: writes are not transactionaly, meaning if they can write half the data and then fail'
        if not self._config.WRITE:
            return False
        if not self.alignment_gcd or not self.minimum_size:
            self.calculate_alloc_stats()
        position = vaddr
        length = len(buf)
        remaining = len(buf)
        while remaining > 0:
            alloc_remaining = self.alignment_gcd - vaddr % self.alignment_gcd
            paddr = self.translate(position)
            datalen = min(remaining, alloc_remaining)
            if paddr is None:
                return False
            result = self.base.write(paddr, buf[:datalen])
            if not result:
                return False
            buf = buf[datalen:]
            position += datalen
            remaining -= datalen
            assert vaddr + length == position + remaining, 'Address + length != position + remaining (' + hex(vaddr + length) + ' != ' + hex(position + remaining) + ') in ' + self.base.__class__.__name__
        return True