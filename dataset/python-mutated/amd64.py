import volatility.plugins.addrspaces.paged as paged
import volatility.obj as obj
import struct
ptrs_page = 2048
entry_size = 8
pde_shift = 21
ptrs_per_pde = 512
page_shift = 12
ptrs_per_pae_pgd = 512
ptrs_per_pae_pte = 512

class AMD64PagedMemory(paged.AbstractWritablePagedMemory):
    """ Standard AMD 64-bit address space.

    This class implements the AMD64/IA-32E paging address space. It is responsible
    for translating each virtual (linear) address to a physical address.
    This is accomplished using hierachical paging structures.
    Every paging structure is 4096 bytes and is composed of entries.
    Each entry is 64 bits.  The first paging structure is located at the
    physical address found in CR3 (dtb).

    Additional Resources:
     - Intel(R) 64 and IA-32 Architectures Software Developer's Manual
       Volume 3A: System Programming Guide. Section 4.3
       http://www.intel.com/products/processor/manuals/index.htm
     - AMD64 Architecture Programmer's Manual Volume 2: System Programming
       http://support.amd.com/us/Processor_TechDocs/24593_APM_v2.pdf
     - N. Petroni, A. Walters, T. Fraser, and W. Arbaugh, "FATKit: A Framework
       for the Extraction and Analysis of Digital Forensic Data from Volatile
       System Memory" ,Digital Investigation Journal 3(4):197-210, December 2006.
       (submitted February 2006)
     - N. P. Maclean, "Acquisition and Analysis of Windows Memory,"
       University of Strathclyde, Glasgow, April 2006.
     - Russinovich, M., & Solomon, D., & Ionescu, A.
       "Windows Internals, 5th Edition", Microsoft Press, 2009.
    """
    order = 60
    pae = False
    checkname = 'AMD64ValidAS'
    paging_address_space = True
    minimum_size = 4096
    alignment_gcd = 4096
    _longlong_struct = struct.Struct('<Q')
    skip_duplicate_entries = False

    def entry_present(self, entry):
        if False:
            for i in range(10):
                print('nop')
        return entry and entry & 1

    def page_size_flag(self, entry):
        if False:
            for i in range(10):
                print('nop')
        if entry & 1 << 7 == 1 << 7:
            return True
        return False

    def is_user_page(self, entry):
        if False:
            while True:
                i = 10
        return entry & 1 << 2 == 1 << 2

    def is_supervisor_page(self, entry):
        if False:
            return 10
        return not self.is_user_page(entry)

    def is_writeable(self, entry):
        if False:
            for i in range(10):
                print('nop')
        return entry & 1 << 1 == 1 << 1

    def is_dirty(self, entry):
        if False:
            print('Hello World!')
        return entry & 1 << 6 == 1 << 6

    def is_nx(self, entry):
        if False:
            return 10
        return entry & 1 << 63 == 1 << 63

    def is_accessed(self, entry):
        if False:
            i = 10
            return i + 15
        return entry & 1 << 5 == 1 << 5

    def is_copyonwrite(self, entry):
        if False:
            return 10
        return entry & 1 << 9 == 1 << 9

    def is_prototype(self, entry):
        if False:
            for i in range(10):
                print('nop')
        return entry & 1 << 10 == 1 << 10

    def get_2MB_paddr(self, vaddr, pgd_entry):
        if False:
            print('Hello World!')
        paddr = pgd_entry & 4503599625273344 | vaddr & 2097151
        return paddr

    def is_valid_profile(self, profile):
        if False:
            while True:
                i = 10
        '\n        This method checks to make sure the address space is being\n        used with a supported profile.\n        '
        return profile.metadata.get('memory_model', '32bit') == '64bit' or profile.metadata.get('os', 'Unknown').lower() == 'mac'

    def pml4e_index(self, vaddr):
        if False:
            while True:
                i = 10
        '\n        This method returns the Page Map Level 4 Entry Index\n        number from the given  virtual address. The index number is\n        in bits 47:39.\n        '
        return (vaddr & 280925220896768) >> 39

    def get_pml4e(self, vaddr):
        if False:
            return 10
        '\n        This method returns the Page Map Level 4 (PML4) entry for the\n        virtual address. Bits 47:39 are used to the select the\n        appropriate 8 byte entry in the Page Map Level 4 Table.\n\n        "Bits 51:12 are from CR3" [Intel]\n        "Bits 11:3 are bits 47:39 of the linear address" [Intel]\n        "Bits 2:0 are 0" [Intel]\n        '
        pml4e_paddr = self.dtb & 4503599627366400 | (vaddr & 280925220896768) >> 36
        return self.read_long_long_phys(pml4e_paddr)

    def get_pdpi(self, vaddr, pml4e):
        if False:
            i = 10
            return i + 15
        '\n        This method returns the Page Directory Pointer entry for the\n        virtual address. Bits 32:30 are used to select the appropriate\n        8 byte entry in the Page Directory Pointer table.\n\n        "Bits 51:12 are from the PML4E" [Intel]\n        "Bits 11:3 are bits 38:30 of the linear address" [Intel]\n        "Bits 2:0 are all 0" [Intel]\n        '
        pdpte_paddr = pml4e & 4503599627366400 | (vaddr & 548682072064) >> 27
        return self.read_long_long_phys(pdpte_paddr)

    def get_1GB_paddr(self, vaddr, pdpte):
        if False:
            for i in range(10):
                print('nop')
        '\n        If the Page Directory Pointer Table entry represents a 1-GByte\n        page, this method extracts the physical address of the page.\n\n        "Bits 51:30 are from the PDPTE" [Intel]\n        "Bits 29:0 are from the original linear address" [Intel]\n        '
        return pdpte & 4503598553628672 | vaddr & 1073741823

    def pde_index(self, vaddr):
        if False:
            return 10
        return vaddr >> pde_shift & ptrs_per_pde - 1

    def pdba_base(self, pdpe):
        if False:
            i = 10
            return i + 15
        return pdpe & 4503599627366400

    def get_pgd(self, vaddr, pdpe):
        if False:
            for i in range(10):
                print('nop')
        pgd_entry = self.pdba_base(pdpe) + self.pde_index(vaddr) * entry_size
        return self.read_long_long_phys(pgd_entry)

    def pte_index(self, vaddr):
        if False:
            return 10
        return vaddr >> page_shift & ptrs_per_pde - 1

    def ptba_base(self, pde):
        if False:
            return 10
        return pde & 4503599627366400

    def get_pte(self, vaddr, pgd):
        if False:
            return 10
        pgd_val = self.ptba_base(pgd) + self.pte_index(vaddr) * entry_size
        return self.read_long_long_phys(pgd_val)

    def pte_pfn(self, pte):
        if False:
            print('Hello World!')
        return pte & 4503599627366400

    def get_paddr(self, vaddr, pte):
        if False:
            i = 10
            return i + 15
        return self.pte_pfn(pte) | vaddr & (1 << page_shift) - 1

    def vtop(self, vaddr):
        if False:
            for i in range(10):
                print('nop')
        '\n        This method translates an address in the virtual\n        address space to its associated physical address.\n        Invalid entries should be handled with operating\n        system abstractions.\n        '
        vaddr = long(vaddr)
        retVal = None
        pml4e = self.get_pml4e(vaddr)
        if not self.entry_present(pml4e):
            return None
        pdpe = self.get_pdpi(vaddr, pml4e)
        if not self.entry_present(pdpe):
            return retVal
        if self.page_size_flag(pdpe):
            return self.get_1GB_paddr(vaddr, pdpe)
        pgd = self.get_pgd(vaddr, pdpe)
        if self.entry_present(pgd):
            if self.page_size_flag(pgd):
                retVal = self.get_2MB_paddr(vaddr, pgd)
            else:
                pte = self.get_pte(vaddr, pgd)
                if self.entry_present(pte):
                    retVal = self.get_paddr(vaddr, pte)
        return retVal

    def read_long_long_phys(self, addr):
        if False:
            i = 10
            return i + 15
        '\n        This method returns a 64-bit little endian\n        unsigned integer from the specified address in the\n        physical address space. If the address cannot be accessed,\n        then the method returns None.\n\n        This code was derived directly from legacyintel.py\n        '
        try:
            string = self.base.read(addr, 8)
        except IOError:
            string = None
        if not string:
            return obj.NoneObject('Unable to read_long_long_phys at ' + hex(addr))
        (longlongval,) = self._longlong_struct.unpack(string)
        return longlongval

    def get_available_pages(self, with_pte=False):
        if False:
            print('Hello World!')
        '\n        This method generates a list of pages that are\n        available within the address space. The entries in\n        are composed of the virtual address of the page\n        and the size of the particular page (address, size).\n        It walks the 0x1000/0x8 (0x200) entries in each Page Map,\n        Page Directory, and Page Table to determine which pages\n        are accessible.\n        '
        pml4 = self.base.read(self.dtb & 4503599627366400, 512 * 8)
        if pml4 is None:
            return
        pml4_entries = struct.unpack('<512Q', pml4)
        for pml4e in range(0, 512):
            vaddr = pml4e << 39
            pml4e_value = pml4_entries[pml4e]
            if not self.entry_present(pml4e_value):
                continue
            pdpt_base = pml4e_value & 4503599627366400
            pdpt = self.base.read(pdpt_base, 512 * 8)
            if pdpt is None:
                continue
            pdpt_entries = struct.unpack('<512Q', pdpt)
            for pdpte in range(0, 512):
                vaddr = pml4e << 39 | pdpte << 30
                pdpte_value = pdpt_entries[pdpte]
                if not self.entry_present(pdpte_value):
                    continue
                if self.page_size_flag(pdpte_value):
                    if with_pte:
                        yield (pdpte_value, vaddr, 1073741824)
                    else:
                        yield (vaddr, 1073741824)
                    continue
                pd_base = self.pdba_base(pdpte_value)
                pd = self.base.read(pd_base, 512 * 8)
                if pd is None:
                    continue
                pd_entries = struct.unpack('<512Q', pd)
                prev_pd_entry = None
                for j in range(0, 512):
                    soffset = j * 512 * 512 * 8
                    entry = pd_entries[j]
                    if self.skip_duplicate_entries and entry == prev_pd_entry:
                        continue
                    prev_pd_entry = entry
                    if self.entry_present(entry) and self.page_size_flag(entry):
                        if with_pte:
                            yield (entry, vaddr + soffset, 2097152)
                        else:
                            yield (vaddr + soffset, 2097152)
                    elif self.entry_present(entry):
                        pt_base = entry & 4503599627366400
                        pt = self.base.read(pt_base, 512 * 8)
                        if pt is None:
                            continue
                        pt_entries = struct.unpack('<512Q', pt)
                        prev_pt_entry = None
                        for k in range(0, 512):
                            pt_entry = pt_entries[k]
                            if self.skip_duplicate_entries and pt_entry == prev_pt_entry:
                                continue
                            prev_pt_entry = pt_entry
                            if self.entry_present(pt_entry):
                                if with_pte:
                                    yield (pt_entry, vaddr + soffset + k * 4096, 4096)
                                else:
                                    yield (vaddr + soffset + k * 4096, 4096)

    @classmethod
    def address_mask(cls, addr):
        if False:
            print('Hello World!')
        return addr & 281474976710655

class WindowsAMD64PagedMemory(AMD64PagedMemory):
    """Windows-specific AMD 64-bit address space.

    This class is a specialized version of AMD64PagedMemory that leverages
    Windows-specific paging logic.
    """
    order = 55

    def is_valid_profile(self, profile):
        if False:
            return 10
        '\n        This method checks to make sure the address space is being\n        used with a Windows profile.\n        '
        valid = AMD64PagedMemory.is_valid_profile(self, profile)
        return valid and profile.metadata.get('os', 'Unknown').lower() == 'windows'

    def entry_present(self, entry):
        if False:
            i = 10
            return i + 15
        present = AMD64PagedMemory.entry_present(self, entry)
        return present or (entry & 1 << 11 and (not entry & 1 << 10))

class SkipDuplicatesAMD64PagedMemory(WindowsAMD64PagedMemory):
    """Windows 8/10-specific AMD 64-bit address space.

    This class is used to filter out large sections of kernel mappings that are
    duplicates in recent versions of Windows 8/10.
    """
    order = 53
    skip_duplicate_entries = True

    def is_valid_profile(self, profile):
        if False:
            i = 10
            return i + 15
        '\n        This address space should only be used with recent Windows 8/10 profiles\n        '
        valid = WindowsAMD64PagedMemory.is_valid_profile(self, profile)
        major = profile.metadata.get('major', 0)
        minor = profile.metadata.get('minor', 0)
        return valid and major >= 6 and (minor >= 2)

class LinuxAMD64PagedMemory(AMD64PagedMemory):
    """Linux-specific AMD 64-bit address space.

    This class is a specialized version of AMD64PagedMemory that leverages
    Linux-specific paging logic.
    """
    order = 55

    def is_valid_profile(self, profile):
        if False:
            for i in range(10):
                print('nop')
        '\n        This method checks to make sure the address space is being\n        used with a Linux profile.\n        '
        valid = AMD64PagedMemory.is_valid_profile(self, profile)
        return valid and profile.metadata.get('os', 'Unknown').lower() == 'linux'

    def entry_present(self, entry):
        if False:
            return 10
        present = AMD64PagedMemory.entry_present(self, entry)
        return present or entry & 1 << 8