"""
Provides PEFile, a class for reading MS portable executable files.

Primary doc sources:
http://www.csn.ul.ie/~caolan/pub/winresdump/winresdump/doc/pefile2.html
http://en.wikibooks.org/wiki/X86_Disassembly/Windows_Executable_Files
"""
from __future__ import annotations
import typing
from .....util.filelike.stream import StreamFragment
from .....util.struct import NamedStruct
if typing.TYPE_CHECKING:
    from openage.convert.value_object.read.media.peresource import PEResources
    from openage.util.fslike.wrapper import GuardedFile

class PEDOSHeader(NamedStruct):
    """
    The (legacy) DOS-compatible PE header.

    In all modern PE files, only the 'lfanew' pointer is relevant.
    """
    endianness = '<'
    signature = '2s'
    bytes_lastpage = 'H'
    count_pages = 'H'
    crlc = 'H'
    cparhdr = 'H'
    minalloc = 'H'
    maxalloc = 'H'
    initial_ss = 'H'
    initial_sp = 'H'
    checksum = 'H'
    initial_ip = 'H'
    initial_cs = 'H'
    lfarlc = 'H'
    ovno = 'H'
    reserved0 = '8s'
    oemid = 'H'
    oeminfo = 'H'
    reserved1 = '20s'
    coffheaderpos = 'I'

class PECOFFHeader(NamedStruct):
    """
    The new (win32) PE and object file header.
    """
    endianness = '<'
    signature = '4s'
    machine = 'H'
    number_of_sections = 'H'
    time_stamp = 'I'
    symbol_table_ptr = 'I'
    symbol_count = 'I'
    opt_header_size = 'H'
    characteristics = 'H'

class PEOptionalHeader(NamedStruct):
    """
    This "optional" header is required for linked files (but not object files).
    """
    endianness = '<'
    signature = 'H'
    major_linker_ver = 'B'
    minor_linker_ver = 'B'
    size_of_code = 'I'
    size_of_data = 'I'
    size_of_bss = 'I'
    entry_point_addr = 'I'
    base_of_code = 'I'
    base_of_data = 'I'
    image_base = 'I'
    section_alignment = 'I'
    file_alignment = 'I'
    major_os_ver = 'H'
    minor_os_ver = 'H'
    major_img_ver = 'H'
    minor_img_ver = 'H'
    major_subsys_ver = 'H'
    minor_subsys_ver = 'H'
    reserved = 'I'
    size_of_image = 'I'
    size_of_headers = 'I'
    checksum = 'I'
    subsystem = 'H'
    dll_characteristics = 'H'
    stack_reserve_size = 'I'
    stack_commit_size = 'I'
    heap_reserve_size = 'I'
    heap_commit_size = 'I'
    loader_flags = 'I'
    data_directory_count = 'I'
    data_directories = None

class PEDataDirectory(NamedStruct):
    """
    Provides the locations of various metadata structures,
    which are used to set up the execution environment.
    """
    endianness = '<'
    rva = 'I'
    size = 'I'

class PESection(NamedStruct):
    """
    Describes a section in a PE file (like an ELF section).
    """
    endianness = '<'
    name = '8s'
    virtual_size = 'I'
    virtual_address = 'I'
    size_on_disk = 'I'
    file_offset = 'I'
    reserved = '12s'
    flags = 'I'

class PEFile:
    """
    Reads Microsoft PE files.

    The constructor takes a file-like object.
    """

    def __init__(self, fileobj: GuardedFile):
        if False:
            print('Hello World!')
        doshdr = PEDOSHeader.read(fileobj)
        if doshdr.signature != b'MZ':
            raise SyntaxError('not a PE file')
        fileobj.seek(doshdr.coffheaderpos)
        coffhdr = PECOFFHeader.read(fileobj)
        if coffhdr.signature != b'PE\x00\x00':
            raise SyntaxError('not a Win32 PE file')
        if coffhdr.opt_header_size != 224:
            raise SyntaxError('unknown optional header size')
        opthdr = PEOptionalHeader.read(fileobj)
        if opthdr.signature not in {267, 523}:
            raise SyntaxError('Not an x86{_64} file')
        opthdr.data_directories = []
        for _ in range(opthdr.data_directory_count):
            opthdr.data_directories.append(PEDataDirectory.read(fileobj))
        sections: dict[str, tuple] = {}
        for _ in range(coffhdr.number_of_sections):
            section = PESection.read(fileobj)
            section.name = section.name.decode('ascii').rstrip('\x00')
            if not section.name.startswith('.'):
                raise SyntaxError('Invalid section name: ' + section.name)
            sections[section.name] = section
        self.fileobj = fileobj
        self.doshdr = doshdr
        self.coffhdr = coffhdr
        self.opthdr = opthdr
        self.sections = sections

    def open_section(self, section_name: str) -> StreamFragment:
        if False:
            print('Hello World!')
        '\n        Returns a tuple of data, va for the given section.\n\n        data is a file-like object (StreamFragment),\n        and va is the RVA of the section start.\n        '
        if section_name not in self.sections:
            raise SyntaxError('no such section in PE file: ' + section_name)
        section = self.sections[section_name]
        return (StreamFragment(self.fileobj, section.file_offset, section.virtual_size), section.virtual_address)

    def resources(self) -> PEResources:
        if False:
            while True:
                i = 10
        '\n        Returns a PEResources object for self.\n        '
        from .peresource import PEResources
        return PEResources(self)