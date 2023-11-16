import volatility.plugins.crashinfo as crashinfo

class MachOInfo(crashinfo.CrashInfo):
    """Dump Mach-O file format information"""
    target_as = ['MachOAddressSpace']

    def render_text(self, outfd, data):
        if False:
            return 10
        header = data.get_header()
        outfd.write('Magic: {0:#x}\n'.format(header.magic))
        outfd.write('Architecture: {0}-bit\n'.format(data.bits))
        self.table_header(outfd, [('File Offset', '[addrpad]'), ('Memory Offset', '[addrpad]'), ('Size', '[addrpad]'), ('Name', '')])
        for seg in data.segs:
            self.table_row(outfd, seg.fileoff, seg.vmaddr, seg.vmsize, seg.segname)