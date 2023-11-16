import os
import volatility.plugins.crashinfo as crashinfo
import volatility.utils as utils

class VMwareInfo(crashinfo.CrashInfo):
    """Dump VMware VMSS/VMSN information"""
    target_as = ['VMWareAddressSpace', 'VMWareMetaAddressSpace']

    def __init__(self, config, *args, **kwargs):
        if False:
            return 10
        crashinfo.CrashInfo.__init__(self, config, *args, **kwargs)
        config.add_option('DUMP-DIR', short_option='D', default=None, help='Directory in which to dump the screenshot (if available)')

    @staticmethod
    def is_valid_profile(profile):
        if False:
            return 10
        return True

    def render_text(self, outfd, data):
        if False:
            return 10
        header = data.get_header()
        outfd.write('Magic: {0:#x} (Version {1})\n'.format(header.Magic, header.Version))
        outfd.write('Group count: {0:#x}\n'.format(header.GroupCount))
        self.table_header(outfd, [('File Offset', '#018x'), ('PhysMem Offset', '#018x'), ('Size', '#018x')])
        for (memory_offset, file_offset, length) in data.get_runs():
            self.table_row(outfd, file_offset, memory_offset, length)
        outfd.write('\n')
        self.table_header(outfd, [('DataOffset', '#018x'), ('DataSize', '#018x'), ('Name', '50'), ('Value', '')])
        for group in header.Groups:
            for tag in group.Tags:
                indices = ''
                for i in tag.TagIndices:
                    indices += '[{0}]'.format(i)
                if tag.DataMemSize == 0:
                    value = ''
                elif tag.DataMemSize == 1:
                    value = '{0}'.format(tag.cast_as('unsigned char'))
                elif tag.DataMemSize == 2:
                    value = '{0}'.format(tag.cast_as('unsigned short'))
                elif tag.DataMemSize == 4:
                    value = '{0:#x}'.format(tag.cast_as('unsigned int'))
                elif tag.DataMemSize == 8:
                    value = '{0:#x}'.format(tag.cast_as('unsigned long long'))
                else:
                    value = ''
                self.table_row(outfd, tag.RealDataOffset, tag.DataMemSize, '{0}/{1}{2}'.format(group.Name, tag.Name, indices), value)
                if self._config.VERBOSE and tag.DataMemSize > 0 and (str(group.Name) != 'memory') and (value == ''):
                    addr = tag.RealDataOffset
                    data = tag.obj_vm.read(addr, tag.DataMemSize)
                    outfd.write(''.join(['{0:#010x}  {1:<48}  {2}\n'.format(addr + o, h, ''.join(c)) for (o, h, c) in utils.Hexdump(data)]))
                    if self._config.DUMP_DIR and str(group.Name) == 'MKSVMX' and (str(tag.Name) == 'imageData'):
                        full_path = os.path.join(self._config.DUMP_DIR, 'screenshot.png')
                        with open(full_path, 'wb') as fh:
                            fh.write(data)
                            outfd.write('Wrote screenshot to: {0}\n'.format(full_path))