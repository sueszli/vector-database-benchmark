from volatility import renderers
from volatility.commands import Command
import volatility.plugins.crashinfo as crashinfo
from volatility.renderers.basic import Address, Hex

class VBoxInfo(crashinfo.CrashInfo):
    """Dump virtualbox information"""
    target_as = ['VirtualBoxCoreDumpElf64']

    def unified_output(self, data):
        if False:
            i = 10
            return i + 15
        return renderers.TreeGrid([('FileOffset', Address), ('Memory Offset', Address), ('Size', Hex)], self.generator(data))

    def generator(self, data):
        if False:
            i = 10
            return i + 15
        for (memory_offset, file_offset, length) in data.get_runs():
            yield (0, [Address(file_offset), Address(memory_offset), Hex(length)])

    def render_text(self, outfd, data):
        if False:
            for i in range(10):
                print('nop')
        header = data.get_header()
        outfd.write('Magic: {0:#x}\n'.format(header.u32Magic))
        outfd.write('Format: {0:#x}\n'.format(header.u32FmtVersion))
        outfd.write('VirtualBox {0}.{1}.{2} (revision {3})\n'.format(header.Major, header.Minor, header.Build, header.u32VBoxRevision))
        outfd.write('CPUs: {0}\n\n'.format(header.cCpus))
        Command.render_text(self, outfd, data)

class QemuInfo(VBoxInfo):
    """Dump Qemu information"""
    target_as = ['QemuCoreDumpElf']

    def render_text(self, outfd, data):
        if False:
            while True:
                i = 10
        Command.render_text(self, outfd, data)