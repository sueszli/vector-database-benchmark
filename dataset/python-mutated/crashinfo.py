import volatility.utils as utils
import volatility.plugins.common as common
import volatility.cache as cache
import volatility.debug as debug
import volatility.obj as obj
from volatility.renderers import TreeGrid
from volatility.renderers.basic import Address
import datetime

class _DMP_HEADER(obj.CType):
    """A class for crash dumps"""

    @property
    def SystemUpTime(self):
        if False:
            i = 10
            return i + 15
        'Returns a string uptime'
        if self.m('SystemUpTime') == 4992030524978970960:
            return obj.NoneObject('No uptime recorded')
        msec = self.m('SystemUpTime') / 10
        return datetime.timedelta(microseconds=msec)

class CrashInfoModification(obj.ProfileModification):
    """Applies overlays for crash dump headers"""
    conditions = {'os': lambda x: x == 'windows'}
    before = ['WindowsVTypes', 'WindowsObjectClasses']

    def modification(self, profile):
        if False:
            i = 10
            return i + 15
        profile.merge_overlay({'_DMP_HEADER': [None, {'Comment': [None, ['String', dict(length=128)]], 'DumpType': [None, ['Enumeration', dict(choices={1: 'Full Dump', 2: 'Kernel Dump', 5: 'BitMap Dump'})]], 'SystemTime': [None, ['WinTimeStamp', dict(is_utc=True)]]}], '_DMP_HEADER64': [None, {'Comment': [None, ['String', dict(length=128)]], 'DumpType': [None, ['Enumeration', dict(choices={1: 'Full Dump', 2: 'Kernel Dump', 5: 'BitMap Dump'})]], 'SystemTime': [None, ['WinTimeStamp', dict(is_utc=True)]]}]})
        profile.object_classes.update({'_DMP_HEADER': _DMP_HEADER, '_DMP_HEADER64': _DMP_HEADER})

class CrashInfo(common.AbstractWindowsCommand):
    """Dump crash-dump information"""
    target_as = ['WindowsCrashDumpSpace32', 'WindowsCrashDumpSpace64', 'WindowsCrashDumpSpace64BitMap']

    @cache.CacheDecorator('tests/crashinfo')
    def calculate(self):
        if False:
            i = 10
            return i + 15
        'Determines the address space'
        addr_space = utils.load_as(self._config, astype='physical')
        result = None
        adrs = addr_space
        while adrs:
            if adrs.__class__.__name__ in self.target_as:
                result = adrs
            adrs = adrs.base
        if result is None:
            debug.error('Memory Image could not be identified as {0}'.format(self.target_as))
        return result

    def unified_output(self, data):
        if False:
            print('Hello World!')
        return TreeGrid([('HeaderName', str), ('Majorversion', Address), ('Minorversion', Address), ('KdSecondaryVersion', Address), ('DirectoryTableBase', Address), ('PfnDataBase', Address), ('PsLoadedModuleList', Address), ('PsActiveProcessHead', Address), ('MachineImageType', Address), ('NumberProcessors', Address), ('BugCheckCode', Address), ('PaeEnabled', Address), ('KdDebuggerDataBlock', Address), ('ProductType', Address), ('SuiteMask', Address), ('WriterStatus', Address), ('Comment', str), ('DumpType', str), ('SystemTime', str), ('SystemUpTime', str), ('NumRuns', int)], self.generator(data))

    def generator(self, data):
        if False:
            print('Hello World!')
        hdr = data.get_header()
        pae = -1
        if hdr.obj_name != '_DMP_HEADER64':
            pae = hdr.PaeEnabled
        yield (0, [str(hdr.obj_name), Address(hdr.MajorVersion), Address(hdr.MinorVersion), Address(hdr.KdSecondaryVersion), Address(hdr.DirectoryTableBase), Address(hdr.PfnDataBase), Address(hdr.PsLoadedModuleList), Address(hdr.PsActiveProcessHead), Address(hdr.MachineImageType), Address(hdr.NumberProcessors), Address(hdr.BugCheckCode), Address(pae), Address(hdr.KdDebuggerDataBlock), Address(hdr.ProductType), Address(hdr.SuiteMask), Address(hdr.WriterStatus), str(hdr.Comment), str(hdr.DumpType), str(hdr.SystemTime or ''), str(hdr.SystemUpTime or ''), len(data.get_runs())])

    def render_text(self, outfd, data):
        if False:
            print('Hello World!')
        'Renders the crashdump header as text'
        hdr = data.get_header()
        runs = data.get_runs()
        outfd.write('{0}:\n'.format(hdr.obj_name))
        outfd.write(' Majorversion:         0x{0:08x} ({1})\n'.format(hdr.MajorVersion, hdr.MajorVersion))
        outfd.write(' Minorversion:         0x{0:08x} ({1})\n'.format(hdr.MinorVersion, hdr.MinorVersion))
        outfd.write(' KdSecondaryVersion    0x{0:08x}\n'.format(hdr.KdSecondaryVersion))
        outfd.write(' DirectoryTableBase    0x{0:08x}\n'.format(hdr.DirectoryTableBase))
        outfd.write(' PfnDataBase           0x{0:08x}\n'.format(hdr.PfnDataBase))
        outfd.write(' PsLoadedModuleList    0x{0:08x}\n'.format(hdr.PsLoadedModuleList))
        outfd.write(' PsActiveProcessHead   0x{0:08x}\n'.format(hdr.PsActiveProcessHead))
        outfd.write(' MachineImageType      0x{0:08x}\n'.format(hdr.MachineImageType))
        outfd.write(' NumberProcessors      0x{0:08x}\n'.format(hdr.NumberProcessors))
        outfd.write(' BugCheckCode          0x{0:08x}\n'.format(hdr.BugCheckCode))
        if hdr.obj_name != '_DMP_HEADER64':
            outfd.write(' PaeEnabled            0x{0:08x}\n'.format(hdr.PaeEnabled))
        outfd.write(' KdDebuggerDataBlock   0x{0:08x}\n'.format(hdr.KdDebuggerDataBlock))
        outfd.write(' ProductType           0x{0:08x}\n'.format(hdr.ProductType))
        outfd.write(' SuiteMask             0x{0:08x}\n'.format(hdr.SuiteMask))
        outfd.write(' WriterStatus          0x{0:08x}\n'.format(hdr.WriterStatus))
        outfd.write(' Comment               {0}\n'.format(hdr.Comment))
        outfd.write(' DumpType              {0}\n'.format(hdr.DumpType))
        outfd.write(' SystemTime            {0}\n'.format(str(hdr.SystemTime or '')))
        outfd.write(' SystemUpTime          {0}\n'.format(str(hdr.SystemUpTime or '')))
        outfd.write('\nPhysical Memory Description:\n')
        outfd.write('Number of runs: {0}\n'.format(len(runs)))
        outfd.write('FileOffset    Start Address    Length\n')
        foffset = 4096
        if hdr.obj_name == '_DMP_HEADER64':
            foffset = 8192
        run = []
        for run in runs:
            outfd.write('{0:08x}      {1:08x}         {2:08x}\n'.format(foffset, run[0], run[2]))
            foffset += run[2]
        outfd.write('{0:08x}      {1:08x}\n'.format(foffset - 4096, run[0] + run[2] - 4096))