import volatility.obj as obj
import volatility.scan as scan
import volatility.cache as cache
import volatility.plugins.common as common
import volatility.addrspace as addrspace
import volatility.registry as registry
import volatility.utils as utils
import volatility.exceptions as exceptions

class MultiStringFinderCheck(scan.ScannerCheck):
    """ Checks for multiple strings per page """

    def __init__(self, address_space, needles=None):
        if False:
            print('Hello World!')
        scan.ScannerCheck.__init__(self, address_space)
        if not needles:
            needles = []
        self.needles = needles
        self.maxlen = 0
        for needle in needles:
            self.maxlen = max(self.maxlen, len(needle))
        if not self.maxlen:
            raise RuntimeError('No needles of any length were found for the ' + self.__class__.__name__)

    def check(self, offset):
        if False:
            i = 10
            return i + 15
        verify = self.address_space.read(offset, self.maxlen)
        for match in self.needles:
            if verify[:len(match)] == match:
                return True
        return False

    def skip(self, data, offset):
        if False:
            print('Hello World!')
        nextval = len(data)
        for needle in self.needles:
            dindex = data.find(needle, offset + 1)
            if dindex > -1:
                nextval = min(nextval, dindex)
        return nextval - offset

class MultiPrefixFinderCheck(MultiStringFinderCheck):
    """ Checks for multiple strings per page, finishing at the offset """

    def check(self, offset):
        if False:
            while True:
                i = 10
        verify = self.address_space.read(offset - self.maxlen, self.maxlen)
        for match in self.needles:
            if verify.endswith(match):
                return True
        return False

class KDBGScanner(scan.BaseScanner):
    checks = []

    def __init__(self, window_size=8, needles=None):
        if False:
            print('Hello World!')
        oses = set()
        arches = set()
        for needle in needles:
            header = str(needle).split('KDBG')
            arches.add(header[0])
            oses.add('KDBG' + header[1])
        self.checks = [('PoolTagCheck', {'tag': 'KDBG'}), ('MultiPrefixFinderCheck', {'needles': arches}), ('MultiStringFinderCheck', {'needles': oses})]
        scan.BaseScanner.__init__(self, window_size)

    def scan(self, address_space, offset=0, maxlen=None):
        if False:
            i = 10
            return i + 15
        for offset in scan.BaseScanner.scan(self, address_space, offset, maxlen):
            offset = offset - 16
            yield offset

class KDBGScan(common.AbstractWindowsCommand):
    """Search for and dump potential KDBG values"""

    @staticmethod
    def register_options(config):
        if False:
            for i in range(10):
                print('nop')
        config.add_option('KDBG', short_option='g', default=None, type='int', help='Specify a KDBG virtual address (Note: for 64-bit Windows 8 and above this is the address of KdCopyDataBlock)')
        config.add_option('FORCE', default=False, action='store_true', help='Force utilization of suspect profile')

    @cache.CacheDecorator(lambda self: 'tests/kdbgscan/kdbg={0}'.format(self._config.KDBG))
    def calculate(self):
        if False:
            while True:
                i = 10
        'Determines the address space'
        profilelist = [p.__name__ for p in registry.get_plugin_classes(obj.Profile).values()]
        encrypted_kdbg_profiles = []
        proflens = {}
        maxlen = 0
        origprofile = self._config.PROFILE
        for p in profilelist:
            self._config.update('PROFILE', p)
            buf = addrspace.BufferAddressSpace(self._config)
            if buf.profile.metadata.get('os', 'unknown') == 'windows':
                proflens[p] = str(obj.VolMagic(buf).KDBGHeader)
                maxlen = max(maxlen, len(proflens[p]))
                if buf.profile.metadata.get('memory_model', '64bit') == '64bit' and (buf.profile.metadata.get('major', 0), buf.profile.metadata.get('minor', 0)) >= (6, 2):
                    encrypted_kdbg_profiles.append(p)
        self._config.update('PROFILE', origprofile)
        count = 0
        if origprofile not in encrypted_kdbg_profiles:
            scanner = KDBGScanner(needles=proflens.values())
            aspace = utils.load_as(self._config, astype='any')
            suspects = []
            for offset in scanner.scan(aspace):
                val = aspace.read(offset, maxlen + 16)
                for l in proflens:
                    if val.find(proflens[l]) >= 0:
                        kdbg = obj.Object('_KDDEBUGGER_DATA64', offset=offset, vm=aspace)
                        suspects.append((l, kdbg))
                        count += 1
            for (p, k) in suspects:
                if not self._config.FORCE:
                    yield (p, k)
                    continue
                self._config.update('PROFILE', p)
                nspace = utils.load_as(self._config, astype='any')
                for offset in scanner.scan(nspace):
                    val = nspace.read(offset, maxlen + 16)
                    if val.find(proflens[p]) >= 0:
                        kdbg = obj.Object('_KDDEBUGGER_DATA64', offset=offset, vm=nspace)
                        yield (p, kdbg)
            self._config.update('PROFILE', origprofile)
        if count == 0:
            if origprofile in encrypted_kdbg_profiles:
                encrypted_kdbg_profiles = [origprofile]
            for profile in encrypted_kdbg_profiles:
                self._config.update('PROFILE', profile)
                aspace = utils.load_as(self._config, astype='any')
                if hasattr(aspace, 'vtop'):
                    for kdbg in obj.VolMagic(aspace).KDBG.generate_suggestions():
                        yield (profile, kdbg)

    def render_text(self, outfd, data):
        if False:
            for i in range(10):
                print('nop')
        'Renders the KPCR values as text'
        for (profile, kdbg) in data:
            outfd.write('*' * 50 + '\n')
            outfd.write('Instantiating KDBG using: {0} {1} ({2}.{3}.{4} {5})\n'.format(kdbg.obj_vm.name, kdbg.obj_vm.profile.__class__.__name__, kdbg.obj_vm.profile.metadata.get('major', 0), kdbg.obj_vm.profile.metadata.get('minor', 0), kdbg.obj_vm.profile.metadata.get('build', 0), kdbg.obj_vm.profile.metadata.get('memory_model', '32bit')))
            has_vtop = hasattr(kdbg.obj_native_vm, 'vtop')
            if has_vtop:
                outfd.write('{0:<30}: {1:#x}\n'.format('Offset (V)', kdbg.obj_offset))
                outfd.write('{0:<30}: {1:#x}\n'.format('Offset (P)', kdbg.obj_native_vm.vtop(kdbg.obj_offset)))
            else:
                outfd.write('{0:<30}: {1:#x}\n'.format('Offset (P)', kdbg.obj_offset))
            if hasattr(kdbg, 'KdCopyDataBlock'):
                outfd.write('{0:<30}: {1:#x}\n'.format('KdCopyDataBlock (V)', kdbg.KdCopyDataBlock))
            if hasattr(kdbg, 'block_encoded'):
                outfd.write('{0:<30}: {1}\n'.format('Block encoded', 'Yes' if kdbg.block_encoded == 1 else 'No'))
            if hasattr(kdbg, 'wait_never'):
                outfd.write('{0:<30}: {1:#x}\n'.format('Wait never', kdbg.wait_never))
            if hasattr(kdbg, 'wait_always'):
                outfd.write('{0:<30}: {1:#x}\n'.format('Wait always', kdbg.wait_always))
            outfd.write('{0:<30}: {1}\n'.format('KDBG owner tag check', str(kdbg.is_valid())))
            outfd.write('{0:<30}: {1}\n'.format('Profile suggestion (KDBGHeader)', profile))
            verinfo = kdbg.dbgkd_version64()
            if verinfo:
                outfd.write('{0:<30}: {1:#x} (Major: {2}, Minor: {3})\n'.format('Version64', verinfo.obj_offset, verinfo.MajorVersion, verinfo.MinorVersion))
            if has_vtop:
                outfd.write('{0:<30}: {1}\n'.format('Service Pack (CmNtCSDVersion)', kdbg.ServicePack))
                outfd.write('{0:<30}: {1}\n'.format('Build string (NtBuildLab)', kdbg.NtBuildLab.dereference()))
                try:
                    num_tasks = len(list(kdbg.processes()))
                except AttributeError:
                    num_tasks = 0
                try:
                    num_modules = len(list(kdbg.modules()))
                except AttributeError:
                    num_modules = 0
                cpu_blocks = list(kdbg.kpcrs())
                outfd.write('{0:<30}: {1:#x} ({2} processes)\n'.format('PsActiveProcessHead', kdbg.PsActiveProcessHead, num_tasks))
                outfd.write('{0:<30}: {1:#x} ({2} modules)\n'.format('PsLoadedModuleList', kdbg.PsLoadedModuleList, num_modules))
                outfd.write('{0:<30}: {1:#x} (Matches MZ: {2})\n'.format('KernelBase', kdbg.KernBase, str(kdbg.obj_native_vm.read(kdbg.KernBase, 2) == 'MZ')))
                try:
                    dos_header = obj.Object('_IMAGE_DOS_HEADER', offset=kdbg.KernBase, vm=kdbg.obj_native_vm)
                    nt_header = dos_header.get_nt_header()
                except (ValueError, exceptions.SanityCheckException):
                    pass
                else:
                    outfd.write('{0:<30}: {1}\n'.format('Major (OptionalHeader)', nt_header.OptionalHeader.MajorOperatingSystemVersion))
                    outfd.write('{0:<30}: {1}\n'.format('Minor (OptionalHeader)', nt_header.OptionalHeader.MinorOperatingSystemVersion))
                for kpcr in cpu_blocks:
                    outfd.write('{0:<30}: {1:#x} (CPU {2})\n'.format('KPCR', kpcr.obj_offset, kpcr.ProcessorBlock.Number))
            else:
                outfd.write('{0:<30}: {1:#x}\n'.format('PsActiveProcessHead', kdbg.PsActiveProcessHead))
                outfd.write('{0:<30}: {1:#x}\n'.format('PsLoadedModuleList', kdbg.PsLoadedModuleList))
                outfd.write('{0:<30}: {1:#x}\n'.format('KernelBase', kdbg.KernBase))
            outfd.write('\n')