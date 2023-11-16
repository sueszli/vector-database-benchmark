import re, copy
import sys, os
import zipfile
import struct
import time
import string
from operator import attrgetter
import volatility.plugins as plugins
import volatility.debug as debug
import volatility.obj as obj
import volatility.plugins.overlays.basic as basic
import volatility.addrspace as addrspace
import volatility.scan as scan
import volatility.plugins.addrspaces.amd64 as amd64
import volatility.plugins.addrspaces.intel as intel
import volatility.plugins.overlays.native_types as native_types
import volatility.utils as utils
import volatility.plugins.mac.common as common
import volatility.plugins.malware.malfind as malfind
try:
    import yara
    has_yara = True
except ImportError:
    has_yara = False
x64_native_types = copy.deepcopy(native_types.x64_native_types)
x64_native_types['long'] = [8, '<q']
x64_native_types['unsigned long'] = [8, '<Q']
dynamic_env_hint = None
dyld_vtypes = {'dyld32_image_info': [12, {'imageLoadAddress': [0, ['pointer', ['unsigned int']]], 'imageFilePath': [4, ['pointer', ['char']]], 'imageFileModDate': [8, ['pointer', ['unsigned int']]]}], 'dyld32_all_image_infos': [20, {'version': [0, ['unsigned int']], 'infoArrayCount': [4, ['unsigned int']], 'infoArray': [8, ['pointer', ['dyld32_image_info']]], 'notification': [12, ['pointer', ['void']]], 'processDetachedFromSharedRegion': [16, ['unsigned int']]}], 'dyld64_image_info': [24, {'imageLoadAddress': [0, ['pointer', ['unsigned int']]], 'imageFilePath': [8, ['pointer', ['char']]], 'imageFileModDate': [16, ['pointer', ['unsigned int']]]}], 'dyld64_all_image_infos': [28, {'version': [0, ['unsigned int']], 'infoArrayCount': [4, ['unsigned int']], 'infoArray': [8, ['pointer', ['dyld64_image_info']]], 'notification': [16, ['pointer', ['void']]], 'processDetachedFromSharedRegion': [24, ['unsigned int']]}]}

class BashEnvYaraScanner(malfind.BaseYaraScanner):
    """A scanner over all memory regions of a process."""

    def __init__(self, task=None, **kwargs):
        if False:
            return 10
        'Scan the process address space through the VMAs.\n\n        Args:\n          task: The task_struct object for this task.\n        '
        self.task = task
        malfind.BaseYaraScanner.__init__(self, address_space=task.get_process_address_space(), **kwargs)

    def scan(self, offset=0, maxlen=None, max_size=None):
        if False:
            print('Hello World!')
        shared_start = self.task.task.shared_region.sr_base_address
        shared_end = shared_start + self.task.task.shared_region.sr_size
        for map in self.task.get_proc_maps():
            start = map.links.start.v()
            end = map.links.end.v()
            length = end - start
            if length >= 16777216:
                continue
            if shared_start <= start <= shared_end:
                continue
            if map.get_perms() != 'rw-' or map.get_path() != '':
                continue
            for match in malfind.BaseYaraScanner.scan(self, start, length):
                yield match

class DyldTypes(obj.ProfileModification):
    conditions = {'os': lambda x: x in ['mac']}

    def modification(self, profile):
        if False:
            for i in range(10):
                print('nop')
        profile.vtypes.update(dyld_vtypes)
mig_vtypes_32 = {'mig_hash_entry': [16, {'num': [0, ['int']], 'routine': [4, ['pointer', ['void']]], 'size': [8, ['int']], 'callcount': [12, ['unsigned int']]}]}
mig_vtypes_64 = {'mig_hash_entry': [24, {'num': [0, ['int']], 'routine': [8, ['pointer', ['void']]], 'size': [16, ['int']], 'callcount': [20, ['unsigned int']]}]}

class MigTypes(obj.ProfileModification):
    conditions = {'os': lambda x: x in ['mac']}

    def modification(self, profile):
        if False:
            i = 10
            return i + 15
        if profile.metadata.get('memory_model', '32bit') == '32bit':
            profile.vtypes.update(mig_vtypes_32)
        else:
            profile.vtypes.update(mig_vtypes_64)
cnode_vtypes = {'cat_attr': [120, {'ca_fileid': [0, ['unsigned int']], 'ca_mode': [4, ['unsigned short']], 'ca_recflags': [6, ['unsigned short']], 'ca_linkcount': [8, ['unsigned int']], 'ca_uid': [12, ['unsigned int']], 'ca_gid': [16, ['unsigned int']], 'ca_atime': [24, ['long']], 'ca_atimeondisk': [32, ['long']], 'ca_mtime': [40, ['long']], 'ca_ctime': [48, ['long']], 'ca_itime': [56, ['long']], 'ca_btime': [64, ['long']], 'ca_flags': [72, ['unsigned int']]}], 'cnode': [328, {'c_flag': [64, ['unsigned int']], 'c_hflag': [68, ['unsigned int']], 'c_vp': [72, ['pointer', ['vnode']]], 'c_rsrc_vp': [80, ['pointer', ['vnode']]], 'c_childhint': [104, ['unsigned int']], 'c_dirthreadhint': [108, ['unsigned int']], 'c_attr': [136, ['cat_attr']], 'c_dirhinttag': [288, ['short']], 'c_dirchangecnt': [292, ['unsigned int']], 'c_touch_acctime': [312, ['unsigned char']], 'c_touch_chgtime': [313, ['unsigned char']], 'c_touch_modtime': [314, ['unsigned char']], 'c_update_txn': [316, ['unsigned int']]}]}

class CNodeTypes(obj.ProfileModification):
    conditions = {'os': lambda x: x in ['mac']}

    def modification(self, profile):
        if False:
            while True:
                i = 10
        if not profile.vtypes.get('cnode'):
            profile.vtypes.update(cnode_vtypes)

class catfishScan(scan.BaseScanner):
    """ Scanner for Catfish string for Mountain Lion """
    checks = []

    def __init__(self, needles=None):
        if False:
            i = 10
            return i + 15
        self.needles = needles
        self.checks = [('MultiStringFinderCheck', {'needles': needles})]
        scan.BaseScanner.__init__(self)

    def scan(self, address_space, offset=0, maxlen=None):
        if False:
            return 10
        for offset in scan.BaseScanner.scan(self, address_space, offset, maxlen):
            yield offset

class VolatilityDTB(obj.VolatilityMagic):
    """A scanner for DTB values."""

    def _get_dtb_pre_m_lion(self):
        if False:
            i = 10
            return i + 15
        profile = self.obj_vm.profile
        if self.obj_vm.profile.metadata.get('memory_model', '32bit') == '32bit':
            ret = profile.get_symbol('_IdlePDPT')
            if ret % 4096:
                ret = self.obj_vm.read(ret, 4)
                ret = struct.unpack('<I', ret)[0]
        else:
            ret = profile.get_symbol('_IdlePML4')
            if ret > 18446743523953737728:
                ret = ret - 18446743523953737728
        return ret

    def _get_dtb_m_lion(self):
        if False:
            for i in range(10):
                print('nop')
        tbl = self.obj_vm.profile.sys_map['kernel']
        config = self.obj_vm.get_config()
        if config.SHIFT:
            shift_address = config.SHIFT
        else:
            ver_addr = tbl['_version'][0][0] - 18446743523953737728
            scanner = catfishScan(needles=['Catfish \x00\x00'])
            for catfish_offset in scanner.scan(self.obj_vm):
                tmp_shift_address = catfish_offset - tbl['_lowGlo'][0][0] % 4294967168
                tmp_ver_addr = ver_addr + tmp_shift_address
                test_buf = self.obj_vm.zread(tmp_ver_addr, 16)
                if test_buf and test_buf.startswith('Darwin'):
                    shift_address = tmp_shift_address
                    break
        self.obj_vm.profile.shift_address = shift_address
        bootpml4 = tbl['_BootPML4'][0][0] % 4294967168 + shift_address
        boot_pml4_dtb = amd64.AMD64PagedMemory(self.obj_vm, config, dtb=bootpml4)
        idlepml4_addr = tbl['_IdlePML4'][0][0] + shift_address
        idlepml4_ptr = obj.Object('unsigned int', offset=idlepml4_addr, vm=boot_pml4_dtb)
        return idlepml4_ptr.v()

    def generate_suggestions(self):
        if False:
            return 10
        profile = self.obj_vm.profile
        bootpml = profile.get_symbol('_BootPML4')
        if bootpml:
            ret = self._get_dtb_m_lion()
        else:
            ret = self._get_dtb_pre_m_lion()
        yield ret

class VolatilityMacIntelValidAS(obj.VolatilityMagic):
    """An object to check that an address space is a valid Mac Intel Paged space"""

    def _set_profile_metadata(self, version):
        if False:
            while True:
                i = 10
        start = version[len('Darwin Kernel Version '):]
        idx = start.find(':')
        (major, minor, _) = [int(x) for x in start[:idx].split('.')]
        setattr(self.obj_vm.profile, '_md_major', major)
        setattr(self.obj_vm.profile, '_md_minor', minor)

    def generate_suggestions(self):
        if False:
            for i in range(10):
                print('nop')
        version_addr = self.obj_vm.profile.get_symbol('_version')
        string = self.obj_vm.read(version_addr, 60)
        if string and string.startswith('Darwin'):
            self._set_profile_metadata(string)
            yield True
        else:
            yield False

class ifnet(obj.CType):

    def sockaddr_dl(self):
        if False:
            while True:
                i = 10
        if hasattr(self, 'if_lladdr'):
            ret = obj.Object('sockaddr_dl', offset=self.if_lladdr.ifa_addr.v(), vm=self.obj_vm)
        else:
            ret = obj.Object('sockaddr_dl', offset=self.if_addrhead.tqh_first.ifa_addr.v(), vm=self.obj_vm)
        return ret

class vnode(obj.CType):

    def is_dir(self):
        if False:
            for i in range(10):
                print('nop')
        return self.v_type == 2

    def is_reg(self):
        if False:
            i = 10
            return i + 15
        return self.v_type == 1

    def _do_calc_path(self, ret, vnodeobj, vname, vnode_offsets):
        if False:
            print('Hello World!')
        if vnodeobj == None:
            return
        if vnodeobj.v() in vnode_offsets:
            return
        vnode_offsets.append(vnodeobj.v())
        if vname:
            ret.append(vname)
        if vnodeobj.v_flag.v() & 1 != 0 and vnodeobj.v_mount.v() != 0:
            if vnodeobj.v_mount.mnt_vnodecovered.v() != 0:
                self._do_calc_path(ret, vnodeobj.v_mount.mnt_vnodecovered, vnodeobj.v_mount.mnt_vnodecovered.v_name, vnode_offsets)
        else:
            self._do_calc_path(ret, vnodeobj.v_parent, vnodeobj.v_parent.v_name, vnode_offsets)

    def full_path(self):
        if False:
            i = 10
            return i + 15
        if self.v_flag.v() & 1 != 0 and self.v_mount.v() != 0 and (self.v_mount.mnt_flag.v() & 16384 != 0):
            ret = '/'
        else:
            elements = []
            files = []
            self._do_calc_path(elements, self, self.v_name, [])
            elements.reverse()
            for e in elements:
                files.append(str(e.dereference()))
            ret = '/'.join(files)
            if ret:
                ret = '/' + ret
        return ret
    '\n    static inline uintptr_t vm_page_unpack_ptr(uintptr_t p)\n    {\n            if (!p) \n                    return ((uintptr_t)0);\n\n            if (p & VM_PACKED_FROM_VM_PAGES_ARRAY)\n                    return ((uintptr_t)(&vm_pages[(uint32_t)(p & ~VM_PACKED_FROM_VM_PAGES_ARRAY)]));\n            return (((p << VM_PACKED_POINTER_SHIFT) + (uintptr_t) VM_MIN_KERNEL_AND_KEXT_ADDRESS));\n    }\n    '

    def _get_next_page(self, memq):
        if False:
            return 10
        p = memq.m('next')
        if p == 0 or p == None:
            ret = None
        elif self.obj_vm.profile.metadata.get('memory_model', 0) == '64bit' and p.size() == 4:
            if p & 2147483648 != 0:
                vm_pages_ptr = self.obj_vm.profile.get_symbol('_vm_pages')
                vm_pages_addr = obj.Object('unsigned long long', offset=vm_pages_ptr, vm=self.obj_vm)
                ret_addr = vm_pages_addr + (p & ~2147483648) * self.obj_vm.profile.get_obj_size('vm_page')
            else:
                ret_addr = (p << 6) + 18446743521806254080
            ret = obj.Object('vm_page', offset=ret_addr, vm=self.obj_vm)
        else:
            ret = p.dereference_as('vm_page')
        return ret

    def get_contents(self):
        if False:
            i = 10
            return i + 15
        memq = self.v_un.vu_ubcinfo.ui_control.moc_object.memq
        cur = self._get_next_page(memq)
        file_size = self.v_un.vu_ubcinfo.ui_size
        phys_as = self.obj_vm.base
        idx = 0
        written = 0
        while cur and cur.is_valid() and (cur.get_offset() < file_size):
            if cur.get_offset() != 0 and cur.get_offset() == idx:
                break
            if cur.get_phys_page() != 0 and cur.get_offset() >= 0:
                sz = 4096
                if file_size - written < 4096:
                    sz = file_size - written
                buf = phys_as.zread(cur.get_phys_page() * 4096, sz)
                yield (cur.get_offset().v(), buf)
            idx = idx + 1
            written = written + 4096
            cur = self._get_next_page(cur.get_listq())

class vm_page(obj.CType):

    def _get_vmp_member(self, memb):
        if False:
            i = 10
            return i + 15
        ret = self.members.get(memb)
        if ret:
            ret = self.m(memb)
        else:
            ret = self.m('vmp_' + memb)
        return ret

    def get_offset(self):
        if False:
            return 10
        return self._get_vmp_member('offset')

    def get_phys_page(self):
        if False:
            return 10
        return self._get_vmp_member('phys_page')

    def get_listq(self):
        if False:
            i = 10
            return i + 15
        return self._get_vmp_member('listq')

class fileglob(obj.CType):

    @property
    def fg_type(self):
        if False:
            while True:
                i = 10
        ret = self.members.get('fg_type')
        if ret:
            ret = self.m('fg_type')
        elif self.fg_ops.is_valid():
            ret = self.fg_ops.fo_type
        else:
            ret = 'INVALID'
        ret = str(ret)
        return ret

class kauth_scope(obj.CType):

    @property
    def ks_identifier(self):
        if False:
            print('Hello World!')
        ident_ptr = self.m('ks_identifier')
        ident = self.obj_vm.read(ident_ptr, 256)
        if ident:
            idx = ident.find('\x00')
            if idx != -1:
                ident = ident[:idx]
        return ident

    def listeners(self):
        if False:
            i = 10
            return i + 15
        ls_array = obj.Object(theType='Array', targetType='kauth_local_listener', offset=self.m('ks_listeners').obj_offset, vm=self.obj_vm, count=16)
        for ls in ls_array:
            if ls.is_valid() and ls.kll_callback != 0:
                yield ls

class thread(obj.CType):

    def start_time(self):
        if False:
            i = 10
            return i + 15
        baddr = self.obj_vm.profile.get_symbol('_clock_boottime')
        boot_time = obj.Object('unsigned long long', offset=baddr, vm=self.obj_vm)
        abs_time = boot_time + self.sched_stamp
        try:
            data = struct.pack('<I', abs_time)
        except struct.error:
            return ''
        bufferas = addrspace.BufferAddressSpace(self.obj_vm.get_config(), data=data)
        dt = obj.Object('UnixTimeStamp', offset=0, vm=bufferas, is_utc=True)
        return dt

class proc(obj.CType):

    def __init__(self, theType, offset, vm, name=None, **kwargs):
        if False:
            print('Hello World!')
        self.pack_fmt = ''
        self.pack_size = 0
        self.addr_type = ''
        obj.CType.__init__(self, theType, offset, vm, name, **kwargs)
        bit_string = str(self.task.map.pmap.pm_task_map or '')[9:]
        if bit_string.find('64BIT') == -1:
            self.pack_fmt = '<I'
            self.pack_size = 4
            self.addr_type = 'unsigned int'
        else:
            self.pack_fmt = '<Q'
            self.pack_size = 8
            self.addr_type = 'unsigned long long'

    def bash_hash_entries(self):
        if False:
            i = 10
            return i + 15
        proc_as = self.get_process_address_space()
        if not proc_as:
            return
        shared_start = self.task.shared_region.sr_base_address
        shared_end = shared_start + self.task.shared_region.sr_size
        bit_string = str(self.task.map.pmap.pm_task_map or '')[9:]
        if bit_string.find('64BIT') == -1:
            addr_type = 'unsigned int'
            bucket_contents_type = 'mac32_bucket_contents'
            htable_type = 'mac32_bash_hash_table'
            nbuckets_offset = self.obj_vm.profile.get_obj_offset(htable_type, 'nbuckets')
        else:
            addr_type = 'unsigned long long'
            bucket_contents_type = 'mac64_bucket_contents'
            htable_type = 'mac64_bash_hash_table'
            nbuckets_offset = self.obj_vm.profile.get_obj_offset(htable_type, 'nbuckets')
        range_end = 4096 - nbuckets_offset - 8
        for map in self.get_proc_maps():
            if shared_start <= map.start <= shared_end:
                continue
            if map.get_perms() != 'rw-':
                continue
            if map.get_path() != '':
                continue
            if map.end - map.start > 1073741824:
                continue
            chunk_off = int(map.start)
            end = int(map.end)
            while chunk_off < end:
                data = proc_as.read(chunk_off, 4096)
                prev_off = chunk_off
                chunk_off = chunk_off + 4096
                if data == None:
                    continue
                off = 0
                while off < range_end:
                    read_off = prev_off + off
                    dr = data[off + nbuckets_offset:off + nbuckets_offset + 4]
                    test = struct.unpack('<I', dr)[0]
                    if test != 64:
                        off = off + 4
                        continue
                    htable = obj.Object(htable_type, offset=read_off, vm=proc_as)
                    if htable.is_valid():
                        bucket_array = obj.Object(theType='Array', targetType=addr_type, offset=htable.bucket_array, vm=htable.nbuckets.obj_vm, count=64)
                        seen = set()
                        for bucket_ptr in bucket_array:
                            bucket = obj.Object(bucket_contents_type, offset=bucket_ptr, vm=htable.nbuckets.obj_vm)
                            while bucket != None and bucket.times_found > 0:
                                if bucket.v() in seen:
                                    break
                                seen.add(bucket.v())
                                pdata = bucket.data
                                if pdata == None:
                                    bucket = bucket.next_bucket()
                                    continue
                                if bucket.key != None and bucket.data != None and pdata.is_valid() and (0 <= pdata.flags <= 2):
                                    if (len(str(bucket.key)) > 0 or len(str(bucket.data.path)) > 0) and 0 < bucket.times_found <= 1024:
                                        yield bucket
                                bucket = bucket.next_bucket()
                    off = off + 4

    def bash_history_entries(self):
        if False:
            print('Hello World!')
        proc_as = self.get_process_address_space()
        bit_string = str(self.task.map.pmap.pm_task_map or '')[9:]
        if bit_string.find('64BIT') == -1:
            pack_format = '<I'
            hist_struct = 'bash32_hist_entry'
        else:
            pack_format = '<Q'
            hist_struct = 'bash64_hist_entry'
        ts_offset = proc_as.profile.get_obj_offset(hist_struct, 'timestamp')
        history_entries = []
        bang_addrs = []
        for ptr_hash in self.search_process_memory_rw_nofile(['#']):
            addr = struct.pack(pack_format, ptr_hash)
            bang_addrs.append(addr)
        for (idx, ptr_string) in enumerate(self.search_process_memory_rw_nofile(bang_addrs)):
            hist = obj.Object(hist_struct, offset=ptr_string - ts_offset, vm=proc_as)
            if hist.is_valid():
                history_entries.append(hist)
        for hist in sorted(history_entries, key=attrgetter('time_as_integer')):
            yield hist

    def _get_libc_range(self, proc_as):
        if False:
            for i in range(10):
                print('nop')
        libc_map = None
        mapping = None
        for dmap in self.get_dyld_maps():
            if dmap.imageFilePath.endswith('libsystem_c.dylib'):
                libc_map = dmap
                break
        if libc_map:
            mh = obj.Object('macho_header', offset=libc_map.imageLoadAddress, vm=proc_as)
            for seg in mh.segments():
                if str(seg.segname) == '__DATA':
                    mapping = [[seg.vmaddr, seg.vmaddr + seg.vmsize, seg.vmsize]]
        return mapping

    def _get_env_mappings(self, proc_as):
        if False:
            while True:
                i = 10
        mappings = self._get_libc_range(proc_as)
        if not mappings:
            mappings = []
            for mapping in self.get_proc_maps():
                if str(mapping.get_perms()) != 'rw-' or mapping.get_path() == '':
                    continue
                mappings.append([mapping.start, mapping.end, mapping.end - mapping.start])
            mappings.reverse()
        return mappings

    def _carve_mappings_for_env(self, proc_as, mappings):
        if False:
            return 10
        global dynamic_env_hint
        seen_ptrs = {}
        seen_firsts = {}
        env_start = 0
        for (start, end, length) in mappings:
            if env_start:
                break
            if length >= 16777216:
                continue
            chunk_offset = start
            while chunk_offset < end:
                if env_start:
                    break
                data = proc_as.read(chunk_offset, 4096)
                chunk_offset = chunk_offset + 4096
                if data == None:
                    continue
                off = 0
                while off < 4096 - 4:
                    addrstr = data[off:off + self.pack_size]
                    off = off + 4
                    addr = struct.unpack(self.pack_fmt, addrstr)[0]
                    if addr in seen_ptrs:
                        continue
                    seen_ptrs[addr] = 1
                    if addr:
                        firstaddrstr = proc_as.read(addr, self.pack_size)
                        if not firstaddrstr or len(firstaddrstr) != self.pack_size:
                            continue
                        firstaddr = struct.unpack(self.pack_fmt, firstaddrstr)[0]
                        if firstaddr in seen_firsts:
                            continue
                        seen_firsts[firstaddr] = 1
                        buf = proc_as.read(firstaddr, 64)
                        if not buf:
                            continue
                        eqidx = buf.find('=')
                        if eqidx > 0:
                            nullidx = buf.find('\x00')
                            if nullidx >= eqidx:
                                env_start = addr
                                if not dynamic_env_hint:
                                    dynamic_env_hint = [start, end, length]
                                break
        return env_start

    def _get_env_vars(self, proc_as, env_start):
        if False:
            for i in range(10):
                print('nop')
        good_vars = []
        envars = obj.Object(theType='Array', targetType=self.addr_type, vm=proc_as, offset=env_start, count=256)
        for var in envars:
            if not var or not var.is_valid():
                break
            sizes = [32, 64, 128, 256, 8, 16, 384, 512, 1024, 2048, 4096]
            good_varstr = None
            for size in sizes:
                varstr = proc_as.read(var, size)
                if not varstr:
                    break
                eqidx = varstr.find('=')
                idx = varstr.find('\x00')
                if idx == -1 or eqidx == -1 or idx < eqidx:
                    continue
                good_varstr = varstr
                break
            if good_varstr:
                good_varstr = good_varstr[:idx]
                key = good_varstr[:eqidx]
                val = good_varstr[eqidx + 1:]
                if len(key) > 0 and len(val) > 0 and self._valid_string(key) and self._valid_string(val):
                    good_vars.append((key, val))
            else:
                break
        return good_vars

    def _dynamic_env(self, proc_as, pack_format, addr_sz):
        if False:
            for i in range(10):
                print('nop')
        env_start = 0
        if dynamic_env_hint:
            mappings = [dynamic_env_hint]
            env_start = self._carve_mappings_for_env(proc_as, mappings)
            good_vars = self._get_env_vars(proc_as, env_start)
            if len(good_vars) < 2:
                env_start = 0
        if env_start == 0:
            mappings = self._get_env_mappings(proc_as)
            env_start = self._carve_mappings_for_env(proc_as, mappings)
        if env_start != 0:
            good_vars = self._get_env_vars(proc_as, env_start)
        else:
            good_vars = []
        return good_vars

    def _valid_string(self, test_string):
        if False:
            print('Hello World!')
        valid = True
        test_string = str(test_string)
        for s in test_string:
            if not s in string.printable:
                valid = False
                break
        return valid

    def _shell_variables(self, proc_as, pack_format, addr_sz, htable_type):
        if False:
            while True:
                i = 10
        if has_yara == False:
            return
        nbuckets_offset = self.obj_vm.profile.get_obj_offset(htable_type, 'nbuckets')
        if addr_sz == 4:
            edata_type = 'mac32_envdata'
        else:
            edata_type = 'mac64_envdata'
        seen_ptr = {}
        s = '{ 40 00 00 00 }'
        rules = yara.compile(sources={'n': 'rule r1 {strings: $a = ' + s + ' condition: $a}'})
        scanner = BashEnvYaraScanner(task=self, rules=rules)
        for (hit, off) in scanner.scan():
            htable = obj.Object(htable_type, offset=off - addr_sz, vm=proc_as)
            if not htable.is_valid():
                continue
            for ent in htable:
                if not ent.m('key').is_valid():
                    continue
                if self._valid_string(ent.key):
                    key = str(ent.key)
                else:
                    key = ''
                val_addr = ent.data.dereference_as(edata_type).value
                if val_addr.is_valid() and self._valid_string(val_addr.dereference()):
                    val = str(val_addr.dereference())
                else:
                    val = ''
                if len(key) > 0 and len(val) > 0:
                    yield (key, val)

    def _load_time_env(self, proc_as):
        if False:
            i = 10
            return i + 15
        start = self.user_stack - self.p_argslen
        skip = len(self.get_arguments())
        end = self.p_argslen
        to_read = end - skip
        vars_buf = proc_as.read(start + skip, to_read)
        if vars_buf:
            ents = vars_buf.split('\x00')
            for varstr in ents:
                eqidx = varstr.find('=')
                if eqidx == -1:
                    continue
                key = varstr[:eqidx]
                val = varstr[eqidx + 1:]
                yield (key, val)

    def psenv(self):
        if False:
            i = 10
            return i + 15
        proc_as = self.get_process_address_space()
        if not proc_as:
            return
        if self.p_pid == 0:
            return
        if self.obj_vm.profile.metadata.get('memory_model', '32bit') == '32bit':
            pack_format = '<I'
            addr_sz = 4
            htable_type = 'mac32_bash_hash_table'
        else:
            pack_format = '<Q'
            addr_sz = 8
            htable_type = 'mac64_bash_hash_table'
        env_count = 0
        for (key, val) in self._dynamic_env(proc_as, pack_format, addr_sz):
            yield (key, val)
            env_count = env_count + 1
        if env_count < 3:
            for (key, val) in self._load_time_env(proc_as):
                yield (key, val)
        if str(self.p_comm) == 'bash':
            for (key, val) in self._shell_variables(proc_as, pack_format, addr_sz, htable_type):
                yield (key, val)

    def netstat(self):
        if False:
            return 10
        for (filp, _, _) in self.lsof():
            if filp.f_fglob.is_valid() and filp.f_fglob.fg_type == 'DTYPE_SOCKET':
                socket = filp.f_fglob.fg_data.dereference_as('socket')
                family = socket.family
                if family == 1:
                    upcb = socket.so_pcb.dereference_as('unpcb')
                    path = upcb.unp_addr.sun_path
                    yield (family, (socket.v(), path))
                elif family in [2, 30]:
                    proto = socket.protocol
                    state = socket.state
                    vals = socket.get_connection_info()
                    if vals:
                        (lip, lport, rip, rport) = vals
                        yield (family, (socket, proto, lip, lport, rip, rport, state))

    @property
    def p_gid(self):
        if False:
            print('Hello World!')
        cred = self.p_ucred
        if not cred.is_valid():
            return -1
        if hasattr(cred, 'cr_posix'):
            try:
                ret = cred.cr_posix.cr_groups[0]
            except IndexError:
                ret = obj.Object('unsigned int', offset=cred.cr_posix.cr_groups.obj_offset, vm=self.obj_vm)
        else:
            ret = cred.cr_groups[0]
        return ret

    @property
    def p_uid(self):
        if False:
            print('Hello World!')
        cred = self.p_ucred
        if not cred.is_valid():
            return -1
        if hasattr(cred, 'cr_posix'):
            ret = cred.cr_posix.cr_uid
        else:
            ret = cred.cr_uid
        return ret

    def threads(self):
        if False:
            i = 10
            return i + 15
        threads = []
        seen_threads = []
        qentry = self.task.threads
        for thread in qentry.thread_walk_list(qentry.obj_offset):
            if thread.obj_offset in seen_threads:
                break
            seen_threads.append(thread.obj_offset)
            threads.append(thread)
        return threads

    def get_process_address_space(self):
        if False:
            return 10
        cr3 = self.task.map.pmap.pm_cr3
        map_val = str(self.task.map.pmap.pm_task_map or '')
        is_64bit_cap = common.is_64bit_capable(self.obj_vm)
        if map_val == 'TASK_MAP_32BIT' and is_64bit_cap:
            try:
                proc_as = amd64.AMD64PagedMemory(self.obj_vm.base, self.obj_vm.get_config(), dtb=cr3, skip_as_check=True)
            except IOError:
                proc_as = self.obj_vm
        elif map_val == 'TASK_MAP_32BIT':
            proc_as = intel.IA32PagedMemoryPae(self.obj_vm.base, self.obj_vm.get_config(), dtb=cr3, skip_as_check=True)
        elif map_val == 'TASK_MAP_64BIT_SHARED' and self.obj_vm.profile.metadata.get('memory_model', '32bit') == '32bit':
            proc_as = amd64.AMD64PagedMemory(self.obj_vm.base, self.obj_vm.get_config(), dtb=cr3, skip_as_check=True)
        elif map_val in ['TASK_MAP_64BIT', 'TASK_MAP_64BIT_SHARED']:
            cr3 &= 4294967264
            proc_as = amd64.AMD64PagedMemory(self.obj_vm.base, self.obj_vm.get_config(), dtb=cr3, skip_as_check=True)
        else:
            proc_as = obj.NoneObject('Cannot get process AS for pm_task_map: {0}'.format(map_val))
        return proc_as

    def start_time(self):
        if False:
            for i in range(10):
                print('nop')
        nsecs_per = 1000000
        start_time = self.p_start
        start_secs = start_time.tv_sec + start_time.tv_usec / nsecs_per
        try:
            data = struct.pack('<I', start_secs)
        except struct.error:
            return ''
        bufferas = addrspace.BufferAddressSpace(self.obj_vm.get_config(), data=data)
        dt = obj.Object('UnixTimeStamp', offset=0, vm=bufferas, is_utc=True)
        return dt

    def text_start(self):
        if False:
            print('Hello World!')
        text_start = 0
        wanted_vnode = self.p_textvp.v()
        if wanted_vnode:
            for map in self.get_proc_maps():
                vnode = map.get_vnode()
                if vnode and vnode != 'sub_map' and (vnode.v() == wanted_vnode) and (map.get_perms() == 'r-x'):
                    text_start = map.start.v()
                    break
        if text_start == 0:
            found_map = None
            for map in self.get_dyld_maps():
                found_map = map
                break
            if found_map:
                text_start = found_map.imageLoadAddress
        return text_start

    def get_macho(self, exe_address):
        if False:
            while True:
                i = 10
        proc_as = self.get_process_address_space()
        m = obj.Object('macho_header', offset=exe_address, vm=proc_as)
        if not m.is_valid():
            return
        buffer = ''
        for seg in m.segments():
            if str(seg.segname) == '__PAGEZERO':
                continue
            if seg.vmsize == 0 or seg.vmsize > 100000000:
                continue
            if str(seg.segname) == '__LINKEDIT' and seg.vmsize > 20000000:
                continue
            cur = seg.vmaddr
            end = seg.vmaddr + seg.vmsize
            while cur < end:
                buffer = buffer + proc_as.zread(cur, 4096)
                cur = cur + 4096
        return buffer

    def procdump(self):
        if False:
            print('Hello World!')
        start = self.text_start()
        if start:
            ret = self.get_macho(start)
        else:
            ret = ''
        return ret

    def get_dyld_maps(self):
        if False:
            while True:
                i = 10
        proc_as = self.get_process_address_space()
        if proc_as == None:
            return
        if self.pack_size == 4:
            dtype = 'dyld32_all_image_infos'
            itype = 'dyld32_image_info'
        else:
            dtype = 'dyld64_all_image_infos'
            itype = 'dyld64_image_info'
        infos = obj.Object(dtype, offset=self.task.all_image_info_addr, vm=proc_as)
        if not infos:
            return
        info_buf = proc_as.read(infos.infoArray.obj_offset, self.pack_size)
        if not info_buf:
            return
        info_addr = struct.unpack(self.pack_fmt, info_buf)[0]
        if not proc_as.is_valid_address(info_addr):
            return
        cnt = infos.infoArrayCount
        if cnt > 4096:
            cnt = 1024
        img_infos = obj.Object(theType='Array', targetType=itype, offset=info_addr, count=cnt, vm=proc_as)
        for info_addr in img_infos:
            if info_addr and info_addr.is_valid():
                yield info_addr

    def get_proc_maps(self):
        if False:
            i = 10
            return i + 15
        map = self.task.map.hdr.links.next
        seen = set()
        for i in xrange(self.task.map.hdr.nentries):
            if map.v() in seen:
                break
            seen.add(map.v())
            if not map:
                break
            map_size = int(map.links.end - map.links.start)
            if 4095 < map_size < 140737488355328 and map_size % 4096 == 0:
                yield map
            map = map.links.next

    def find_heap_map(self):
        if False:
            i = 10
            return i + 15
        ret = None
        for pmap in self.get_proc_maps():
            if pmap.get_special_path() == '[heap]':
                ret = pmap
                break
        return None

    def find_map(self, addr):
        if False:
            i = 10
            return i + 15
        ret = None
        for vma in self.get_proc_maps():
            if int(vma.links.start) <= int(addr) <= int(vma.links.end):
                ret = vma
                break
        return ret

    def find_map_path(self, addr):
        if False:
            print('Hello World!')
        path = ''
        m = self.find_map(addr)
        if m:
            path = m.get_path()
            if path == '':
                path = m.get_special_path()
        return path

    def search_process_memory(self, s):
        if False:
            for i in range(10):
                print('nop')
        'Search process memory. \n\n        @param s: a list of strings like ["one", "two"]\n        '
        overlap = 1024
        scan_blk_sz = 1024 * 1024 * 10
        addr_space = self.get_process_address_space()
        for vma in self.get_proc_maps():
            offset = vma.links.start
            out_of_range = vma.links.start + (vma.links.end - vma.links.start)
            while offset < out_of_range:
                to_read = min(scan_blk_sz + overlap, out_of_range - offset)
                data = addr_space.zread(offset, to_read)
                if not data:
                    break
                for x in s:
                    for hit in utils.iterfind(data, x):
                        yield (offset + hit)
                offset += min(to_read, scan_blk_sz)

    def search_process_memory_rw_nofile(self, s):
        if False:
            print('Hello World!')
        'Search process memory. \n\n        @param s: a list of strings like ["one", "two"]\n        '
        overlap = 1024
        scan_blk_sz = 1024 * 1024 * 10
        addr_space = self.get_process_address_space()
        for vma in self.get_proc_maps():
            if vma.get_perms() != 'rw-' or vma.get_path() != '':
                if vma.get_special_path() != '[heap]':
                    continue
            offset = vma.links.start
            out_of_range = vma.links.start + (vma.links.end - vma.links.start)
            while offset < out_of_range:
                to_read = min(scan_blk_sz + overlap, out_of_range - offset)
                data = addr_space.zread(offset, to_read)
                if not data:
                    break
                for x in s:
                    for hit in utils.iterfind(data, x):
                        yield (offset + hit)
                offset += min(to_read, scan_blk_sz)

    def get_environment(self):
        if False:
            print('Hello World!')
        env = ''
        for (k, v) in self.psenv():
            env = env + '{0}={1} '.format(k, v)
        return env

    def get_arguments(self):
        if False:
            while True:
                i = 10
        proc_as = self.get_process_address_space()
        if not proc_as:
            return ''
        argsstart = self.user_stack - self.p_argslen
        if not proc_as.is_valid_address(argsstart) or self.p_argslen == 0 or self.p_argc == 0:
            return ''
        argc = self.p_argc + 1
        args = []
        if argc > 1024:
            return ''
        while argc > 0:
            arg = obj.Object('String', offset=argsstart, vm=proc_as, length=256)
            if not arg:
                break
            argsstart += len(str(arg)) + 1
            if len(args) == 0:
                while proc_as.read(argsstart, 1) == '\x00' and argsstart < self.user_stack:
                    argsstart += 1
                args.append(arg)
            elif str(arg) != str(args[0]):
                args.append(arg)
            argc -= 1
        return ' '.join([str(s) for s in args])

    def lsof(self):
        if False:
            print('Hello World!')
        num_fds = self.p_fd.fd_lastfile
        nfiles = self.p_fd.fd_nfiles
        if nfiles > num_fds:
            num_fds = nfiles
        if num_fds > 4096:
            num_fds = 1024
        fds = obj.Object('Array', offset=self.p_fd.fd_ofiles, vm=self.obj_vm, targetType='Pointer', count=num_fds)
        for (i, fd) in enumerate(fds):
            f = fd.dereference_as('fileproc')
            if f and f.f_fglob.is_valid():
                ftype = f.f_fglob.fg_type
                if ftype == 'DTYPE_VNODE':
                    vnode = f.f_fglob.fg_data.dereference_as('vnode')
                    path = vnode.full_path()
                else:
                    path = '<%s>' % ftype.replace('DTYPE_', '').lower()
                yield (f, path, i)

class rtentry(obj.CType):

    def is_valid(self):
        if False:
            i = 10
            return i + 15
        return str(self.source_ip) != '' and str(self.dest_ip) != '' and (0 <= int(self.sent) < 50000000000) and (0 <= int(self.rx) < 50000000000)

    def get_time(self):
        if False:
            return 10
        if not hasattr(self, 'base_calendartime'):
            return 'N/A'
        data = struct.pack('<I', self.base_calendartime)
        bufferas = addrspace.BufferAddressSpace(self.obj_vm.get_config(), data=data)
        dt = obj.Object('UnixTimeStamp', offset=0, vm=bufferas, is_utc=True)
        return dt

    @property
    def sent(self):
        if False:
            i = 10
            return i + 15
        if hasattr(self, 'rt_stats'):
            ret = self.rt_stats.nstat_txpackets
        else:
            ret = 'N/A'
        return ret

    @property
    def rx(self):
        if False:
            for i in range(10):
                print('nop')
        if hasattr(self, 'rt_stats'):
            ret = self.rt_stats.nstat_rxpackets
        else:
            ret = 'N/A'
        return ret

    @property
    def delta(self):
        if False:
            while True:
                i = 10
        if self.expire() == 0:
            ret = 0
        else:
            ret = self.expire() - self.base_uptime
        return ret

    def expire(self):
        if False:
            for i in range(10):
                print('nop')
        if hasattr(self, 'rt_expire'):
            ret = self.rt_expire
        else:
            ret = 0
        return ret

    @property
    def name(self):
        if False:
            while True:
                i = 10
        return '{}{}'.format(self.rt_ifp.if_name.dereference(), self.rt_ifp.if_unit)

    @property
    def source_ip(self):
        if False:
            for i in range(10):
                print('nop')
        try:
            node = self.rt_nodes[0]
        except IndexError:
            node = obj.Object('radix_node', offset=self.rt_nodes.obj_offset, vm=self.obj_vm)
        return node.rn_u.rn_leaf.rn_Key.dereference_as('sockaddr').get_address()

    @property
    def dest_ip(self):
        if False:
            while True:
                i = 10
        return self.rt_gateway.get_address()

class queue_entry(obj.CType):

    def walk_list(self, list_head):
        if False:
            i = 10
            return i + 15
        n = self.next.dereference_as('task')
        while n and n.obj_offset != list_head:
            yield n
            n = n.tasks.next.dereference_as('task')
        p = self.prev.dereference_as('task')
        while p and p.obj_offset != list_head:
            yield p
            p = p.tasks.prev.dereference_as('task')

class zone(obj.CType):

    def is_valid(self):
        if False:
            for i in range(10):
                print('nop')
        return self.elem_size > 0

    def _get_from_active_zones(self):
        if False:
            i = 10
            return i + 15
        ret = []
        first_elem = self.active_zones
        elem = first_elem
        sz = 16
        i = 0
        while elem != first_elem.v() or i == 0:
            a = elem.v()
            b = sz
            off = a + b
            ret.append(off)
            i = i + 1
            if i == 4:
                break
            elem = elem.m('next')
        return ret

    def get_active_elements(self, elem_type, zone_idx=-1):
        if False:
            return 10
        ret = []
        if hasattr(self, 'active_zones'):
            objs = self._get_from_active_zones()
        else:
            debug.error('zone does not have active zones.')
        for o in objs:
            val = obj.Object(elem_type, offset=o, vm=self.obj_vm)
            ret.append(val)
        return ret

    def get_free_elements(self, elem_type):
        if False:
            i = 10
            return i + 15
        ret = []
        nxt = obj.Object('zone_free_element', offset=self.free_elements, vm=self.obj_vm)
        while nxt:
            o = nxt.obj_offset
            val = obj.Object(elem_type, offset=o, vm=self.obj_vm)
            ret.append(val)
            nxt = nxt.m('next')
        return ret

class sysctl_oid(obj.CType):

    def get_perms(self):
        if False:
            while True:
                i = 10
        '\n        # define CTLFLAG_RD      0x80000000      /* Allow reads of variable */\n        # define CTLFLAG_WR      0x40000000      /* Allow writes to the variable */\n        # define CTLFLAG_LOCKED  0x00800000      /* node will handle locking itself */\n        '
        ret = ''
        checks = [2147483648, 1073741824, 8388608]
        perms = ['R', 'W', 'L']
        for (i, c) in enumerate(checks):
            if c & self.oid_kind:
                ret = ret + perms[i]
            else:
                ret = ret + '-'
        return ret

    def get_ctltype(self):
        if False:
            i = 10
            return i + 15
        '\n        #define CTLTYPE_NODE    1\n        #define CTLTYPE_INT     2       /* name describes an integer */\n        #define CTLTYPE_STRING  3       /* name describes a string */\n        #define CTLTYPE_QUAD    4       /* name describes a 64-bit number */\n        #define CTLTYPE_OPAQUE  5       /* name describes a structure */\n        #define CTLTYPE_STRUCT  CTLTYPE_OPAQUE  /* name describes a structure */\n        '
        types = {1: 'CTLTYPE_NODE', 2: 'CTLTYPE_INT', 3: 'CTLTYPE_STRING', 4: 'CTLTYPE_QUAD', 5: 'CTLTYPE_OPAQUE'}
        ctltype = self.oid_kind & 15
        try:
            return types[ctltype]
        except KeyError:
            return 'INVALID -1'

class OSString(obj.CType):

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        if self.string == 0:
            return ''
        if self.length > 4096:
            return ''
        string_object = obj.Object('String', offset=self.string, vm=self.obj_vm, length=self.length)
        return str(string_object or '')

class vm_map_object(obj.CType):

    def object(self):
        if False:
            print('Hello World!')
        if hasattr(self, 'vm_object'):
            ret = self.m('vm_object')
        else:
            ret = self.vmo_object
        return ret

class vm_map_entry(obj.CType):

    @property
    def start(self):
        if False:
            while True:
                i = 10
        return self.links.start

    @property
    def end(self):
        if False:
            for i in range(10):
                print('nop')
        return self.links.end

    def get_perms(self):
        if False:
            while True:
                i = 10
        permask = 'rwx'
        perms = ''
        for (ctr, i) in enumerate([1, 3, 5]):
            if self.protection & i == i:
                perms = perms + permask[ctr]
            else:
                perms = perms + '-'
        return perms

    def range_alias(self):
        if False:
            print('Hello World!')
        if hasattr(self, 'alias'):
            ret = self.alias.v()
        else:
            ret = self.vme_offset.v() & 4095
        return ret

    def get_special_path(self):
        if False:
            return 10
        check = self.range_alias()
        if 0 < check < 10:
            ret = '[heap]'
        elif check == 30:
            ret = '[stack]'
        else:
            ret = ''
        return ret

    def get_path(self):
        if False:
            for i in range(10):
                print('nop')
        vnode = self.get_vnode()
        if type(vnode) == str and vnode == 'sub_map':
            ret = vnode
        elif vnode:
            path = []
            seen = set()
            while vnode and vnode.v() not in seen:
                seen.add(vnode.v())
                path.append(str(vnode.v_name.dereference() or ''))
                vnode = vnode.v_parent
            path.reverse()
            ret = '/'.join(path)
        else:
            ret = ''
        return ret

    @property
    def object(self):
        if False:
            return 10
        if hasattr(self, 'vme_object'):
            ret = self.vme_object
        else:
            ret = self.m('object')
        return ret

    @property
    def offset(self):
        if False:
            print('Hello World!')
        if hasattr(self, 'vme_offset'):
            ret = self.vme_offset
        else:
            ret = self.m('offset')
        return ret

    def get_vnode(self):
        if False:
            i = 10
            return i + 15
        map_obj = self
        if self.is_sub_map == 1:
            return 'sub_map'
        vnode_object = map_obj.object.object()
        seen = set()
        while vnode_object.shadow.dereference() != None and vnode_object.v() not in seen:
            vnode_object = vnode_object.shadow.dereference()
            seen.add(vnode_object.v())
        ops = vnode_object.pager.mo_pager_ops.v()
        if ops == self.obj_vm.profile.get_symbol('_vnode_pager_ops'):
            vpager = obj.Object('vnode_pager', offset=vnode_object.pager, vm=self.obj_vm)
            ret = vpager.vnode_handle
        else:
            ret = None
        return ret

    def resident_count(self):
        if False:
            print('Hello World!')
        vmobj = self.object.object()
        if not vmobj:
            return 0
        if hasattr(vmobj, 'all_reusable'):
            if vmobj.all_reusable == 1:
                count = vmobj.wired_page_count.v()
            else:
                count = vmobj.resident_page_count.v() - vmobj.reusable_page_count.v()
        else:
            count = vmobj.resident_page_count.v()
        return count

    def is_suspicious(self):
        if False:
            print('Hello World!')
        ret = False
        perms = self.get_perms()
        if perms == 'rwx':
            ret = True
        elif perms == 'r-x' and self.get_path() == '':
            ret = True
        return ret

class inpcb(obj.CType):

    def get_tcp_state(self):
        if False:
            for i in range(10):
                print('nop')
        tcp_states = ('CLOSED', 'LISTEN', 'SYN_SENT', 'SYN_RECV', 'ESTABLISHED', 'CLOSE_WAIT', 'FIN_WAIT1', 'CLOSING', 'LAST_ACK', 'FIN_WAIT2', 'TIME_WAIT')
        tcpcb = self.inp_ppcb.dereference_as('tcpcb')
        state_type = tcpcb.t_state
        if state_type:
            state = tcp_states[state_type]
        else:
            state = ''
        return state

    def ipv4_info(self):
        if False:
            while True:
                i = 10
        lip = self.inp_dependladdr.inp46_local.ia46_addr4.s_addr.v()
        lport = self.inp_lport
        rip = self.inp_dependfaddr.inp46_foreign.ia46_addr4.s_addr.v()
        rport = self.inp_fport
        return [lip, lport, rip, rport]

    def ipv6_info(self):
        if False:
            print('Hello World!')
        lip = self.inp_dependladdr.inp6_local.__u6_addr.v()
        lport = self.inp_lport
        rip = self.inp_dependfaddr.inp6_foreign.__u6_addr.v()
        rport = self.inp_fport
        return [lip, lport, rip, rport]

class inpcbinfo(obj.CType):

    @property
    def hashbase(self):
        if False:
            i = 10
            return i + 15
        ret = self.members.get('hashbase')
        if ret is None:
            ret = self.ipi_hashbase
        else:
            ret = self.m('hashbase')
        return ret

    @property
    def hashmask(self):
        if False:
            i = 10
            return i + 15
        ret = self.members.get('hashmask')
        if ret is None:
            ret = self.ipi_hashmask
        else:
            ret = self.m('hashmask')
        return ret

    @property
    def listhead(self):
        if False:
            print('Hello World!')
        ret = self.members.get('listhead')
        if ret is None:
            ret = self.ipi_listhead
        else:
            ret = self.m('listhead')
        return ret

class socket(obj.CType):

    @property
    def family(self):
        if False:
            return 10
        return self.so_proto.pr_domain.dom_family

    @property
    def protocol(self):
        if False:
            i = 10
            return i + 15
        proto = self.so_proto.pr_protocol
        if proto == 6:
            ret = 'TCP'
        elif proto == 17:
            ret = 'UDP'
        else:
            ret = ''
        return ret

    def _get_tcp_state(self):
        if False:
            print('Hello World!')
        tcp_states = ('CLOSED', 'LISTEN', 'SYN_SENT', 'SYN_RECV', 'ESTABLISHED', 'CLOSE_WAIT', 'FIN_WAIT1', 'CLOSING', 'LAST_ACK', 'FIN_WAIT2', 'TIME_WAIT')
        inpcb = self.so_pcb.dereference_as('inpcb')
        tcpcb = inpcb.inp_ppcb.dereference_as('tcpcb')
        state = tcpcb.t_state
        if state:
            ret = tcp_states[tcpcb.t_state]
        else:
            ret = '<INVALID>'
        return ret

    @property
    def state(self):
        if False:
            return 10
        if self.so_proto.pr_protocol == 6:
            ret = self._get_tcp_state()
        else:
            ret = ''
        return ret

    def get_connection_info(self):
        if False:
            i = 10
            return i + 15
        if not self.so_pcb.is_valid():
            return None
        ipcb = self.so_pcb.dereference_as('inpcb')
        if self.family == 2:
            ret = ipcb.ipv4_info()
        else:
            ret = ipcb.ipv6_info()
        return ret

class sockaddr_dl(obj.CType):

    def v(self):
        if False:
            while True:
                i = 10
        'Get the value of the sockaddr_dl object.'
        ret = ''
        for i in xrange(self.sdl_alen):
            try:
                e = self.sdl_data[self.sdl_nlen + i]
                e = ord(e.v())
            except IndexError:
                e = 0
            ret = ret + '%.02x:' % e
        if ret and ret[-1] == ':':
            ret = ret[:-1]
        return ret

class sockaddr(obj.CType):

    def get_address(self):
        if False:
            for i in range(10):
                print('nop')
        family = self.sa_family
        ip = ''
        if family == 2:
            addr_in = obj.Object('sockaddr_in', offset=self.obj_offset, vm=self.obj_vm)
            ip = addr_in.sin_addr.s_addr.v()
        elif family == 30:
            addr_in6 = obj.Object('sockaddr_in6', offset=self.obj_offset, vm=self.obj_vm)
            ip = addr_in6.sin6_addr.__u6_addr.v()
        elif family == 18:
            addr_dl = obj.Object('sockaddr_dl', offset=self.obj_offset, vm=self.obj_vm)
            ip = addr_dl.v()
        return ip

class dyld32_image_info(obj.CType):

    def is_valid(self):
        if False:
            print('Hello World!')
        return len(self.imageFilePath) > 1 and self.imageLoadAddress > 4096

    def _read_ptr(self, addr):
        if False:
            for i in range(10):
                print('nop')
        addr = self.obj_vm.read(addr, 4)
        if not addr:
            ret = None
        else:
            ret = struct.unpack('<I', addr)[0]
        return ret

    @property
    def imageFilePath(self):
        if False:
            print('Hello World!')
        addr = self.m('imageFilePath').obj_offset
        addr = self._read_ptr(addr)
        if addr == None:
            return ''
        buf = self.obj_vm.zread(addr, 256)
        if buf:
            idx = buf.find('\x00')
            if idx != -1:
                buf = buf[:idx]
        return buf

    @property
    def imageLoadAddress(self):
        if False:
            return 10
        addr = self.m('imageLoadAddress').obj_offset
        addr = self._read_ptr(addr)
        return addr

class dyld64_image_info(obj.CType):

    def is_valid(self):
        if False:
            return 10
        return len(self.imageFilePath) > 1 and self.imageLoadAddress > 4096

    def _read_ptr(self, addr):
        if False:
            return 10
        addr = self.obj_vm.read(addr, 8)
        if addr == None:
            ret = None
        else:
            ret = struct.unpack('<Q', addr)[0]
        return ret

    @property
    def imageFilePath(self):
        if False:
            return 10
        addr = self.m('imageFilePath').obj_offset
        addr = self._read_ptr(addr)
        if addr == None:
            return ''
        buf = self.obj_vm.zread(addr, 256)
        if buf:
            idx = buf.find('\x00')
            if idx != -1:
                buf = buf[:idx]
        return buf

    @property
    def imageLoadAddress(self):
        if False:
            while True:
                i = 10
        addr = self.m('imageLoadAddress').obj_offset
        addr = self._read_ptr(addr)
        return addr

def exec_vtypes(filename):
    if False:
        while True:
            i = 10
    env = {}
    exec(filename, dict(__builtins__=None), env)
    return env['mac_types']

def parse_dsymutil(data, module):
    if False:
        return 10
    'Parse the symbol file.'
    sys_map = {}
    sys_map[module] = {}
    want_lower = ['_IdlePML4']
    type_map = {}
    type_map[module] = {}
    arch = ''
    for line in data.splitlines():
        ents = line.split()
        match = re.search("\\[.*?\\(([^\\)]+)\\)\\s+[0-9A-Fa-z]+\\s+\\d+\\s+([0-9A-Fa-f]+)\\s'(\\w+)'", line)
        if match:
            (sym_type, addr, name) = match.groups()
            sym_type = sym_type.strip()
            addr = int(addr, 16)
            if addr == 0 or name == '':
                continue
            if not name in sys_map[module]:
                sys_map[module][name] = [(addr, sym_type)]
            oldaddr = sys_map[module][name][0][0]
            if addr < oldaddr and name in want_lower:
                sys_map[module][name] = [(addr, sym_type)]
            if not addr in type_map[module]:
                type_map[module][addr] = (name, [sym_type])
            type_map[module][addr][1].append(sym_type)
        elif line.find('Symbol table for') != -1:
            if line.find('i386') != -1:
                arch = '32bit'
            else:
                arch = '64bit'
    if arch == '':
        return None
    return (arch, sys_map, type_map)

def MacProfileFactory(profpkg):
    if False:
        while True:
            i = 10
    vtypesvar = {}
    sysmapvar = {}
    typesmapvar = {}
    (memmodel, arch) = ('32bit', 'x86')
    profilename = os.path.splitext(os.path.basename(profpkg.filename))[0]
    for f in profpkg.filelist:
        if 'symbol.dsymutil' in f.filename.lower():
            (memmodel, sysmap, typemap) = parse_dsymutil(profpkg.read(f.filename), 'kernel')
            if memmodel == '64bit':
                arch = 'x64'
            sysmapvar.update(sysmap)
            typesmapvar.update(typemap)
            debug.debug('{2}: Found system file {0} with {1} symbols'.format(f.filename, len(sysmapvar.keys()), profilename))
        elif f.filename.endswith('.vtypes'):
            v = exec_vtypes(profpkg.read(f.filename))
            vtypesvar.update(v)
    if not sysmapvar or not vtypesvar:
        return None

    class AbstractMacProfile(obj.Profile):
        __doc__ = 'A Profile for Mac ' + profilename + ' ' + arch
        _md_os = 'mac'
        _md_memory_model = memmodel
        native_mapping = {'32bit': native_types.x86_native_types, '64bit': x64_native_types}

        def __init__(self, *args, **kwargs):
            if False:
                for i in range(10):
                    print('nop')
            self._init_vars()
            obj.Profile.__init__(self, *args, **kwargs)

        def _init_vars(self):
            if False:
                print('Hello World!')
            self.sys_map = {}
            self.type_map = {}
            self.shift_address = 0
            self.sba_cache = {}
            self.sbat_cache = {}

        def clear(self):
            if False:
                i = 10
                return i + 15
            'Clear out the system map, and everything else'
            self._init_vars()
            obj.Profile.clear(self)

        def reset(self):
            if False:
                for i in range(10):
                    print('nop')
            'Reset the vtypes, sysmap and apply modifications, then compile'
            self.clear()
            self.load_vtypes()
            self.load_sysmap()
            self.load_modifications()
            self.compile()

        def load_vtypes(self):
            if False:
                for i in range(10):
                    print('nop')
            'Loads up the vtypes data'
            ntvar = self.metadata.get('memory_model', '32bit')
            self.native_types = copy.deepcopy(self.native_mapping.get(ntvar))
            self.vtypes.update(vtypesvar)

        def load_sysmap(self):
            if False:
                i = 10
                return i + 15
            'Loads up the system map data'
            self.sys_map.update(sysmapvar)
            self.type_map.update(typesmapvar)

        def get_all_symbols(self, module='kernel'):
            if False:
                print('Hello World!')
            ' Gets all the symbol tuples for the given module '
            ret = []
            symtable = self.sys_map
            if module in symtable:
                mod = symtable[module]
                for (name, addrs) in mod.items():
                    addr = addrs[0][0]
                    if self.shift_address and addr:
                        addr = addr + self.shift_address
                    ret.append([name, addr])
            else:
                debug.info('All symbols requested for non-existent module %s' % module)
            return ret

        def get_all_addresses(self, module='kernel'):
            if False:
                while True:
                    i = 10
            ' Gets all the symbol addresses for the given module '
            ret = {}
            symbols = self.get_all_symbols(module)
            for (_name, addr) in symbols:
                ret[addr] = 1
            return ret

        def get_all_function_symbols(self, module='kernel'):
            if False:
                print('Hello World!')
            ' Gets all the function tuples for the given module '
            ret = []
            symtable = self.type_map
            if module in symtable:
                mod = symtable[module]
                for (addr, (name, _sym_types)) in mod.items():
                    if self.shift_address and addr:
                        addr = addr + self.shift_address
                    ret.append([name, addr])
            else:
                debug.info('All symbols requested for non-existent module %s' % module)
            return ret

        def get_all_function_addresses(self, module='kernel'):
            if False:
                return 10
            ' Gets all the function addresses for the given module '
            ret = {}
            symbols = self.get_all_function_symbols(module)
            for (_name, addr) in symbols:
                ret[addr] = 1
            return ret

        def _get_symbol_by_address_type(self, module, wanted_sym_address, wanted_sym_type):
            if False:
                i = 10
                return i + 15
            ret = ''
            symtable = self.type_map
            mod = symtable[module]
            for (addr, (name, sym_types)) in mod.items():
                for sym_type in sym_types:
                    key = '%s|%x|%s' % (module, addr, sym_type)
                    self.sbat_cache[key] = name
                    if (wanted_sym_address == addr or wanted_sym_address == self.shift_address + addr) and wanted_sym_type == sym_type:
                        ret = name
                        break
            return ret

        def get_symbol_by_address_type(self, module, sym_address, sym_type):
            if False:
                i = 10
                return i + 15
            key = '%s|%x|%s' % (module, sym_address, sym_type)
            if key in self.sbat_cache:
                ret = self.sbat_cache[key]
            else:
                ret = self._get_symbol_by_address_type(module, sym_address, sym_type)
            return ret

        def _fill_sba_cache(self):
            if False:
                while True:
                    i = 10
            ret = ''
            symtable = self.sys_map
            mod = symtable['kernel']
            for (name, addrs) in mod.items():
                for (addr, _) in addrs:
                    key = '%s|%x' % ('kernel', addr)
                    self.sba_cache[key] = name
                    key = '%s|%x' % ('kernel', addr + self.shift_address)
                    self.sba_cache[key] = name

        def get_symbol_by_address(self, module, sym_address):
            if False:
                print('Hello World!')
            if self.sba_cache == {}:
                self._fill_sba_cache()
            key = '%s|%x' % (module, sym_address)
            if key in self.sba_cache:
                ret = self.sba_cache[key]
            else:
                ret = ''
            return ret

        def get_all_symbol_names(self, module='kernel'):
            if False:
                return 10
            symtable = self.sys_map
            if module in symtable:
                ret = symtable[module].keys()
            else:
                debug.error('get_all_symbol_names called on non-existent module')
            return ret

        def get_next_symbol_address(self, sym_name, module='kernel'):
            if False:
                return 10
            '\n            This is used to find the address of the next symbol in the profile\n            For some data structures, we cannot determine their size automaticlaly so this\n            can be used to figure it out on the fly\n            '
            high_addr = 18446744073709551615
            table_addr = self.get_symbol(sym_name, module=module)
            addrs = self.get_all_addresses(module=module)
            for addr in addrs.keys():
                if table_addr < addr < high_addr:
                    high_addr = addr
            return high_addr

        def get_symbol(self, sym_name, nm_type='', module='kernel'):
            if False:
                while True:
                    i = 10
            "Gets a symbol out of the profile\n            \n            sym_name -> name of the symbol\n            nm_tyes  -> types as defined by 'nm' (man nm for examples)\n            module   -> which module to get the symbol from, default is kernel, otherwise can be any name seen in 'lsmod'\n    \n            This fixes a few issues from the old static hash table method:\n            1) Conflicting symbols can be handled, if a symbol is found to conflict on any profile, \n               then the plugin will need to provide the nm_type to differentiate, otherwise the plugin will be errored out\n            2) Can handle symbols gathered from modules on disk as well from the static kernel\n    \n            symtable is stored as a hash table of:\n            \n            symtable[module][sym_name] = [(symbol address, symbol type), (symbol addres, symbol type), ...]\n    \n            The function has overly verbose error checking on purpose...\n            "
            symtable = self.sys_map
            ret = None
            if module in symtable:
                mod = symtable[module]
                if sym_name in mod:
                    sym_list = mod[sym_name]
                    if len(sym_list) > 1:
                        if nm_type == '':
                            debug.error('Requested symbol {0:s} in module {1:s} has multiple definitions and no type given\n'.format(sym_name, module))
                        else:
                            for (addr, stype) in sym_list:
                                if stype == nm_type:
                                    ret = addr
                                    break
                            if ret == None:
                                debug.error('Requested symbol {0:s} in module {1:s} could not be found\n'.format(sym_name, module))
                    else:
                        ret = sym_list[0][0]
                else:
                    debug.debug('Requested symbol {0:s} not found in module {1:s}\n'.format(sym_name, module))
            else:
                debug.info('Requested module {0:s} not found in symbol table\n'.format(module))
            if self.shift_address and ret:
                ret = ret + self.shift_address
            return ret
    cls = AbstractMacProfile
    cls.__name__ = 'Mac' + profilename.replace('.', '_') + arch
    return cls
new_classes = []
for path in set(plugins.__path__):
    for (path, _, files) in os.walk(path):
        for fn in files:
            if zipfile.is_zipfile(os.path.join(path, fn)):
                new_classes.append(MacProfileFactory(zipfile.ZipFile(os.path.join(path, fn))))
kext_overlay = {'kmod_info_class': [None, {'name': [None, ['String', dict(length=64)]]}]}

class KextOverlay(obj.ProfileModification):
    conditions = {'os': lambda x: x == 'mac'}
    before = ['BasicObjectClasses']

    def modification(self, profile):
        if False:
            print('Hello World!')
        if 'kmod_info_class' in profile.vtypes:
            profile.merge_overlay(kext_overlay)

class MacOverlay(obj.ProfileModification):
    conditions = {'os': lambda x: x == 'mac'}
    before = ['BasicObjectClasses']

    def modification(self, profile):
        if False:
            while True:
                i = 10
        profile.merge_overlay(mac_overlay)

class MacObjectClasses(obj.ProfileModification):
    conditions = {'os': lambda x: x == 'mac'}
    before = ['BasicObjectClasses']

    def modification(self, profile):
        if False:
            print('Hello World!')
        profile.object_classes.update({'VolatilityDTB': VolatilityDTB, 'VolatilityMacIntelValidAS': VolatilityMacIntelValidAS, 'proc': proc, 'thread': thread, 'kauth_scope': kauth_scope, 'dyld32_image_info': dyld32_image_info, 'dyld64_image_info': dyld64_image_info, 'fileglob': fileglob, 'vnode': vnode, 'ifnet': ifnet, 'socket': socket, 'inpcbinfo': inpcbinfo, 'inpcb': inpcb, 'zone': zone, 'OSString': OSString, 'OSString_class': OSString, 'sysctl_oid': sysctl_oid, 'IpAddress': basic.IpAddress, 'Ipv6Address': basic.Ipv6Address, 'sockaddr': sockaddr, 'sockaddr_dl': sockaddr_dl, 'vm_map_entry': vm_map_entry, 'vm_map_object': vm_map_object, 'rtentry': rtentry, 'queue_entry': queue_entry, 'vm_page': vm_page})
mac_overlay = {'VOLATILITY_MAGIC': [None, {'DTB': [0, ['VolatilityDTB', dict(configname='DTB')]], 'IA32ValidAS': [0, ['VolatilityMacIntelValidAS']], 'AMD64ValidAS': [0, ['VolatilityMacIntelValidAS']]}], 'session': [None, {'s_login': [None, ['String', dict(length=256)]]}], 'kfs_event': [None, {'str': [None, ['pointer', ['String', dict(length=256)]]]}], 'zone': [None, {'zone_name': [None, ['pointer', ['String', dict(length=256)]]]}], 'mac_policy_conf': [None, {'mpc_name': [None, ['pointer', ['String', dict(length=256)]]]}], 'proc': [None, {'p_comm': [None, ['String', dict(length=17)]], 'task': [None, ['pointer', ['task']]]}], 'ifnet': [None, {'if_name': [None, ['pointer', ['String', dict(length=256)]]]}], 'vnode': [None, {'v_name': [None, ['pointer', ['String', dict(length=256)]]]}], 'boot_args': [None, {'CommandLine': [None, ['String', dict(length=1024)]]}], 'vfsstatfs': [None, {'f_fstypename': [None, ['String', dict(length=16)]], 'f_mntonname': [None, ['String', dict(length=1024)]], 'f_mntfromname': [None, ['String', dict(length=1024)]]}], 'kmod_info': [None, {'name': [None, ['String', dict(length=64)]], 'version': [None, ['String', dict(length=64)]]}], 'ipf_filter': [None, {'name': [None, ['pointer', ['String', dict(length=256)]]]}], 'sysctl_oid': [None, {'oid_name': [None, ['pointer', ['String', dict(length=256)]]]}], 'sockaddr_un': [None, {'sun_path': [None, ['String', dict(length=104)]]}], 'in_addr': [None, {'s_addr': [None, ['IpAddress']]}], 'in6_addr': [None, {'__u6_addr': [None, ['Ipv6Address']]}], 'inpcb': [None, {'inp_lport': [None, ['unsigned be short']], 'inp_fport': [None, ['unsigned be short']]}]}