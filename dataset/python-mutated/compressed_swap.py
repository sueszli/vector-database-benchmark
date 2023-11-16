"""
@author:       Golden G. Richard III
@license:      GNU General Public License 2.0
@contact:      golden@arcanealloy.com
@organization: Arcane Alloy, LLC
"""
import volatility.obj as obj
import volatility.debug as debug
import volatility.plugins.mac.common as common
from struct import pack
import WKdm

class mac_compressed_swap(common.AbstractMacCommand):
    """ Prints Mac OS X VM compressor stats and dumps all compressed pages """

    def __init__(self, config, *args, **kwargs):
        if False:
            return 10
        common.AbstractMacCommand.__init__(self, config, *args, **kwargs)
        if config:
            self._config.add_option('SKIP-WRITING', short_option='t', help='Skip writing decompressed pages, just print stats and test decompression', action='store_true', default=False)
        self.C_SEG_BUFSIZE = 1024 * 256
        self.C_SEG_ALLOCSIZE = self.C_SEG_BUFSIZE + 4096
        self.C_SEG_SLOT_ARRAYS = 6
        self.C_SEG_SLOT_ARRAY_SIZE = 64
        self.COMPRESSOR_SLOTS_CHUNK_SIZE = 512
        self.COMPRESSOR_SLOTS_PER_CHUNK = 128
        self.wkdm = WKdm.WKdm()
        self.dest = [0] * self.wkdm.PAGE_SIZE_IN_BYTES

    def calculate(self):
        if False:
            for i in range(10):
                print('nop')
        common.set_plugin_members(self)
        com_obj_addr = self.addr_space.profile.get_symbol('_compressor_object_store')
        if not com_obj_addr:
            debug.error('The given memory sample does not utilize compressed swap.')
        compressor_object = obj.Object('vm_object', offset=com_obj_addr, vm=self.addr_space)
        c_segment_count = obj.Object('unsigned int', offset=self.addr_space.profile.get_symbol('_c_segment_count'), vm=self.addr_space)
        c_segments_ptr = obj.Object('Pointer', offset=self.addr_space.profile.get_symbol('_c_segments'), vm=self.addr_space)
        c_segments = obj.Object('Array', targetType='c_segu', count=c_segment_count, offset=c_segments_ptr, vm=self.addr_space)
        c_segments_available = obj.Object('unsigned int', offset=self.addr_space.profile.get_symbol('_c_segments_available'), vm=self.addr_space)
        c_segments_busy = obj.Object('unsigned int', offset=self.addr_space.profile.get_symbol('_c_segments_busy'), vm=self.addr_space)
        c_segment_compressed_bytes = obj.Object('long long', offset=self.addr_space.profile.get_symbol('_c_segment_compressed_bytes'), vm=self.addr_space)
        compressor_bytes_used = obj.Object('long long', offset=self.addr_space.profile.get_symbol('_compressor_bytes_used'), vm=self.addr_space)
        yield ('Compressor memory used', compressor_bytes_used, 'bytes')
        vm_page_active_count = obj.Object('unsigned int', offset=self.addr_space.profile.get_symbol('_vm_page_active_count'), vm=self.addr_space)
        vm_page_inactive_count = obj.Object('unsigned int', offset=self.addr_space.profile.get_symbol('_vm_page_inactive_count'), vm=self.addr_space)
        vm_page_free_count = obj.Object('unsigned int', offset=self.addr_space.profile.get_symbol('_vm_page_free_count'), vm=self.addr_space)
        vm_page_speculative_count = obj.Object('unsigned int', offset=self.addr_space.profile.get_symbol('_vm_page_speculative_count'), vm=self.addr_space)
        available_uncompressed = vm_page_active_count + vm_page_inactive_count + vm_page_free_count + vm_page_speculative_count
        yield ('Available uncompressed memory', available_uncompressed, 'pages')
        available_memory = available_uncompressed + compressor_object.resident_page_count
        yield ('Available memory', available_memory, 'pages')
        yield ('Segments available', c_segments_available, 'segments')
        yield ('Segments busy', c_segments_busy, 'segments')
        yield ('Current segment count', c_segment_count, 'segments')
        for i in range(c_segment_count):
            if not c_segments[i].c_seg.is_valid():
                yield ('Segment ' + str(i) + ' is invalid', 'SKIPPING', '')
                continue
            if c_segments[i].c_seg.c_ondisk == 1:
                yield ('Segment ' + str(i) + ' is swapped out', 'SKIPPING', '')
                continue
            if c_segments[i].c_seg.c_bytes_used < 1 or c_segments[i].c_seg.c_bytes_used > self.C_SEG_ALLOCSIZE:
                yield ('Segment ' + str(i) + ' size is invalid', 'SKIPPING', '')
                continue
            yield ('Segment ' + str(i), c_segments[i].c_seg.c_bytes_used, 'bytes used')
            yield ('Segment ' + str(i), c_segments[i].c_seg.c_bytes_unused, 'bytes unused')
            j1 = 0
            j2 = 0
            c_nextslot = c_segments[i].c_seg.c_nextslot
            yield ('Last valid slot', str((c_nextslot - 1) / self.C_SEG_SLOT_ARRAY_SIZE) + ', ' + str((c_nextslot - 1) % self.C_SEG_SLOT_ARRAY_SIZE), '')
            while j1 < self.C_SEG_SLOT_ARRAYS and j1 * self.C_SEG_SLOT_ARRAY_SIZE + j2 < c_nextslot:
                cslot_array = c_segments[i].c_seg.c_slots[j1]
                if cslot_array.is_valid():
                    cslots = obj.Object('Array', offset=cslot_array, targetType='c_slot', count=self.C_SEG_SLOT_ARRAY_SIZE, vm=self.addr_space)
                    while j2 < self.C_SEG_SLOT_ARRAY_SIZE and j1 * self.C_SEG_SLOT_ARRAY_SIZE + j2 < c_nextslot:
                        cslot = cslots[j2]
                        (csize, compressed, status) = (4096 / 4, False, 'UNCOMPRESSED') if cslot.c_size == 4095 else (cslot.c_size / 4, True, 'COMPRESSED')
                        if csize > 0:
                            yield ('  Slot ' + str(j1) + ', ' + str(j2) + ' offset', str(cslot.c_offset * 4), 'bytes')
                            yield ('  Slot ' + str(j1) + ', ' + str(j2) + ' size', str(csize * 4), 'bytes ' + status)
                            cslot_data = obj.Object('Array', offset=c_segments[i].c_seg.c_store.c_buffer + cslot.c_offset * 4, targetType='int', count=csize, vm=self.addr_space)
                            yield ('  Processing page at slot ' + str(j1) + ', ' + str(j2), '', '')
                            if compressed:
                                decompressed = self.wkdm.WKdm_decompress(cslot_data, self.dest)
                                if decompressed > 0:
                                    if not self._config.SKIP_WRITING:
                                        f = open(str(i) + '-' + str(j1) + '-' + str(j2) + '-decompressed.out', 'wb')
                                        for k in range(decompressed):
                                            f.write(pack('<i', self.dest[k]))
                                        f.close()
                                else:
                                    yield ('  Decompression failed on slot ' + str(j1) + ', ' + str(j2), '', 'SKIPPING')
                            elif not self._config.SKIP_WRITING:
                                f = open(str(i) + '-' + str(j1) + '-' + str(j2) + '-uncompressed.out', 'wb')
                                for k in range(0, csize):
                                    f.write(pack('<i', cslot_data[k]))
                                f.close()
                        j2 += 1
                    j2 = 0
                else:
                    yield ('  Slot array ' + str(j1) + ' is invalid', '', 'SKIPPING')
                j1 += 1

    def render_text(self, outfd, data):
        if False:
            while True:
                i = 10
        for (k, v1, v2) in data:
            outfd.write('{0:<36} : {1:>12} {2}\n'.format(k, v1, v2))