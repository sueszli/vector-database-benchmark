import lief
import subprocess
from utils import get_sample, is_apple_m1, sign, chmod_exe

def process_crypt_and_hash(path: str, delta: int=0):
    if False:
        return 10
    '\n    Test on a regular Mach-O binary that contains rebase fixups\n    '
    fat = lief.MachO.parse(path)
    target = fat.take(lief.MachO.CPU_TYPES.ARM64)
    assert target.has(lief.MachO.LOAD_COMMAND_TYPES.DYLD_CHAINED_FIXUPS)
    dyld_chained = target.get(lief.MachO.LOAD_COMMAND_TYPES.DYLD_CHAINED_FIXUPS)
    assert dyld_chained.fixups_version == 0
    assert dyld_chained.starts_offset == 32
    assert dyld_chained.imports_offset == 112
    assert dyld_chained.symbols_offset == 272
    assert dyld_chained.imports_count == 40
    assert dyld_chained.imports_format == lief.MachO.DYLD_CHAINED_FORMAT.IMPORT
    assert len(dyld_chained.chained_starts_in_segments) == 5
    assert len(dyld_chained.bindings) == 41
    start_in_segment = dyld_chained.chained_starts_in_segments[2]
    assert start_in_segment.offset == 24
    assert start_in_segment.size == 26
    assert start_in_segment.page_size == 16384
    assert start_in_segment.segment_offset == 409600 + delta
    assert start_in_segment.pointer_format == lief.MachO.DYLD_CHAINED_PTR_FORMAT.PTR_64_OFFSET
    assert start_in_segment.max_valid_pointer == 0
    assert start_in_segment.page_count == 2
    assert start_in_segment.segment.name == '__DATA_CONST'
    assert start_in_segment.page_start[0] == 0
    assert start_in_segment.page_start[1] == 16
    rebases = start_in_segment.segment.relocations
    assert len(rebases) == 1247
    assert (rebases[0].address, rebases[0].target) == (4295377232 + delta, 4295265630 + delta)
    assert (rebases[1246].address, rebases[1246].target) == (4295406440 + delta, 4295401872 + delta)
    assert (rebases[389].address, rebases[389].target) == (4295385088 + delta, 4295342648 + delta)
    start_in_segment = dyld_chained.chained_starts_in_segments[3]
    rebases = start_in_segment.segment.relocations
    assert len(rebases) == 15
    assert (rebases[0].address, rebases[0].target) == (4295409664 + delta, 4295314352 + delta)
    assert (rebases[14].address, rebases[14].target) == (4295409784 + delta, 4295278339 + delta)

def test_1():
    if False:
        return 10
    '\n    Simple test on the regular id binary comming from an Apple M1\n    This sample does not contains rebase fixups\n    '
    fat = lief.MachO.parse(get_sample('MachO/8119b2bd6a15b78b5c0bc2245eb63673173cb8fe9e0638f19aea7e68da668696_id.macho'))
    target = fat.take(lief.MachO.CPU_TYPES.ARM64)
    assert target.has(lief.MachO.LOAD_COMMAND_TYPES.DYLD_CHAINED_FIXUPS)
    dyld_chained = target.get(lief.MachO.LOAD_COMMAND_TYPES.DYLD_CHAINED_FIXUPS)
    assert dyld_chained.fixups_version == 0
    assert dyld_chained.starts_offset == 32
    assert dyld_chained.imports_offset == 80
    assert dyld_chained.symbols_offset == 192
    assert dyld_chained.imports_count == 28
    assert dyld_chained.imports_format == lief.MachO.DYLD_CHAINED_FORMAT.IMPORT
    assert len(dyld_chained.chained_starts_in_segments) == 5
    assert len(dyld_chained.bindings) == 28
    start_in_segment: lief.MachO.DyldChainedFixups.chained_starts_in_segment = dyld_chained.chained_starts_in_segments[2]
    assert start_in_segment.offset == 24
    assert start_in_segment.size == 24
    assert start_in_segment.page_size == 16384
    assert start_in_segment.pointer_format == lief.MachO.DYLD_CHAINED_PTR_FORMAT.PTR_ARM64E_USERLAND24
    assert start_in_segment.max_valid_pointer == 0
    assert start_in_segment.page_count == 1
    assert start_in_segment.segment.name == '__DATA_CONST'
    assert start_in_segment.page_start[0] == 0
    assert len(start_in_segment.segment.relocations) == 0
    bindings = dyld_chained.bindings
    assert len(bindings) == 28
    bnd_0 = bindings[0]
    assert bnd_0.offset == 16384
    assert bnd_0.format == lief.MachO.DYLD_CHAINED_FORMAT.IMPORT
    assert bnd_0.ptr_format == lief.MachO.DYLD_CHAINED_PTR_FORMAT.PTR_ARM64E_USERLAND24
    assert bnd_0.symbol.name == '_err'
    assert bnd_0.segment.name == '__DATA_CONST'
    assert bnd_0.library.name == '/usr/lib/libSystem.B.dylib'
    assert bnd_0.address == 4294983680
    assert bnd_0.sign_extended_addend == 0
    assert not bnd_0.weak_import
    bnd_14 = bindings[14]
    assert bnd_14.offset == 16496
    assert bnd_14.format == lief.MachO.DYLD_CHAINED_FORMAT.IMPORT
    assert bnd_14.ptr_format == lief.MachO.DYLD_CHAINED_PTR_FORMAT.PTR_ARM64E_USERLAND24
    assert bnd_14.symbol.name == '_getopt'
    assert bnd_14.segment.name == '__DATA_CONST'
    assert bnd_14.library.name == '/usr/lib/libSystem.B.dylib'
    assert bnd_14.address == 4294983792
    assert not bnd_14.weak_import
    assert bnd_14.sign_extended_addend == 0
    bnd_27 = bindings[27]
    assert bnd_27.offset == 16600
    assert bnd_27.format == lief.MachO.DYLD_CHAINED_FORMAT.IMPORT
    assert bnd_27.ptr_format == lief.MachO.DYLD_CHAINED_PTR_FORMAT.PTR_ARM64E_USERLAND24
    assert bnd_27.symbol.name == '_optind'
    assert bnd_27.segment.name == '__DATA_CONST'
    assert bnd_27.library.name == '/usr/lib/libSystem.B.dylib'
    assert bnd_27.address == 4294983896
    assert not bnd_27.weak_import
    assert bnd_27.sign_extended_addend == 0

def test_2():
    if False:
        print('Hello World!')
    process_crypt_and_hash(get_sample('MachO/9edfb04c55289c6c682a25211a4b30b927a86fe50b014610d04d6055bd4ac23d_crypt_and_hash.macho'))

def test_3():
    if False:
        i = 10
        return i + 15
    '\n    Test on dyld which contains DYLD_CHAINED_PTR_FORMAT.PTR_32\n    '
    fat = lief.MachO.parse(get_sample('MachO/42d4f6b799d5d3ff88c50d4c6966773d269d19793226724b5e893212091bf737_dyld.macho'))
    target = fat.take(lief.MachO.CPU_TYPES.x86)
    assert target.has(lief.MachO.LOAD_COMMAND_TYPES.DYLD_CHAINED_FIXUPS)
    dyld_chained = target.get(lief.MachO.LOAD_COMMAND_TYPES.DYLD_CHAINED_FIXUPS)
    assert dyld_chained.fixups_version == 0
    assert dyld_chained.starts_offset == 32
    assert dyld_chained.imports_offset == 616
    assert dyld_chained.symbols_offset == 616
    assert dyld_chained.imports_count == 0
    assert dyld_chained.imports_format == lief.MachO.DYLD_CHAINED_FORMAT.IMPORT
    assert dyld_chained.symbols_format == 0
    assert len(dyld_chained.chained_starts_in_segments) == 4
    assert len(dyld_chained.bindings) == 0
    start_in_segment = dyld_chained.chained_starts_in_segments[1]
    assert start_in_segment.offset == 24
    assert start_in_segment.size == 278
    assert start_in_segment.page_size == 16384
    assert start_in_segment.segment_offset == 360448
    assert start_in_segment.pointer_format == lief.MachO.DYLD_CHAINED_PTR_FORMAT.PTR_32
    assert start_in_segment.max_valid_pointer == 1048576
    assert start_in_segment.page_count == 1
    assert start_in_segment.segment.name == '__DATA_CONST'
    assert start_in_segment.page_start[0] == 228
    rebases = start_in_segment.segment.relocations
    assert len(rebases) == 952
    start_in_segment = dyld_chained.chained_starts_in_segments[2]
    assert start_in_segment.offset == 304
    assert start_in_segment.size == 278
    assert start_in_segment.page_size == 16384
    assert start_in_segment.segment_offset == 442368
    assert start_in_segment.pointer_format == lief.MachO.DYLD_CHAINED_PTR_FORMAT.PTR_32
    assert start_in_segment.max_valid_pointer == 1048576
    assert start_in_segment.page_count == 1
    assert start_in_segment.segment.name == '__DATA'
    assert start_in_segment.page_start[0] == 32769
    rebases = start_in_segment.segment.relocations
    assert len(rebases) == 33
    assert (rebases[0].address, rebases[0].target) == (442368, 283688)
    assert (rebases[23].address, rebases[23].target) == (442460, 0)
    assert (rebases[32].address, rebases[32].target) == (442888, 323321)

def test_builder(tmp_path):
    if False:
        print('Hello World!')
    binary_name = 'crypt_and_hash'
    fat = lief.MachO.parse(get_sample('MachO/9edfb04c55289c6c682a25211a4b30b927a86fe50b014610d04d6055bd4ac23d_crypt_and_hash.macho'))
    target = fat.take(lief.MachO.CPU_TYPES.ARM64)
    output = f'{tmp_path}/{binary_name}.built'
    target.write(output)
    process_crypt_and_hash(output)
    if is_apple_m1():
        chmod_exe(output)
        sign(output)
        with subprocess.Popen([output], universal_newlines=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT) as proc:
            stdout = proc.stdout.read()
            assert 'CAMELLIA-256-CCM*-NO-TAG' in stdout
            assert 'AES-128-CCM*-NO-TAG' in stdout

def test_linkedit_shift(tmp_path):
    if False:
        for i in range(10):
            print('nop')
    binary_name = 'crypt_and_hash'
    fat = lief.MachO.parse(get_sample('MachO/9edfb04c55289c6c682a25211a4b30b927a86fe50b014610d04d6055bd4ac23d_crypt_and_hash.macho'))
    target: lief.MachO.Binary = fat.take(lief.MachO.CPU_TYPES.ARM64)
    target.shift_linkedit(16384)
    output = f'{tmp_path}/{binary_name}.built'
    target.remove_signature()
    target.write(output)
    process_crypt_and_hash(output)
    if is_apple_m1():
        chmod_exe(output)
        sign(output)
        with subprocess.Popen([output], universal_newlines=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT) as proc:
            stdout = proc.stdout.read()
            assert 'CAMELLIA-256-CCM*-NO-TAG' in stdout
            assert 'AES-128-CCM*-NO-TAG' in stdout

def test_shift(tmp_path):
    if False:
        print('Hello World!')
    DELTA = 16384
    binary_name = 'crypt_and_hash'
    fat = lief.MachO.parse(get_sample('MachO/9edfb04c55289c6c682a25211a4b30b927a86fe50b014610d04d6055bd4ac23d_crypt_and_hash.macho'))
    target = fat.take(lief.MachO.CPU_TYPES.ARM64)
    target.shift(DELTA)
    output = f'{tmp_path}/{binary_name}.built'
    target.write(output)
    process_crypt_and_hash(output, DELTA)
    if is_apple_m1():
        chmod_exe(output)
        sign(output)
        with subprocess.Popen([output], universal_newlines=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT) as proc:
            stdout = proc.stdout.read()
            assert 'CAMELLIA-256-CCM*-NO-TAG' in stdout
            assert 'AES-128-CCM*-NO-TAG' in stdout

def test_issue_804(tmp_path):
    if False:
        for i in range(10):
            print('nop')
    fat = lief.MachO.parse(get_sample('MachO/test_issue_804.bin'))
    target = fat.take(lief.MachO.CPU_TYPES.ARM64)
    bindings = target.dyld_chained_fixups.bindings
    assert len(bindings) == 5
    objc_nsobj = set((binding.address for binding in bindings if binding.symbol.name == '_OBJC_METACLASS_$_NSObject'))
    assert objc_nsobj == {4295000208, 4295000216}
    output = f'{tmp_path}/test_issue_804.built'
    target.write(output)
    fat = lief.MachO.parse(output)
    target = fat.take(lief.MachO.CPU_TYPES.ARM64)
    bindings = target.dyld_chained_fixups.bindings
    assert len(bindings) == 5
    objc_nsobj = set((binding.address for binding in bindings if binding.symbol.name == '_OBJC_METACLASS_$_NSObject'))
    assert objc_nsobj == {4295000208, 4295000216}
    if is_apple_m1():
        chmod_exe(output)
        sign(output)
        with subprocess.Popen([output], universal_newlines=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT) as proc:
            stdout = proc.stdout.read()

def test_issue_853(tmp_path):
    if False:
        for i in range(10):
            print('nop')
    ios14 = lief.parse(get_sample('MachO/issue_853_classes_14.bin'))
    relocations = ios14.relocations
    assert len(relocations) == 31
    assert all((0 < r.target - ios14.imagebase and r.target - ios14.imagebase < ios14.imagebase for r in relocations))
    output = f'{tmp_path}/test_issue_853_ios14.bin'
    ios14.write(output)
    ios14_built = lief.parse(output)
    assert len(ios14_built.relocations) == 31
    assert ios14_built.relocations[0].target == 4294999720
    ios15 = lief.parse(get_sample('MachO/issue_853_classes_15.bin'))
    relocations = ios15.relocations
    assert len(relocations) == 31
    assert all((0 < r.target - ios15.imagebase and r.target - ios15.imagebase < ios15.imagebase for r in relocations))
    output = f'{tmp_path}/test_issue_853_ios15.bin'
    ios15.write(output)
    ios15_built = lief.parse(output)
    assert len(ios15_built.relocations) == 31
    assert ios15_built.relocations[0].target == 4294999720