import lief
from utils import get_sample
import hashlib

def test_function_starts():
    if False:
        while True:
            i = 10
    dd = lief.parse(get_sample('MachO/MachO64_x86-64_binary_dd.bin'))
    functions = [4294972801, 4294973132, 4294973388, 4294973923, 4294973955, 4294973981, 4294975661, 4294976246, 4294976495, 4294976619, 4294976652, 4294977242, 4294977364, 4294977643, 4294977812, 4294978520, 4294978536, 4294978603, 4294978658, 4294978852, 4294978906, 4294978961, 4294979029, 4294979046, 4294979068, 4294979136, 4294979153, 4294979175, 4294979486]
    assert dd.function_starts.data_offset == 21168
    assert dd.function_starts.data_size == 48
    text_segment = list(filter(lambda e: e.name == '__TEXT', dd.segments))[0]
    functions_dd = map(text_segment.virtual_address.__add__, dd.function_starts.functions)
    assert functions == list(functions_dd)

def test_version_min():
    if False:
        for i in range(10):
            print('nop')
    sshd = lief.parse(get_sample('MachO/MachO64_x86-64_binary_sshd.bin'))
    assert sshd.version_min.version == [10, 11, 0]
    assert sshd.version_min.sdk == [10, 11, 0]

def test_va2offset():
    if False:
        for i in range(10):
            print('nop')
    dd = lief.parse(get_sample('MachO/MachO64_x86-64_binary_dd.bin'))
    assert dd.virtual_address_to_offset(4294983764) == 16468

def test_thread_cmd():
    if False:
        while True:
            i = 10
    micromacho = lief.parse(get_sample('MachO/MachO32_x86_binary_micromacho.bin'))
    assert micromacho.has_thread_command
    assert micromacho.thread_command.pc == 104
    assert micromacho.thread_command.flavor == 1
    assert micromacho.thread_command.count == 16
    assert micromacho.entrypoint == 104

def test_rpath_cmd():
    if False:
        while True:
            i = 10
    rpathmacho = lief.parse(get_sample('MachO/MachO64_x86-64_binary_rpathtest.bin'))
    assert rpathmacho.rpath.path == '@executable_path/../lib'

def test_rpaths():
    if False:
        return 10
    macho = lief.parse(get_sample('MachO/rpath_291.bin'))
    assert len(macho.rpaths) == 2
    assert macho.rpaths[0].path == '/tmp'
    assert macho.rpaths[1].path == '/var'

def test_relocations():
    if False:
        return 10
    helloworld = lief.parse(get_sample('MachO/MachO64_x86-64_object_HelloWorld64.o'))
    text_section = helloworld.get_section('__text')
    relocations = text_section.relocations
    assert len(relocations) == 2
    assert relocations[0].address == 563
    assert relocations[0].type == 2
    assert relocations[0].size == 32
    assert not relocations[0].is_scattered
    assert relocations[0].has_symbol
    assert relocations[0].symbol.name == '_printf'
    assert relocations[0].has_section
    assert relocations[0].section.name == text_section.name
    assert relocations[1].address == 539
    assert relocations[1].type == 1
    assert relocations[1].size == 32
    assert not relocations[1].is_scattered
    assert not relocations[1].has_symbol
    assert relocations[1].has_section
    assert relocations[1].section.name == text_section.name
    cunwind_section = helloworld.get_section('__compact_unwind')
    relocations = cunwind_section.relocations
    assert len(relocations) == 1
    assert relocations[0].address == 583
    assert relocations[0].type == 0
    assert relocations[0].size == 32
    assert not relocations[0].is_scattered
    assert not relocations[0].has_symbol
    assert relocations[0].has_section
    assert relocations[0].section.name == '__cstring'

def test_data_in_code():
    if False:
        return 10
    binary = lief.parse(get_sample('MachO/MachO32_ARM_binary_data-in-code-LLVM.bin'))
    assert binary.has_data_in_code
    dcode = binary.data_in_code
    assert dcode.data_offset == 284
    assert dcode.data_size == 32
    assert len(dcode.entries) == 4
    assert dcode.entries[0].type == lief.MachO.DataCodeEntry.TYPES.DATA
    assert dcode.entries[0].offset == 0
    assert dcode.entries[0].length == 4
    assert dcode.entries[1].type == lief.MachO.DataCodeEntry.TYPES.JUMP_TABLE_32
    assert dcode.entries[1].offset == 4
    assert dcode.entries[1].length == 4
    assert dcode.entries[2].type == lief.MachO.DataCodeEntry.TYPES.JUMP_TABLE_16
    assert dcode.entries[2].offset == 8
    assert dcode.entries[2].length == 2
    assert dcode.entries[3].type == lief.MachO.DataCodeEntry.TYPES.JUMP_TABLE_8
    assert dcode.entries[3].offset == 10
    assert dcode.entries[3].length == 1

def test_segment_split_info():
    if False:
        print('Hello World!')
    binary = lief.parse(get_sample('MachO/FAT_MachO_x86_x86-64_library_libdyld.dylib'))
    assert binary.has_segment_split_info
    ssi = binary.segment_split_info
    assert ssi.data_offset == 32852
    assert ssi.data_size == 292

def test_dyld_environment():
    if False:
        print('Hello World!')
    binary = lief.parse(get_sample('MachO/MachO64_x86-64_binary_safaridriver.bin'))
    assert binary.has_dyld_environment
    assert binary.dyld_environment.value == 'DYLD_VERSIONED_FRAMEWORK_PATH=/System/Library/StagedFrameworks/Safari'

def test_sub_framework():
    if False:
        for i in range(10):
            print('nop')
    binary = lief.parse(get_sample('MachO/FAT_MachO_x86_x86-64_library_libdyld.dylib'))
    assert binary.has_sub_framework
    assert binary.sub_framework.umbrella == 'System'

def test_unwind():
    if False:
        return 10
    binary = lief.parse(get_sample('MachO/MachO64_x86-64_binary_sshd.bin'))
    functions = sorted(binary.functions, key=lambda f: f.address)
    assert len(functions) == 2619
    assert functions[0].address == 2624
    assert functions[0].size == 0
    assert functions[0].name == ''
    assert functions[-1].address == 4295642981
    assert functions[-1].size == 0
    assert functions[-1].name == 'ctor_0'

def test_build_version():
    if False:
        return 10
    binary = lief.MachO.parse(get_sample('MachO/FAT_MachO_arm-arm64-binary-helloworld.bin'))
    target = binary[1]
    assert target.has_build_version
    build_version = target.build_version
    assert build_version.minos == [12, 1, 0]
    assert build_version.sdk == [12, 1, 0]
    assert build_version.platform == lief.MachO.BuildVersion.PLATFORMS.IOS
    tools = build_version.tools
    assert len(tools) == 1
    assert tools[0].version == [409, 12, 0]
    assert tools[0].tool == lief.MachO.BuildToolVersion.TOOLS.LD

def test_segment_index():
    if False:
        print('Hello World!')
    binary = lief.parse(get_sample('MachO/MachO64_x86-64_binary_safaridriver.bin'))
    assert binary.get_segment('__LINKEDIT').index == len(binary.segments) - 1
    original_data_index = binary.get_segment('__DATA').index
    segment = lief.MachO.SegmentCommand('__LIEF', [96] * 256)
    segment = binary.add(segment)
    assert segment.index == binary.get_segment('__LINKEDIT').index - 1
    assert segment.index == original_data_index + 1
    binary = lief.parse(get_sample('MachO/MachO64_x86-64_binary_safaridriver.bin'))
    text_segment = binary.get_segment('__TEXT')
    original_data_index = binary.get_segment('__DATA').index
    binary.remove(text_segment)
    assert binary.get_segment('__DATA').index == original_data_index - 1
    assert binary.get_segment('__LINKEDIT').index == original_data_index
    assert binary.get_segment('__PAGEZERO').index == 0

def test_offset_to_va():
    if False:
        return 10
    sample = get_sample('MachO/MachO64_x86-64_binary_large-bss.bin')
    large_bss = lief.parse(sample)
    assert large_bss.segment_from_offset(0).name == '__TEXT'
    assert large_bss.segment_from_offset(16385).name == '__DATA_CONST'
    assert large_bss.segment_from_offset(49152).name == '__LINKEDIT'
    assert large_bss.segment_from_offset(49153).name == '__LINKEDIT'

def test_get_section():
    if False:
        while True:
            i = 10
    sample = get_sample('MachO/MachO64_x86-64_binary_large-bss.bin')
    macho = lief.parse(sample)
    assert macho.get_section('__DATA_CONST', '__got') is not None

def test_segment_add_section():
    if False:
        print('Hello World!')
    binary = lief.parse(get_sample('MachO/MachO64_x86-64_binary_safaridriver.bin'))
    section = lief.MachO.Section('__bar', [1, 2, 3])
    existing_segment = binary.get_segment('__TEXT')
    new_segment = lief.MachO.SegmentCommand('__FOO')
    for segment in (existing_segment, new_segment):
        assert not segment.has_section(section.name)
        assert not segment.has(section)
        assert segment.numberof_sections == len(segment.sections)
        numberof_sections = segment.numberof_sections
        section = segment.add_section(section)
        assert segment.numberof_sections == numberof_sections + 1
        assert segment.has_section(section.name)
        assert segment.has(section)
        assert section in segment.sections

def test_issue_728():
    if False:
        print('Hello World!')
    x86_64_binary = lief.parse(get_sample('MachO/MachO64_x86-64_binary_safaridriver.bin'))
    arm64_binary = lief.MachO.parse(get_sample('MachO/FAT_MachO_arm-arm64-binary-helloworld.bin')).take(lief.MachO.CPU_TYPES.ARM64)
    segment = lief.MachO.SegmentCommand('__FOO')
    segment.add_section(lief.MachO.Section('__bar', [1, 2, 3]))
    for parsed in (x86_64_binary, arm64_binary):
        new_segment = parsed.add(segment)
        assert new_segment.virtual_size == parsed.page_size

def test_twolevel_hints():
    if False:
        for i in range(10):
            print('nop')
    sample = lief.MachO.parse(get_sample('MachO/ios1-expr.bin'))[0]
    tw_hints: lief.MachO.TwoLevelHints = sample[lief.MachO.LOAD_COMMAND_TYPES.TWOLEVEL_HINTS]
    assert tw_hints is not None
    print(tw_hints)
    hints = tw_hints.hints
    assert len(hints) == 26
    print(hints[0])
    assert sum(hints) == 10854400
    assert hints[0] == 54528
    assert hashlib.sha256(tw_hints.data).hexdigest() == 'e44cef3a83eb89954557a9ad2a36ebf4794ce0385da5a39381fdadc3e6037beb'
    assert tw_hints.command_offset == 1552
    print(lief.to_json(tw_hints))