import lief
from utils import get_sample

def test_header():
    if False:
        i = 10
        return i + 15
    CallDeviceId = lief.parse(get_sample('OAT/OAT_138_AArch64_android.uid.systemui.oat'))
    header = CallDeviceId.header
    assert header.magic == [111, 97, 116, 10]
    assert header.version == 138
    assert header.checksum == 1550111048
    assert header.instruction_set == lief.OAT.INSTRUCTION_SETS.ARM_64
    assert header.nb_dex_files == 1
    assert header.oat_dex_files_offset == 3289146
    assert header.executable_offset == 3293184
    assert header.i2i_bridge_offset == 0
    assert header.i2c_code_bridge_offset == 0
    assert header.jni_dlsym_lookup_offset == 0
    assert header.quick_generic_jni_trampoline_offset == 0
    assert header.quick_imt_conflict_trampoline_offset == 0
    assert header.quick_resolution_trampoline_offset == 0
    assert header.quick_to_interpreter_bridge_offset == 0
    assert header.image_patch_delta == 0
    assert header.image_file_location_oat_checksum == 2394378138
    assert header.image_file_location_oat_data_begin == 1898192896