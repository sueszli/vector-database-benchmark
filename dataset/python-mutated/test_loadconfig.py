import lief
from utils import get_sample
lief.logging.set_level(lief.logging.LOGGING_LEVEL.INFO)

def test_winapp():
    if False:
        return 10
    winapp = lief.parse(get_sample('PE/PE64_x86-64_binary_WinApp.exe'))
    assert winapp.has_configuration
    lconf = winapp.load_configuration
    assert lconf.version == lief.PE.WIN_VERSION.WIN10_0_15002
    assert lconf.characteristics == 248
    assert lconf.timedatestamp == 0
    assert lconf.major_version == 0
    assert lconf.minor_version == 0
    assert lconf.global_flags_clear == 0
    assert lconf.global_flags_set == 0
    assert lconf.critical_section_default_timeout == 0
    assert lconf.decommit_free_block_threshold == 0
    assert lconf.decommit_total_free_threshold == 0
    assert lconf.lock_prefix_table == 0
    assert lconf.maximum_allocation_size == 0
    assert lconf.virtual_memory_threshold == 0
    assert lconf.process_affinity_mask == 0
    assert lconf.process_heap_flags == 0
    assert lconf.csd_version == 0
    assert lconf.reserved1 == 0
    assert lconf.editlist == 0
    assert lconf.security_cookie == 1073795080
    assert lconf.se_handler_table == 0
    assert lconf.se_handler_count == 0
    assert lconf.guard_cf_check_function_pointer == 5368782848
    assert lconf.guard_cf_dispatch_function_pointer == 5368782864
    assert lconf.guard_cf_function_table == 5368778752
    assert lconf.guard_cf_function_count == 15
    expected_flags = lief.PE.GUARD_CF_FLAGS.GCF_LONGJUMP_TABLE_PRESENT
    expected_flags |= lief.PE.GUARD_CF_FLAGS.GCF_FUNCTION_TABLE_PRESENT
    expected_flags |= lief.PE.GUARD_CF_FLAGS.GCF_INSTRUMENTED
    assert lconf.guard_flags == expected_flags
    code_integrity = lconf.code_integrity
    assert code_integrity.flags == 0
    assert code_integrity.catalog == 0
    assert code_integrity.catalog_offset == 0
    assert code_integrity.reserved == 0
    assert print(code_integrity) is None
    assert lconf.guard_address_taken_iat_entry_table == 0
    assert lconf.guard_address_taken_iat_entry_count == 0
    assert lconf.guard_long_jump_target_table == 0
    assert lconf.guard_long_jump_target_count == 0
    assert lconf.dynamic_value_reloc_table == 0
    assert lconf.hybrid_metadata_pointer == 0
    assert lconf.guard_rf_failure_routine == 5368713280
    assert lconf.guard_rf_failure_routine_function_pointer == 5368782880
    assert lconf.dynamic_value_reloctable_offset == 0
    assert lconf.dynamic_value_reloctable_section == 0
    assert lconf.reserved2 == 0
    assert lconf.guard_rf_verify_stackpointer_function_pointer == 5368782896
    assert lconf.hotpatch_table_offset == 0
    assert print(lconf) is None

def test_v8():
    if False:
        print('Hello World!')
    pe = lief.parse(get_sample('PE/ANCUtility.dll'))
    assert pe.has_configuration
    lconf = pe.load_configuration
    assert lconf.version == lief.PE.WIN_VERSION.WIN10_0_18362
    assert isinstance(lconf, lief.PE.LoadConfigurationV8)
    assert lconf.volatile_metadata_pointer == 0
    assert print(lconf) is None

def test_v9():
    if False:
        for i in range(10):
            print('nop')
    pe = lief.parse(get_sample('PE/ucrtbase.dll'))
    assert pe.has_configuration
    lconf = pe.load_configuration
    assert lconf.version == lief.PE.WIN_VERSION.WIN10_0_19534
    assert isinstance(lconf, lief.PE.LoadConfigurationV9)
    assert lconf.guard_eh_continuation_table == 6443210608
    assert lconf.guard_eh_continuation_count == 34
    assert print(lconf) is None

def test_v11():
    if False:
        print('Hello World!')
    pe = lief.parse(get_sample('PE/hostfxr.dll'))
    assert pe.has_configuration
    lconf = pe.load_configuration
    assert lconf.version == lief.PE.WIN_VERSION.WIN10_0_MSVC_2019_16
    assert isinstance(lconf, lief.PE.LoadConfigurationV11)
    assert lconf.guard_xfg_check_function_pointer == 6442718424
    assert lconf.guard_xfg_dispatch_function_pointer == 6442718440
    assert lconf.guard_xfg_table_dispatch_function_pointer == 6442718448
    assert lconf.cast_guard_os_determined_failure_mode == 6442810904
    assert print(lconf) is None
    assert lconf.copy() == lconf