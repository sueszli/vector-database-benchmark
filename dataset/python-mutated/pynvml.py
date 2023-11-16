from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
_nvmlEnableState_t = c_uint
NVML_FEATURE_DISABLED = 0
NVML_FEATURE_ENABLED = 1
_nvmlBrandType_t = c_uint
NVML_BRAND_UNKNOWN = 0
NVML_BRAND_QUADRO = 1
NVML_BRAND_TESLA = 2
NVML_BRAND_NVS = 3
NVML_BRAND_GRID = 4
NVML_BRAND_GEFORCE = 5
NVML_BRAND_TITAN = 6
NVML_BRAND_NVIDIA_VAPPS = 7
NVML_BRAND_NVIDIA_VPC = 8
NVML_BRAND_NVIDIA_VCS = 9
NVML_BRAND_NVIDIA_VWS = 10
NVML_BRAND_NVIDIA_CLOUD_GAMING = 11
NVML_BRAND_NVIDIA_VGAMING = NVML_BRAND_NVIDIA_CLOUD_GAMING
NVML_BRAND_QUADRO_RTX = 12
NVML_BRAND_NVIDIA_RTX = 13
NVML_BRAND_NVIDIA = 14
NVML_BRAND_GEFORCE_RTX = 15
NVML_BRAND_TITAN_RTX = 16
NVML_BRAND_COUNT = 17
_nvmlTemperatureThresholds_t = c_uint
NVML_TEMPERATURE_THRESHOLD_SHUTDOWN = 0
NVML_TEMPERATURE_THRESHOLD_SLOWDOWN = 1
NVML_TEMPERATURE_THRESHOLD_MEM_MAX = 2
NVML_TEMPERATURE_THRESHOLD_GPU_MAX = 3
NVML_TEMPERATURE_THRESHOLD_ACOUSTIC_MIN = 4
NVML_TEMPERATURE_THRESHOLD_ACOUSTIC_CURR = 5
NVML_TEMPERATURE_THRESHOLD_ACOUSTIC_MAX = 6
NVML_TEMPERATURE_THRESHOLD_COUNT = 7
_nvmlTemperatureSensors_t = c_uint
NVML_TEMPERATURE_GPU = 0
NVML_TEMPERATURE_COUNT = 1
_nvmlComputeMode_t = c_uint
NVML_COMPUTEMODE_DEFAULT = 0
NVML_COMPUTEMODE_EXCLUSIVE_THREAD = 1
NVML_COMPUTEMODE_PROHIBITED = 2
NVML_COMPUTEMODE_EXCLUSIVE_PROCESS = 3
NVML_COMPUTEMODE_COUNT = 4
_nvmlMemoryLocation_t = c_uint
NVML_MEMORY_LOCATION_L1_CACHE = 0
NVML_MEMORY_LOCATION_L2_CACHE = 1
NVML_MEMORY_LOCATION_DEVICE_MEMORY = 2
NVML_MEMORY_LOCATION_DRAM = 2
NVML_MEMORY_LOCATION_REGISTER_FILE = 3
NVML_MEMORY_LOCATION_TEXTURE_MEMORY = 4
NVML_MEMORY_LOCATION_TEXTURE_SHM = 5
NVML_MEMORY_LOCATION_CBU = 6
NVML_MEMORY_LOCATION_SRAM = 7
NVML_MEMORY_LOCATION_COUNT = 8
NVML_NVLINK_MAX_LINKS = 18
NVML_NVLINK_MAX_LANES = NVML_NVLINK_MAX_LINKS
_nvmlNvLinkErrorCounter_t = c_uint
NVML_NVLINK_ERROR_DL_REPLAY = 0
NVML_NVLINK_ERROR_DL_RECOVERY = 1
NVML_NVLINK_ERROR_DL_CRC_FLIT = 2
NVML_NVLINK_ERROR_DL_CRC_DATA = 3
NVML_NVLINK_ERROR_DL_ECC_DATA = 4
NVML_NVLINK_ERROR_COUNT = 5
_nvmlNvLinkEccLaneErrorCounter_t = c_uint
NVML_NVLINK_ERROR_DL_ECC_LANE0 = 0
NVML_NVLINK_ERROR_DL_ECC_LANE1 = 1
NVML_NVLINK_ERROR_DL_ECC_LANE2 = 2
NVML_NVLINK_ERROR_DL_ECC_LANE3 = 3
NVML_NVLINK_ERROR_DL_ECC_COUNT = 5
_nvmlNvLinkCapability_t = c_uint
NVML_NVLINK_CAP_P2P_SUPPORTED = 0
NVML_NVLINK_CAP_SYSMEM_ACCESS = 1
NVML_NVLINK_CAP_P2P_ATOMICS = 2
NVML_NVLINK_CAP_SYSMEM_ATOMICS = 3
NVML_NVLINK_CAP_SLI_BRIDGE = 4
NVML_NVLINK_CAP_VALID = 5
NVML_NVLINK_CAP_COUNT = 6
_nvmlNvLinkUtilizationCountPktTypes_t = c_uint
NVML_NVLINK_COUNTER_PKTFILTER_NOP = 1
NVML_NVLINK_COUNTER_PKTFILTER_READ = 2
NVML_NVLINK_COUNTER_PKTFILTER_WRITE = 4
NVML_NVLINK_COUNTER_PKTFILTER_RATOM = 8
NVML_NVLINK_COUNTER_PKTFILTER_NRATOM = 16
NVML_NVLINK_COUNTER_PKTFILTER_FLUSH = 32
NVML_NVLINK_COUNTER_PKTFILTER_RESPDATA = 64
NVML_NVLINK_COUNTER_PKTFILTER_RESPNODATA = 128
NVML_NVLINK_COUNTER_PKTFILTER_ALL = 255
_nvmlNvLinkUtilizationCountUnits_t = c_uint
NVML_NVLINK_COUNTER_UNIT_CYCLES = 0
NVML_NVLINK_COUNTER_UNIT_PACKETS = 1
NVML_NVLINK_COUNTER_UNIT_BYTES = 2
NVML_NVLINK_COUNTER_UNIT_RESERVED = 3
NVML_NVLINK_COUNTER_UNIT_COUNT = 4
_nvmlNvLinkDeviceType_t = c_uint
NVML_NVLINK_DEVICE_TYPE_GPU = 0
NVML_NVLINK_DEVICE_TYPE_IBMNPU = 1
NVML_NVLINK_DEVICE_TYPE_SWITCH = 2
NVML_NVLINK_DEVICE_TYPE_UNKNOWN = 255
_nvmlEccBitType_t = c_uint
NVML_SINGLE_BIT_ECC = 0
NVML_DOUBLE_BIT_ECC = 1
NVML_ECC_ERROR_TYPE_COUNT = 2
_nvmlEccCounterType_t = c_uint
NVML_VOLATILE_ECC = 0
NVML_AGGREGATE_ECC = 1
NVML_ECC_COUNTER_TYPE_COUNT = 2
_nvmlMemoryErrorType_t = c_uint
NVML_MEMORY_ERROR_TYPE_CORRECTED = 0
NVML_MEMORY_ERROR_TYPE_UNCORRECTED = 1
NVML_MEMORY_ERROR_TYPE_COUNT = 2
_nvmlClockType_t = c_uint
NVML_CLOCK_GRAPHICS = 0
NVML_CLOCK_SM = 1
NVML_CLOCK_MEM = 2
NVML_CLOCK_VIDEO = 3
NVML_CLOCK_COUNT = 4
_nvmlClockId_t = c_uint
NVML_CLOCK_ID_CURRENT = 0
NVML_CLOCK_ID_APP_CLOCK_TARGET = 1
NVML_CLOCK_ID_APP_CLOCK_DEFAULT = 2
NVML_CLOCK_ID_CUSTOMER_BOOST_MAX = 3
NVML_CLOCK_ID_COUNT = 4
_nvmlDriverModel_t = c_uint
NVML_DRIVER_WDDM = 0
NVML_DRIVER_WDM = 1
NVML_DRIVER_MCDM = 2
NVML_MAX_GPU_PERF_PSTATES = 16
_nvmlPstates_t = c_uint
NVML_PSTATE_0 = 0
NVML_PSTATE_1 = 1
NVML_PSTATE_2 = 2
NVML_PSTATE_3 = 3
NVML_PSTATE_4 = 4
NVML_PSTATE_5 = 5
NVML_PSTATE_6 = 6
NVML_PSTATE_7 = 7
NVML_PSTATE_8 = 8
NVML_PSTATE_9 = 9
NVML_PSTATE_10 = 10
NVML_PSTATE_11 = 11
NVML_PSTATE_12 = 12
NVML_PSTATE_13 = 13
NVML_PSTATE_14 = 14
NVML_PSTATE_15 = 15
NVML_PSTATE_UNKNOWN = 32
_nvmlInforomObject_t = c_uint
NVML_INFOROM_OEM = 0
NVML_INFOROM_ECC = 1
NVML_INFOROM_POWER = 2
NVML_INFOROM_COUNT = 3
_nvmlReturn_t = c_uint
NVML_SUCCESS = 0
NVML_ERROR_UNINITIALIZED = 1
NVML_ERROR_INVALID_ARGUMENT = 2
NVML_ERROR_NOT_SUPPORTED = 3
NVML_ERROR_NO_PERMISSION = 4
NVML_ERROR_ALREADY_INITIALIZED = 5
NVML_ERROR_NOT_FOUND = 6
NVML_ERROR_INSUFFICIENT_SIZE = 7
NVML_ERROR_INSUFFICIENT_POWER = 8
NVML_ERROR_DRIVER_NOT_LOADED = 9
NVML_ERROR_TIMEOUT = 10
NVML_ERROR_IRQ_ISSUE = 11
NVML_ERROR_LIBRARY_NOT_FOUND = 12
NVML_ERROR_FUNCTION_NOT_FOUND = 13
NVML_ERROR_CORRUPTED_INFOROM = 14
NVML_ERROR_GPU_IS_LOST = 15
NVML_ERROR_RESET_REQUIRED = 16
NVML_ERROR_OPERATING_SYSTEM = 17
NVML_ERROR_LIB_RM_VERSION_MISMATCH = 18
NVML_ERROR_IN_USE = 19
NVML_ERROR_MEMORY = 20
NVML_ERROR_NO_DATA = 21
NVML_ERROR_VGPU_ECC_NOT_SUPPORTED = 22
NVML_ERROR_INSUFFICIENT_RESOURCES = 23
NVML_ERROR_FREQ_NOT_SUPPORTED = 24
NVML_ERROR_ARGUMENT_VERSION_MISMATCH = 25
NVML_ERROR_DEPRECATED = 26
NVML_ERROR_NOT_READY = 27
NVML_ERROR_UNKNOWN = 999
_nvmlFanState_t = c_uint
NVML_FAN_NORMAL = 0
NVML_FAN_FAILED = 1
_nvmlFanControlPolicy_t = c_uint
NVML_FAN_POLICY_TEMPERATURE_CONTINOUS_SW = 0
NVML_FAN_POLICY_MANUAL = 1
_nvmlLedColor_t = c_uint
NVML_LED_COLOR_GREEN = 0
NVML_LED_COLOR_AMBER = 1
_nvmlGpuOperationMode_t = c_uint
NVML_GOM_ALL_ON = 0
NVML_GOM_COMPUTE = 1
NVML_GOM_LOW_DP = 2
_nvmlPageRetirementCause_t = c_uint
NVML_PAGE_RETIREMENT_CAUSE_MULTIPLE_SINGLE_BIT_ECC_ERRORS = 0
NVML_PAGE_RETIREMENT_CAUSE_DOUBLE_BIT_ECC_ERROR = 1
NVML_PAGE_RETIREMENT_CAUSE_COUNT = 2
_nvmlRestrictedAPI_t = c_uint
NVML_RESTRICTED_API_SET_APPLICATION_CLOCKS = 0
NVML_RESTRICTED_API_SET_AUTO_BOOSTED_CLOCKS = 1
NVML_RESTRICTED_API_COUNT = 2
_nvmlBridgeChipType_t = c_uint
NVML_BRIDGE_CHIP_PLX = 0
NVML_BRIDGE_CHIP_BRO4 = 1
NVML_MAX_PHYSICAL_BRIDGE = 128
_nvmlValueType_t = c_uint
NVML_VALUE_TYPE_DOUBLE = 0
NVML_VALUE_TYPE_UNSIGNED_INT = 1
NVML_VALUE_TYPE_UNSIGNED_LONG = 2
NVML_VALUE_TYPE_UNSIGNED_LONG_LONG = 3
NVML_VALUE_TYPE_SIGNED_LONG_LONG = 4
NVML_VALUE_TYPE_SIGNED_INT = 5
NVML_VALUE_TYPE_COUNT = 6
_nvmlPerfPolicyType_t = c_uint
NVML_PERF_POLICY_POWER = 0
NVML_PERF_POLICY_THERMAL = 1
NVML_PERF_POLICY_SYNC_BOOST = 2
NVML_PERF_POLICY_BOARD_LIMIT = 3
NVML_PERF_POLICY_LOW_UTILIZATION = 4
NVML_PERF_POLICY_RELIABILITY = 5
NVML_PERF_POLICY_TOTAL_APP_CLOCKS = 10
NVML_PERF_POLICY_TOTAL_BASE_CLOCKS = 11
NVML_PERF_POLICY_COUNT = 12
_nvmlEncoderQueryType_t = c_uint
NVML_ENCODER_QUERY_H264 = 0
NVML_ENCODER_QUERY_HEVC = 1
_nvmlFBCSessionType_t = c_uint
NVML_FBC_SESSION_TYPE_UNKNOWN = 0
NVML_FBC_SESSION_TYPE_TOSYS = 1
NVML_FBC_SESSION_TYPE_CUDA = 2
NVML_FBC_SESSION_TYPE_VID = 3
NVML_FBC_SESSION_TYPE_HWENC = 4
_nvmlDetachGpuState_t = c_uint
NVML_DETACH_GPU_KEEP = 0
NVML_DETACH_GPU_REMOVE = 1
_nvmlPcieLinkState_t = c_uint
NVML_PCIE_LINK_KEEP = 0
NVML_PCIE_LINK_SHUT_DOWN = 1
_nvmlSamplingType_t = c_uint
NVML_TOTAL_POWER_SAMPLES = 0
NVML_GPU_UTILIZATION_SAMPLES = 1
NVML_MEMORY_UTILIZATION_SAMPLES = 2
NVML_ENC_UTILIZATION_SAMPLES = 3
NVML_DEC_UTILIZATION_SAMPLES = 4
NVML_PROCESSOR_CLK_SAMPLES = 5
NVML_MEMORY_CLK_SAMPLES = 6
NVML_MODULE_POWER_SAMPLES = 7
NVML_SAMPLINGTYPE_COUNT = 8
_nvmlPcieUtilCounter_t = c_uint
NVML_PCIE_UTIL_TX_BYTES = 0
NVML_PCIE_UTIL_RX_BYTES = 1
NVML_PCIE_UTIL_COUNT = 2
_nvmlGpuTopologyLevel_t = c_uint
NVML_TOPOLOGY_INTERNAL = 0
NVML_TOPOLOGY_SINGLE = 10
NVML_TOPOLOGY_MULTIPLE = 20
NVML_TOPOLOGY_HOSTBRIDGE = 30
NVML_TOPOLOGY_NODE = 40
NVML_TOPOLOGY_CPU = NVML_TOPOLOGY_NODE
NVML_TOPOLOGY_SYSTEM = 50
_nvmlGpuP2PCapsIndex_t = c_uint
NVML_P2P_CAPS_INDEX_READ = (0,)
NVML_P2P_CAPS_INDEX_WRITE = 1
NVML_P2P_CAPS_INDEX_NVLINK = 2
NVML_P2P_CAPS_INDEX_ATOMICS = 3
NVML_P2P_CAPS_INDEX_PROP = 4
NVML_P2P_CAPS_INDEX_LOOPBACK = 5
NVML_P2P_CAPS_INDEX_UNKNOWN = 6
_nvmlGpuP2PStatus_t = c_uint
NVML_P2P_STATUS_OK = 0
NVML_P2P_STATUS_CHIPSET_NOT_SUPPORED = 1
NVML_P2P_STATUS_CHIPSET_NOT_SUPPORTED = NVML_P2P_STATUS_CHIPSET_NOT_SUPPORED
NVML_P2P_STATUS_GPU_NOT_SUPPORTED = 2
NVML_P2P_STATUS_IOH_TOPOLOGY_NOT_SUPPORTED = 3
NVML_P2P_STATUS_DISABLED_BY_REGKEY = 4
NVML_P2P_STATUS_NOT_SUPPORTED = 5
NVML_P2P_STATUS_UNKNOWN = 6
_nvmlDeviceArchitecture_t = c_uint
NVML_DEVICE_ARCH_KEPLER = 2
NVML_DEVICE_ARCH_MAXWELL = 3
NVML_DEVICE_ARCH_PASCAL = 4
NVML_DEVICE_ARCH_VOLTA = 5
NVML_DEVICE_ARCH_TURING = 6
NVML_DEVICE_ARCH_AMPERE = 7
NVML_DEVICE_ARCH_ADA = 8
NVML_DEVICE_ARCH_HOPPER = 9
NVML_DEVICE_ARCH_UNKNOWN = 4294967295
_nvmlBusType_t = c_uint
NVML_BUS_TYPE_UNKNOWN = 0
NVML_BUS_TYPE_PCI = 1
NVML_BUS_TYPE_PCIE = 2
NVML_BUS_TYPE_FPCI = 3
NVML_BUS_TYPE_AGP = 4
_nvmlPowerSource_t = c_uint
NVML_POWER_SOURCE_AC = 0
NVML_POWER_SOURCE_BATTERY = 1
NVML_POWER_SOURCE_UNDERSIZED = 2
_nvmlAdaptiveClockInfoStatus_t = c_uint
NVML_ADAPTIVE_CLOCKING_INFO_STATUS_DISABLED = 0
NVML_ADAPTIVE_CLOCKING_INFO_STATUS_ENABLED = 1
_nvmlClockLimitId_t = c_uint
NVML_CLOCK_LIMIT_ID_RANGE_START = 4294967040
NVML_CLOCK_LIMIT_ID_TDP = 4294967041
NVML_CLOCK_LIMIT_ID_UNLIMITED = 4294967042
_nvmlPcieLinkMaxSpeed_t = c_uint
NVML_PCIE_LINK_MAX_SPEED_INVALID = 0
NVML_PCIE_LINK_MAX_SPEED_2500MBPS = 1
NVML_PCIE_LINK_MAX_SPEED_5000MBPS = 2
NVML_PCIE_LINK_MAX_SPEED_8000MBPS = 3
NVML_PCIE_LINK_MAX_SPEED_16000MBPS = 4
NVML_PCIE_LINK_MAX_SPEED_32000MBPS = 5
NVML_PCIE_LINK_MAX_SPEED_64000MBPS = 6
_nvmlAffinityScope_t = c_uint
NVML_AFFINITY_SCOPE_NODE = 0
NVML_AFFINITY_SCOPE_SOCKET = 1
nvmlFlagDefault = 0
nvmlFlagForce = 1
NVML_INIT_FLAG_NO_GPUS = 1
NVML_INIT_FLAG_NO_ATTACH = 2
NVML_MAX_GPC_COUNT = 32
NVML_DEVICE_INFOROM_VERSION_BUFFER_SIZE = 16
NVML_DEVICE_UUID_BUFFER_SIZE = 80
NVML_DEVICE_UUID_V2_BUFFER_SIZE = 96
NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE = 80
NVML_SYSTEM_NVML_VERSION_BUFFER_SIZE = 80
NVML_DEVICE_NAME_BUFFER_SIZE = 64
NVML_DEVICE_NAME_V2_BUFFER_SIZE = 96
NVML_DEVICE_SERIAL_BUFFER_SIZE = 30
NVML_DEVICE_PART_NUMBER_BUFFER_SIZE = 80
NVML_DEVICE_GPU_PART_NUMBER_BUFFER_SIZE = 80
NVML_DEVICE_VBIOS_VERSION_BUFFER_SIZE = 32
NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE = 32
NVML_DEVICE_PCI_BUS_ID_BUFFER_V2_SIZE = 16
NVML_GRID_LICENSE_BUFFER_SIZE = 128
NVML_VGPU_NAME_BUFFER_SIZE = 64
NVML_GRID_LICENSE_FEATURE_MAX_COUNT = 3
NVML_VGPU_METADATA_OPAQUE_DATA_SIZE = sizeof(c_uint) + 256
NVML_VGPU_PGPU_METADATA_OPAQUE_DATA_SIZE = 256
NVML_DEVICE_GPU_FRU_PART_NUMBER_BUFFER_SIZE = 20
NVML_DEVICE_PCI_BUS_ID_LEGACY_FMT = '%04X:%02X:%02X.0'
NVML_DEVICE_PCI_BUS_ID_FMT = '%08X:%02X:%02X.0'
NVML_VALUE_NOT_AVAILABLE_ulonglong = c_ulonglong(-1)
NVML_VALUE_NOT_AVAILABLE_uint = c_uint(-1)
'\n Field Identifiers.\n\n All Identifiers pertain to a device. Each ID is only used once and is guaranteed never to change.\n'
NVML_FI_DEV_ECC_CURRENT = 1
NVML_FI_DEV_ECC_PENDING = 2
NVML_FI_DEV_ECC_SBE_VOL_TOTAL = 3
NVML_FI_DEV_ECC_DBE_VOL_TOTAL = 4
NVML_FI_DEV_ECC_SBE_AGG_TOTAL = 5
NVML_FI_DEV_ECC_DBE_AGG_TOTAL = 6
NVML_FI_DEV_ECC_SBE_VOL_L1 = 7
NVML_FI_DEV_ECC_DBE_VOL_L1 = 8
NVML_FI_DEV_ECC_SBE_VOL_L2 = 9
NVML_FI_DEV_ECC_DBE_VOL_L2 = 10
NVML_FI_DEV_ECC_SBE_VOL_DEV = 11
NVML_FI_DEV_ECC_DBE_VOL_DEV = 12
NVML_FI_DEV_ECC_SBE_VOL_REG = 13
NVML_FI_DEV_ECC_DBE_VOL_REG = 14
NVML_FI_DEV_ECC_SBE_VOL_TEX = 15
NVML_FI_DEV_ECC_DBE_VOL_TEX = 16
NVML_FI_DEV_ECC_DBE_VOL_CBU = 17
NVML_FI_DEV_ECC_SBE_AGG_L1 = 18
NVML_FI_DEV_ECC_DBE_AGG_L1 = 19
NVML_FI_DEV_ECC_SBE_AGG_L2 = 20
NVML_FI_DEV_ECC_DBE_AGG_L2 = 21
NVML_FI_DEV_ECC_SBE_AGG_DEV = 22
NVML_FI_DEV_ECC_DBE_AGG_DEV = 23
NVML_FI_DEV_ECC_SBE_AGG_REG = 24
NVML_FI_DEV_ECC_DBE_AGG_REG = 25
NVML_FI_DEV_ECC_SBE_AGG_TEX = 26
NVML_FI_DEV_ECC_DBE_AGG_TEX = 27
NVML_FI_DEV_ECC_DBE_AGG_CBU = 28
NVML_FI_DEV_RETIRED_SBE = 29
NVML_FI_DEV_RETIRED_DBE = 30
NVML_FI_DEV_RETIRED_PENDING = 31
NVML_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L0 = 32
NVML_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L1 = 33
NVML_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L2 = 34
NVML_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L3 = 35
NVML_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L4 = 36
NVML_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L5 = 37
NVML_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_TOTAL = 38
NVML_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L0 = 39
NVML_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L1 = 40
NVML_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L2 = 41
NVML_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L3 = 42
NVML_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L4 = 43
NVML_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L5 = 44
NVML_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_TOTAL = 45
NVML_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L0 = 46
NVML_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L1 = 47
NVML_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L2 = 48
NVML_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L3 = 49
NVML_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L4 = 50
NVML_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L5 = 51
NVML_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_TOTAL = 52
NVML_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L0 = 53
NVML_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L1 = 54
NVML_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L2 = 55
NVML_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L3 = 56
NVML_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L4 = 57
NVML_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L5 = 58
NVML_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_TOTAL = 59
NVML_FI_DEV_NVLINK_BANDWIDTH_C0_L0 = 60
NVML_FI_DEV_NVLINK_BANDWIDTH_C0_L1 = 61
NVML_FI_DEV_NVLINK_BANDWIDTH_C0_L2 = 62
NVML_FI_DEV_NVLINK_BANDWIDTH_C0_L3 = 63
NVML_FI_DEV_NVLINK_BANDWIDTH_C0_L4 = 64
NVML_FI_DEV_NVLINK_BANDWIDTH_C0_L5 = 65
NVML_FI_DEV_NVLINK_BANDWIDTH_C0_TOTAL = 66
NVML_FI_DEV_NVLINK_BANDWIDTH_C1_L0 = 67
NVML_FI_DEV_NVLINK_BANDWIDTH_C1_L1 = 68
NVML_FI_DEV_NVLINK_BANDWIDTH_C1_L2 = 69
NVML_FI_DEV_NVLINK_BANDWIDTH_C1_L3 = 70
NVML_FI_DEV_NVLINK_BANDWIDTH_C1_L4 = 71
NVML_FI_DEV_NVLINK_BANDWIDTH_C1_L5 = 72
NVML_FI_DEV_NVLINK_BANDWIDTH_C1_TOTAL = 73
NVML_FI_DEV_PERF_POLICY_POWER = 74
NVML_FI_DEV_PERF_POLICY_THERMAL = 75
NVML_FI_DEV_PERF_POLICY_SYNC_BOOST = 76
NVML_FI_DEV_PERF_POLICY_BOARD_LIMIT = 77
NVML_FI_DEV_PERF_POLICY_LOW_UTILIZATION = 78
NVML_FI_DEV_PERF_POLICY_RELIABILITY = 79
NVML_FI_DEV_PERF_POLICY_TOTAL_APP_CLOCKS = 80
NVML_FI_DEV_PERF_POLICY_TOTAL_BASE_CLOCKS = 81
NVML_FI_DEV_MEMORY_TEMP = 82
NVML_FI_DEV_TOTAL_ENERGY_CONSUMPTION = 83
NVML_FI_DEV_NVLINK_SPEED_MBPS_L0 = 84
NVML_FI_DEV_NVLINK_SPEED_MBPS_L1 = 85
NVML_FI_DEV_NVLINK_SPEED_MBPS_L2 = 86
NVML_FI_DEV_NVLINK_SPEED_MBPS_L3 = 87
NVML_FI_DEV_NVLINK_SPEED_MBPS_L4 = 88
NVML_FI_DEV_NVLINK_SPEED_MBPS_L5 = 89
NVML_FI_DEV_NVLINK_SPEED_MBPS_COMMON = 90
NVML_FI_DEV_NVLINK_LINK_COUNT = 91
NVML_FI_DEV_RETIRED_PENDING_SBE = 92
NVML_FI_DEV_RETIRED_PENDING_DBE = 93
NVML_FI_DEV_PCIE_REPLAY_COUNTER = 94
NVML_FI_DEV_PCIE_REPLAY_ROLLOVER_COUNTER = 95
NVML_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L6 = 96
NVML_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L7 = 97
NVML_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L8 = 98
NVML_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L9 = 99
NVML_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L10 = 100
NVML_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L11 = 101
NVML_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L6 = 102
NVML_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L7 = 103
NVML_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L8 = 104
NVML_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L9 = 105
NVML_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L10 = 106
NVML_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L11 = 107
NVML_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L6 = 108
NVML_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L7 = 109
NVML_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L8 = 110
NVML_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L9 = 111
NVML_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L10 = 112
NVML_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L11 = 113
NVML_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L6 = 114
NVML_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L7 = 115
NVML_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L8 = 116
NVML_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L9 = 117
NVML_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L10 = 118
NVML_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L11 = 119
NVML_FI_DEV_NVLINK_BANDWIDTH_C0_L6 = 120
NVML_FI_DEV_NVLINK_BANDWIDTH_C0_L7 = 121
NVML_FI_DEV_NVLINK_BANDWIDTH_C0_L8 = 122
NVML_FI_DEV_NVLINK_BANDWIDTH_C0_L9 = 123
NVML_FI_DEV_NVLINK_BANDWIDTH_C0_L10 = 124
NVML_FI_DEV_NVLINK_BANDWIDTH_C0_L11 = 125
NVML_FI_DEV_NVLINK_BANDWIDTH_C1_L6 = 126
NVML_FI_DEV_NVLINK_BANDWIDTH_C1_L7 = 127
NVML_FI_DEV_NVLINK_BANDWIDTH_C1_L8 = 128
NVML_FI_DEV_NVLINK_BANDWIDTH_C1_L9 = 129
NVML_FI_DEV_NVLINK_BANDWIDTH_C1_L10 = 130
NVML_FI_DEV_NVLINK_BANDWIDTH_C1_L11 = 131
NVML_FI_DEV_NVLINK_SPEED_MBPS_L6 = 132
NVML_FI_DEV_NVLINK_SPEED_MBPS_L7 = 133
NVML_FI_DEV_NVLINK_SPEED_MBPS_L8 = 134
NVML_FI_DEV_NVLINK_SPEED_MBPS_L9 = 135
NVML_FI_DEV_NVLINK_SPEED_MBPS_L10 = 136
NVML_FI_DEV_NVLINK_SPEED_MBPS_L11 = 137
NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_TX = 138
NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_RX = 139
NVML_FI_DEV_NVLINK_THROUGHPUT_RAW_TX = 140
NVML_FI_DEV_NVLINK_THROUGHPUT_RAW_RX = 141
NVML_FI_DEV_REMAPPED_COR = 142
NVML_FI_DEV_REMAPPED_UNC = 143
NVML_FI_DEV_REMAPPED_PENDING = 144
NVML_FI_DEV_REMAPPED_FAILURE = 145
NVML_FI_DEV_NVLINK_REMOTE_NVLINK_ID = 146
NVML_FI_DEV_NVSWITCH_CONNECTED_LINK_COUNT = 147
NVML_FI_DEV_NVLINK_ECC_DATA_ERROR_COUNT_L0 = 148
NVML_FI_DEV_NVLINK_ECC_DATA_ERROR_COUNT_L1 = 149
NVML_FI_DEV_NVLINK_ECC_DATA_ERROR_COUNT_L2 = 150
NVML_FI_DEV_NVLINK_ECC_DATA_ERROR_COUNT_L3 = 151
NVML_FI_DEV_NVLINK_ECC_DATA_ERROR_COUNT_L4 = 152
NVML_FI_DEV_NVLINK_ECC_DATA_ERROR_COUNT_L5 = 153
NVML_FI_DEV_NVLINK_ECC_DATA_ERROR_COUNT_L6 = 154
NVML_FI_DEV_NVLINK_ECC_DATA_ERROR_COUNT_L7 = 155
NVML_FI_DEV_NVLINK_ECC_DATA_ERROR_COUNT_L8 = 156
NVML_FI_DEV_NVLINK_ECC_DATA_ERROR_COUNT_L9 = 157
NVML_FI_DEV_NVLINK_ECC_DATA_ERROR_COUNT_L10 = 158
NVML_FI_DEV_NVLINK_ECC_DATA_ERROR_COUNT_L11 = 159
NVML_FI_DEV_NVLINK_ECC_DATA_ERROR_COUNT_TOTAL = 160
NVML_FI_DEV_NVLINK_ERROR_DL_REPLAY = 161
NVML_FI_DEV_NVLINK_ERROR_DL_RECOVERY = 162
NVML_FI_DEV_NVLINK_ERROR_DL_CRC = 163
NVML_FI_DEV_NVLINK_GET_SPEED = 164
NVML_FI_DEV_NVLINK_GET_STATE = 165
NVML_FI_DEV_NVLINK_GET_VERSION = 166
NVML_FI_DEV_NVLINK_GET_POWER_STATE = 167
NVML_FI_DEV_NVLINK_GET_POWER_THRESHOLD = 168
NVML_FI_DEV_PCIE_L0_TO_RECOVERY_COUNTER = 169
NVML_FI_DEV_C2C_LINK_COUNT = 170
NVML_FI_DEV_C2C_LINK_GET_STATUS = 171
NVML_FI_DEV_C2C_LINK_GET_MAX_BW = 172
NVML_FI_DEV_PCIE_COUNT_CORRECTABLE_ERRORS = 173
NVML_FI_DEV_PCIE_COUNT_NAKS_RECEIVED = 174
NVML_FI_DEV_PCIE_COUNT_RECEIVER_ERROR = 175
NVML_FI_DEV_PCIE_COUNT_BAD_TLP = 176
NVML_FI_DEV_PCIE_COUNT_NAKS_SENT = 177
NVML_FI_DEV_PCIE_COUNT_BAD_DLLP = 178
NVML_FI_DEV_PCIE_COUNT_NON_FATAL_ERROR = 179
NVML_FI_DEV_PCIE_COUNT_FATAL_ERROR = 180
NVML_FI_DEV_PCIE_COUNT_UNSUPPORTED_REQ = 181
NVML_FI_DEV_PCIE_COUNT_LCRC_ERROR = 182
NVML_FI_DEV_PCIE_COUNT_LANE_ERROR = 183
NVML_FI_DEV_IS_RESETLESS_MIG_SUPPORTED = 184
NVML_FI_DEV_POWER_AVERAGE = 185
NVML_FI_DEV_POWER_INSTANT = 186
NVML_FI_DEV_POWER_MIN_LIMIT = 187
NVML_FI_DEV_POWER_MAX_LIMIT = 188
NVML_FI_DEV_POWER_DEFAULT_LIMIT = 189
NVML_FI_DEV_POWER_CURRENT_LIMIT = 190
NVML_FI_DEV_ENERGY = 191
NVML_FI_DEV_POWER_REQUESTED_LIMIT = 192
NVML_FI_DEV_TEMPERATURE_SHUTDOWN_TLIMIT = 193
NVML_FI_DEV_TEMPERATURE_SLOWDOWN_TLIMIT = 194
NVML_FI_DEV_TEMPERATURE_MEM_MAX_TLIMIT = 195
NVML_FI_DEV_TEMPERATURE_GPU_MAX_TLIMIT = 196
NVML_FI_MAX = 197
NVML_GPU_VIRTUALIZATION_MODE_NONE = 0
NVML_GPU_VIRTUALIZATION_MODE_PASSTHROUGH = 1
NVML_GPU_VIRTUALIZATION_MODE_VGPU = 2
NVML_GPU_VIRTUALIZATION_MODE_HOST_VGPU = 3
NVML_GPU_VIRTUALIZATION_MODE_HOST_VSGA = 4
nvmlLib = None
libLoadLock = threading.Lock()
_nvmlLib_refcount = 0
_nvmlVgpuTypeId_t = c_uint
_nvmlVgpuInstance_t = c_uint
_nvmlVgpuVmIdType_t = c_uint
NVML_VGPU_VM_ID_DOMAIN_ID = 0
NVML_VGPU_VM_ID_UUID = 1
_nvmlGridLicenseFeatureCode_t = c_uint
NVML_GRID_LICENSE_FEATURE_CODE_UNKNOWN = 0
NVML_GRID_LICENSE_FEATURE_CODE_VGPU = 1
NVML_GRID_LICENSE_FEATURE_CODE_NVIDIA_RTX = 2
NVML_GRID_LICENSE_FEATURE_CODE_VWORKSTATION = 2
NVML_GRID_LICENSE_FEATURE_CODE_GAMING = 3
NVML_GRID_LICENSE_FEATURE_CODE_COMPUTE = 4
_nvmlGridLicenseExpiryStatus_t = c_uint8
NVML_GRID_LICENSE_EXPIRY_NOT_AVAILABLE = (0,)
NVML_GRID_LICENSE_EXPIRY_INVALID = (1,)
NVML_GRID_LICENSE_EXPIRY_VALID = (2,)
NVML_GRID_LICENSE_EXPIRY_NOT_APPLICABLE = (3,)
NVML_GRID_LICENSE_EXPIRY_PERMANENT = (4,)
_nvmlVgpuCapability_t = c_uint
NVML_VGPU_CAP_NVLINK_P2P = 0
NVML_VGPU_CAP_GPUDIRECT = 1
NVML_VGPU_CAP_MULTI_VGPU_EXCLUSIVE = 2
NVML_VGPU_CAP_EXCLUSIVE_TYPE = 3
NVML_VGPU_CAP_EXCLUSIVE_SIZE = 4
NVML_VGPU_CAP_COUNT = 5
_nvmlVgpuDriverCapability_t = c_uint
NVML_VGPU_DRIVER_CAP_HETEROGENEOUS_MULTI_VGPU = 0
NVML_VGPU_DRIVER_CAP_COUNT = 1
_nvmlDeviceVgpuCapability_t = c_uint
NVML_DEVICE_VGPU_CAP_FRACTIONAL_MULTI_VGPU = 0
NVML_DEVICE_VGPU_CAP_HETEROGENEOUS_TIMESLICE_PROFILES = 1
NVML_DEVICE_VGPU_CAP_HETEROGENEOUS_TIMESLICE_SIZES = 2
NVML_DEVICE_VGPU_CAP_READ_DEVICE_BUFFER_BW = 3
NVML_DEVICE_VGPU_CAP_WRITE_DEVICE_BUFFER_BW = 4
NVML_DEVICE_VGPU_CAP_COUNT = 5
_nvmlVgpuGuestInfoState_t = c_uint
NVML_VGPU_INSTANCE_GUEST_INFO_STATE_UNINITIALIZED = 0
NVML_VGPU_INSTANCE_GUEST_INFO_STATE_INITIALIZED = 1
_nvmlVgpuVmCompatibility_t = c_uint
NVML_VGPU_VM_COMPATIBILITY_NONE = 0
NVML_VGPU_VM_COMPATIBILITY_COLD = 1
NVML_VGPU_VM_COMPATIBILITY_HIBERNATE = 2
NVML_VGPU_VM_COMPATIBILITY_SLEEP = 4
NVML_VGPU_VM_COMPATIBILITY_LIVE = 8
_nvmlVgpuPgpuCompatibilityLimitCode_t = c_uint
NVML_VGPU_COMPATIBILITY_LIMIT_NONE = 0
NVML_VGPU_COMPATIBILITY_LIMIT_HOST_DRIVER = 1
NVML_VGPU_COMPATIBILITY_LIMIT_GUEST_DRIVER = 2
NVML_VGPU_COMPATIBILITY_LIMIT_GPU = 4
NVML_VGPU_COMPATIBILITY_LIMIT_OTHER = 2147483648
_nvmlHostVgpuMode_t = c_uint
NVML_HOST_VGPU_MODE_NON_SRIOV = 0
NVML_HOST_VGPU_MODE_SRIOV = 1
_nvmlConfComputeGpusReadyState_t = c_uint
NVML_CC_ACCEPTING_CLIENT_REQUESTS_FALSE = 0
NVML_CC_ACCEPTING_CLIENT_REQUESTS_TRUE = 1
_nvmlConfComputeGpuCaps_t = c_uint
NVML_CC_SYSTEM_GPUS_CC_NOT_CAPABLE = 0
NVML_CC_SYSTEM_GPUS_CC_CAPABLE = 1
_nvmlConfComputeCpuCaps_t = c_uint
NVML_CC_SYSTEM_CPU_CAPS_NONE = 0
NVML_CC_SYSTEM_CPU_CAPS_AMD_SEV = 1
NVML_CC_SYSTEM_CPU_CAPS_INTEL_TDX = 2
_nvmlConfComputeDevToolsMode_t = c_uint
NVML_CC_SYSTEM_DEVTOOLS_MODE_OFF = 0
NVML_CC_SYSTEM_DEVTOOLS_MODE_ON = 1
NVML_CC_SYSTEM_ENVIRONMENT_UNAVAILABLE = 0
NVML_CC_SYSTEM_ENVIRONMENT_SIM = 1
NVML_CC_SYSTEM_ENVIRONMENT_PROD = 2
_nvmlConfComputeCcFeature_t = c_uint
NVML_CC_SYSTEM_FEATURE_DISABLED = 0
NVML_CC_SYSTEM_FEATURE_ENABLED = 1
NVML_GSP_FIRMWARE_VERSION_BUF_SIZE = 64

class NVMLError(Exception):
    _valClassMapping = dict()
    _errcode_to_string = {NVML_ERROR_UNINITIALIZED: 'Uninitialized', NVML_ERROR_INVALID_ARGUMENT: 'Invalid Argument', NVML_ERROR_NOT_SUPPORTED: 'Not Supported', NVML_ERROR_NO_PERMISSION: 'Insufficient Permissions', NVML_ERROR_ALREADY_INITIALIZED: 'Already Initialized', NVML_ERROR_NOT_FOUND: 'Not Found', NVML_ERROR_INSUFFICIENT_SIZE: 'Insufficient Size', NVML_ERROR_INSUFFICIENT_POWER: 'Insufficient External Power', NVML_ERROR_DRIVER_NOT_LOADED: 'Driver Not Loaded', NVML_ERROR_TIMEOUT: 'Timeout', NVML_ERROR_IRQ_ISSUE: 'Interrupt Request Issue', NVML_ERROR_LIBRARY_NOT_FOUND: 'NVML Shared Library Not Found', NVML_ERROR_FUNCTION_NOT_FOUND: 'Function Not Found', NVML_ERROR_CORRUPTED_INFOROM: 'Corrupted infoROM', NVML_ERROR_GPU_IS_LOST: 'GPU is lost', NVML_ERROR_RESET_REQUIRED: 'GPU requires restart', NVML_ERROR_OPERATING_SYSTEM: 'The operating system has blocked the request.', NVML_ERROR_LIB_RM_VERSION_MISMATCH: 'RM has detected an NVML/RM version mismatch.', NVML_ERROR_MEMORY: 'Insufficient Memory', NVML_ERROR_UNKNOWN: 'Unknown Error'}

    def __new__(typ, value):
        if False:
            for i in range(10):
                print('nop')
        '\n        Maps value to a proper subclass of NVMLError.\n        See _extractNVMLErrorsAsClasses function for more details\n        '
        if typ == NVMLError:
            typ = NVMLError._valClassMapping.get(value, typ)
        obj = Exception.__new__(typ)
        obj.value = value
        return obj

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        try:
            if self.value not in NVMLError._errcode_to_string:
                NVMLError._errcode_to_string[self.value] = str(nvmlErrorString(self.value))
            return NVMLError._errcode_to_string[self.value]
        except NVMLError:
            return 'NVML Error with code %d' % self.value

    def __eq__(self, other):
        if False:
            return 10
        return self.value == other.value

def nvmlExceptionClass(nvmlErrorCode):
    if False:
        i = 10
        return i + 15
    if nvmlErrorCode not in NVMLError._valClassMapping:
        raise ValueError('nvmlErrorCode %s is not valid' % nvmlErrorCode)
    return NVMLError._valClassMapping[nvmlErrorCode]

def _extractNVMLErrorsAsClasses():
    if False:
        while True:
            i = 10
    "\n    Generates a hierarchy of classes on top of NVMLError class.\n\n    Each NVML Error gets a new NVMLError subclass. This way try,except blocks can filter appropriate\n    exceptions more easily.\n\n    NVMLError is a parent class. Each NVML_ERROR_* gets it's own subclass.\n    e.g. NVML_ERROR_ALREADY_INITIALIZED will be turned into NVMLError_AlreadyInitialized\n    "
    this_module = sys.modules[__name__]
    nvmlErrorsNames = [x for x in dir(this_module) if x.startswith('NVML_ERROR_')]
    for err_name in nvmlErrorsNames:
        class_name = 'NVMLError_' + string.capwords(err_name.replace('NVML_ERROR_', ''), '_').replace('_', '')
        err_val = getattr(this_module, err_name)

        def gen_new(val):
            if False:
                print('Hello World!')

            def new(typ):
                if False:
                    for i in range(10):
                        print('nop')
                obj = NVMLError.__new__(typ, val)
                return obj
            return new
        new_error_class = type(class_name, (NVMLError,), {'__new__': gen_new(err_val)})
        new_error_class.__module__ = __name__
        setattr(this_module, class_name, new_error_class)
        NVMLError._valClassMapping[err_val] = new_error_class
_extractNVMLErrorsAsClasses()

def _nvmlCheckReturn(ret):
    if False:
        return 10
    if ret != NVML_SUCCESS:
        raise NVMLError(ret)
    return ret
_nvmlGetFunctionPointer_cache = dict()

def _nvmlGetFunctionPointer(name):
    if False:
        print('Hello World!')
    global nvmlLib
    if name in _nvmlGetFunctionPointer_cache:
        return _nvmlGetFunctionPointer_cache[name]
    libLoadLock.acquire()
    try:
        if nvmlLib == None:
            raise NVMLError(NVML_ERROR_UNINITIALIZED)
        try:
            _nvmlGetFunctionPointer_cache[name] = getattr(nvmlLib, name)
            return _nvmlGetFunctionPointer_cache[name]
        except AttributeError:
            raise NVMLError(NVML_ERROR_FUNCTION_NOT_FOUND)
    finally:
        libLoadLock.release()

class nvmlFriendlyObject(object):

    def __init__(self, dictionary):
        if False:
            while True:
                i = 10
        for x in dictionary:
            setattr(self, x, dictionary[x])

    def __str__(self):
        if False:
            while True:
                i = 10
        return self.__dict__.__str__()

def nvmlStructToFriendlyObject(struct):
    if False:
        print('Hello World!')
    d = {}
    for x in struct._fields_:
        key = x[0]
        value = getattr(struct, key)
        d[key] = value.decode() if isinstance(value, bytes) else value
    obj = nvmlFriendlyObject(d)
    return obj

def nvmlFriendlyObjectToStruct(obj, model):
    if False:
        for i in range(10):
            print('nop')
    for x in model._fields_:
        key = x[0]
        value = obj.__dict__[key]
        if sys.version_info >= (3,):
            setattr(model, key, value.encode())
        else:
            setattr(model, key, value)
    return model

class struct_c_nvmlUnit_t(Structure):
    pass
c_nvmlUnit_t = POINTER(struct_c_nvmlUnit_t)

class _PrintableStructure(Structure):
    """
    Abstract class that produces nicer __str__ output than ctypes.Structure.
    e.g. instead of:
      >> print str(obj)
      <class_name object at 0x7fdf82fef9e0>
    this class will print
      class_name(field_name: formatted_value, field_name: formatted_value)

    _fmt_ dictionary of <str _field_ name> -> <str format>
    e.g. class that has _field_ 'hex_value', c_uint could be formatted with
      _fmt_ = {"hex_value" : "%08X"}
    to produce nicer output.
    Default fomratting string for all fields can be set with key "<default>" like:
      _fmt_ = {"<default>" : "%d MHz"} # e.g all values are numbers in MHz.
    If not set it's assumed to be just "%s"

    Exact format of returned str from this class is subject to change in the future.
    """
    _fmt_ = {}

    def __str__(self):
        if False:
            print('Hello World!')
        result = []
        for x in self._fields_:
            key = x[0]
            value = getattr(self, key)
            fmt = '%s'
            if key in self._fmt_:
                fmt = self._fmt_[key]
            elif '<default>' in self._fmt_:
                fmt = self._fmt_['<default>']
            result.append(('%s: ' + fmt) % (key, value))
        return self.__class__.__name__ + '(' + ', '.join(result) + ')'

    def __getattribute__(self, name):
        if False:
            for i in range(10):
                print('nop')
        res = super(_PrintableStructure, self).__getattribute__(name)
        if isinstance(res, bytes):
            if isinstance(res, str):
                return res
            return res.decode()
        return res

    def __setattr__(self, name, value):
        if False:
            return 10
        if isinstance(value, str):
            value = value.encode()
        super(_PrintableStructure, self).__setattr__(name, value)

class c_nvmlUnitInfo_t(_PrintableStructure):
    _fields_ = [('name', c_char * 96), ('id', c_char * 96), ('serial', c_char * 96), ('firmwareVersion', c_char * 96)]

class c_nvmlLedState_t(_PrintableStructure):
    _fields_ = [('cause', c_char * 256), ('color', _nvmlLedColor_t)]

class c_nvmlPSUInfo_t(_PrintableStructure):
    _fields_ = [('state', c_char * 256), ('current', c_uint), ('voltage', c_uint), ('power', c_uint)]

class c_nvmlUnitFanInfo_t(_PrintableStructure):
    _fields_ = [('speed', c_uint), ('state', _nvmlFanState_t)]

class c_nvmlUnitFanSpeeds_t(_PrintableStructure):
    _fields_ = [('fans', c_nvmlUnitFanInfo_t * 24), ('count', c_uint)]

class struct_c_nvmlDevice_t(Structure):
    pass
c_nvmlDevice_t = POINTER(struct_c_nvmlDevice_t)

class nvmlPciInfo_v2_t(_PrintableStructure):
    _fields_ = [('busId', c_char * NVML_DEVICE_PCI_BUS_ID_BUFFER_V2_SIZE), ('domain', c_uint), ('bus', c_uint), ('device', c_uint), ('pciDeviceId', c_uint), ('pciSubSystemId', c_uint), ('reserved0', c_uint), ('reserved1', c_uint), ('reserved2', c_uint), ('reserved3', c_uint)]
    _fmt_ = {'domain': '0x%04X', 'bus': '0x%02X', 'device': '0x%02X', 'pciDeviceId': '0x%08X', 'pciSubSystemId': '0x%08X'}

class nvmlPciInfo_t(_PrintableStructure):
    _fields_ = [('busIdLegacy', c_char * NVML_DEVICE_PCI_BUS_ID_BUFFER_V2_SIZE), ('domain', c_uint), ('bus', c_uint), ('device', c_uint), ('pciDeviceId', c_uint), ('pciSubSystemId', c_uint), ('busId', c_char * NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE)]
    _fmt_ = {'domain': '0x%08X', 'bus': '0x%02X', 'device': '0x%02X', 'pciDeviceId': '0x%08X', 'pciSubSystemId': '0x%08X'}

class c_nvmlExcludedDeviceInfo_t(_PrintableStructure):
    _fields_ = [('pci', nvmlPciInfo_t), ('uuid', c_char * NVML_DEVICE_UUID_BUFFER_SIZE)]

class nvmlNvLinkUtilizationControl_t(_PrintableStructure):
    _fields_ = [('units', _nvmlNvLinkUtilizationCountUnits_t), ('pktfilter', _nvmlNvLinkUtilizationCountPktTypes_t)]

class c_nvmlMemory_t(_PrintableStructure):
    _fields_ = [('total', c_ulonglong), ('free', c_ulonglong), ('used', c_ulonglong)]
    _fmt_ = {'<default>': '%d B'}

class c_nvmlMemory_v2_t(_PrintableStructure):
    _fields_ = [('version', c_uint), ('total', c_ulonglong), ('reserved', c_ulonglong), ('free', c_ulonglong), ('used', c_ulonglong)]
    _fmt_ = {'<default>': '%d B'}
nvmlMemory_v2 = 33554472

class c_nvmlBAR1Memory_t(_PrintableStructure):
    _fields_ = [('bar1Total', c_ulonglong), ('bar1Free', c_ulonglong), ('bar1Used', c_ulonglong)]
    _fmt_ = {'<default>': '%d B'}

class nvmlClkMonFaultInfo_t(Structure):
    _fields_ = [('clkApiDomain', c_uint), ('clkDomainFaultMask', c_uint)]

class nvmlClkMonStatus_t(Structure):
    _fields_ = [('bGlobalStatus', c_uint), ('clkMonListSize', c_uint), ('clkMonList', nvmlClkMonFaultInfo_t)]

class c_nvmlProcessInfo_v2_t(_PrintableStructure):
    _fields_ = [('pid', c_uint), ('usedGpuMemory', c_ulonglong), ('gpuInstanceId', c_uint), ('computeInstanceId', c_uint)]
    _fmt_ = {'usedGpuMemory': '%d B'}
c_nvmlProcessInfo_t = c_nvmlProcessInfo_v2_t
_nvmlProcessMode_t = c_uint
NVML_PROCESS_MODE_COMPUTE = 0
NVML_PROCESS_MODE_GRAPHICS = 1
NVML_PROCESS_MODE_MPS = 2

class c_nvmlProcessDetail_v1_t(Structure):
    _fields_ = [('pid', c_uint), ('usedGpuMemory', c_ulonglong), ('gpuInstanceId', c_uint), ('computeInstanceId', c_uint), ('usedGpuCcProtectedMemory', c_ulonglong)]

class c_nvmlProcessDetailList_v1_t(_PrintableStructure):
    _fields_ = [('version', c_uint), ('mode', _nvmlProcessMode_t), ('numProcArrayEntries', c_uint), ('procArray', POINTER(c_nvmlProcessDetail_v1_t))]
    _fmt_ = {'numProcArrayEntries': '%d B'}
c_nvmlProcessDetailList_t = c_nvmlProcessDetailList_v1_t
nvmlProcessDetailList_v1 = 16777232

class c_nvmlBridgeChipInfo_t(_PrintableStructure):
    _fields_ = [('type', _nvmlBridgeChipType_t), ('fwVersion', c_uint)]

class c_nvmlBridgeChipHierarchy_t(_PrintableStructure):
    _fields_ = [('bridgeCount', c_uint), ('bridgeChipInfo', c_nvmlBridgeChipInfo_t * 128)]

class c_nvmlEccErrorCounts_t(_PrintableStructure):
    _fields_ = [('l1Cache', c_ulonglong), ('l2Cache', c_ulonglong), ('deviceMemory', c_ulonglong), ('registerFile', c_ulonglong)]

class c_nvmlUtilization_t(_PrintableStructure):
    _fields_ = [('gpu', c_uint), ('memory', c_uint)]
    _fmt_ = {'<default>': '%d %%'}

class c_nvmlHwbcEntry_t(_PrintableStructure):
    _fields_ = [('hwbcId', c_uint), ('firmwareVersion', c_char * 32)]

class c_nvmlValue_t(Union):
    _fields_ = [('dVal', c_double), ('uiVal', c_uint), ('ulVal', c_ulong), ('ullVal', c_ulonglong), ('sllVal', c_longlong), ('siVal', c_int)]

class c_nvmlSample_t(_PrintableStructure):
    _fields_ = [('timeStamp', c_ulonglong), ('sampleValue', c_nvmlValue_t)]

class c_nvmlViolationTime_t(_PrintableStructure):
    _fields_ = [('referenceTime', c_ulonglong), ('violationTime', c_ulonglong)]

class c_nvmlFieldValue_t(_PrintableStructure):
    _fields_ = [('fieldId', c_uint32), ('scopeId', c_uint32), ('timestamp', c_int64), ('latencyUsec', c_int64), ('valueType', _nvmlValueType_t), ('nvmlReturn', _nvmlReturn_t), ('value', c_nvmlValue_t)]

class c_nvmlVgpuInstanceUtilizationSample_t(_PrintableStructure):
    _fields_ = [('vgpuInstance', _nvmlVgpuInstance_t), ('timeStamp', c_ulonglong), ('smUtil', c_nvmlValue_t), ('memUtil', c_nvmlValue_t), ('encUtil', c_nvmlValue_t), ('decUtil', c_nvmlValue_t)]

class c_nvmlVgpuProcessUtilizationSample_t(_PrintableStructure):
    _fields_ = [('vgpuInstance', _nvmlVgpuInstance_t), ('pid', c_uint), ('processName', c_char * NVML_VGPU_NAME_BUFFER_SIZE), ('timeStamp', c_ulonglong), ('smUtil', c_uint), ('memUtil', c_uint), ('encUtil', c_uint), ('decUtil', c_uint)]

class c_nvmlVgpuLicenseExpiry_t(_PrintableStructure):
    _fields_ = [('year', c_uint32), ('month', c_uint16), ('day', c_uint16), ('hour', c_uint16), ('min', c_uint16), ('sec', c_uint16), ('status', c_uint8)]
NVML_GRID_LICENSE_STATE_UNKNOWN = 0
NVML_GRID_LICENSE_STATE_UNINITIALIZED = 1
NVML_GRID_LICENSE_STATE_UNLICENSED_UNRESTRICTED = 2
NVML_GRID_LICENSE_STATE_UNLICENSED_RESTRICTED = 3
NVML_GRID_LICENSE_STATE_UNLICENSED = 4
NVML_GRID_LICENSE_STATE_LICENSED = 5

class c_nvmlVgpuLicenseInfo_t(_PrintableStructure):
    _fields_ = [('isLicensed', c_uint8), ('licenseExpiry', c_nvmlVgpuLicenseExpiry_t), ('currentState', c_uint)]

class c_nvmlEncoderSession_t(_PrintableStructure):
    _fields_ = [('sessionId', c_uint), ('pid', c_uint), ('vgpuInstance', _nvmlVgpuInstance_t), ('codecType', c_uint), ('hResolution', c_uint), ('vResolution', c_uint), ('averageFps', c_uint), ('encodeLatency', c_uint)]

class c_nvmlProcessUtilizationSample_t(_PrintableStructure):
    _fields_ = [('pid', c_uint), ('timeStamp', c_ulonglong), ('smUtil', c_uint), ('memUtil', c_uint), ('encUtil', c_uint), ('decUtil', c_uint)]

class c_nvmlGridLicenseExpiry_t(_PrintableStructure):
    _fields_ = [('year', c_uint32), ('month', c_uint16), ('day', c_uint16), ('hour', c_uint16), ('min', c_uint16), ('sec', c_uint16), ('status', c_uint8)]

class c_nvmlGridLicensableFeature_v4_t(_PrintableStructure):
    _fields_ = [('featureCode', _nvmlGridLicenseFeatureCode_t), ('featureState', c_uint), ('licenseInfo', c_char * NVML_GRID_LICENSE_BUFFER_SIZE), ('productName', c_char * NVML_GRID_LICENSE_BUFFER_SIZE), ('featureEnabled', c_uint), ('licenseExpiry', c_nvmlGridLicenseExpiry_t)]

class c_nvmlGridLicensableFeatures_v4_t(_PrintableStructure):
    _fields_ = [('isGridLicenseSupported', c_int), ('licensableFeaturesCount', c_uint), ('gridLicensableFeatures', c_nvmlGridLicensableFeature_v4_t * NVML_GRID_LICENSE_FEATURE_MAX_COUNT)]

class c_nvmlGridLicensableFeature_v3_t(_PrintableStructure):
    _fields_ = [('featureCode', _nvmlGridLicenseFeatureCode_t), ('featureState', c_uint), ('licenseInfo', c_char * NVML_GRID_LICENSE_BUFFER_SIZE), ('productName', c_char * NVML_GRID_LICENSE_BUFFER_SIZE), ('featureEnabled', c_uint)]

class c_nvmlGridLicensableFeatures_v3_t(_PrintableStructure):
    _fields_ = [('isGridLicenseSupported', c_int), ('licensableFeaturesCount', c_uint), ('gridLicensableFeatures', c_nvmlGridLicensableFeature_v3_t * NVML_GRID_LICENSE_FEATURE_MAX_COUNT)]

class c_nvmlGridLicensableFeature_v2_t(_PrintableStructure):
    _fields_ = [('featureCode', _nvmlGridLicenseFeatureCode_t), ('featureState', c_uint), ('licenseInfo', c_char * NVML_GRID_LICENSE_BUFFER_SIZE), ('productName', c_char * NVML_GRID_LICENSE_BUFFER_SIZE)]

class c_nvmlGridLicensableFeatures_v2_t(_PrintableStructure):
    _fields_ = [('isGridLicenseSupported', c_int), ('licensableFeaturesCount', c_uint), ('gridLicensableFeatures', c_nvmlGridLicensableFeature_v2_t * NVML_GRID_LICENSE_FEATURE_MAX_COUNT)]

class c_nvmlGridLicensableFeature_t(_PrintableStructure):
    _fields_ = [('featureCode', _nvmlGridLicenseFeatureCode_t), ('featureState', c_uint), ('licenseInfo', c_char * NVML_GRID_LICENSE_BUFFER_SIZE)]

class c_nvmlGridLicensableFeatures_t(_PrintableStructure):
    _fields_ = [('isGridLicenseSupported', c_int), ('licensableFeaturesCount', c_uint), ('gridLicensableFeatures', c_nvmlGridLicensableFeature_t * NVML_GRID_LICENSE_FEATURE_MAX_COUNT)]

class struct_c_nvmlEventSet_t(Structure):
    pass
c_nvmlEventSet_t = POINTER(struct_c_nvmlEventSet_t)
nvmlEventTypeSingleBitEccError = 1
nvmlEventTypeDoubleBitEccError = 2
nvmlEventTypePState = 4
nvmlEventTypeXidCriticalError = 8
nvmlEventTypeClock = 16
nvmlEventTypePowerSourceChange = 128
nvmlEventMigConfigChange = 256
nvmlEventTypeNone = 0
nvmlEventTypeAll = nvmlEventTypeNone | nvmlEventTypeSingleBitEccError | nvmlEventTypeDoubleBitEccError | nvmlEventTypePState | nvmlEventTypeClock | nvmlEventTypePowerSourceChange | nvmlEventTypeXidCriticalError | nvmlEventMigConfigChange
nvmlClocksEventReasonGpuIdle = 1
nvmlClocksEventReasonApplicationsClocksSetting = 2
nvmlClocksEventReasonUserDefinedClocks = nvmlClocksEventReasonApplicationsClocksSetting
nvmlClocksEventReasonSwPowerCap = 4
nvmlClocksEventReasonHwSlowdown = 8
nvmlClocksEventReasonSyncBoost = 16
nvmlClocksEventReasonSwThermalSlowdown = 32
nvmlClocksEventReasonHwThermalSlowdown = 64
nvmlClocksEventReasonHwPowerBrakeSlowdown = 128
nvmlClocksEventReasonDisplayClockSetting = 256
nvmlClocksEventReasonNone = 0
nvmlClocksEventReasonAll = nvmlClocksEventReasonNone | nvmlClocksEventReasonGpuIdle | nvmlClocksEventReasonApplicationsClocksSetting | nvmlClocksEventReasonSwPowerCap | nvmlClocksEventReasonHwSlowdown | nvmlClocksEventReasonSyncBoost | nvmlClocksEventReasonSwThermalSlowdown | nvmlClocksEventReasonHwThermalSlowdown | nvmlClocksEventReasonHwPowerBrakeSlowdown | nvmlClocksEventReasonDisplayClockSetting
nvmlClocksThrottleReasonGpuIdle = 1
nvmlClocksThrottleReasonApplicationsClocksSetting = 2
nvmlClocksThrottleReasonUserDefinedClocks = nvmlClocksThrottleReasonApplicationsClocksSetting
nvmlClocksThrottleReasonSwPowerCap = 4
nvmlClocksThrottleReasonHwSlowdown = 8
nvmlClocksThrottleReasonSyncBoost = 16
nvmlClocksThrottleReasonSwThermalSlowdown = 32
nvmlClocksThrottleReasonHwThermalSlowdown = 64
nvmlClocksThrottleReasonHwPowerBrakeSlowdown = 128
nvmlClocksThrottleReasonDisplayClockSetting = 256
nvmlClocksThrottleReasonNone = 0
nvmlClocksThrottleReasonAll = nvmlClocksThrottleReasonNone | nvmlClocksThrottleReasonGpuIdle | nvmlClocksThrottleReasonApplicationsClocksSetting | nvmlClocksThrottleReasonSwPowerCap | nvmlClocksThrottleReasonHwSlowdown | nvmlClocksThrottleReasonSyncBoost | nvmlClocksThrottleReasonSwThermalSlowdown | nvmlClocksThrottleReasonHwThermalSlowdown | nvmlClocksThrottleReasonHwPowerBrakeSlowdown | nvmlClocksThrottleReasonDisplayClockSetting

class c_nvmlEventData_t(_PrintableStructure):
    _fields_ = [('device', c_nvmlDevice_t), ('eventType', c_ulonglong), ('eventData', c_ulonglong), ('gpuInstanceId', c_uint), ('computeInstanceId', c_uint)]
    _fmt_ = {'eventType': '0x%08X'}

class c_nvmlAccountingStats_t(_PrintableStructure):
    _fields_ = [('gpuUtilization', c_uint), ('memoryUtilization', c_uint), ('maxMemoryUsage', c_ulonglong), ('time', c_ulonglong), ('startTime', c_ulonglong), ('isRunning', c_uint), ('reserved', c_uint * 5)]

class c_nvmlVgpuVersion_t(Structure):
    _fields_ = [('minVersion', c_uint), ('maxVersion', c_uint)]

class c_nvmlVgpuMetadata_t(_PrintableStructure):
    _fields_ = [('version', c_uint), ('revision', c_uint), ('guestInfoState', _nvmlVgpuGuestInfoState_t), ('guestDriverVersion', c_char * NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE), ('hostDriverVersion', c_char * NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE), ('reserved', c_uint * 6), ('vgpuVirtualizationCaps', c_uint), ('guestVgpuVersion', c_uint), ('opaqueDataSize', c_uint), ('opaqueData', c_char * NVML_VGPU_METADATA_OPAQUE_DATA_SIZE)]

class c_nvmlVgpuPgpuMetadata_t(_PrintableStructure):
    _fields_ = [('version', c_uint), ('revision', c_uint), ('hostDriverVersion', c_char * NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE), ('pgpuVirtualizationCaps', c_uint), ('reserved', c_uint * 5), ('hostSupportedVgpuRange', c_nvmlVgpuVersion_t), ('opaqueDataSize', c_uint), ('opaqueData', c_char * NVML_VGPU_PGPU_METADATA_OPAQUE_DATA_SIZE)]

class c_nvmlVgpuPgpuCompatibility_t(Structure):
    _fields_ = [('vgpuVmCompatibility', _nvmlVgpuVmCompatibility_t), ('compatibilityLimitCode', _nvmlVgpuPgpuCompatibilityLimitCode_t)]
NVML_VGPU_SCHEDULER_POLICY_UNKNOWN = 0
NVML_VGPU_SCHEDULER_POLICY_BEST_EFFORT = 1
NVML_VGPU_SCHEDULER_POLICY_EQUAL_SHARE = 2
NVML_VGPU_SCHEDULER_POLICY_FIXED_SHARE = 3
NVML_SUPPORTED_VGPU_SCHEDULER_POLICY_COUNT = 3
NVML_SCHEDULER_SW_MAX_LOG_ENTRIES = 200
NVML_VGPU_SCHEDULER_ARR_DEFAULT = 0
NVML_VGPU_SCHEDULER_ARR_DISABLE = 1
NVML_VGPU_SCHEDULER_ARR_ENABLE = 2

class c_nvmlVgpuSchedDataWithARR_t(_PrintableStructure):
    _fields_ = [('avgFactor', c_uint), ('timeslice', c_uint)]

class c_nvmlVgpuSchedData_t(_PrintableStructure):
    _fields_ = [('timeslice', c_uint)]

class c_nvmlVgpuSchedulerParams_t(Union):
    _fields_ = [('vgpuSchedDataWithARR', c_nvmlVgpuSchedDataWithARR_t), ('vgpuSchedData', c_nvmlVgpuSchedData_t)]

class c_nvmlVgpuSchedulerLogEntry_t(_PrintableStructure):
    _fields_ = [('timestamp', c_ulonglong), ('timeRunTotal', c_ulonglong), ('timeRun', c_ulonglong), ('swRunlistId', c_uint), ('targetTimeSlice', c_ulonglong), ('cumulativePreemptionTime', c_ulonglong)]

class c_nvmlVgpuSchedulerLog_t(_PrintableStructure):
    _fields_ = [('engineId', c_uint), ('schedulerPolicy', c_uint), ('arrMode', c_uint), ('schedulerParams', c_nvmlVgpuSchedulerParams_t), ('entriesCount', c_uint), ('logEntries', c_nvmlVgpuSchedulerLogEntry_t * NVML_SCHEDULER_SW_MAX_LOG_ENTRIES)]

class c_nvmlVgpuSchedulerGetState_t(_PrintableStructure):
    _fields_ = [('schedulerPolicy', c_uint), ('arrMode', c_uint), ('schedulerParams', c_nvmlVgpuSchedulerParams_t)]

class c_nvmlVgpuSchedSetDataWithARR_t(_PrintableStructure):
    _fields_ = [('avgFactor', c_uint), ('frequency', c_uint)]

class c_nvmlVgpuSchedSetData_t(_PrintableStructure):
    _fields_ = [('timeslice', c_uint)]

class c_nvmlVgpuSchedulerSetParams_t(Union):
    _fields_ = [('vgpuSchedDataWithARR', c_nvmlVgpuSchedSetDataWithARR_t), ('vgpuSchedData', c_nvmlVgpuSchedSetData_t)]

class c_nvmlVgpuSchedulerSetState_t(_PrintableStructure):
    _fields_ = [('schedulerPolicy', c_uint), ('enableARRMode', c_uint), ('schedulerParams', c_nvmlVgpuSchedulerSetParams_t)]

class c_nvmlVgpuSchedulerCapabilities_t(_PrintableStructure):
    _fields_ = [('supportedSchedulers', c_uint * NVML_SUPPORTED_VGPU_SCHEDULER_POLICY_COUNT), ('maxTimeslice', c_uint), ('minTimeslice', c_uint), ('isArrModeSupported', c_uint), ('maxFrequencyForARR', c_uint), ('minFrequencyForARR', c_uint), ('maxAvgFactorForARR', c_uint), ('minAvgFactorForARR', c_uint)]

class c_nvmlFBCStats_t(Structure):
    _fields_ = [('sessionsCount', c_uint), ('averageFPS', c_uint), ('averageLatency', c_uint)]

class c_nvmlFBCSession_t(_PrintableStructure):
    _fields_ = [('sessionId', c_uint), ('pid', c_uint), ('vgpuInstance', _nvmlVgpuInstance_t), ('displayOrdinal', c_uint), ('sessionType', c_uint), ('sessionFlags', c_uint), ('hMaxResolution', c_uint), ('vMaxResolution', c_uint), ('hResolution', c_uint), ('vResolution', c_uint), ('averageFPS', c_uint), ('averageLatency', c_uint)]
NVML_DEVICE_MIG_DISABLE = 0
NVML_DEVICE_MIG_ENABLE = 1
NVML_GPU_INSTANCE_PROFILE_1_SLICE = 0
NVML_GPU_INSTANCE_PROFILE_2_SLICE = 1
NVML_GPU_INSTANCE_PROFILE_3_SLICE = 2
NVML_GPU_INSTANCE_PROFILE_4_SLICE = 3
NVML_GPU_INSTANCE_PROFILE_7_SLICE = 4
NVML_GPU_INSTANCE_PROFILE_8_SLICE = 5
NVML_GPU_INSTANCE_PROFILE_6_SLICE = 6
NVML_GPU_INSTANCE_PROFILE_1_SLICE_REV1 = 7
NVML_GPU_INSTANCE_PROFILE_2_SLICE_REV1 = 8
NVML_GPU_INSTANCE_PROFILE_1_SLICE_REV2 = 9
NVML_GPU_INSTANCE_PROFILE_COUNT = 10

class c_nvmlGpuInstancePlacement_t(Structure):
    _fields_ = [('start', c_uint), ('size', c_uint)]

class c_nvmlGpuInstanceProfileInfo_t(Structure):
    _fields_ = [('id', c_uint), ('isP2pSupported', c_uint), ('sliceCount', c_uint), ('instanceCount', c_uint), ('multiprocessorCount', c_uint), ('copyEngineCount', c_uint), ('decoderCount', c_uint), ('encoderCount', c_uint), ('jpegCount', c_uint), ('ofaCount', c_uint), ('memorySizeMB', c_ulonglong)]
nvmlGpuInstanceProfileInfo_v2 = 33554584

class c_nvmlGpuInstanceProfileInfo_v2_t(_PrintableStructure):
    _fields_ = [('version', c_uint), ('id', c_uint), ('isP2pSupported', c_uint), ('sliceCount', c_uint), ('instanceCount', c_uint), ('multiprocessorCount', c_uint), ('copyEngineCount', c_uint), ('decoderCount', c_uint), ('encoderCount', c_uint), ('jpegCount', c_uint), ('ofaCount', c_uint), ('memorySizeMB', c_ulonglong), ('name', c_char * NVML_DEVICE_NAME_V2_BUFFER_SIZE)]

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super(c_nvmlGpuInstanceProfileInfo_v2_t, self).__init__(version=nvmlGpuInstanceProfileInfo_v2)

class c_nvmlGpuInstanceInfo_t(Structure):
    _fields_ = [('device', c_nvmlDevice_t), ('id', c_uint), ('profileId', c_uint), ('placement', c_nvmlGpuInstancePlacement_t)]

class struct_c_nvmlGpuInstance_t(Structure):
    pass
c_nvmlGpuInstance_t = POINTER(struct_c_nvmlGpuInstance_t)
NVML_COMPUTE_INSTANCE_PROFILE_1_SLICE = 0
NVML_COMPUTE_INSTANCE_PROFILE_2_SLICE = 1
NVML_COMPUTE_INSTANCE_PROFILE_3_SLICE = 2
NVML_COMPUTE_INSTANCE_PROFILE_4_SLICE = 3
NVML_COMPUTE_INSTANCE_PROFILE_7_SLICE = 4
NVML_COMPUTE_INSTANCE_PROFILE_8_SLICE = 5
NVML_COMPUTE_INSTANCE_PROFILE_6_SLICE = 6
NVML_COMPUTE_INSTANCE_PROFILE_1_SLICE_REV1 = 7
NVML_COMPUTE_INSTANCE_PROFILE_COUNT = 8
NVML_COMPUTE_INSTANCE_ENGINE_PROFILE_SHARED = 0
NVML_COMPUTE_INSTANCE_ENGINE_PROFILE_COUNT = 1

class c_nvmlComputeInstancePlacement_t(Structure):
    _fields_ = [('start', c_uint), ('size', c_uint)]

class c_nvmlComputeInstanceProfileInfo_t(Structure):
    _fields_ = [('id', c_uint), ('sliceCount', c_uint), ('instanceCount', c_uint), ('multiprocessorCount', c_uint), ('sharedCopyEngineCount', c_uint), ('sharedDecoderCount', c_uint), ('sharedEncoderCount', c_uint), ('sharedJpegCount', c_uint), ('sharedOfaCount', c_uint)]
nvmlComputeInstanceProfileInfo_v2 = 33554568

class c_nvmlComputeInstanceProfileInfo_v2_t(_PrintableStructure):
    _fields_ = [('version', c_uint), ('id', c_uint), ('sliceCount', c_uint), ('instanceCount', c_uint), ('multiprocessorCount', c_uint), ('sharedCopyEngineCount', c_uint), ('sharedDecoderCount', c_uint), ('sharedEncoderCount', c_uint), ('sharedJpegCount', c_uint), ('sharedOfaCount', c_uint), ('name', c_char * NVML_DEVICE_NAME_V2_BUFFER_SIZE)]

    def __init__(self):
        if False:
            return 10
        super(c_nvmlComputeInstanceProfileInfo_v2_t, self).__init__(version=nvmlComputeInstanceProfileInfo_v2)

class c_nvmlComputeInstanceInfo_t(Structure):
    _fields_ = [('device', c_nvmlDevice_t), ('gpuInstance', c_nvmlGpuInstance_t), ('id', c_uint), ('profileId', c_uint), ('placement', c_nvmlComputeInstancePlacement_t)]
NVML_MAX_GPU_UTILIZATIONS = 8
NVML_GPU_UTILIZATION_DOMAIN_GPU = 0
NVML_GPU_UTILIZATION_DOMAIN_FB = 1
NVML_GPU_UTILIZATION_DOMAIN_VID = 2
NVML_GPU_UTILIZATION_DOMAIN_BUS = 3

class c_nvmlGpuDynamicPstatesUtilization_t(Structure):
    _fields_ = [('bIsPresent', c_uint, 1), ('percentage', c_uint), ('incThreshold', c_uint), ('decThreshold', c_uint)]

class c_nvmlGpuDynamicPstatesInfo_t(Structure):
    _fields_ = [('flags', c_uint), ('utilization', c_nvmlGpuDynamicPstatesUtilization_t * NVML_MAX_GPU_UTILIZATIONS)]
NVML_MAX_THERMAL_SENSORS_PER_GPU = 3
NVML_THERMAL_TARGET_NONE = 0
NVML_THERMAL_TARGET_GPU = 1
NVML_THERMAL_TARGET_MEMORY = 2
NVML_THERMAL_TARGET_POWER_SUPPLY = 4
NVML_THERMAL_TARGET_BOARD = 8
NVML_THERMAL_TARGET_VCD_BOARD = 9
NVML_THERMAL_TARGET_VCD_INLET = 10
NVML_THERMAL_TARGET_VCD_OUTLET = 11
NVML_THERMAL_TARGET_ALL = 15
NVML_THERMAL_TARGET_UNKNOWN = -1
NVML_THERMAL_CONTROLLER_NONE = 0
NVML_THERMAL_CONTROLLER_GPU_INTERNAL = 1
NVML_THERMAL_CONTROLLER_ADM1032 = 2
NVML_THERMAL_CONTROLLER_ADT7461 = 3
NVML_THERMAL_CONTROLLER_MAX6649 = 4
NVML_THERMAL_CONTROLLER_MAX1617 = 5
NVML_THERMAL_CONTROLLER_LM99 = 6
NVML_THERMAL_CONTROLLER_LM89 = 7
NVML_THERMAL_CONTROLLER_LM64 = 8
NVML_THERMAL_CONTROLLER_G781 = 9
NVML_THERMAL_CONTROLLER_ADT7473 = 10
NVML_THERMAL_CONTROLLER_SBMAX6649 = 11
NVML_THERMAL_CONTROLLER_VBIOSEVT = 12
NVML_THERMAL_CONTROLLER_OS = 13
NVML_THERMAL_CONTROLLER_NVSYSCON_CANOAS = 14
NVML_THERMAL_CONTROLLER_NVSYSCON_E551 = 15
NVML_THERMAL_CONTROLLER_MAX6649R = 16
NVML_THERMAL_CONTROLLER_ADT7473S = 17
NVML_THERMAL_CONTROLLER_UNKNOWN = -1

class c_nvmlGpuThermalSensor_t(Structure):
    _fields_ = [('controller', c_int), ('defaultMinTemp', c_int), ('defaultMaxTemp', c_int), ('currentTemp', c_int), ('target', c_int)]

class c_nvmlGpuThermalSettings_t(Structure):
    _fields_ = [('count', c_uint), ('sensor', c_nvmlGpuThermalSensor_t * NVML_MAX_THERMAL_SENSORS_PER_GPU)]

class struct_c_nvmlComputeInstance_t(Structure):
    pass
c_nvmlComputeInstance_t = POINTER(struct_c_nvmlComputeInstance_t)

class c_nvmlDeviceAttributes(Structure):
    _fields_ = [('multiprocessorCount', c_uint), ('sharedCopyEngineCount', c_uint), ('sharedDecoderCount', c_uint), ('sharedEncoderCount', c_uint), ('sharedJpegCount', c_uint), ('sharedOfaCount', c_uint), ('gpuInstanceSliceCount', c_uint), ('computeInstanceSliceCount', c_uint), ('memorySizeMB', c_ulonglong)]

class c_nvmlRowRemapperHistogramValues(Structure):
    _fields_ = [('max', c_uint), ('high', c_uint), ('partial', c_uint), ('low', c_uint), ('none', c_uint)]
NVML_GPU_CERT_CHAIN_SIZE = 4096
NVML_GPU_ATTESTATION_CERT_CHAIN_SIZE = 5120
NVML_CC_GPU_CEC_NONCE_SIZE = 32
NVML_CC_GPU_ATTESTATION_REPORT_SIZE = 8192
NVML_CC_GPU_CEC_ATTESTATION_REPORT_SIZE = 4096
NVML_CC_CEC_ATTESTATION_REPORT_NOT_PRESENT = 0
NVML_CC_CEC_ATTESTATION_REPORT_PRESENT = 1

class c_nvmlConfComputeSystemState_t(Structure):
    _fields_ = [('environment', c_uint), ('ccFeature', c_uint), ('devToolsMode', c_uint)]

class c_nvmlConfComputeSystemCaps_t(Structure):
    _fields_ = [('cpuCaps', c_uint), ('gpusCaps', c_uint)]

class c_nvmlConfComputeMemSizeInfo_t(Structure):
    _fields_ = [('protectedMemSizeKib', c_ulonglong), ('unprotectedMemSizeKib', c_ulonglong)]

class c_nvmlConfComputeGpuCertificate_t(Structure):
    _fields_ = [('certChainSize', c_uint), ('attestationCertChainSize', c_uint), ('certChain', c_uint8 * NVML_GPU_CERT_CHAIN_SIZE), ('attestationCertChain', c_uint8 * NVML_GPU_ATTESTATION_CERT_CHAIN_SIZE)]

class c_nvmlConfComputeGpuAttestationReport_t(Structure):
    _fields_ = [('isCecAttestationReportPresent', c_uint), ('attestationReportSize', c_uint), ('cecAttestationReportSize', c_uint), ('nonce', c_uint8 * NVML_CC_GPU_CEC_NONCE_SIZE), ('attestationReport', c_uint8 * NVML_CC_GPU_ATTESTATION_REPORT_SIZE), ('cecAttestationReport', c_uint8 * NVML_CC_GPU_CEC_ATTESTATION_REPORT_SIZE)]

def convertStrBytes(func):
    if False:
        print('Hello World!')
    "\n    In python 3, strings are unicode instead of bytes, and need to be converted for ctypes\n    Args from caller: (1, 'string', <__main__.c_nvmlDevice_t at 0xFFFFFFFF>)\n    Args passed to function: (1, b'string', <__main__.c_nvmlDevice_t at 0xFFFFFFFF)>\n    ----\n    Returned from function: b'returned string'\n    Returned to caller: 'returned string'\n    "

    @wraps(func)
    def wrapper(*args, **kwargs):
        if False:
            print('Hello World!')
        args = [arg.encode() if isinstance(arg, str) else arg for arg in args]
        res = func(*args, **kwargs)
        if isinstance(res, bytes):
            if isinstance(res, str):
                return res
            return res.decode()
        return res
    if sys.version_info >= (3,):
        return wrapper
    return func

def nvmlInitWithFlags(flags):
    if False:
        i = 10
        return i + 15
    _LoadNvmlLibrary()
    fn = _nvmlGetFunctionPointer('nvmlInitWithFlags')
    ret = fn(flags)
    _nvmlCheckReturn(ret)
    global _nvmlLib_refcount
    libLoadLock.acquire()
    _nvmlLib_refcount += 1
    libLoadLock.release()
    return None

def nvmlInit():
    if False:
        return 10
    nvmlInitWithFlags(0)
    return None

def _LoadNvmlLibrary():
    if False:
        while True:
            i = 10
    "\n    Load the library if it isn't loaded already\n    "
    global nvmlLib
    if nvmlLib == None:
        libLoadLock.acquire()
        try:
            if nvmlLib == None:
                try:
                    if sys.platform[:3] == 'win':
                        try:
                            nvmlLib = CDLL(os.path.join(os.getenv('WINDIR', 'C:/Windows'), 'System32/nvml.dll'))
                        except OSError as ose:
                            nvmlLib = CDLL(os.path.join(os.getenv('ProgramFiles', 'C:/Program Files'), 'NVIDIA Corporation/NVSMI/nvml.dll'))
                    else:
                        nvmlLib = CDLL('libnvidia-ml.so.1')
                except OSError as ose:
                    _nvmlCheckReturn(NVML_ERROR_LIBRARY_NOT_FOUND)
                if nvmlLib == None:
                    _nvmlCheckReturn(NVML_ERROR_LIBRARY_NOT_FOUND)
        finally:
            libLoadLock.release()

def nvmlShutdown():
    if False:
        return 10
    fn = _nvmlGetFunctionPointer('nvmlShutdown')
    ret = fn()
    _nvmlCheckReturn(ret)
    global _nvmlLib_refcount
    libLoadLock.acquire()
    if 0 < _nvmlLib_refcount:
        _nvmlLib_refcount -= 1
    libLoadLock.release()
    return None

@convertStrBytes
def nvmlErrorString(result):
    if False:
        print('Hello World!')
    fn = _nvmlGetFunctionPointer('nvmlErrorString')
    fn.restype = c_char_p
    ret = fn(result)
    return ret

@convertStrBytes
def nvmlSystemGetNVMLVersion():
    if False:
        for i in range(10):
            print('nop')
    c_version = create_string_buffer(NVML_SYSTEM_NVML_VERSION_BUFFER_SIZE)
    fn = _nvmlGetFunctionPointer('nvmlSystemGetNVMLVersion')
    ret = fn(c_version, c_uint(NVML_SYSTEM_NVML_VERSION_BUFFER_SIZE))
    _nvmlCheckReturn(ret)
    return c_version.value

def nvmlSystemGetCudaDriverVersion():
    if False:
        for i in range(10):
            print('nop')
    c_cuda_version = c_int()
    fn = _nvmlGetFunctionPointer('nvmlSystemGetCudaDriverVersion')
    ret = fn(byref(c_cuda_version))
    _nvmlCheckReturn(ret)
    return c_cuda_version.value

def nvmlSystemGetCudaDriverVersion_v2():
    if False:
        for i in range(10):
            print('nop')
    c_cuda_version = c_int()
    fn = _nvmlGetFunctionPointer('nvmlSystemGetCudaDriverVersion_v2')
    ret = fn(byref(c_cuda_version))
    _nvmlCheckReturn(ret)
    return c_cuda_version.value

@convertStrBytes
def nvmlSystemGetProcessName(pid):
    if False:
        i = 10
        return i + 15
    c_name = create_string_buffer(1024)
    fn = _nvmlGetFunctionPointer('nvmlSystemGetProcessName')
    ret = fn(c_uint(pid), c_name, c_uint(1024))
    _nvmlCheckReturn(ret)
    return c_name.value

@convertStrBytes
def nvmlSystemGetDriverVersion():
    if False:
        for i in range(10):
            print('nop')
    c_version = create_string_buffer(NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE)
    fn = _nvmlGetFunctionPointer('nvmlSystemGetDriverVersion')
    ret = fn(c_version, c_uint(NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE))
    _nvmlCheckReturn(ret)
    return c_version.value

def nvmlSystemGetHicVersion():
    if False:
        i = 10
        return i + 15
    c_count = c_uint(0)
    hics = None
    fn = _nvmlGetFunctionPointer('nvmlSystemGetHicVersion')
    ret = fn(byref(c_count), None)
    if ret != NVML_SUCCESS and ret != NVML_ERROR_INSUFFICIENT_SIZE:
        raise NVMLError(ret)
    if c_count.value == 0:
        return []
    hic_array = c_nvmlHwbcEntry_t * c_count.value
    hics = hic_array()
    ret = fn(byref(c_count), hics)
    _nvmlCheckReturn(ret)
    return hics

def nvmlUnitGetCount():
    if False:
        while True:
            i = 10
    c_count = c_uint()
    fn = _nvmlGetFunctionPointer('nvmlUnitGetCount')
    ret = fn(byref(c_count))
    _nvmlCheckReturn(ret)
    return c_count.value

def nvmlUnitGetHandleByIndex(index):
    if False:
        i = 10
        return i + 15
    c_index = c_uint(index)
    unit = c_nvmlUnit_t()
    fn = _nvmlGetFunctionPointer('nvmlUnitGetHandleByIndex')
    ret = fn(c_index, byref(unit))
    _nvmlCheckReturn(ret)
    return unit

def nvmlUnitGetUnitInfo(unit):
    if False:
        return 10
    c_info = c_nvmlUnitInfo_t()
    fn = _nvmlGetFunctionPointer('nvmlUnitGetUnitInfo')
    ret = fn(unit, byref(c_info))
    _nvmlCheckReturn(ret)
    return c_info

def nvmlUnitGetLedState(unit):
    if False:
        for i in range(10):
            print('nop')
    c_state = c_nvmlLedState_t()
    fn = _nvmlGetFunctionPointer('nvmlUnitGetLedState')
    ret = fn(unit, byref(c_state))
    _nvmlCheckReturn(ret)
    return c_state

def nvmlUnitGetPsuInfo(unit):
    if False:
        i = 10
        return i + 15
    c_info = c_nvmlPSUInfo_t()
    fn = _nvmlGetFunctionPointer('nvmlUnitGetPsuInfo')
    ret = fn(unit, byref(c_info))
    _nvmlCheckReturn(ret)
    return c_info

def nvmlUnitGetTemperature(unit, type):
    if False:
        for i in range(10):
            print('nop')
    c_temp = c_uint()
    fn = _nvmlGetFunctionPointer('nvmlUnitGetTemperature')
    ret = fn(unit, c_uint(type), byref(c_temp))
    _nvmlCheckReturn(ret)
    return c_temp.value

def nvmlUnitGetFanSpeedInfo(unit):
    if False:
        return 10
    c_speeds = c_nvmlUnitFanSpeeds_t()
    fn = _nvmlGetFunctionPointer('nvmlUnitGetFanSpeedInfo')
    ret = fn(unit, byref(c_speeds))
    _nvmlCheckReturn(ret)
    return c_speeds

def nvmlUnitGetDeviceCount(unit):
    if False:
        return 10
    c_count = c_uint(0)
    fn = _nvmlGetFunctionPointer('nvmlUnitGetDevices')
    ret = fn(unit, byref(c_count), None)
    if ret == NVML_ERROR_INSUFFICIENT_SIZE:
        ret = NVML_SUCCESS
    _nvmlCheckReturn(ret)
    return c_count.value

def nvmlUnitGetDevices(unit):
    if False:
        for i in range(10):
            print('nop')
    c_count = c_uint(nvmlUnitGetDeviceCount(unit))
    device_array = c_nvmlDevice_t * c_count.value
    c_devices = device_array()
    fn = _nvmlGetFunctionPointer('nvmlUnitGetDevices')
    ret = fn(unit, byref(c_count), c_devices)
    _nvmlCheckReturn(ret)
    return c_devices

def nvmlDeviceGetCount():
    if False:
        return 10
    c_count = c_uint()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetCount_v2')
    ret = fn(byref(c_count))
    _nvmlCheckReturn(ret)
    return c_count.value

def nvmlDeviceGetHandleByIndex(index):
    if False:
        while True:
            i = 10
    c_index = c_uint(index)
    device = c_nvmlDevice_t()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetHandleByIndex_v2')
    ret = fn(c_index, byref(device))
    _nvmlCheckReturn(ret)
    return device

@convertStrBytes
def nvmlDeviceGetHandleBySerial(serial):
    if False:
        i = 10
        return i + 15
    c_serial = c_char_p(serial)
    device = c_nvmlDevice_t()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetHandleBySerial')
    ret = fn(c_serial, byref(device))
    _nvmlCheckReturn(ret)
    return device

@convertStrBytes
def nvmlDeviceGetHandleByUUID(uuid):
    if False:
        for i in range(10):
            print('nop')
    c_uuid = c_char_p(uuid)
    device = c_nvmlDevice_t()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetHandleByUUID')
    ret = fn(c_uuid, byref(device))
    _nvmlCheckReturn(ret)
    return device

@convertStrBytes
def nvmlDeviceGetHandleByPciBusId(pciBusId):
    if False:
        print('Hello World!')
    c_busId = c_char_p(pciBusId)
    device = c_nvmlDevice_t()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetHandleByPciBusId_v2')
    ret = fn(c_busId, byref(device))
    _nvmlCheckReturn(ret)
    return device

@convertStrBytes
def nvmlDeviceGetName(handle):
    if False:
        i = 10
        return i + 15
    c_name = create_string_buffer(NVML_DEVICE_NAME_V2_BUFFER_SIZE)
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetName')
    ret = fn(handle, c_name, c_uint(NVML_DEVICE_NAME_V2_BUFFER_SIZE))
    _nvmlCheckReturn(ret)
    return c_name.value

def nvmlDeviceGetBoardId(handle):
    if False:
        print('Hello World!')
    c_id = c_uint()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetBoardId')
    ret = fn(handle, byref(c_id))
    _nvmlCheckReturn(ret)
    return c_id.value

def nvmlDeviceGetMultiGpuBoard(handle):
    if False:
        for i in range(10):
            print('nop')
    c_multiGpu = c_uint()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetMultiGpuBoard')
    ret = fn(handle, byref(c_multiGpu))
    _nvmlCheckReturn(ret)
    return c_multiGpu.value

def nvmlDeviceGetBrand(handle):
    if False:
        i = 10
        return i + 15
    c_type = _nvmlBrandType_t()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetBrand')
    ret = fn(handle, byref(c_type))
    _nvmlCheckReturn(ret)
    return c_type.value

@convertStrBytes
def nvmlDeviceGetBoardPartNumber(handle):
    if False:
        return 10
    c_part_number = create_string_buffer(NVML_DEVICE_PART_NUMBER_BUFFER_SIZE)
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetBoardPartNumber')
    ret = fn(handle, c_part_number, c_uint(NVML_DEVICE_PART_NUMBER_BUFFER_SIZE))
    _nvmlCheckReturn(ret)
    return c_part_number.value

@convertStrBytes
def nvmlDeviceGetSerial(handle):
    if False:
        return 10
    c_serial = create_string_buffer(NVML_DEVICE_SERIAL_BUFFER_SIZE)
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetSerial')
    ret = fn(handle, c_serial, c_uint(NVML_DEVICE_SERIAL_BUFFER_SIZE))
    _nvmlCheckReturn(ret)
    return c_serial.value

def nvmlDeviceGetModuleId(handle, moduleId):
    if False:
        return 10
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetModuleId')
    ret = fn(handle, moduleId)
    return ret

def nvmlDeviceGetMemoryAffinity(handle, nodeSetSize, scope):
    if False:
        while True:
            i = 10
    affinity_array = c_ulonglong * nodeSetSize
    c_affinity = affinity_array()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetMemoryAffinity')
    ret = fn(handle, nodeSetSize, byref(c_affinity), _nvmlAffinityScope_t(scope))
    _nvmlCheckReturn(ret)
    return c_affinity

def nvmlDeviceGetCpuAffinityWithinScope(handle, cpuSetSize, scope):
    if False:
        print('Hello World!')
    affinity_array = c_ulonglong * cpuSetSize
    c_affinity = affinity_array()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetCpuAffinityWithinScope')
    ret = fn(handle, cpuSetSize, byref(c_affinity), _nvmlAffinityScope_t(scope))
    _nvmlCheckReturn(ret)
    return c_affinity

def nvmlDeviceGetCpuAffinity(handle, cpuSetSize):
    if False:
        print('Hello World!')
    affinity_array = c_ulonglong * cpuSetSize
    c_affinity = affinity_array()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetCpuAffinity')
    ret = fn(handle, cpuSetSize, byref(c_affinity))
    _nvmlCheckReturn(ret)
    return c_affinity

def nvmlDeviceSetCpuAffinity(handle):
    if False:
        while True:
            i = 10
    fn = _nvmlGetFunctionPointer('nvmlDeviceSetCpuAffinity')
    ret = fn(handle)
    _nvmlCheckReturn(ret)
    return None

def nvmlDeviceClearCpuAffinity(handle):
    if False:
        while True:
            i = 10
    fn = _nvmlGetFunctionPointer('nvmlDeviceClearCpuAffinity')
    ret = fn(handle)
    _nvmlCheckReturn(ret)
    return None

def nvmlDeviceGetMinorNumber(handle):
    if False:
        i = 10
        return i + 15
    c_minor_number = c_uint()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetMinorNumber')
    ret = fn(handle, byref(c_minor_number))
    _nvmlCheckReturn(ret)
    return c_minor_number.value

@convertStrBytes
def nvmlDeviceGetUUID(handle):
    if False:
        return 10
    c_uuid = create_string_buffer(NVML_DEVICE_UUID_V2_BUFFER_SIZE)
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetUUID')
    ret = fn(handle, c_uuid, c_uint(NVML_DEVICE_UUID_V2_BUFFER_SIZE))
    _nvmlCheckReturn(ret)
    return c_uuid.value

@convertStrBytes
def nvmlDeviceGetInforomVersion(handle, infoRomObject):
    if False:
        return 10
    c_version = create_string_buffer(NVML_DEVICE_INFOROM_VERSION_BUFFER_SIZE)
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetInforomVersion')
    ret = fn(handle, _nvmlInforomObject_t(infoRomObject), c_version, c_uint(NVML_DEVICE_INFOROM_VERSION_BUFFER_SIZE))
    _nvmlCheckReturn(ret)
    return c_version.value

@convertStrBytes
def nvmlDeviceGetInforomImageVersion(handle):
    if False:
        return 10
    c_version = create_string_buffer(NVML_DEVICE_INFOROM_VERSION_BUFFER_SIZE)
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetInforomImageVersion')
    ret = fn(handle, c_version, c_uint(NVML_DEVICE_INFOROM_VERSION_BUFFER_SIZE))
    _nvmlCheckReturn(ret)
    return c_version.value

def nvmlDeviceGetInforomConfigurationChecksum(handle):
    if False:
        print('Hello World!')
    c_checksum = c_uint()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetInforomConfigurationChecksum')
    ret = fn(handle, byref(c_checksum))
    _nvmlCheckReturn(ret)
    return c_checksum.value

def nvmlDeviceValidateInforom(handle):
    if False:
        while True:
            i = 10
    fn = _nvmlGetFunctionPointer('nvmlDeviceValidateInforom')
    ret = fn(handle)
    _nvmlCheckReturn(ret)
    return None

def nvmlDeviceGetDisplayMode(handle):
    if False:
        print('Hello World!')
    c_mode = _nvmlEnableState_t()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetDisplayMode')
    ret = fn(handle, byref(c_mode))
    _nvmlCheckReturn(ret)
    return c_mode.value

def nvmlDeviceGetDisplayActive(handle):
    if False:
        for i in range(10):
            print('nop')
    c_mode = _nvmlEnableState_t()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetDisplayActive')
    ret = fn(handle, byref(c_mode))
    _nvmlCheckReturn(ret)
    return c_mode.value

def nvmlDeviceGetPersistenceMode(handle):
    if False:
        print('Hello World!')
    c_state = _nvmlEnableState_t()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetPersistenceMode')
    ret = fn(handle, byref(c_state))
    _nvmlCheckReturn(ret)
    return c_state.value

def nvmlDeviceGetPciInfo_v3(handle):
    if False:
        return 10
    c_info = nvmlPciInfo_t()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetPciInfo_v3')
    ret = fn(handle, byref(c_info))
    _nvmlCheckReturn(ret)
    return c_info

def nvmlDeviceGetPciInfo(handle):
    if False:
        for i in range(10):
            print('nop')
    return nvmlDeviceGetPciInfo_v3(handle)

def nvmlDeviceGetClockInfo(handle, type):
    if False:
        for i in range(10):
            print('nop')
    c_clock = c_uint()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetClockInfo')
    ret = fn(handle, _nvmlClockType_t(type), byref(c_clock))
    _nvmlCheckReturn(ret)
    return c_clock.value

def nvmlDeviceGetMaxClockInfo(handle, type):
    if False:
        for i in range(10):
            print('nop')
    c_clock = c_uint()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetMaxClockInfo')
    ret = fn(handle, _nvmlClockType_t(type), byref(c_clock))
    _nvmlCheckReturn(ret)
    return c_clock.value

def nvmlDeviceGetApplicationsClock(handle, type):
    if False:
        i = 10
        return i + 15
    c_clock = c_uint()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetApplicationsClock')
    ret = fn(handle, _nvmlClockType_t(type), byref(c_clock))
    _nvmlCheckReturn(ret)
    return c_clock.value

def nvmlDeviceGetMaxCustomerBoostClock(handle, type):
    if False:
        while True:
            i = 10
    c_clock = c_uint()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetMaxCustomerBoostClock')
    ret = fn(handle, _nvmlClockType_t(type), byref(c_clock))
    _nvmlCheckReturn(ret)
    return c_clock.value

def nvmlDeviceGetClock(handle, type, id):
    if False:
        print('Hello World!')
    c_clock = c_uint()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetClock')
    ret = fn(handle, _nvmlClockType_t(type), _nvmlClockId_t(id), byref(c_clock))
    _nvmlCheckReturn(ret)
    return c_clock.value

def nvmlDeviceGetDefaultApplicationsClock(handle, type):
    if False:
        return 10
    c_clock = c_uint()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetDefaultApplicationsClock')
    ret = fn(handle, _nvmlClockType_t(type), byref(c_clock))
    _nvmlCheckReturn(ret)
    return c_clock.value

def nvmlDeviceGetSupportedMemoryClocks(handle):
    if False:
        i = 10
        return i + 15
    c_count = c_uint(0)
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetSupportedMemoryClocks')
    ret = fn(handle, byref(c_count), None)
    if ret == NVML_SUCCESS:
        return []
    elif ret == NVML_ERROR_INSUFFICIENT_SIZE:
        clocks_array = c_uint * c_count.value
        c_clocks = clocks_array()
        ret = fn(handle, byref(c_count), c_clocks)
        _nvmlCheckReturn(ret)
        procs = []
        for i in range(c_count.value):
            procs.append(c_clocks[i])
        return procs
    else:
        raise NVMLError(ret)

def nvmlDeviceGetSupportedGraphicsClocks(handle, memoryClockMHz):
    if False:
        return 10
    c_count = c_uint(0)
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetSupportedGraphicsClocks')
    ret = fn(handle, c_uint(memoryClockMHz), byref(c_count), None)
    if ret == NVML_SUCCESS:
        return []
    elif ret == NVML_ERROR_INSUFFICIENT_SIZE:
        clocks_array = c_uint * c_count.value
        c_clocks = clocks_array()
        ret = fn(handle, c_uint(memoryClockMHz), byref(c_count), c_clocks)
        _nvmlCheckReturn(ret)
        procs = []
        for i in range(c_count.value):
            procs.append(c_clocks[i])
        return procs
    else:
        raise NVMLError(ret)

def nvmlDeviceGetFanSpeed(handle):
    if False:
        return 10
    c_speed = c_uint()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetFanSpeed')
    ret = fn(handle, byref(c_speed))
    _nvmlCheckReturn(ret)
    return c_speed.value

def nvmlDeviceGetFanSpeed_v2(handle, fan):
    if False:
        return 10
    c_speed = c_uint()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetFanSpeed_v2')
    ret = fn(handle, fan, byref(c_speed))
    _nvmlCheckReturn(ret)
    return c_speed.value

def nvmlDeviceGetTargetFanSpeed(handle, fan):
    if False:
        for i in range(10):
            print('nop')
    c_speed = c_uint()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetTargetFanSpeed')
    ret = fn(handle, fan, byref(c_speed))
    _nvmlCheckReturn(ret)
    return c_speed.value

def nvmlDeviceGetNumFans(device):
    if False:
        print('Hello World!')
    c_numFans = c_uint()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetNumFans')
    ret = fn(device, byref(c_numFans))
    _nvmlCheckReturn(ret)
    return c_numFans.value

def nvmlDeviceSetDefaultFanSpeed_v2(handle, index):
    if False:
        print('Hello World!')
    fn = _nvmlGetFunctionPointer('nvmlDeviceSetDefaultFanSpeed_v2')
    ret = fn(handle, index)
    _nvmlCheckReturn(ret)
    return ret

def nvmlDeviceGetMinMaxFanSpeed(handle, minSpeed, maxSpeed):
    if False:
        print('Hello World!')
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetMinMaxFanSpeed')
    ret = fn(handle, minSpeed, maxSpeed)
    _nvmlCheckReturn(ret)
    return ret

def nvmlDeviceGetFanControlPolicy_v2(handle, fan, fanControlPolicy):
    if False:
        while True:
            i = 10
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetFanControlPolicy_v2')
    ret = fn(handle, fan, fanControlPolicy)
    _nvmlCheckReturn(ret)
    return ret

def nvmlDeviceSetFanControlPolicy(handle, fan, fanControlPolicy):
    if False:
        for i in range(10):
            print('nop')
    fn = _nvmlGetFunctionPointer('nvmlDeviceSetFanControlPolicy')
    ret = fn(handle, fan, _nvmlFanControlPolicy_t(fanControlPolicy))
    _nvmlCheckReturn(ret)
    return ret

def nvmlDeviceGetTemperature(handle, sensor):
    if False:
        i = 10
        return i + 15
    c_temp = c_uint()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetTemperature')
    ret = fn(handle, _nvmlTemperatureSensors_t(sensor), byref(c_temp))
    _nvmlCheckReturn(ret)
    return c_temp.value

def nvmlDeviceGetTemperatureThreshold(handle, threshold):
    if False:
        print('Hello World!')
    c_temp = c_uint()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetTemperatureThreshold')
    ret = fn(handle, _nvmlTemperatureThresholds_t(threshold), byref(c_temp))
    _nvmlCheckReturn(ret)
    return c_temp.value

def nvmlDeviceSetTemperatureThreshold(handle, threshold, temp):
    if False:
        while True:
            i = 10
    c_temp = c_uint()
    c_temp.value = temp
    fn = _nvmlGetFunctionPointer('nvmlDeviceSetTemperatureThreshold')
    ret = fn(handle, _nvmlTemperatureThresholds_t(threshold), byref(c_temp))
    _nvmlCheckReturn(ret)
    return None

def nvmlDeviceGetPowerState(handle):
    if False:
        return 10
    c_pstate = _nvmlPstates_t()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetPowerState')
    ret = fn(handle, byref(c_pstate))
    _nvmlCheckReturn(ret)
    return c_pstate.value

def nvmlDeviceGetPerformanceState(handle):
    if False:
        return 10
    c_pstate = _nvmlPstates_t()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetPerformanceState')
    ret = fn(handle, byref(c_pstate))
    _nvmlCheckReturn(ret)
    return c_pstate.value

def nvmlDeviceGetPowerManagementMode(handle):
    if False:
        i = 10
        return i + 15
    c_pcapMode = _nvmlEnableState_t()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetPowerManagementMode')
    ret = fn(handle, byref(c_pcapMode))
    _nvmlCheckReturn(ret)
    return c_pcapMode.value

def nvmlDeviceGetPowerManagementLimit(handle):
    if False:
        return 10
    c_limit = c_uint()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetPowerManagementLimit')
    ret = fn(handle, byref(c_limit))
    _nvmlCheckReturn(ret)
    return c_limit.value

def nvmlDeviceGetPowerManagementLimitConstraints(handle):
    if False:
        while True:
            i = 10
    c_minLimit = c_uint()
    c_maxLimit = c_uint()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetPowerManagementLimitConstraints')
    ret = fn(handle, byref(c_minLimit), byref(c_maxLimit))
    _nvmlCheckReturn(ret)
    return [c_minLimit.value, c_maxLimit.value]

def nvmlDeviceGetPowerManagementDefaultLimit(handle):
    if False:
        i = 10
        return i + 15
    c_limit = c_uint()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetPowerManagementDefaultLimit')
    ret = fn(handle, byref(c_limit))
    _nvmlCheckReturn(ret)
    return c_limit.value

def nvmlDeviceGetEnforcedPowerLimit(handle):
    if False:
        i = 10
        return i + 15
    c_limit = c_uint()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetEnforcedPowerLimit')
    ret = fn(handle, byref(c_limit))
    _nvmlCheckReturn(ret)
    return c_limit.value

def nvmlDeviceGetPowerUsage(handle):
    if False:
        print('Hello World!')
    c_watts = c_uint()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetPowerUsage')
    ret = fn(handle, byref(c_watts))
    _nvmlCheckReturn(ret)
    return c_watts.value

def nvmlDeviceGetTotalEnergyConsumption(handle):
    if False:
        i = 10
        return i + 15
    c_millijoules = c_uint64()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetTotalEnergyConsumption')
    ret = fn(handle, byref(c_millijoules))
    _nvmlCheckReturn(ret)
    return c_millijoules.value

def nvmlDeviceGetGpuOperationMode(handle):
    if False:
        return 10
    c_currState = _nvmlGpuOperationMode_t()
    c_pendingState = _nvmlGpuOperationMode_t()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetGpuOperationMode')
    ret = fn(handle, byref(c_currState), byref(c_pendingState))
    _nvmlCheckReturn(ret)
    return [c_currState.value, c_pendingState.value]

def nvmlDeviceGetCurrentGpuOperationMode(handle):
    if False:
        while True:
            i = 10
    return nvmlDeviceGetGpuOperationMode(handle)[0]

def nvmlDeviceGetPendingGpuOperationMode(handle):
    if False:
        while True:
            i = 10
    return nvmlDeviceGetGpuOperationMode(handle)[1]

def nvmlDeviceGetMemoryInfo(handle, version=None):
    if False:
        for i in range(10):
            print('nop')
    if not version:
        c_memory = c_nvmlMemory_t()
        fn = _nvmlGetFunctionPointer('nvmlDeviceGetMemoryInfo')
    else:
        c_memory = c_nvmlMemory_v2_t()
        c_memory.version = version
        fn = _nvmlGetFunctionPointer('nvmlDeviceGetMemoryInfo_v2')
    ret = fn(handle, byref(c_memory))
    _nvmlCheckReturn(ret)
    return c_memory

def nvmlDeviceGetBAR1MemoryInfo(handle):
    if False:
        i = 10
        return i + 15
    c_bar1_memory = c_nvmlBAR1Memory_t()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetBAR1MemoryInfo')
    ret = fn(handle, byref(c_bar1_memory))
    _nvmlCheckReturn(ret)
    return c_bar1_memory

def nvmlDeviceGetComputeMode(handle):
    if False:
        while True:
            i = 10
    c_mode = _nvmlComputeMode_t()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetComputeMode')
    ret = fn(handle, byref(c_mode))
    _nvmlCheckReturn(ret)
    return c_mode.value

def nvmlDeviceGetCudaComputeCapability(handle):
    if False:
        print('Hello World!')
    c_major = c_int()
    c_minor = c_int()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetCudaComputeCapability')
    ret = fn(handle, byref(c_major), byref(c_minor))
    _nvmlCheckReturn(ret)
    return (c_major.value, c_minor.value)

def nvmlDeviceGetEccMode(handle):
    if False:
        print('Hello World!')
    c_currState = _nvmlEnableState_t()
    c_pendingState = _nvmlEnableState_t()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetEccMode')
    ret = fn(handle, byref(c_currState), byref(c_pendingState))
    _nvmlCheckReturn(ret)
    return [c_currState.value, c_pendingState.value]

def nvmlDeviceGetCurrentEccMode(handle):
    if False:
        print('Hello World!')
    return nvmlDeviceGetEccMode(handle)[0]

def nvmlDeviceGetPendingEccMode(handle):
    if False:
        while True:
            i = 10
    return nvmlDeviceGetEccMode(handle)[1]

def nvmlDeviceGetDefaultEccMode(handle):
    if False:
        return 10
    c_defaultState = _nvmlEnableState_t()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetDefaultEccMode')
    ret = fn(handle, byref(c_defaultState))
    _nvmlCheckReturn(ret)
    return [c_defaultState.value]

def nvmlDeviceGetTotalEccErrors(handle, errorType, counterType):
    if False:
        for i in range(10):
            print('nop')
    c_count = c_ulonglong()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetTotalEccErrors')
    ret = fn(handle, _nvmlMemoryErrorType_t(errorType), _nvmlEccCounterType_t(counterType), byref(c_count))
    _nvmlCheckReturn(ret)
    return c_count.value

def nvmlDeviceGetDetailedEccErrors(handle, errorType, counterType):
    if False:
        return 10
    c_counts = c_nvmlEccErrorCounts_t()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetDetailedEccErrors')
    ret = fn(handle, _nvmlMemoryErrorType_t(errorType), _nvmlEccCounterType_t(counterType), byref(c_counts))
    _nvmlCheckReturn(ret)
    return c_counts

def nvmlDeviceGetMemoryErrorCounter(handle, errorType, counterType, locationType):
    if False:
        for i in range(10):
            print('nop')
    c_count = c_ulonglong()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetMemoryErrorCounter')
    ret = fn(handle, _nvmlMemoryErrorType_t(errorType), _nvmlEccCounterType_t(counterType), _nvmlMemoryLocation_t(locationType), byref(c_count))
    _nvmlCheckReturn(ret)
    return c_count.value

def nvmlDeviceGetUtilizationRates(handle):
    if False:
        print('Hello World!')
    c_util = c_nvmlUtilization_t()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetUtilizationRates')
    ret = fn(handle, byref(c_util))
    _nvmlCheckReturn(ret)
    return c_util

def nvmlDeviceGetEncoderUtilization(handle):
    if False:
        return 10
    c_util = c_uint()
    c_samplingPeriod = c_uint()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetEncoderUtilization')
    ret = fn(handle, byref(c_util), byref(c_samplingPeriod))
    _nvmlCheckReturn(ret)
    return [c_util.value, c_samplingPeriod.value]

def nvmlDeviceGetDecoderUtilization(handle):
    if False:
        print('Hello World!')
    c_util = c_uint()
    c_samplingPeriod = c_uint()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetDecoderUtilization')
    ret = fn(handle, byref(c_util), byref(c_samplingPeriod))
    _nvmlCheckReturn(ret)
    return [c_util.value, c_samplingPeriod.value]

def nvmlDeviceGetJpgUtilization(handle):
    if False:
        print('Hello World!')
    c_util = c_uint()
    c_samplingPeriod = c_uint()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetJpgUtilization')
    ret = fn(handle, byref(c_util), byref(c_samplingPeriod))
    _nvmlCheckReturn(ret)
    return [c_util.value, c_samplingPeriod.value]

def nvmlDeviceGetOfaUtilization(handle):
    if False:
        i = 10
        return i + 15
    c_util = c_uint()
    c_samplingPeriod = c_uint()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetOfaUtilization')
    ret = fn(handle, byref(c_util), byref(c_samplingPeriod))
    _nvmlCheckReturn(ret)
    return [c_util.value, c_samplingPeriod.value]

def nvmlDeviceGetPcieReplayCounter(handle):
    if False:
        return 10
    c_replay = c_uint()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetPcieReplayCounter')
    ret = fn(handle, byref(c_replay))
    _nvmlCheckReturn(ret)
    return c_replay.value

def nvmlDeviceGetDriverModel(handle):
    if False:
        return 10
    c_currModel = _nvmlDriverModel_t()
    c_pendingModel = _nvmlDriverModel_t()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetDriverModel')
    ret = fn(handle, byref(c_currModel), byref(c_pendingModel))
    _nvmlCheckReturn(ret)
    return [c_currModel.value, c_pendingModel.value]

def nvmlDeviceGetCurrentDriverModel(handle):
    if False:
        print('Hello World!')
    return nvmlDeviceGetDriverModel(handle)[0]

def nvmlDeviceGetPendingDriverModel(handle):
    if False:
        print('Hello World!')
    return nvmlDeviceGetDriverModel(handle)[1]

@convertStrBytes
def nvmlDeviceGetVbiosVersion(handle):
    if False:
        for i in range(10):
            print('nop')
    c_version = create_string_buffer(NVML_DEVICE_VBIOS_VERSION_BUFFER_SIZE)
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetVbiosVersion')
    ret = fn(handle, c_version, c_uint(NVML_DEVICE_VBIOS_VERSION_BUFFER_SIZE))
    _nvmlCheckReturn(ret)
    return c_version.value

def nvmlDeviceGetComputeRunningProcesses_v3(handle):
    if False:
        while True:
            i = 10
    c_count = c_uint(0)
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetComputeRunningProcesses_v3')
    ret = fn(handle, byref(c_count), None)
    if ret == NVML_SUCCESS:
        return []
    elif ret == NVML_ERROR_INSUFFICIENT_SIZE:
        c_count.value = c_count.value * 2 + 5
        proc_array = c_nvmlProcessInfo_t * c_count.value
        c_procs = proc_array()
        ret = fn(handle, byref(c_count), c_procs)
        _nvmlCheckReturn(ret)
        procs = []
        for i in range(c_count.value):
            obj = nvmlStructToFriendlyObject(c_procs[i])
            if obj.usedGpuMemory == NVML_VALUE_NOT_AVAILABLE_ulonglong.value:
                obj.usedGpuMemory = None
            procs.append(obj)
        return procs
    else:
        raise NVMLError(ret)

def nvmlDeviceGetComputeRunningProcesses(handle):
    if False:
        print('Hello World!')
    return nvmlDeviceGetComputeRunningProcesses_v3(handle)

def nvmlDeviceGetGraphicsRunningProcesses_v3(handle):
    if False:
        return 10
    c_count = c_uint(0)
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetGraphicsRunningProcesses_v3')
    ret = fn(handle, byref(c_count), None)
    if ret == NVML_SUCCESS:
        return []
    elif ret == NVML_ERROR_INSUFFICIENT_SIZE:
        c_count.value = c_count.value * 2 + 5
        proc_array = c_nvmlProcessInfo_t * c_count.value
        c_procs = proc_array()
        ret = fn(handle, byref(c_count), c_procs)
        _nvmlCheckReturn(ret)
        procs = []
        for i in range(c_count.value):
            obj = nvmlStructToFriendlyObject(c_procs[i])
            if obj.usedGpuMemory == NVML_VALUE_NOT_AVAILABLE_ulonglong.value:
                obj.usedGpuMemory = None
            procs.append(obj)
        return procs
    else:
        raise NVMLError(ret)

def nvmlDeviceGetGraphicsRunningProcesses(handle):
    if False:
        while True:
            i = 10
    return nvmlDeviceGetGraphicsRunningProcesses_v3(handle)

def nvmlDeviceGetMPSComputeRunningProcesses(handle):
    if False:
        print('Hello World!')
    return nvmlDeviceGetMPSComputeRunningProcesses_v3(handle)

def nvmlDeviceGetMPSComputeRunningProcesses_v3(handle):
    if False:
        i = 10
        return i + 15
    c_count = c_uint(0)
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetMPSComputeRunningProcesses_v3')
    ret = fn(handle, byref(c_count), None)
    if ret == NVML_SUCCESS:
        return []
    elif ret == NVML_ERROR_INSUFFICIENT_SIZE:
        c_count.value = c_count.value * 2 + 5
        proc_array = c_nvmlProcessInfo_t * c_count.value
        c_procs = proc_array()
        ret = fn(handle, byref(c_count), c_procs)
        _nvmlCheckReturn(ret)
        procs = []
        for i in range(c_count.value):
            obj = nvmlStructToFriendlyObject(c_procs[i])
            if obj.usedGpuMemory == NVML_VALUE_NOT_AVAILABLE_ulonglong.value:
                obj.usedGpuMemory = None
            procs.append(obj)
        return procs
    else:
        raise NVMLError(ret)

def nvmlDeviceGetRunningProcessDetailList(handle, version, mode):
    if False:
        return 10
    c_processDetailList = c_nvmlProcessDetailList_t()
    c_processDetailList.version = version
    c_processDetailList.mode = mode
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetRunningProcessDetailList')
    ret = fn(handle, c_processDetailList)
    return ret

def nvmlDeviceGetAutoBoostedClocksEnabled(handle):
    if False:
        for i in range(10):
            print('nop')
    c_isEnabled = _nvmlEnableState_t()
    c_defaultIsEnabled = _nvmlEnableState_t()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetAutoBoostedClocksEnabled')
    ret = fn(handle, byref(c_isEnabled), byref(c_defaultIsEnabled))
    _nvmlCheckReturn(ret)
    return [c_isEnabled.value, c_defaultIsEnabled.value]

def nvmlUnitSetLedState(unit, color):
    if False:
        return 10
    fn = _nvmlGetFunctionPointer('nvmlUnitSetLedState')
    ret = fn(unit, _nvmlLedColor_t(color))
    _nvmlCheckReturn(ret)
    return None

def nvmlDeviceSetPersistenceMode(handle, mode):
    if False:
        return 10
    fn = _nvmlGetFunctionPointer('nvmlDeviceSetPersistenceMode')
    ret = fn(handle, _nvmlEnableState_t(mode))
    _nvmlCheckReturn(ret)
    return None

def nvmlDeviceSetComputeMode(handle, mode):
    if False:
        print('Hello World!')
    fn = _nvmlGetFunctionPointer('nvmlDeviceSetComputeMode')
    ret = fn(handle, _nvmlComputeMode_t(mode))
    _nvmlCheckReturn(ret)
    return None

def nvmlDeviceSetEccMode(handle, mode):
    if False:
        i = 10
        return i + 15
    fn = _nvmlGetFunctionPointer('nvmlDeviceSetEccMode')
    ret = fn(handle, _nvmlEnableState_t(mode))
    _nvmlCheckReturn(ret)
    return None

def nvmlDeviceClearEccErrorCounts(handle, counterType):
    if False:
        print('Hello World!')
    fn = _nvmlGetFunctionPointer('nvmlDeviceClearEccErrorCounts')
    ret = fn(handle, _nvmlEccCounterType_t(counterType))
    _nvmlCheckReturn(ret)
    return None

def nvmlDeviceSetDriverModel(handle, model):
    if False:
        while True:
            i = 10
    fn = _nvmlGetFunctionPointer('nvmlDeviceSetDriverModel')
    ret = fn(handle, _nvmlDriverModel_t(model))
    _nvmlCheckReturn(ret)
    return None

def nvmlDeviceSetAutoBoostedClocksEnabled(handle, enabled):
    if False:
        return 10
    fn = _nvmlGetFunctionPointer('nvmlDeviceSetAutoBoostedClocksEnabled')
    ret = fn(handle, _nvmlEnableState_t(enabled))
    _nvmlCheckReturn(ret)
    return None

def nvmlDeviceSetDefaultAutoBoostedClocksEnabled(handle, enabled, flags):
    if False:
        print('Hello World!')
    fn = _nvmlGetFunctionPointer('nvmlDeviceSetDefaultAutoBoostedClocksEnabled')
    ret = fn(handle, _nvmlEnableState_t(enabled), c_uint(flags))
    _nvmlCheckReturn(ret)
    return None

def nvmlDeviceSetGpuLockedClocks(handle, minGpuClockMHz, maxGpuClockMHz):
    if False:
        print('Hello World!')
    fn = _nvmlGetFunctionPointer('nvmlDeviceSetGpuLockedClocks')
    ret = fn(handle, c_uint(minGpuClockMHz), c_uint(maxGpuClockMHz))
    _nvmlCheckReturn(ret)
    return None

def nvmlDeviceResetGpuLockedClocks(handle):
    if False:
        print('Hello World!')
    fn = _nvmlGetFunctionPointer('nvmlDeviceResetGpuLockedClocks')
    ret = fn(handle)
    _nvmlCheckReturn(ret)
    return None

def nvmlDeviceSetMemoryLockedClocks(handle, minMemClockMHz, maxMemClockMHz):
    if False:
        print('Hello World!')
    fn = _nvmlGetFunctionPointer('nvmlDeviceSetMemoryLockedClocks')
    ret = fn(handle, c_uint(minMemClockMHz), c_uint(maxMemClockMHz))
    _nvmlCheckReturn(ret)
    return None

def nvmlDeviceResetMemoryLockedClocks(handle):
    if False:
        i = 10
        return i + 15
    fn = _nvmlGetFunctionPointer('nvmlDeviceResetMemoryLockedClocks')
    ret = fn(handle)
    _nvmlCheckReturn(ret)
    return None

def nvmlDeviceGetClkMonStatus(handle, c_clkMonInfo):
    if False:
        i = 10
        return i + 15
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetClkMonStatus')
    ret = fn(handle, c_clkMonInfo)
    return ret

def nvmlDeviceSetApplicationsClocks(handle, maxMemClockMHz, maxGraphicsClockMHz):
    if False:
        print('Hello World!')
    fn = _nvmlGetFunctionPointer('nvmlDeviceSetApplicationsClocks')
    ret = fn(handle, c_uint(maxMemClockMHz), c_uint(maxGraphicsClockMHz))
    _nvmlCheckReturn(ret)
    return None

def nvmlDeviceResetApplicationsClocks(handle):
    if False:
        return 10
    fn = _nvmlGetFunctionPointer('nvmlDeviceResetApplicationsClocks')
    ret = fn(handle)
    _nvmlCheckReturn(ret)
    return None

def nvmlDeviceSetPowerManagementLimit(handle, limit):
    if False:
        while True:
            i = 10
    fn = _nvmlGetFunctionPointer('nvmlDeviceSetPowerManagementLimit')
    ret = fn(handle, c_uint(limit))
    _nvmlCheckReturn(ret)
    return None

def nvmlDeviceSetGpuOperationMode(handle, mode):
    if False:
        for i in range(10):
            print('nop')
    fn = _nvmlGetFunctionPointer('nvmlDeviceSetGpuOperationMode')
    ret = fn(handle, _nvmlGpuOperationMode_t(mode))
    _nvmlCheckReturn(ret)
    return None

def nvmlEventSetCreate():
    if False:
        i = 10
        return i + 15
    fn = _nvmlGetFunctionPointer('nvmlEventSetCreate')
    eventSet = c_nvmlEventSet_t()
    ret = fn(byref(eventSet))
    _nvmlCheckReturn(ret)
    return eventSet

def nvmlDeviceRegisterEvents(handle, eventTypes, eventSet):
    if False:
        while True:
            i = 10
    fn = _nvmlGetFunctionPointer('nvmlDeviceRegisterEvents')
    ret = fn(handle, c_ulonglong(eventTypes), eventSet)
    _nvmlCheckReturn(ret)
    return None

def nvmlDeviceGetSupportedEventTypes(handle):
    if False:
        i = 10
        return i + 15
    c_eventTypes = c_ulonglong()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetSupportedEventTypes')
    ret = fn(handle, byref(c_eventTypes))
    _nvmlCheckReturn(ret)
    return c_eventTypes.value

def nvmlEventSetWait_v2(eventSet, timeoutms):
    if False:
        print('Hello World!')
    fn = _nvmlGetFunctionPointer('nvmlEventSetWait_v2')
    data = c_nvmlEventData_t()
    ret = fn(eventSet, byref(data), c_uint(timeoutms))
    _nvmlCheckReturn(ret)
    return data

def nvmlEventSetWait(eventSet, timeoutms):
    if False:
        for i in range(10):
            print('nop')
    return nvmlEventSetWait_v2(eventSet, timeoutms)

def nvmlEventSetFree(eventSet):
    if False:
        i = 10
        return i + 15
    fn = _nvmlGetFunctionPointer('nvmlEventSetFree')
    ret = fn(eventSet)
    _nvmlCheckReturn(ret)
    return None

def nvmlDeviceOnSameBoard(handle1, handle2):
    if False:
        return 10
    fn = _nvmlGetFunctionPointer('nvmlDeviceOnSameBoard')
    onSameBoard = c_int()
    ret = fn(handle1, handle2, byref(onSameBoard))
    _nvmlCheckReturn(ret)
    return onSameBoard.value != 0

def nvmlDeviceGetCurrPcieLinkGeneration(handle):
    if False:
        i = 10
        return i + 15
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetCurrPcieLinkGeneration')
    gen = c_uint()
    ret = fn(handle, byref(gen))
    _nvmlCheckReturn(ret)
    return gen.value

def nvmlDeviceGetMaxPcieLinkGeneration(handle):
    if False:
        for i in range(10):
            print('nop')
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetMaxPcieLinkGeneration')
    gen = c_uint()
    ret = fn(handle, byref(gen))
    _nvmlCheckReturn(ret)
    return gen.value

def nvmlDeviceGetCurrPcieLinkWidth(handle):
    if False:
        for i in range(10):
            print('nop')
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetCurrPcieLinkWidth')
    width = c_uint()
    ret = fn(handle, byref(width))
    _nvmlCheckReturn(ret)
    return width.value

def nvmlDeviceGetMaxPcieLinkWidth(handle):
    if False:
        while True:
            i = 10
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetMaxPcieLinkWidth')
    width = c_uint()
    ret = fn(handle, byref(width))
    _nvmlCheckReturn(ret)
    return width.value

def nvmlDeviceGetGpuMaxPcieLinkGeneration(handle):
    if False:
        return 10
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetGpuMaxPcieLinkGeneration')
    gen = c_uint()
    ret = fn(handle, byref(gen))
    _nvmlCheckReturn(ret)
    return gen.value

def nvmlDeviceGetSupportedClocksThrottleReasons(handle):
    if False:
        while True:
            i = 10
    c_reasons = c_ulonglong()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetSupportedClocksThrottleReasons')
    ret = fn(handle, byref(c_reasons))
    _nvmlCheckReturn(ret)
    return c_reasons.value

def nvmlDeviceGetSupportedClocksEventReasons(handle):
    if False:
        while True:
            i = 10
    c_reasons = c_ulonglong()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetSupportedClocksEventReasons')
    ret = fn(handle, byref(c_reasons))
    _nvmlCheckReturn(ret)
    return c_reasons.value

def nvmlDeviceGetCurrentClocksThrottleReasons(handle):
    if False:
        print('Hello World!')
    c_reasons = c_ulonglong()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetCurrentClocksThrottleReasons')
    ret = fn(handle, byref(c_reasons))
    _nvmlCheckReturn(ret)
    return c_reasons.value

def nvmlDeviceGetCurrentClocksEventReasons(handle):
    if False:
        return 10
    c_reasons = c_ulonglong()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetCurrentClocksEventReasons')
    ret = fn(handle, byref(c_reasons))
    _nvmlCheckReturn(ret)
    return c_reasons.value

def nvmlDeviceGetIndex(handle):
    if False:
        print('Hello World!')
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetIndex')
    c_index = c_uint()
    ret = fn(handle, byref(c_index))
    _nvmlCheckReturn(ret)
    return c_index.value

def nvmlDeviceGetAccountingMode(handle):
    if False:
        i = 10
        return i + 15
    c_mode = _nvmlEnableState_t()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetAccountingMode')
    ret = fn(handle, byref(c_mode))
    _nvmlCheckReturn(ret)
    return c_mode.value

def nvmlDeviceSetAccountingMode(handle, mode):
    if False:
        i = 10
        return i + 15
    fn = _nvmlGetFunctionPointer('nvmlDeviceSetAccountingMode')
    ret = fn(handle, _nvmlEnableState_t(mode))
    _nvmlCheckReturn(ret)
    return None

def nvmlDeviceClearAccountingPids(handle):
    if False:
        while True:
            i = 10
    fn = _nvmlGetFunctionPointer('nvmlDeviceClearAccountingPids')
    ret = fn(handle)
    _nvmlCheckReturn(ret)
    return None

def nvmlDeviceGetAccountingStats(handle, pid):
    if False:
        return 10
    stats = c_nvmlAccountingStats_t()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetAccountingStats')
    ret = fn(handle, c_uint(pid), byref(stats))
    _nvmlCheckReturn(ret)
    if stats.maxMemoryUsage == NVML_VALUE_NOT_AVAILABLE_ulonglong.value:
        stats.maxMemoryUsage = None
    return stats

def nvmlDeviceGetAccountingPids(handle):
    if False:
        for i in range(10):
            print('nop')
    count = c_uint(nvmlDeviceGetAccountingBufferSize(handle))
    pids = (c_uint * count.value)()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetAccountingPids')
    ret = fn(handle, byref(count), pids)
    _nvmlCheckReturn(ret)
    return list(map(int, pids[0:count.value]))

def nvmlDeviceGetAccountingBufferSize(handle):
    if False:
        i = 10
        return i + 15
    bufferSize = c_uint()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetAccountingBufferSize')
    ret = fn(handle, byref(bufferSize))
    _nvmlCheckReturn(ret)
    return int(bufferSize.value)

def nvmlDeviceGetRetiredPages(device, sourceFilter):
    if False:
        while True:
            i = 10
    c_source = _nvmlPageRetirementCause_t(sourceFilter)
    c_count = c_uint(0)
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetRetiredPages')
    ret = fn(device, c_source, byref(c_count), None)
    if ret != NVML_SUCCESS and ret != NVML_ERROR_INSUFFICIENT_SIZE:
        raise NVMLError(ret)
    c_count.value = c_count.value * 2 + 5
    page_array = c_ulonglong * c_count.value
    c_pages = page_array()
    ret = fn(device, c_source, byref(c_count), c_pages)
    _nvmlCheckReturn(ret)
    return list(map(int, c_pages[0:c_count.value]))

def nvmlDeviceGetRetiredPages_v2(device, sourceFilter):
    if False:
        while True:
            i = 10
    c_source = _nvmlPageRetirementCause_t(sourceFilter)
    c_count = c_uint(0)
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetRetiredPages_v2')
    ret = fn(device, c_source, byref(c_count), None)
    if ret != NVML_SUCCESS and ret != NVML_ERROR_INSUFFICIENT_SIZE:
        raise NVMLError(ret)
    c_count.value = c_count.value * 2 + 5
    page_array = c_ulonglong * c_count.value
    c_pages = page_array()
    times_array = c_ulonglong * c_count.value
    c_times = times_array()
    ret = fn(device, c_source, byref(c_count), c_pages, c_times)
    _nvmlCheckReturn(ret)
    return [{'address': int(c_pages[i]), 'timestamp': int(c_times[i])} for i in range(c_count.value)]

def nvmlDeviceGetRetiredPagesPendingStatus(device):
    if False:
        return 10
    c_pending = _nvmlEnableState_t()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetRetiredPagesPendingStatus')
    ret = fn(device, byref(c_pending))
    _nvmlCheckReturn(ret)
    return int(c_pending.value)

def nvmlDeviceGetAPIRestriction(device, apiType):
    if False:
        while True:
            i = 10
    c_permission = _nvmlEnableState_t()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetAPIRestriction')
    ret = fn(device, _nvmlRestrictedAPI_t(apiType), byref(c_permission))
    _nvmlCheckReturn(ret)
    return int(c_permission.value)

def nvmlDeviceSetAPIRestriction(handle, apiType, isRestricted):
    if False:
        while True:
            i = 10
    fn = _nvmlGetFunctionPointer('nvmlDeviceSetAPIRestriction')
    ret = fn(handle, _nvmlRestrictedAPI_t(apiType), _nvmlEnableState_t(isRestricted))
    _nvmlCheckReturn(ret)
    return None

def nvmlDeviceGetBridgeChipInfo(handle):
    if False:
        i = 10
        return i + 15
    bridgeHierarchy = c_nvmlBridgeChipHierarchy_t()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetBridgeChipInfo')
    ret = fn(handle, byref(bridgeHierarchy))
    _nvmlCheckReturn(ret)
    return bridgeHierarchy

def nvmlDeviceGetSamples(device, sampling_type, timeStamp):
    if False:
        while True:
            i = 10
    c_sampling_type = _nvmlSamplingType_t(sampling_type)
    c_time_stamp = c_ulonglong(timeStamp)
    c_sample_count = c_uint(0)
    c_sample_value_type = _nvmlValueType_t()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetSamples')
    ret = fn(device, c_sampling_type, c_time_stamp, byref(c_sample_value_type), byref(c_sample_count), None)
    if ret != NVML_SUCCESS:
        raise NVMLError(ret)
    sampleArray = c_sample_count.value * c_nvmlSample_t
    c_samples = sampleArray()
    ret = fn(device, c_sampling_type, c_time_stamp, byref(c_sample_value_type), byref(c_sample_count), c_samples)
    _nvmlCheckReturn(ret)
    return (c_sample_value_type.value, c_samples[0:c_sample_count.value])

def nvmlDeviceGetViolationStatus(device, perfPolicyType):
    if False:
        return 10
    c_perfPolicy_type = _nvmlPerfPolicyType_t(perfPolicyType)
    c_violTime = c_nvmlViolationTime_t()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetViolationStatus')
    ret = fn(device, c_perfPolicy_type, byref(c_violTime))
    _nvmlCheckReturn(ret)
    return c_violTime

def nvmlDeviceGetPcieThroughput(device, counter):
    if False:
        return 10
    c_util = c_uint()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetPcieThroughput')
    ret = fn(device, _nvmlPcieUtilCounter_t(counter), byref(c_util))
    _nvmlCheckReturn(ret)
    return c_util.value

def nvmlSystemGetTopologyGpuSet(cpuNumber):
    if False:
        i = 10
        return i + 15
    c_count = c_uint(0)
    fn = _nvmlGetFunctionPointer('nvmlSystemGetTopologyGpuSet')
    ret = fn(cpuNumber, byref(c_count), None)
    if ret != NVML_SUCCESS:
        raise NVMLError(ret)
    device_array = c_nvmlDevice_t * c_count.value
    c_devices = device_array()
    ret = fn(cpuNumber, byref(c_count), c_devices)
    _nvmlCheckReturn(ret)
    return list(c_devices[0:c_count.value])

def nvmlDeviceGetTopologyNearestGpus(device, level):
    if False:
        i = 10
        return i + 15
    c_count = c_uint(0)
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetTopologyNearestGpus')
    ret = fn(device, level, byref(c_count), None)
    if ret != NVML_SUCCESS:
        raise NVMLError(ret)
    device_array = c_nvmlDevice_t * c_count.value
    c_devices = device_array()
    ret = fn(device, level, byref(c_count), c_devices)
    _nvmlCheckReturn(ret)
    return list(c_devices[0:c_count.value])

def nvmlDeviceGetTopologyCommonAncestor(device1, device2):
    if False:
        for i in range(10):
            print('nop')
    c_level = _nvmlGpuTopologyLevel_t()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetTopologyCommonAncestor')
    ret = fn(device1, device2, byref(c_level))
    _nvmlCheckReturn(ret)
    return c_level.value

def nvmlDeviceGetNvLinkUtilizationCounter(device, link, counter):
    if False:
        for i in range(10):
            print('nop')
    c_rxcounter = c_ulonglong()
    c_txcounter = c_ulonglong()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetNvLinkUtilizationCounter')
    ret = fn(device, link, counter, byref(c_rxcounter), byref(c_txcounter))
    _nvmlCheckReturn(ret)
    return (c_rxcounter.value, c_txcounter.value)

def nvmlDeviceFreezeNvLinkUtilizationCounter(device, link, counter, freeze):
    if False:
        print('Hello World!')
    fn = _nvmlGetFunctionPointer('nvmlDeviceFreezeNvLinkUtilizationCounter')
    ret = fn(device, link, counter, freeze)
    _nvmlCheckReturn(ret)
    return None

def nvmlDeviceResetNvLinkUtilizationCounter(device, link, counter):
    if False:
        print('Hello World!')
    fn = _nvmlGetFunctionPointer('nvmlDeviceResetNvLinkUtilizationCounter')
    ret = fn(device, link, counter)
    _nvmlCheckReturn(ret)
    return None

def nvmlDeviceSetNvLinkUtilizationControl(device, link, counter, control, reset):
    if False:
        return 10
    fn = _nvmlGetFunctionPointer('nvmlDeviceSetNvLinkUtilizationControl')
    ret = fn(device, link, counter, byref(control), reset)
    _nvmlCheckReturn(ret)
    return None

def nvmlDeviceGetNvLinkUtilizationControl(device, link, counter):
    if False:
        while True:
            i = 10
    c_control = nvmlNvLinkUtilizationControl_t()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetNvLinkUtilizationControl')
    ret = fn(device, link, counter, byref(c_control))
    _nvmlCheckReturn(ret)
    return c_control

def nvmlDeviceGetNvLinkCapability(device, link, capability):
    if False:
        while True:
            i = 10
    c_capResult = c_uint()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetNvLinkCapability')
    ret = fn(device, link, capability, byref(c_capResult))
    _nvmlCheckReturn(ret)
    return c_capResult.value

def nvmlDeviceGetNvLinkErrorCounter(device, link, counter):
    if False:
        for i in range(10):
            print('nop')
    c_result = c_ulonglong()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetNvLinkErrorCounter')
    ret = fn(device, link, counter, byref(c_result))
    _nvmlCheckReturn(ret)
    return c_result.value

def nvmlDeviceResetNvLinkErrorCounters(device, link):
    if False:
        while True:
            i = 10
    fn = _nvmlGetFunctionPointer('nvmlDeviceResetNvLinkErrorCounters')
    ret = fn(device, link)
    _nvmlCheckReturn(ret)
    return None

def nvmlDeviceGetNvLinkRemotePciInfo(device, link):
    if False:
        for i in range(10):
            print('nop')
    c_pci = nvmlPciInfo_t()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetNvLinkRemotePciInfo_v2')
    ret = fn(device, link, byref(c_pci))
    _nvmlCheckReturn(ret)
    return c_pci

def nvmlDeviceGetNvLinkRemoteDeviceType(handle, link):
    if False:
        for i in range(10):
            print('nop')
    c_type = _nvmlNvLinkDeviceType_t()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetNvLinkRemoteDeviceType')
    ret = fn(handle, link, byref(c_type))
    _nvmlCheckReturn(ret)
    return c_type.value

def nvmlDeviceGetNvLinkState(device, link):
    if False:
        return 10
    c_isActive = c_uint()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetNvLinkState')
    ret = fn(device, link, byref(c_isActive))
    _nvmlCheckReturn(ret)
    return c_isActive.value

def nvmlDeviceGetNvLinkVersion(device, link):
    if False:
        while True:
            i = 10
    c_version = c_uint()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetNvLinkVersion')
    ret = fn(device, link, byref(c_version))
    _nvmlCheckReturn(ret)
    return c_version.value

def nvmlDeviceModifyDrainState(pciInfo, newState):
    if False:
        return 10
    fn = _nvmlGetFunctionPointer('nvmlDeviceModifyDrainState')
    ret = fn(pointer(pciInfo), newState)
    _nvmlCheckReturn(ret)
    return None

def nvmlDeviceQueryDrainState(pciInfo):
    if False:
        for i in range(10):
            print('nop')
    c_newState = c_uint()
    fn = _nvmlGetFunctionPointer('nvmlDeviceQueryDrainState')
    ret = fn(pointer(pciInfo), byref(c_newState))
    _nvmlCheckReturn(ret)
    return c_newState.value

def nvmlDeviceRemoveGpu(pciInfo):
    if False:
        print('Hello World!')
    fn = _nvmlGetFunctionPointer('nvmlDeviceRemoveGpu')
    ret = fn(pointer(pciInfo))
    _nvmlCheckReturn(ret)
    return None

def nvmlDeviceDiscoverGpus(pciInfo):
    if False:
        i = 10
        return i + 15
    fn = _nvmlGetFunctionPointer('nvmlDeviceDiscoverGpus')
    ret = fn(pointer(pciInfo))
    _nvmlCheckReturn(ret)
    return None

def nvmlDeviceGetFieldValues(handle, fieldIds):
    if False:
        i = 10
        return i + 15
    values_arr = c_nvmlFieldValue_t * len(fieldIds)
    values = values_arr()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetFieldValues')
    for (i, fieldId) in enumerate(fieldIds):
        try:
            (values[i].fieldId, values[i].scopeId) = fieldId
        except TypeError:
            values[i].fieldId = fieldId
    ret = fn(handle, c_int32(len(fieldIds)), byref(values))
    _nvmlCheckReturn(ret)
    return values

def nvmlDeviceClearFieldValues(handle, fieldIds):
    if False:
        i = 10
        return i + 15
    values_arr = c_nvmlFieldValue_t * len(fieldIds)
    values = values_arr()
    fn = _nvmlGetFunctionPointer('nvmlDeviceClearFieldValues')
    for (i, fieldId) in enumerate(fieldIds):
        try:
            (values[i].fieldId, values[i].scopeId) = fieldId
        except TypeError:
            values[i].fieldId = fieldId
    ret = fn(handle, c_int32(len(fieldIds)), byref(values))
    _nvmlCheckReturn(ret)
    return values

def nvmlDeviceGetVirtualizationMode(handle):
    if False:
        return 10
    c_virtualization_mode = c_ulonglong()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetVirtualizationMode')
    ret = fn(handle, byref(c_virtualization_mode))
    _nvmlCheckReturn(ret)
    return c_virtualization_mode.value

def nvmlDeviceSetVirtualizationMode(handle, virtualization_mode):
    if False:
        while True:
            i = 10
    fn = _nvmlGetFunctionPointer('nvmlDeviceSetVirtualizationMode')
    return fn(handle, virtualization_mode)

def nvmlGetVgpuDriverCapabilities(capability):
    if False:
        while True:
            i = 10
    c_capResult = c_uint()
    fn = _nvmlGetFunctionPointer('nvmlGetVgpuDriverCapabilities')
    ret = fn(_nvmlVgpuDriverCapability_t(capability), byref(c_capResult))
    _nvmlCheckReturn(ret)
    return c_capResult.value

def nvmlDeviceGetVgpuCapabilities(handle, capability):
    if False:
        print('Hello World!')
    c_capResult = c_uint()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetVgpuCapabilities')
    ret = fn(handle, _nvmlDeviceVgpuCapability_t(capability), byref(c_capResult))
    _nvmlCheckReturn(ret)
    return c_capResult.value

def nvmlDeviceGetSupportedVgpus(handle):
    if False:
        while True:
            i = 10
    c_vgpu_count = c_uint(0)
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetSupportedVgpus')
    ret = fn(handle, byref(c_vgpu_count), None)
    if ret == NVML_SUCCESS:
        return []
    elif ret == NVML_ERROR_INSUFFICIENT_SIZE:
        vgpu_type_ids_array = _nvmlVgpuTypeId_t * c_vgpu_count.value
        c_vgpu_type_ids = vgpu_type_ids_array()
        ret = fn(handle, byref(c_vgpu_count), c_vgpu_type_ids)
        _nvmlCheckReturn(ret)
        vgpus = []
        for i in range(c_vgpu_count.value):
            vgpus.append(c_vgpu_type_ids[i])
        return vgpus
    else:
        raise NVMLError(ret)

def nvmlDeviceGetCreatableVgpus(handle):
    if False:
        print('Hello World!')
    c_vgpu_count = c_uint(0)
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetCreatableVgpus')
    ret = fn(handle, byref(c_vgpu_count), None)
    if ret == NVML_SUCCESS:
        return []
    elif ret == NVML_ERROR_INSUFFICIENT_SIZE:
        vgpu_type_ids_array = _nvmlVgpuTypeId_t * c_vgpu_count.value
        c_vgpu_type_ids = vgpu_type_ids_array()
        ret = fn(handle, byref(c_vgpu_count), c_vgpu_type_ids)
        _nvmlCheckReturn(ret)
        vgpus = []
        for i in range(c_vgpu_count.value):
            vgpus.append(c_vgpu_type_ids[i])
        return vgpus
    else:
        raise NVMLError(ret)

def nvmlVgpuTypeGetGpuInstanceProfileId(vgpuTypeId):
    if False:
        while True:
            i = 10
    c_profile_id = c_uint(0)
    fn = _nvmlGetFunctionPointer('nvmlVgpuTypeGetGpuInstanceProfileId')
    ret = fn(vgpuTypeId, byref(c_profile_id))
    _nvmlCheckReturn(ret)
    return c_profile_id.value

@convertStrBytes
def nvmlVgpuTypeGetClass(vgpuTypeId):
    if False:
        return 10
    c_class = create_string_buffer(NVML_DEVICE_NAME_BUFFER_SIZE)
    c_buffer_size = c_uint(NVML_DEVICE_NAME_BUFFER_SIZE)
    fn = _nvmlGetFunctionPointer('nvmlVgpuTypeGetClass')
    ret = fn(vgpuTypeId, c_class, byref(c_buffer_size))
    _nvmlCheckReturn(ret)
    return c_class.value

@convertStrBytes
def nvmlVgpuTypeGetName(vgpuTypeId):
    if False:
        print('Hello World!')
    c_name = create_string_buffer(NVML_DEVICE_NAME_BUFFER_SIZE)
    c_buffer_size = c_uint(NVML_DEVICE_NAME_BUFFER_SIZE)
    fn = _nvmlGetFunctionPointer('nvmlVgpuTypeGetName')
    ret = fn(vgpuTypeId, c_name, byref(c_buffer_size))
    _nvmlCheckReturn(ret)
    return c_name.value

def nvmlVgpuTypeGetDeviceID(vgpuTypeId):
    if False:
        while True:
            i = 10
    c_device_id = c_ulonglong(0)
    c_subsystem_id = c_ulonglong(0)
    fn = _nvmlGetFunctionPointer('nvmlVgpuTypeGetDeviceID')
    ret = fn(vgpuTypeId, byref(c_device_id), byref(c_subsystem_id))
    _nvmlCheckReturn(ret)
    return (c_device_id.value, c_subsystem_id.value)

def nvmlVgpuTypeGetFramebufferSize(vgpuTypeId):
    if False:
        while True:
            i = 10
    c_fb_size = c_ulonglong(0)
    fn = _nvmlGetFunctionPointer('nvmlVgpuTypeGetFramebufferSize')
    ret = fn(vgpuTypeId, byref(c_fb_size))
    _nvmlCheckReturn(ret)
    return c_fb_size.value

def nvmlVgpuTypeGetNumDisplayHeads(vgpuTypeId):
    if False:
        return 10
    c_num_heads = c_uint(0)
    fn = _nvmlGetFunctionPointer('nvmlVgpuTypeGetNumDisplayHeads')
    ret = fn(vgpuTypeId, byref(c_num_heads))
    _nvmlCheckReturn(ret)
    return c_num_heads.value

def nvmlVgpuTypeGetResolution(vgpuTypeId):
    if False:
        for i in range(10):
            print('nop')
    c_xdim = c_uint(0)
    c_ydim = c_uint(0)
    fn = _nvmlGetFunctionPointer('nvmlVgpuTypeGetResolution')
    ret = fn(vgpuTypeId, 0, byref(c_xdim), byref(c_ydim))
    _nvmlCheckReturn(ret)
    return (c_xdim.value, c_ydim.value)

@convertStrBytes
def nvmlVgpuTypeGetLicense(vgpuTypeId):
    if False:
        print('Hello World!')
    c_license = create_string_buffer(NVML_GRID_LICENSE_BUFFER_SIZE)
    c_buffer_size = c_uint(NVML_GRID_LICENSE_BUFFER_SIZE)
    fn = _nvmlGetFunctionPointer('nvmlVgpuTypeGetLicense')
    ret = fn(vgpuTypeId, c_license, c_buffer_size)
    _nvmlCheckReturn(ret)
    return c_license.value

def nvmlVgpuTypeGetFrameRateLimit(vgpuTypeId):
    if False:
        return 10
    c_frl_config = c_uint(0)
    fn = _nvmlGetFunctionPointer('nvmlVgpuTypeGetFrameRateLimit')
    ret = fn(vgpuTypeId, byref(c_frl_config))
    _nvmlCheckReturn(ret)
    return c_frl_config.value

def nvmlVgpuTypeGetMaxInstances(handle, vgpuTypeId):
    if False:
        print('Hello World!')
    c_max_instances = c_uint(0)
    fn = _nvmlGetFunctionPointer('nvmlVgpuTypeGetMaxInstances')
    ret = fn(handle, vgpuTypeId, byref(c_max_instances))
    _nvmlCheckReturn(ret)
    return c_max_instances.value

def nvmlVgpuTypeGetMaxInstancesPerVm(vgpuTypeId):
    if False:
        print('Hello World!')
    c_max_instances_per_vm = c_uint(0)
    fn = _nvmlGetFunctionPointer('nvmlVgpuTypeGetMaxInstancesPerVm')
    ret = fn(vgpuTypeId, byref(c_max_instances_per_vm))
    _nvmlCheckReturn(ret)
    return c_max_instances_per_vm.value

def nvmlDeviceGetActiveVgpus(handle):
    if False:
        while True:
            i = 10
    c_vgpu_count = c_uint(0)
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetActiveVgpus')
    ret = fn(handle, byref(c_vgpu_count), None)
    if ret == NVML_SUCCESS:
        return []
    elif ret == NVML_ERROR_INSUFFICIENT_SIZE:
        vgpu_instance_array = _nvmlVgpuInstance_t * c_vgpu_count.value
        c_vgpu_instances = vgpu_instance_array()
        ret = fn(handle, byref(c_vgpu_count), c_vgpu_instances)
        _nvmlCheckReturn(ret)
        vgpus = []
        for i in range(c_vgpu_count.value):
            vgpus.append(c_vgpu_instances[i])
        return vgpus
    else:
        raise NVMLError(ret)

@convertStrBytes
def nvmlVgpuInstanceGetVmID(vgpuInstance):
    if False:
        while True:
            i = 10
    c_vm_id = create_string_buffer(NVML_DEVICE_UUID_BUFFER_SIZE)
    c_buffer_size = c_uint(NVML_GRID_LICENSE_BUFFER_SIZE)
    c_vm_id_type = c_uint(0)
    fn = _nvmlGetFunctionPointer('nvmlVgpuInstanceGetVmID')
    ret = fn(vgpuInstance, byref(c_vm_id), c_buffer_size, byref(c_vm_id_type))
    _nvmlCheckReturn(ret)
    return (c_vm_id.value, c_vm_id_type.value)

@convertStrBytes
def nvmlVgpuInstanceGetUUID(vgpuInstance):
    if False:
        for i in range(10):
            print('nop')
    c_uuid = create_string_buffer(NVML_DEVICE_UUID_BUFFER_SIZE)
    c_buffer_size = c_uint(NVML_DEVICE_UUID_BUFFER_SIZE)
    fn = _nvmlGetFunctionPointer('nvmlVgpuInstanceGetUUID')
    ret = fn(vgpuInstance, byref(c_uuid), c_buffer_size)
    _nvmlCheckReturn(ret)
    return c_uuid.value

@convertStrBytes
def nvmlVgpuInstanceGetMdevUUID(vgpuInstance):
    if False:
        while True:
            i = 10
    c_uuid = create_string_buffer(NVML_DEVICE_UUID_BUFFER_SIZE)
    c_buffer_size = c_uint(NVML_DEVICE_UUID_BUFFER_SIZE)
    fn = _nvmlGetFunctionPointer('nvmlVgpuInstanceGetMdevUUID')
    ret = fn(vgpuInstance, byref(c_uuid), c_buffer_size)
    _nvmlCheckReturn(ret)
    return c_uuid.value

@convertStrBytes
def nvmlVgpuInstanceGetVmDriverVersion(vgpuInstance):
    if False:
        for i in range(10):
            print('nop')
    c_driver_version = create_string_buffer(NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE)
    c_buffer_size = c_uint(NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE)
    fn = _nvmlGetFunctionPointer('nvmlVgpuInstanceGetVmDriverVersion')
    ret = fn(vgpuInstance, byref(c_driver_version), c_buffer_size)
    _nvmlCheckReturn(ret)
    return c_driver_version.value

def nvmlVgpuInstanceGetLicenseStatus(vgpuInstance):
    if False:
        print('Hello World!')
    c_license_status = c_uint(0)
    fn = _nvmlGetFunctionPointer('nvmlVgpuInstanceGetLicenseStatus')
    ret = fn(vgpuInstance, byref(c_license_status))
    _nvmlCheckReturn(ret)
    return c_license_status.value

def nvmlVgpuInstanceGetLicenseInfo_v2(vgpuInstance):
    if False:
        return 10
    fn = _nvmlGetFunctionPointer('nvmlVgpuInstanceGetLicenseInfo_v2')
    c_license_info = c_nvmlVgpuLicenseInfo_t()
    ret = fn(vgpuInstance, byref(c_license_info))
    _nvmlCheckReturn(ret)
    return c_license_info

def nvmlVgpuInstanceGetLicenseInfo(vgpuInstance):
    if False:
        return 10
    return nvmlVgpuInstanceGetLicenseInfo_v2(vgpuInstance)

def nvmlVgpuInstanceGetFrameRateLimit(vgpuInstance):
    if False:
        print('Hello World!')
    c_frl = c_uint(0)
    fn = _nvmlGetFunctionPointer('nvmlVgpuInstanceGetFrameRateLimit')
    ret = fn(vgpuInstance, byref(c_frl))
    _nvmlCheckReturn(ret)
    return c_frl.value

def nvmlVgpuInstanceGetEccMode(vgpuInstance):
    if False:
        while True:
            i = 10
    c_mode = _nvmlEnableState_t()
    fn = _nvmlGetFunctionPointer('nvmlVgpuInstanceGetEccMode')
    ret = fn(vgpuInstance, byref(c_mode))
    _nvmlCheckReturn(ret)
    return c_mode.value

def nvmlVgpuInstanceGetType(vgpuInstance):
    if False:
        i = 10
        return i + 15
    c_vgpu_type = c_uint(0)
    fn = _nvmlGetFunctionPointer('nvmlVgpuInstanceGetType')
    ret = fn(vgpuInstance, byref(c_vgpu_type))
    _nvmlCheckReturn(ret)
    return c_vgpu_type.value

def nvmlVgpuInstanceGetEncoderCapacity(vgpuInstance):
    if False:
        for i in range(10):
            print('nop')
    c_encoder_capacity = c_ulonglong(0)
    fn = _nvmlGetFunctionPointer('nvmlVgpuInstanceGetEncoderCapacity')
    ret = fn(vgpuInstance, byref(c_encoder_capacity))
    _nvmlCheckReturn(ret)
    return c_encoder_capacity.value

def nvmlVgpuInstanceSetEncoderCapacity(vgpuInstance, encoder_capacity):
    if False:
        while True:
            i = 10
    fn = _nvmlGetFunctionPointer('nvmlVgpuInstanceSetEncoderCapacity')
    return fn(vgpuInstance, encoder_capacity)

def nvmlVgpuInstanceGetFbUsage(vgpuInstance):
    if False:
        i = 10
        return i + 15
    c_fb_usage = c_uint(0)
    fn = _nvmlGetFunctionPointer('nvmlVgpuInstanceGetFbUsage')
    ret = fn(vgpuInstance, byref(c_fb_usage))
    _nvmlCheckReturn(ret)
    return c_fb_usage.value

def nvmlVgpuTypeGetCapabilities(vgpuTypeId, capability):
    if False:
        print('Hello World!')
    c_cap_result = c_uint(0)
    fn = _nvmlGetFunctionPointer('nvmlVgpuTypeGetCapabilities')
    ret = fn(vgpuTypeId, _nvmlVgpuCapability_t(capability), byref(c_cap_result))
    _nvmlCheckReturn(ret)
    return c_cap_result.value

def nvmlVgpuInstanceGetGpuInstanceId(vgpuInstance):
    if False:
        i = 10
        return i + 15
    c_id = c_uint(0)
    fn = _nvmlGetFunctionPointer('nvmlVgpuInstanceGetGpuInstanceId')
    ret = fn(vgpuInstance, byref(c_id))
    _nvmlCheckReturn(ret)
    return c_id.value

@convertStrBytes
def nvmlVgpuInstanceGetGpuPciId(vgpuInstance):
    if False:
        while True:
            i = 10
    c_vgpuPciId = create_string_buffer(NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE)
    fn = _nvmlGetFunctionPointer('nvmlVgpuInstanceGetGpuPciId')
    ret = fn(vgpuInstance, c_vgpuPciId, byref(c_uint(NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE)))
    _nvmlCheckReturn(ret)
    return c_vgpuPciId.value

def nvmlDeviceGetVgpuUtilization(handle, timeStamp):
    if False:
        for i in range(10):
            print('nop')
    c_vgpu_count = c_uint(0)
    c_time_stamp = c_ulonglong(timeStamp)
    c_sample_value_type = _nvmlValueType_t()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetVgpuUtilization')
    ret = fn(handle, c_time_stamp, byref(c_sample_value_type), byref(c_vgpu_count), None)
    if ret == NVML_SUCCESS:
        return []
    elif ret == NVML_ERROR_INSUFFICIENT_SIZE:
        sampleArray = c_vgpu_count.value * c_nvmlVgpuInstanceUtilizationSample_t
        c_samples = sampleArray()
        ret = fn(handle, c_time_stamp, byref(c_sample_value_type), byref(c_vgpu_count), c_samples)
        _nvmlCheckReturn(ret)
        return c_samples[0:c_vgpu_count.value]
    else:
        raise NVMLError(ret)

def nvmlDeviceGetP2PStatus(device1, device2, p2pIndex):
    if False:
        return 10
    c_p2pstatus = _nvmlGpuP2PStatus_t()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetP2PStatus')
    ret = fn(device1, device2, p2pIndex, byref(c_p2pstatus))
    _nvmlCheckReturn(ret)
    return c_p2pstatus.value

def nvmlDeviceGetGridLicensableFeatures_v4(handle):
    if False:
        return 10
    c_get_grid_licensable_features = c_nvmlGridLicensableFeatures_v4_t()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetGridLicensableFeatures_v4')
    ret = fn(handle, byref(c_get_grid_licensable_features))
    _nvmlCheckReturn(ret)
    return c_get_grid_licensable_features

def nvmlDeviceGetGridLicensableFeatures(handle):
    if False:
        while True:
            i = 10
    return nvmlDeviceGetGridLicensableFeatures_v4(handle)

def nvmlDeviceGetGspFirmwareVersion(handle, version):
    if False:
        return 10
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetGspFirmwareVersion')
    ret = fn(handle, version)
    _nvmlCheckReturn(ret)
    return ret

def nvmlDeviceGetGspFirmwareMode(handle, isEnabled, defaultMode):
    if False:
        for i in range(10):
            print('nop')
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetGspFirmwareMode')
    ret = fn(handle, isEnabled, defaultMode)
    _nvmlCheckReturn(ret)
    return ret

def nvmlDeviceGetEncoderCapacity(handle, encoderQueryType):
    if False:
        i = 10
        return i + 15
    c_encoder_capacity = c_ulonglong(0)
    c_encoderQuery_type = _nvmlEncoderQueryType_t(encoderQueryType)
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetEncoderCapacity')
    ret = fn(handle, c_encoderQuery_type, byref(c_encoder_capacity))
    _nvmlCheckReturn(ret)
    return c_encoder_capacity.value

def nvmlDeviceGetVgpuProcessUtilization(handle, timeStamp):
    if False:
        return 10
    c_vgpu_count = c_uint(0)
    c_time_stamp = c_ulonglong(timeStamp)
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetVgpuProcessUtilization')
    ret = fn(handle, c_time_stamp, byref(c_vgpu_count), None)
    if ret == NVML_SUCCESS:
        return []
    elif ret == NVML_ERROR_INSUFFICIENT_SIZE:
        sampleArray = c_vgpu_count.value * c_nvmlVgpuProcessUtilizationSample_t
        c_samples = sampleArray()
        ret = fn(handle, c_time_stamp, byref(c_vgpu_count), c_samples)
        _nvmlCheckReturn(ret)
        return c_samples[0:c_vgpu_count.value]
    else:
        raise NVMLError(ret)

def nvmlDeviceGetEncoderStats(handle):
    if False:
        while True:
            i = 10
    c_encoderCount = c_ulonglong(0)
    c_encodeFps = c_ulonglong(0)
    c_encoderLatency = c_ulonglong(0)
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetEncoderStats')
    ret = fn(handle, byref(c_encoderCount), byref(c_encodeFps), byref(c_encoderLatency))
    _nvmlCheckReturn(ret)
    return (c_encoderCount.value, c_encodeFps.value, c_encoderLatency.value)

def nvmlDeviceGetEncoderSessions(handle):
    if False:
        while True:
            i = 10
    c_session_count = c_uint(0)
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetEncoderSessions')
    ret = fn(handle, byref(c_session_count), None)
    if ret == NVML_SUCCESS:
        if c_session_count.value != 0:
            session_array = c_nvmlEncoderSession_t * c_session_count.value
            c_sessions = session_array()
            ret = fn(handle, byref(c_session_count), c_sessions)
            _nvmlCheckReturn(ret)
            sessions = []
            for i in range(c_session_count.value):
                sessions.append(c_sessions[i])
            return sessions
        else:
            return []
    else:
        raise NVMLError(ret)

def nvmlDeviceGetFBCStats(handle):
    if False:
        for i in range(10):
            print('nop')
    c_fbcStats = c_nvmlFBCStats_t()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetFBCStats')
    ret = fn(handle, byref(c_fbcStats))
    _nvmlCheckReturn(ret)
    return c_fbcStats

def nvmlDeviceGetFBCSessions(handle):
    if False:
        i = 10
        return i + 15
    c_session_count = c_uint(0)
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetFBCSessions')
    ret = fn(handle, byref(c_session_count), None)
    if ret == NVML_SUCCESS:
        if c_session_count.value != 0:
            session_array = c_nvmlFBCSession_t * c_session_count.value
            c_sessions = session_array()
            ret = fn(handle, byref(c_session_count), c_sessions)
            _nvmlCheckReturn(ret)
            sessions = []
            for i in range(c_session_count.value):
                sessions.append(c_sessions[i])
            return sessions
        else:
            return []
    else:
        raise NVMLError(ret)

def nvmlVgpuInstanceGetEncoderStats(vgpuInstance):
    if False:
        i = 10
        return i + 15
    c_encoderCount = c_ulonglong(0)
    c_encodeFps = c_ulonglong(0)
    c_encoderLatency = c_ulonglong(0)
    fn = _nvmlGetFunctionPointer('nvmlVgpuInstanceGetEncoderStats')
    ret = fn(vgpuInstance, byref(c_encoderCount), byref(c_encodeFps), byref(c_encoderLatency))
    _nvmlCheckReturn(ret)
    return (c_encoderCount.value, c_encodeFps.value, c_encoderLatency.value)

def nvmlVgpuInstanceGetEncoderSessions(vgpuInstance):
    if False:
        while True:
            i = 10
    c_session_count = c_uint(0)
    fn = _nvmlGetFunctionPointer('nvmlVgpuInstanceGetEncoderSessions')
    ret = fn(vgpuInstance, byref(c_session_count), None)
    if ret == NVML_SUCCESS:
        if c_session_count.value != 0:
            session_array = c_nvmlEncoderSession_t * c_session_count.value
            c_sessions = session_array()
            ret = fn(vgpuInstance, byref(c_session_count), c_sessions)
            _nvmlCheckReturn(ret)
            sessions = []
            for i in range(c_session_count.value):
                sessions.append(c_sessions[i])
            return sessions
        else:
            return []
    else:
        raise NVMLError(ret)

def nvmlVgpuInstanceGetFBCStats(vgpuInstance):
    if False:
        return 10
    c_fbcStats = c_nvmlFBCStats_t()
    fn = _nvmlGetFunctionPointer('nvmlVgpuInstanceGetFBCStats')
    ret = fn(vgpuInstance, byref(c_fbcStats))
    _nvmlCheckReturn(ret)
    return c_fbcStats

def nvmlVgpuInstanceGetFBCSessions(vgpuInstance):
    if False:
        print('Hello World!')
    c_session_count = c_uint(0)
    fn = _nvmlGetFunctionPointer('nvmlVgpuInstanceGetFBCSessions')
    ret = fn(vgpuInstance, byref(c_session_count), None)
    if ret == NVML_SUCCESS:
        if c_session_count.value != 0:
            session_array = c_nvmlFBCSession_t * c_session_count.value
            c_sessions = session_array()
            ret = fn(vgpuInstance, byref(c_session_count), c_sessions)
            _nvmlCheckReturn(ret)
            sessions = []
            for i in range(c_session_count.value):
                sessions.append(c_sessions[i])
            return sessions
        else:
            return []
    else:
        raise NVMLError(ret)

def nvmlDeviceGetProcessUtilization(handle, timeStamp):
    if False:
        return 10
    c_count = c_uint(0)
    c_time_stamp = c_ulonglong(timeStamp)
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetProcessUtilization')
    ret = fn(handle, None, byref(c_count), c_time_stamp)
    if ret == NVML_ERROR_INSUFFICIENT_SIZE:
        sampleArray = c_count.value * c_nvmlProcessUtilizationSample_t
        c_samples = sampleArray()
        ret = fn(handle, c_samples, byref(c_count), c_time_stamp)
        _nvmlCheckReturn(ret)
        return c_samples[0:c_count.value]
    else:
        raise NVMLError(ret)

def nvmlVgpuInstanceGetMetadata(vgpuInstance):
    if False:
        while True:
            i = 10
    fn = _nvmlGetFunctionPointer('nvmlVgpuInstanceGetMetadata')
    c_vgpuMetadata = c_nvmlVgpuMetadata_t()
    c_bufferSize = c_uint(0)
    ret = fn(vgpuInstance, byref(c_vgpuMetadata), byref(c_bufferSize))
    if ret == NVML_ERROR_INSUFFICIENT_SIZE:
        ret = fn(vgpuInstance, byref(c_vgpuMetadata), byref(c_bufferSize))
        _nvmlCheckReturn(ret)
    else:
        raise NVMLError(ret)
    return c_vgpuMetadata

def nvmlDeviceGetVgpuMetadata(handle):
    if False:
        print('Hello World!')
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetVgpuMetadata')
    c_vgpuPgpuMetadata = c_nvmlVgpuPgpuMetadata_t()
    c_bufferSize = c_uint(0)
    ret = fn(handle, byref(c_vgpuPgpuMetadata), byref(c_bufferSize))
    if ret == NVML_ERROR_INSUFFICIENT_SIZE:
        ret = fn(handle, byref(c_vgpuPgpuMetadata), byref(c_bufferSize))
        _nvmlCheckReturn(ret)
    else:
        raise NVMLError(ret)
    return c_vgpuPgpuMetadata

def nvmlGetVgpuCompatibility(vgpuMetadata, pgpuMetadata):
    if False:
        return 10
    fn = _nvmlGetFunctionPointer('nvmlGetVgpuCompatibility')
    c_vgpuPgpuCompatibility = c_nvmlVgpuPgpuCompatibility_t()
    ret = fn(byref(vgpuMetadata), byref(pgpuMetadata), byref(c_vgpuPgpuCompatibility))
    _nvmlCheckReturn(ret)
    return c_vgpuPgpuCompatibility

@convertStrBytes
def nvmlDeviceGetPgpuMetadataString(handle):
    if False:
        print('Hello World!')
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetPgpuMetadataString')
    c_pgpuMetadata = create_string_buffer(NVML_VGPU_PGPU_METADATA_OPAQUE_DATA_SIZE)
    c_bufferSize = c_uint(0)
    ret = fn(handle, byref(c_pgpuMetadata), byref(c_bufferSize))
    if ret == NVML_ERROR_INSUFFICIENT_SIZE:
        ret = fn(handle, byref(c_pgpuMetadata), byref(c_bufferSize))
        _nvmlCheckReturn(ret)
    else:
        raise NVMLError(ret)
    return (c_pgpuMetadata.value, c_bufferSize.value)

def nvmlDeviceGetVgpuSchedulerLog(handle):
    if False:
        return 10
    c_vgpu_sched_log = c_nvmlVgpuSchedulerLog_t()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetVgpuSchedulerLog')
    ret = fn(handle, byref(c_vgpu_sched_log))
    _nvmlCheckReturn(ret)
    return c_vgpu_sched_log

def nvmlDeviceGetVgpuSchedulerState(handle):
    if False:
        i = 10
        return i + 15
    c_vgpu_sched_state = c_nvmlVgpuSchedulerGetState_t()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetVgpuSchedulerState')
    ret = fn(handle, byref(c_vgpu_sched_state))
    _nvmlCheckReturn(ret)
    return c_vgpu_sched_state

def nvmlDeviceGetVgpuSchedulerCapabilities(handle):
    if False:
        return 10
    c_vgpu_sched_caps = c_nvmlVgpuSchedulerCapabilities_t()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetVgpuSchedulerCapabilities')
    ret = fn(handle, byref(c_vgpu_sched_caps))
    _nvmlCheckReturn(ret)
    return c_vgpu_sched_caps

def nvmlDeviceSetVgpuSchedulerState(handle, sched_state):
    if False:
        while True:
            i = 10
    fn = _nvmlGetFunctionPointer('nvmlDeviceSetVgpuSchedulerState')
    ret = fn(handle, byref(sched_state))
    _nvmlCheckReturn(ret)
    return ret

def nvmlSetVgpuVersion(vgpuVersion):
    if False:
        print('Hello World!')
    fn = _nvmlGetFunctionPointer('nvmlSetVgpuVersion')
    ret = fn(byref(vgpuVersion))
    _nvmlCheckReturn(ret)
    return ret

def nvmlGetVgpuVersion(supported, current):
    if False:
        print('Hello World!')
    fn = _nvmlGetFunctionPointer('nvmlGetVgpuVersion')
    ret = fn(byref(supported), byref(current))
    _nvmlCheckReturn(ret)
    return ret

def nvmlVgpuInstanceGetAccountingMode(vgpuInstance):
    if False:
        while True:
            i = 10
    c_mode = _nvmlEnableState_t()
    fn = _nvmlGetFunctionPointer('nvmlVgpuInstanceGetAccountingMode')
    ret = fn(vgpuInstance, byref(c_mode))
    _nvmlCheckReturn(ret)
    return c_mode.value

def nvmlVgpuInstanceGetAccountingPids(vgpuInstance):
    if False:
        print('Hello World!')
    c_pidCount = c_uint()
    fn = _nvmlGetFunctionPointer('nvmlVgpuInstanceGetAccountingPids')
    ret = fn(vgpuInstance, byref(c_pidCount), None)
    if ret == NVML_ERROR_INSUFFICIENT_SIZE:
        sampleArray = c_pidCount.value * c_uint
        c_pidArray = sampleArray()
        ret = fn(vgpuInstance, byref(c_pidCount), byref(c_pidArray))
        _nvmlCheckReturn(ret)
    else:
        raise NVMLError(ret)
    return (c_pidCount, c_pidArray)

def nvmlVgpuInstanceGetAccountingStats(vgpuInstance, pid):
    if False:
        return 10
    c_accountingStats = c_nvmlAccountingStats_t()
    fn = _nvmlGetFunctionPointer('nvmlVgpuInstanceGetAccountingStats')
    ret = fn(vgpuInstance, pid, byref(c_accountingStats))
    _nvmlCheckReturn(ret)
    return c_accountingStats

def nvmlVgpuInstanceClearAccountingPids(vgpuInstance):
    if False:
        for i in range(10):
            print('nop')
    fn = _nvmlGetFunctionPointer('nvmlVgpuInstanceClearAccountingPids')
    ret = fn(vgpuInstance)
    _nvmlCheckReturn(ret)
    return ret

def nvmlGetExcludedDeviceCount():
    if False:
        i = 10
        return i + 15
    c_count = c_uint()
    fn = _nvmlGetFunctionPointer('nvmlGetExcludedDeviceCount')
    ret = fn(byref(c_count))
    _nvmlCheckReturn(ret)
    return c_count.value

def nvmlGetExcludedDeviceInfoByIndex(index):
    if False:
        while True:
            i = 10
    c_index = c_uint(index)
    info = c_nvmlExcludedDeviceInfo_t()
    fn = _nvmlGetFunctionPointer('nvmlGetExcludedDeviceInfoByIndex')
    ret = fn(c_index, byref(info))
    _nvmlCheckReturn(ret)
    return info

def nvmlDeviceGetHostVgpuMode(handle):
    if False:
        print('Hello World!')
    c_host_vgpu_mode = _nvmlHostVgpuMode_t()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetHostVgpuMode')
    ret = fn(handle, byref(c_host_vgpu_mode))
    _nvmlCheckReturn(ret)
    return c_host_vgpu_mode.value

def nvmlDeviceSetMigMode(device, mode):
    if False:
        return 10
    c_activationStatus = c_uint()
    fn = _nvmlGetFunctionPointer('nvmlDeviceSetMigMode')
    ret = fn(device, mode, byref(c_activationStatus))
    _nvmlCheckReturn(ret)
    return c_activationStatus.value

def nvmlDeviceGetMigMode(device):
    if False:
        for i in range(10):
            print('nop')
    c_currentMode = c_uint()
    c_pendingMode = c_uint()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetMigMode')
    ret = fn(device, byref(c_currentMode), byref(c_pendingMode))
    _nvmlCheckReturn(ret)
    return [c_currentMode.value, c_pendingMode.value]

def nvmlDeviceGetGpuInstanceProfileInfo(device, profile, version=2):
    if False:
        print('Hello World!')
    if version == 2:
        c_info = c_nvmlGpuInstanceProfileInfo_v2_t()
        fn = _nvmlGetFunctionPointer('nvmlDeviceGetGpuInstanceProfileInfoV')
    elif version == 1:
        c_info = c_nvmlGpuInstanceProfileInfo_t()
        fn = _nvmlGetFunctionPointer('nvmlDeviceGetGpuInstanceProfileInfo')
    else:
        raise NVMLError(NVML_ERROR_FUNCTION_NOT_FOUND)
    ret = fn(device, profile, byref(c_info))
    _nvmlCheckReturn(ret)
    return c_info
nvmlDeviceGetGpuInstanceProfileInfoV = nvmlDeviceGetGpuInstanceProfileInfo

def nvmlDeviceGetGpuInstanceRemainingCapacity(device, profileId):
    if False:
        for i in range(10):
            print('nop')
    c_count = c_uint()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetGpuInstanceRemainingCapacity')
    ret = fn(device, profileId, byref(c_count))
    _nvmlCheckReturn(ret)
    return c_count.value

def nvmlDeviceGetGpuInstancePossiblePlacements(device, profileId, placementsRef, countRef):
    if False:
        while True:
            i = 10
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetGpuInstancePossiblePlacements_v2')
    ret = fn(device, profileId, placementsRef, countRef)
    _nvmlCheckReturn(ret)
    return ret

def nvmlDeviceCreateGpuInstance(device, profileId):
    if False:
        return 10
    c_instance = c_nvmlGpuInstance_t()
    fn = _nvmlGetFunctionPointer('nvmlDeviceCreateGpuInstance')
    ret = fn(device, profileId, byref(c_instance))
    _nvmlCheckReturn(ret)
    return c_instance

def nvmlDeviceCreateGpuInstanceWithPlacement(device, profileId, placement):
    if False:
        while True:
            i = 10
    c_instance = c_nvmlGpuInstance_t()
    fn = _nvmlGetFunctionPointer('nvmlDeviceCreateGpuInstanceWithPlacement')
    ret = fn(device, profileId, placement, byref(c_instance))
    _nvmlCheckReturn(ret)
    return c_instance

def nvmlGpuInstanceDestroy(gpuInstance):
    if False:
        for i in range(10):
            print('nop')
    fn = _nvmlGetFunctionPointer('nvmlGpuInstanceDestroy')
    ret = fn(gpuInstance)
    _nvmlCheckReturn(ret)
    return ret

def nvmlDeviceGetGpuInstances(device, profileId, gpuInstancesRef, countRef):
    if False:
        while True:
            i = 10
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetGpuInstances')
    ret = fn(device, profileId, gpuInstancesRef, countRef)
    _nvmlCheckReturn(ret)
    return ret

def nvmlDeviceGetGpuInstanceById(device, gpuInstanceId):
    if False:
        return 10
    c_instance = c_nvmlGpuInstance_t()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetGpuInstanceById')
    ret = fn(device, gpuInstanceId, byref(c_instance))
    _nvmlCheckReturn(ret)
    return c_instance

def nvmlGpuInstanceGetInfo(gpuInstance):
    if False:
        print('Hello World!')
    c_info = c_nvmlGpuInstanceInfo_t()
    fn = _nvmlGetFunctionPointer('nvmlGpuInstanceGetInfo')
    ret = fn(gpuInstance, byref(c_info))
    _nvmlCheckReturn(ret)
    return c_info

def nvmlGpuInstanceGetComputeInstanceProfileInfo(device, profile, engProfile, version=2):
    if False:
        print('Hello World!')
    if version == 2:
        c_info = c_nvmlComputeInstanceProfileInfo_v2_t()
        fn = _nvmlGetFunctionPointer('nvmlGpuInstanceGetComputeInstanceProfileInfoV')
    elif version == 1:
        c_info = c_nvmlComputeInstanceProfileInfo_t()
        fn = _nvmlGetFunctionPointer('nvmlGpuInstanceGetComputeInstanceProfileInfo')
    else:
        raise NVMLError(NVML_ERROR_FUNCTION_NOT_FOUND)
    ret = fn(device, profile, engProfile, byref(c_info))
    _nvmlCheckReturn(ret)
    return c_info
nvmlGpuInstanceGetComputeInstanceProfileInfoV = nvmlGpuInstanceGetComputeInstanceProfileInfo

def nvmlGpuInstanceGetComputeInstanceRemainingCapacity(gpuInstance, profileId):
    if False:
        i = 10
        return i + 15
    c_count = c_uint()
    fn = _nvmlGetFunctionPointer('nvmlGpuInstanceGetComputeInstanceRemainingCapacity')
    ret = fn(gpuInstance, profileId, byref(c_count))
    _nvmlCheckReturn(ret)
    return c_count.value

def nvmlGpuInstanceGetComputeInstancePossiblePlacements(gpuInstance, profileId, placementsRef, countRef):
    if False:
        for i in range(10):
            print('nop')
    fn = _nvmlGetFunctionPointer('nvmlGpuInstanceGetComputeInstancePossiblePlacements')
    ret = fn(gpuInstance, profileId, placementsRef, countRef)
    _nvmlCheckReturn(ret)
    return ret

def nvmlGpuInstanceCreateComputeInstance(gpuInstance, profileId):
    if False:
        for i in range(10):
            print('nop')
    c_instance = c_nvmlComputeInstance_t()
    fn = _nvmlGetFunctionPointer('nvmlGpuInstanceCreateComputeInstance')
    ret = fn(gpuInstance, profileId, byref(c_instance))
    _nvmlCheckReturn(ret)
    return c_instance

def nvmlGpuInstanceCreateComputeInstanceWithPlacement(gpuInstance, profileId, placement):
    if False:
        i = 10
        return i + 15
    c_instance = c_nvmlComputeInstance_t()
    fn = _nvmlGetFunctionPointer('nvmlGpuInstanceCreateComputeInstanceWithPlacement')
    ret = fn(gpuInstance, profileId, placement, byref(c_instance))
    _nvmlCheckReturn(ret)
    return c_instance

def nvmlComputeInstanceDestroy(computeInstance):
    if False:
        for i in range(10):
            print('nop')
    fn = _nvmlGetFunctionPointer('nvmlComputeInstanceDestroy')
    ret = fn(computeInstance)
    _nvmlCheckReturn(ret)
    return ret

def nvmlGpuInstanceGetComputeInstances(gpuInstance, profileId, computeInstancesRef, countRef):
    if False:
        for i in range(10):
            print('nop')
    fn = _nvmlGetFunctionPointer('nvmlGpuInstanceGetComputeInstances')
    ret = fn(gpuInstance, profileId, computeInstancesRef, countRef)
    _nvmlCheckReturn(ret)
    return ret

def nvmlGpuInstanceGetComputeInstanceById(gpuInstance, computeInstanceId):
    if False:
        return 10
    c_instance = c_nvmlComputeInstance_t()
    fn = _nvmlGetFunctionPointer('nvmlGpuInstanceGetComputeInstanceById')
    ret = fn(gpuInstance, computeInstanceId, byref(c_instance))
    _nvmlCheckReturn(ret)
    return c_instance

def nvmlComputeInstanceGetInfo_v2(computeInstance):
    if False:
        for i in range(10):
            print('nop')
    c_info = c_nvmlComputeInstanceInfo_t()
    fn = _nvmlGetFunctionPointer('nvmlComputeInstanceGetInfo_v2')
    ret = fn(computeInstance, byref(c_info))
    _nvmlCheckReturn(ret)
    return c_info

def nvmlComputeInstanceGetInfo(computeInstance):
    if False:
        return 10
    return nvmlComputeInstanceGetInfo_v2(computeInstance)

def nvmlDeviceIsMigDeviceHandle(device):
    if False:
        print('Hello World!')
    c_isMigDevice = c_uint()
    fn = _nvmlGetFunctionPointer('nvmlDeviceIsMigDeviceHandle')
    ret = fn(device, byref(c_isMigDevice))
    _nvmlCheckReturn(ret)
    return c_isMigDevice

def nvmlDeviceGetGpuInstanceId(device):
    if False:
        print('Hello World!')
    c_gpuInstanceId = c_uint()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetGpuInstanceId')
    ret = fn(device, byref(c_gpuInstanceId))
    _nvmlCheckReturn(ret)
    return c_gpuInstanceId.value

def nvmlDeviceGetComputeInstanceId(device):
    if False:
        for i in range(10):
            print('nop')
    c_computeInstanceId = c_uint()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetComputeInstanceId')
    ret = fn(device, byref(c_computeInstanceId))
    _nvmlCheckReturn(ret)
    return c_computeInstanceId.value

def nvmlDeviceGetMaxMigDeviceCount(device):
    if False:
        return 10
    c_count = c_uint()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetMaxMigDeviceCount')
    ret = fn(device, byref(c_count))
    _nvmlCheckReturn(ret)
    return c_count.value

def nvmlDeviceGetMigDeviceHandleByIndex(device, index):
    if False:
        i = 10
        return i + 15
    c_index = c_uint(index)
    migDevice = c_nvmlDevice_t()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetMigDeviceHandleByIndex')
    ret = fn(device, c_index, byref(migDevice))
    _nvmlCheckReturn(ret)
    return migDevice

def nvmlDeviceGetDeviceHandleFromMigDeviceHandle(migDevice):
    if False:
        while True:
            i = 10
    device = c_nvmlDevice_t()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetDeviceHandleFromMigDeviceHandle')
    ret = fn(migDevice, byref(device))
    _nvmlCheckReturn(ret)
    return device

def nvmlDeviceGetAttributes_v2(device):
    if False:
        while True:
            i = 10
    c_attrs = c_nvmlDeviceAttributes()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetAttributes_v2')
    ret = fn(device, byref(c_attrs))
    _nvmlCheckReturn(ret)
    return c_attrs

def nvmlDeviceGetAttributes(device):
    if False:
        print('Hello World!')
    return nvmlDeviceGetAttributes_v2(device)

def nvmlDeviceGetRemappedRows(device):
    if False:
        i = 10
        return i + 15
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetRemappedRows')
    c_corr = c_uint()
    c_unc = c_uint()
    c_bpending = c_uint()
    c_bfailure = c_uint()
    ret = fn(device, byref(c_corr), byref(c_unc), byref(c_bpending), byref(c_bfailure))
    _nvmlCheckReturn(ret)
    return (c_corr.value, c_unc.value, c_bpending.value, c_bfailure.value)

def nvmlDeviceGetRowRemapperHistogram(device):
    if False:
        return 10
    c_vals = c_nvmlRowRemapperHistogramValues()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetRowRemapperHistogram')
    ret = fn(device, byref(c_vals))
    _nvmlCheckReturn(ret)
    return c_vals

def nvmlDeviceGetArchitecture(device):
    if False:
        while True:
            i = 10
    arch = _nvmlDeviceArchitecture_t()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetArchitecture')
    ret = fn(device, byref(arch))
    _nvmlCheckReturn(ret)
    return arch.value

def nvmlDeviceGetBusType(device):
    if False:
        for i in range(10):
            print('nop')
    c_busType = _nvmlBusType_t()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetBusType')
    ret = fn(device, byref(c_busType))
    _nvmlCheckReturn(ret)
    return c_busType.value

def nvmlDeviceGetIrqNum(device):
    if False:
        print('Hello World!')
    c_irqNum = c_uint()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetIrqNum')
    ret = fn(device, byref(c_irqNum))
    _nvmlCheckReturn(ret)
    return c_irqNum.value

def nvmlDeviceGetNumGpuCores(device):
    if False:
        return 10
    c_numCores = c_uint()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetNumGpuCores')
    ret = fn(device, byref(c_numCores))
    _nvmlCheckReturn(ret)
    return c_numCores.value

def nvmlDeviceGetPowerSource(device):
    if False:
        return 10
    c_powerSource = _nvmlPowerSource_t()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetPowerSource')
    ret = fn(device, byref(c_powerSource))
    _nvmlCheckReturn(ret)
    return c_powerSource.value

def nvmlDeviceGetMemoryBusWidth(device):
    if False:
        i = 10
        return i + 15
    c_memBusWidth = c_uint()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetMemoryBusWidth')
    ret = fn(device, byref(c_memBusWidth))
    _nvmlCheckReturn(ret)
    return c_memBusWidth.value

def nvmlDeviceGetPcieLinkMaxSpeed(device):
    if False:
        i = 10
        return i + 15
    c_speed = _nvmlPcieLinkMaxSpeed_t()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetPcieLinkMaxSpeed')
    ret = fn(device, byref(c_speed))
    _nvmlCheckReturn(ret)
    return c_speed.value

def nvmlDeviceGetAdaptiveClockInfoStatus(device):
    if False:
        for i in range(10):
            print('nop')
    c_adaptiveClockInfoStatus = _nvmlAdaptiveClockInfoStatus_t()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetAdaptiveClockInfoStatus')
    ret = fn(device, byref(c_adaptiveClockInfoStatus))
    _nvmlCheckReturn(ret)
    return c_adaptiveClockInfoStatus.value

def nvmlDeviceGetPcieSpeed(device):
    if False:
        i = 10
        return i + 15
    c_speed = c_uint()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetPcieSpeed')
    ret = fn(device, byref(c_speed))
    _nvmlCheckReturn(ret)
    return c_speed.value

def nvmlDeviceGetDynamicPstatesInfo(device, c_dynamicpstatesinfo):
    if False:
        while True:
            i = 10
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetDynamicPstatesInfo')
    ret = fn(device, c_dynamicpstatesinfo)
    _nvmlCheckReturn(ret)
    return ret

def nvmlDeviceSetFanSpeed_v2(handle, index, speed):
    if False:
        for i in range(10):
            print('nop')
    fn = _nvmlGetFunctionPointer('nvmlDeviceSetFanSpeed_v2')
    ret = fn(handle, index, speed)
    _nvmlCheckReturn(ret)
    return ret

def nvmlDeviceGetThermalSettings(device, sensorindex, c_thermalsettings):
    if False:
        i = 10
        return i + 15
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetThermalSettings')
    ret = fn(device, sensorindex, c_thermalsettings)
    _nvmlCheckReturn(ret)
    return ret

def nvmlDeviceGetMinMaxClockOfPState(device, type, pstate, minClockMHz, maxClockMHz):
    if False:
        return 10
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetMinMaxClockOfPState')
    ret = fn(device, _nvmlClockType_t(type), _nvmlClockType_t(pstate), minClockMHz, maxClockMHz)
    _nvmlCheckReturn(ret)
    return ret

def nvmlDeviceGetSupportedPerformanceStates(device):
    if False:
        return 10
    pstates = []
    c_count = c_uint(NVML_MAX_GPU_PERF_PSTATES)
    c_size = sizeof(c_uint) * c_count.value
    pstates_array = _nvmlPstates_t * c_count.value
    c_pstates = pstates_array()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetSupportedPerformanceStates')
    ret = fn(device, c_pstates, c_size)
    _nvmlCheckReturn(ret)
    for value in c_pstates:
        if value != NVML_PSTATE_UNKNOWN:
            pstates.append(value)
    return pstates

def nvmlDeviceGetGpcClkVfOffset(device):
    if False:
        for i in range(10):
            print('nop')
    offset = c_int32()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetGpcClkVfOffset')
    ret = fn(device, byref(offset))
    _nvmlCheckReturn(ret)
    return offset.value

def nvmlDeviceSetGpcClkVfOffset(device, offset):
    if False:
        return 10
    c_offset = c_int32(offset)
    fn = _nvmlGetFunctionPointer('nvmlDeviceSetGpcClkVfOffset')
    ret = fn(device, c_offset)
    _nvmlCheckReturn(ret)
    return ret

def nvmlDeviceGetGpcClkMinMaxVfOffset(device, minOffset, maxOffset):
    if False:
        return 10
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetGpcClkMinMaxVfOffset')
    ret = fn(device, minOffset, maxOffset)
    _nvmlCheckReturn(ret)
    return ret

def nvmlDeviceGetMemClkVfOffset(device):
    if False:
        print('Hello World!')
    offset = c_int32()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetMemClkVfOffset')
    ret = fn(device, byref(offset))
    _nvmlCheckReturn(ret)
    return offset.value

def nvmlDeviceSetMemClkVfOffset(device, offset):
    if False:
        i = 10
        return i + 15
    c_offset = c_int32(offset)
    fn = _nvmlGetFunctionPointer('nvmlDeviceSetMemClkVfOffset')
    ret = fn(device, c_offset)
    _nvmlCheckReturn(ret)
    return ret

def nvmlDeviceGetMemClkMinMaxVfOffset(device, minOffset, maxOffset):
    if False:
        return 10
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetMemClkMinMaxVfOffset')
    ret = fn(device, minOffset, maxOffset)
    _nvmlCheckReturn(ret)
    return ret

def nvmlSystemSetConfComputeGpusReadyState(state):
    if False:
        return 10
    c_state = c_uint(state)
    fn = _nvmlGetFunctionPointer('nvmlSystemSetConfComputeGpusReadyState')
    ret = fn(c_state)
    _nvmlCheckReturn(ret)
    return ret

def nvmlSystemGetConfComputeGpusReadyState():
    if False:
        return 10
    c_state = c_uint()
    fn = _nvmlGetFunctionPointer('nvmlSystemGetConfComputeGpusReadyState')
    ret = fn(byref(c_state))
    _nvmlCheckReturn(ret)
    return c_state.value

def nvmlSystemGetConfComputeCapabilities():
    if False:
        i = 10
        return i + 15
    c_ccSysCaps = c_nvmlConfComputeSystemCaps_t()
    fn = _nvmlGetFunctionPointer('nvmlSystemGetConfComputeCapabilities')
    ret = fn(byref(c_ccSysCaps))
    _nvmlCheckReturn(ret)
    return c_ccSysCaps

def nvmlSystemGetConfComputeState():
    if False:
        return 10
    c_state = c_nvmlConfComputeSystemState_t()
    fn = _nvmlGetFunctionPointer('nvmlSystemGetConfComputeState')
    ret = fn(byref(c_state))
    _nvmlCheckReturn(ret)
    return c_state

def nvmlDeviceSetConfComputeUnprotectedMemSize(device, c_ccMemSize):
    if False:
        i = 10
        return i + 15
    fn = _nvmlGetFunctionPointer('nvmlDeviceSetConfComputeUnprotectedMemSize')
    ret = fn(device, c_ccMemSize)
    _nvmlCheckReturn(ret)
    return ret

def nvmlDeviceGetConfComputeMemSizeInfo(device):
    if False:
        while True:
            i = 10
    c_ccMemSize = c_nvmlConfComputeMemSizeInfo_t()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetConfComputeMemSizeInfo')
    ret = fn(device, byref(c_ccMemSize))
    _nvmlCheckReturn(ret)
    return c_ccMemSize

def nvmlDeviceGetConfComputeProtectedMemoryUsage(device):
    if False:
        i = 10
        return i + 15
    c_memory = c_nvmlMemory_t()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetConfComputeProtectedMemoryUsage')
    ret = fn(device, byref(c_memory))
    _nvmlCheckReturn(ret)
    return c_memory

def nvmlDeviceGetConfComputeGpuCertificate(device):
    if False:
        print('Hello World!')
    c_cert = c_nvmlConfComputeGpuCertificate_t()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetConfComputeGpuCertificate')
    ret = fn(device, byref(c_cert))
    _nvmlCheckReturn(ret)
    return c_cert

def nvmlDeviceGetConfComputeGpuAttestationReport(device, c_nonce):
    if False:
        for i in range(10):
            print('nop')
    c_attestReport = c_nvmlConfComputeGpuAttestationReport_t()
    c_nonce_arr = (c_uint8 * len(c_nonce))(*c_nonce)
    setattr(c_attestReport, 'nonce', c_nonce_arr)
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetConfComputeGpuAttestationReport')
    ret = fn(device, byref(c_attestReport))
    _nvmlCheckReturn(ret)
    return c_attestReport
NVML_GPM_METRIC_GRAPHICS_UTIL = 1
NVML_GPM_METRIC_SM_UTIL = 2
NVML_GPM_METRIC_SM_OCCUPANCY = 3
NVML_GPM_METRIC_INTEGER_UTIL = 4
NVML_GPM_METRIC_ANY_TENSOR_UTIL = 5
NVML_GPM_METRIC_DFMA_TENSOR_UTIL = 6
NVML_GPM_METRIC_HMMA_TENSOR_UTIL = 7
NVML_GPM_METRIC_IMMA_TENSOR_UTIL = 9
NVML_GPM_METRIC_DRAM_BW_UTIL = 10
NVML_GPM_METRIC_FP64_UTIL = 11
NVML_GPM_METRIC_FP32_UTIL = 12
NVML_GPM_METRIC_FP16_UTIL = 13
NVML_GPM_METRIC_PCIE_TX_PER_SEC = 20
NVML_GPM_METRIC_PCIE_RX_PER_SEC = 21
NVML_GPM_METRIC_NVDEC_0_UTIL = 30
NVML_GPM_METRIC_NVDEC_1_UTIL = 31
NVML_GPM_METRIC_NVDEC_2_UTIL = 32
NVML_GPM_METRIC_NVDEC_3_UTIL = 33
NVML_GPM_METRIC_NVDEC_4_UTIL = 34
NVML_GPM_METRIC_NVDEC_5_UTIL = 35
NVML_GPM_METRIC_NVDEC_6_UTIL = 36
NVML_GPM_METRIC_NVDEC_7_UTIL = 37
NVML_GPM_METRIC_NVJPG_0_UTIL = 40
NVML_GPM_METRIC_NVJPG_1_UTIL = 41
NVML_GPM_METRIC_NVJPG_2_UTIL = 42
NVML_GPM_METRIC_NVJPG_3_UTIL = 43
NVML_GPM_METRIC_NVJPG_4_UTIL = 44
NVML_GPM_METRIC_NVJPG_5_UTIL = 45
NVML_GPM_METRIC_NVJPG_6_UTIL = 46
NVML_GPM_METRIC_NVJPG_7_UTIL = 47
NVML_GPM_METRIC_NVOFA_0_UTIL = 50
NVML_GPM_METRIC_NVLINK_TOTAL_RX_PER_SEC = 60
NVML_GPM_METRIC_NVLINK_TOTAL_TX_PER_SEC = 61
NVML_GPM_METRIC_NVLINK_L0_RX_PER_SEC = 62
NVML_GPM_METRIC_NVLINK_L0_TX_PER_SEC = 63
NVML_GPM_METRIC_NVLINK_L1_RX_PER_SEC = 64
NVML_GPM_METRIC_NVLINK_L1_TX_PER_SEC = 65
NVML_GPM_METRIC_NVLINK_L2_RX_PER_SEC = 66
NVML_GPM_METRIC_NVLINK_L2_TX_PER_SEC = 67
NVML_GPM_METRIC_NVLINK_L3_RX_PER_SEC = 68
NVML_GPM_METRIC_NVLINK_L3_TX_PER_SEC = 69
NVML_GPM_METRIC_NVLINK_L4_RX_PER_SEC = 70
NVML_GPM_METRIC_NVLINK_L4_TX_PER_SEC = 71
NVML_GPM_METRIC_NVLINK_L5_RX_PER_SEC = 72
NVML_GPM_METRIC_NVLINK_L5_TX_PER_SEC = 73
NVML_GPM_METRIC_NVLINK_L6_RX_PER_SEC = 74
NVML_GPM_METRIC_NVLINK_L6_TX_PER_SEC = 75
NVML_GPM_METRIC_NVLINK_L7_RX_PER_SEC = 76
NVML_GPM_METRIC_NVLINK_L7_TX_PER_SEC = 77
NVML_GPM_METRIC_NVLINK_L8_RX_PER_SEC = 78
NVML_GPM_METRIC_NVLINK_L8_TX_PER_SEC = 79
NVML_GPM_METRIC_NVLINK_L9_RX_PER_SEC = 80
NVML_GPM_METRIC_NVLINK_L9_TX_PER_SEC = 81
NVML_GPM_METRIC_NVLINK_L10_RX_PER_SEC = 82
NVML_GPM_METRIC_NVLINK_L10_TX_PER_SEC = 83
NVML_GPM_METRIC_NVLINK_L11_RX_PER_SEC = 84
NVML_GPM_METRIC_NVLINK_L11_TX_PER_SEC = 85
NVML_GPM_METRIC_NVLINK_L12_RX_PER_SEC = 86
NVML_GPM_METRIC_NVLINK_L12_TX_PER_SEC = 87
NVML_GPM_METRIC_NVLINK_L13_RX_PER_SEC = 88
NVML_GPM_METRIC_NVLINK_L13_TX_PER_SEC = 89
NVML_GPM_METRIC_NVLINK_L14_RX_PER_SEC = 90
NVML_GPM_METRIC_NVLINK_L14_TX_PER_SEC = 91
NVML_GPM_METRIC_NVLINK_L15_RX_PER_SEC = 92
NVML_GPM_METRIC_NVLINK_L15_TX_PER_SEC = 93
NVML_GPM_METRIC_NVLINK_L16_RX_PER_SEC = 94
NVML_GPM_METRIC_NVLINK_L16_TX_PER_SEC = 95
NVML_GPM_METRIC_NVLINK_L17_RX_PER_SEC = 96
NVML_GPM_METRIC_NVLINK_L17_TX_PER_SEC = 97
NVML_GPM_METRIC_MAX = 98

class c_nvmlUnitInfo_t(_PrintableStructure):
    _fields_ = [('name', c_char * 96), ('id', c_char * 96), ('serial', c_char * 96), ('firmwareVersion', c_char * 96)]

class struct_c_nvmlGpmSample_t(Structure):
    pass
c_nvmlGpmSample_t = POINTER(struct_c_nvmlGpmSample_t)

class c_metricInfo_t(Structure):
    _fields_ = [('shortName', c_char_p), ('longName', c_char_p), ('unit', c_char_p)]

class c_nvmlGpmMetric_t(_PrintableStructure):
    _fields_ = [('metricId', c_uint), ('nvmlReturn', _nvmlReturn_t), ('value', c_double), ('metricInfo', c_metricInfo_t)]

class c_nvmlGpmMetricsGet_t(_PrintableStructure):
    _fields_ = [('version', c_uint), ('numMetrics', c_uint), ('sample1', c_nvmlGpmSample_t), ('sample2', c_nvmlGpmSample_t), ('metrics', c_nvmlGpmMetric_t * NVML_GPM_METRIC_MAX)]
NVML_GPM_METRICS_GET_VERSION = 1

class c_nvmlGpmSupport_t(_PrintableStructure):
    _fields_ = [('version', c_uint), ('isSupportedDevice', c_uint)]
NVML_GPM_SUPPORT_VERSION = 1

def nvmlGpmMetricsGet(metricsGet):
    if False:
        for i in range(10):
            print('nop')
    fn = _nvmlGetFunctionPointer('nvmlGpmMetricsGet')
    ret = fn(byref(metricsGet))
    _nvmlCheckReturn(ret)
    return metricsGet

def nvmlGpmSampleFree(gpmSample):
    if False:
        for i in range(10):
            print('nop')
    fn = _nvmlGetFunctionPointer('nvmlGpmSampleFree')
    ret = fn(gpmSample)
    _nvmlCheckReturn(ret)
    return

def nvmlGpmSampleAlloc():
    if False:
        return 10
    gpmSample = c_nvmlGpmSample_t()
    fn = _nvmlGetFunctionPointer('nvmlGpmSampleAlloc')
    ret = fn(byref(gpmSample))
    _nvmlCheckReturn(ret)
    return gpmSample

def nvmlGpmSampleGet(device, gpmSample):
    if False:
        return 10
    fn = _nvmlGetFunctionPointer('nvmlGpmSampleGet')
    ret = fn(device, gpmSample)
    _nvmlCheckReturn(ret)
    return gpmSample

def nvmlGpmMigSampleGet(device, gpuInstanceId, gpmSample):
    if False:
        i = 10
        return i + 15
    fn = _nvmlGetFunctionPointer('nvmlGpmMigSampleGet')
    ret = fn(device, gpuInstanceId, gpmSample)
    _nvmlCheckReturn(ret)
    return gpmSample

def nvmlGpmQueryDeviceSupport(device):
    if False:
        print('Hello World!')
    gpmSupport = c_nvmlGpmSupport_t()
    gpmSupport.version = NVML_GPM_SUPPORT_VERSION
    fn = _nvmlGetFunctionPointer('nvmlGpmQueryDeviceSupport')
    ret = fn(device, byref(gpmSupport))
    _nvmlCheckReturn(ret)
    return gpmSupport

def nvmlGpmSetStreamingEnabled(device, state):
    if False:
        i = 10
        return i + 15
    c_state = c_uint(state)
    fn = _nvmlGetFunctionPointer('nvmlGpmSetStreamingEnabled')
    ret = fn(device, c_state)
    _nvmlCheckReturn(ret)
    return ret

def nvmlGpmQueryIfStreamingEnabled(device):
    if False:
        i = 10
        return i + 15
    c_state = c_uint()
    fn = _nvmlGetFunctionPointer('nvmlGpmQueryIfStreamingEnabled')
    ret = fn(device, byref(c_state))
    _nvmlCheckReturn(ret)
    return c_state.value

class c_nvmlNvLinkPowerThres_t(Structure):
    _fields_ = [('lowPwrThreshold', c_uint)]

def nvmlDeviceSetNvLinkDeviceLowPowerThreshold(device, l1threshold):
    if False:
        for i in range(10):
            print('nop')
    c_info = c_nvmlNvLinkPowerThres_t()
    c_info.lowPwrThreshold = l1threshold
    fn = _nvmlGetFunctionPointer('nvmlDeviceSetNvLinkDeviceLowPowerThreshold')
    ret = fn(device, byref(c_info))
    _nvmlCheckReturn(ret)
    return ret
_nvmlGpuFabricState_t = c_uint
NVML_GPU_FABRIC_STATE_NOT_SUPPORTED = 0
NVML_GPU_FABRIC_STATE_NOT_STARTED = 1
NVML_GPU_FABRIC_STATE_IN_PROGRESS = 2
NVML_GPU_FABRIC_STATE_COMPLETED = 3

class c_nvmlGpuFabricInfo_t(_PrintableStructure):
    _fields_ = [('clusterUuid', c_char * NVML_DEVICE_UUID_BUFFER_SIZE), ('status', _nvmlReturn_t), ('partitionId', c_uint32), ('state', _nvmlGpuFabricState_t)]

def nvmlDeviceGetGpuFabricInfo(device, gpuFabricInfo):
    if False:
        while True:
            i = 10
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetGpuFabricInfo')
    ret = fn(device, gpuFabricInfo)
    _nvmlCheckReturn(ret)
    return ret
NVML_GPU_NVLINK_BW_MODE_FULL = 0
NVML_GPU_NVLINK_BW_MODE_OFF = 1
NVML_GPU_NVLINK_BW_MODE_MIN = 2
NVML_GPU_NVLINK_BW_MODE_HALF = 3
NVML_GPU_NVLINK_BW_MODE_3QUARTER = 4
NVML_GPU_NVLINK_BW_MODE_COUNT = 5

def nvmlSystemSetNvlinkBwMode(mode):
    if False:
        i = 10
        return i + 15
    fn = _nvmlGetFunctionPointer('nvmlSystemSetNvlinkBwMode')
    ret = fn(mode)
    _nvmlCheckReturn(ret)
    return ret

def nvmlSystemGetNvlinkBwMode():
    if False:
        for i in range(10):
            print('nop')
    mode = c_uint()
    fn = _nvmlGetFunctionPointer('nvmlSystemGetNvlinkBwMode')
    ret = fn(byref(mode))
    _nvmlCheckReturn(ret)
    return mode.value
_nvmlPowerScopeType_t = c_uint
NVML_POWER_SCOPE_GPU = 0
NVML_POWER_SCOPE_MODULE = 1

class c_nvmlPowerValue_v2_t(_PrintableStructure):
    _fields_ = [('version', c_uint), ('powerScope', _nvmlPowerScopeType_t), ('powerValueMw', c_uint)]
    _fmt_ = {'<default>': '%d B'}
nvmlPowerValue_v2 = 33554444

def nvmlDeviceSetPowerManagementLimit_v2(device, powerScope, powerLimit, version=nvmlPowerValue_v2):
    if False:
        return 10
    c_powerScope = _nvmlPowerScopeType_t(powerScope)
    c_powerValue = c_nvmlPowerValue_v2_t()
    c_powerValue.version = c_uint(version)
    c_powerValue.powerScope = c_powerScope
    c_powerValue.powerValueMw = c_uint(powerLimit)
    fn = _nvmlGetFunctionPointer('nvmlDeviceSetPowerManagementLimit_v2')
    ret = fn(device, byref(c_powerValue))
    return ret