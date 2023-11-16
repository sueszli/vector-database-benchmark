from machine import mem8, mem16, mem32
from micropython import const
import stm
SRAM2A_BASE = const(537067520)
OGF_VENDOR = const(63)
OCF_FUS_GET_STATE = const(82)
OCF_FUS_FW_UPGRADE = const(84)
OCF_FUS_FW_DELETE = const(85)
OCF_FUS_START_WS = const(90)
OCF_BLE_INIT = const(102)
TABLE_DEVICE_INFO = const(0)
TABLE_BLE = const(1)
TABLE_SYS = const(3)
TABLE_MEM_MANAGER = const(4)
CHANNEL_BLE = const(1)
CHANNEL_SYS = const(2)
CHANNEL_TRACES = const(4)
CHANNEL_ACL = const(6)
INDICATOR_HCI_COMMAND = const(1)
INDICATOR_HCI_EVENT = const(4)
INDICATOR_FUS_COMMAND = const(16)
INDICATOR_FUS_RESPONSE = const(17)
INDICATOR_FUS_EVENT = const(18)
MAGIC_FUS_ACTIVE = const(2839959225)

def get_ipccdba():
    if False:
        return 10
    return mem32[stm.FLASH + stm.FLASH_IPCCBR] & 16383

def get_ipcc_table(table):
    if False:
        for i in range(10):
            print('nop')
    return mem32[SRAM2A_BASE + get_ipccdba() + table * 4]

def get_ipcc_table_word(table, offset):
    if False:
        for i in range(10):
            print('nop')
    return mem32[get_ipcc_table(table) + offset * 4] & 4294967295

def get_ipcc_table_byte(table, offset):
    if False:
        while True:
            i = 10
    return mem8[get_ipcc_table(table) + offset] & 255

def sram2a_dump(num_words=64, width=8):
    if False:
        while True:
            i = 10
    print('SRAM2A @%08x' % SRAM2A_BASE)
    for i in range((num_words + width - 1) // width):
        print('  %04x ' % (i * 4 * width), end='')
        for j in range(width):
            print(' %08x' % (mem32[SRAM2A_BASE + (i * width + j) * 4] & 4294967295), end='')
        print()
SYS_CMD_BUF = 0
SYS_SYS_QUEUE = 0
MM_BLE_SPARE_EVT_BUF = 0
MM_SYS_SPARE_EVT_BUF = 0
MM_BLE_POOL = 0
MM_BLE_POOL_SIZE = 0
MM_FREE_BUF_QUEUE = 0
MM_EV_POOL = 0
MM_EV_POOL_SIZE = 0
BLE_CMD_BUF = 0
BLE_CS_BUF = 0
BLE_EVT_QUEUE = 0
BLE_HCI_ACL_DATA_BUF = 0

def ipcc_init():
    if False:
        print('Hello World!')
    global SYS_CMD_BUF, SYS_SYS_QUEUE
    SYS_CMD_BUF = get_ipcc_table_word(TABLE_SYS, 0)
    SYS_SYS_QUEUE = get_ipcc_table_word(TABLE_SYS, 1)
    global MM_BLE_SPARE_EVT_BUF, MM_SYS_SPARE_EVT_BUF, MM_BLE_POOL, MM_BLE_POOL_SIZE, MM_FREE_BUF_QUEUE, MM_EV_POOL, MM_EV_POOL_SIZE
    MM_BLE_SPARE_EVT_BUF = get_ipcc_table_word(TABLE_MEM_MANAGER, 0)
    MM_SYS_SPARE_EVT_BUF = get_ipcc_table_word(TABLE_MEM_MANAGER, 1)
    MM_BLE_POOL = get_ipcc_table_word(TABLE_MEM_MANAGER, 2)
    MM_BLE_POOL_SIZE = get_ipcc_table_word(TABLE_MEM_MANAGER, 3)
    MM_FREE_BUF_QUEUE = get_ipcc_table_word(TABLE_MEM_MANAGER, 4)
    MM_EV_POOL = get_ipcc_table_word(TABLE_MEM_MANAGER, 5)
    MM_EV_POOL_SIZE = get_ipcc_table_word(TABLE_MEM_MANAGER, 6)
    global BLE_CMD_BUF, BLE_CS_BUF, BLE_EVT_QUEUE, BLE_HCI_ACL_DATA_BUF
    BLE_CMD_BUF = get_ipcc_table_word(TABLE_BLE, 0)
    BLE_CS_BUF = get_ipcc_table_word(TABLE_BLE, 1)
    BLE_EVT_QUEUE = get_ipcc_table_word(TABLE_BLE, 2)
    BLE_HCI_ACL_DATA_BUF = get_ipcc_table_word(TABLE_BLE, 3)
    mem32[stm.IPCC + stm.IPCC_C1CR] = 0
    print('IPCC initialised')
    print('SYS: 0x%08x 0x%08x' % (SYS_CMD_BUF, SYS_SYS_QUEUE))
    print('BLE: 0x%08x 0x%08x 0x%08x' % (BLE_CMD_BUF, BLE_CS_BUF, BLE_EVT_QUEUE))

def fus_active():
    if False:
        return 10
    return get_ipcc_table_word(TABLE_DEVICE_INFO, 0) == MAGIC_FUS_ACTIVE

def info():
    if False:
        while True:
            i = 10
    sfr = mem32[stm.FLASH + stm.FLASH_SFR]
    srrvr = mem32[stm.FLASH + stm.FLASH_SRRVR]
    print('IPCCDBA  : 0x%08x' % (get_ipccdba() & 16383))
    print('DDS      : %r' % bool(sfr & 1 << 12))
    print('FSD      : %r' % bool(sfr & 1 << 8))
    print('SFSA     : 0x%08x' % (sfr & 255))
    print('C2OPT    : %r' % bool(srrvr & 1 << 31))
    print('NBRSD    : %r' % bool(srrvr & 1 << 30))
    print('SNBRSA   : 0x%08x' % (srrvr >> 25 & 31))
    print('BRSD     : %r' % bool(srrvr & 1 << 23))
    print('SBRSA    : 0x%08x' % (srrvr >> 18 & 31))
    print('SBRV     : 0x%08x' % (srrvr & 262143))

def dev_info():
    if False:
        for i in range(10):
            print('nop')

    def dump_version(offset):
        if False:
            i = 10
            return i + 15
        x = get_ipcc_table_word(TABLE_DEVICE_INFO, offset)
        print('0x%08x (%u.%u.%u.%u.%u)' % (x, x >> 24, x >> 16 & 255, x >> 8 & 255, x >> 4 & 15, x & 15))

    def dump_memory_size(offset):
        if False:
            i = 10
            return i + 15
        x = get_ipcc_table_word(TABLE_DEVICE_INFO, offset)
        print('0x%08x (SRAM2b=%uk SRAM2a=%uk flash=%uk)' % (x, x >> 24, x >> 16 & 255, (x & 255) * 4))
    print('Device information table @%08x:' % get_ipcc_table(TABLE_DEVICE_INFO))
    if fus_active():
        print('FUS is active')
        print('state                    : 0x%08x' % get_ipcc_table_word(TABLE_DEVICE_INFO, 0))
        print('last FUS active state    : 0x%02x' % get_ipcc_table_byte(TABLE_DEVICE_INFO, 5))
        print('last wireless stack state: 0x%02x' % get_ipcc_table_byte(TABLE_DEVICE_INFO, 6))
        print('cur wireless stack type  : 0x%02x' % get_ipcc_table_byte(TABLE_DEVICE_INFO, 7))
        print('safe boot version        : ', end='')
        dump_version(2)
        print('FUS version              : ', end='')
        dump_version(3)
        print('FUS memory size          : ', end='')
        dump_memory_size(4)
        print('wireless stack version   : ', end='')
        dump_version(5)
        print('wireless stack mem size  : ', end='')
        dump_memory_size(6)
        print('wireless FW-BLE info     : 0x%08x' % get_ipcc_table_word(TABLE_DEVICE_INFO, 7))
        print('wireless FW-thread info  : 0x%08x' % get_ipcc_table_word(TABLE_DEVICE_INFO, 8))
        print('UID64                    : 0x%08x 0x%08x' % (get_ipcc_table_word(TABLE_DEVICE_INFO, 9), get_ipcc_table_word(TABLE_DEVICE_INFO, 10)))
        print('device ID                : 0x%04x' % get_ipcc_table_word(TABLE_DEVICE_INFO, 11))
    else:
        print('WS is active')
        print('safe boot version        : ', end='')
        dump_version(0)
        print('FUS version              : ', end='')
        dump_version(1)
        print('FUS memory size          : ', end='')
        dump_memory_size(2)
        print('FUS info                 : 0x%08x' % get_ipcc_table_word(TABLE_DEVICE_INFO, 3))
        print('wireless stack version   : ', end='')
        dump_version(4)
        print('wireless stack mem size  : ', end='')
        dump_memory_size(5)
        print('wireless stack info      : 0x%08x' % get_ipcc_table_word(TABLE_DEVICE_INFO, 7))
        print('wireless reserved        : 0x%08x' % get_ipcc_table_word(TABLE_DEVICE_INFO, 7))

def ipcc_state():
    if False:
        return 10
    print('IPCC:')
    print('  C1CR:     0x%08x' % (mem32[stm.IPCC + stm.IPCC_C1CR] & 4294967295), end='')
    print('  C2CR:     0x%08x' % (mem32[stm.IPCC + stm.IPCC_C2CR] & 4294967295))
    print('  C1MR:     0x%08x' % (mem32[stm.IPCC + stm.IPCC_C1MR] & 4294967295), end='')
    print('  C2MR:     0x%08x' % (mem32[stm.IPCC + stm.IPCC_C2MR] & 4294967295))
    print('  C1TOC2SR: 0x%08x' % (mem32[stm.IPCC + stm.IPCC_C1TOC2SR] & 4294967295), end='')
    print('  C2TOC1SR: 0x%08x' % (mem32[stm.IPCC + stm.IPCC_C2TOC1SR] & 4294967295))
sram2a_dump(264)
ipcc_init()
info()
dev_info()
ipcc_state()