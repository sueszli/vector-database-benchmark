import struct, os
try:
    import machine, stm
    from binascii import crc32
    from micropython import const
except ImportError:
    from binascii import crc32
    machine = stm = None
    const = lambda x: x
_OGF_VENDOR = const(63)
_OCF_FUS_GET_STATE = const(82)
_OCF_FUS_FW_UPGRADE = const(84)
_OCF_FUS_FW_DELETE = const(85)
_OCF_FUS_START_WS = const(90)
_OCF_BLE_INIT = const(102)
_HCI_KIND_VENDOR_RESPONSE = const(17)
_OBFUSCATION_KEY = const(1463506346)
STAGING_AREA_START = 135004160
_MAGIC_FUS_ACTIVE = const(2839959225)
_MAGIC_IPCC_MEM_INCORRECT = const(1038708577)
_FW_VERSION_FUS = const(0)
_FW_VERSION_WS = const(1)
_STATE_IDLE = const(0)
_STATE_FAILED = const(1)
_STATE_WAITING_FOR_FUS = const(2)
_STATE_WAITING_FOR_WS = const(3)
_STATE_DELETING_WS = const(4)
_STATE_COPYING_FUS = const(5)
_STATE_COPYING_WS = const(6)
_STATE_COPIED_FUS = const(7)
_STATE_COPIED_WS = const(8)
_STATE_CHECK_UPDATES = const(9)
_STATE_INSTALLING_WS = const(10)
_STATE_INSTALLING_FUS = const(11)
REASON_OK = const(0)
REASON_FLASH_COPY_FAILED = const(1)
REASON_NO_WS = const(2)
REASON_FLASH_FUS_BAD_STATE = const(3)
REASON_FLASH_WS_BAD_STATE = const(4)
REASON_FUS_NOT_RESPONDING = const(5)
REASON_FUS_NOT_RESPONDING_AFTER_FUS = const(6)
REASON_FUS_NOT_RESPONDING_AFTER_WS = const(7)
REASON_RFCORE_NOT_CONFIGURED = const(8)
REASON_WS_STILL_PRESENT = const(9)
REASON_WS_DELETION_FAILED = const(10)
REASON_FUS_VENDOR = const(16)
REASON_WS_VENDOR = const(48)
_FUS_VERSION_102 = (1, 0, 2, 0, 0)
_FUS_VERSION_110 = (1, 1, 0, 0, 0)
_PATH_FUS_102 = 'fus_102.bin'
_PATH_FUS_110 = 'fus_110.bin'
_PATH_WS_BLE_HCI = 'ws_ble_hci.bin'
_INSTALLING_FUS_GET_STATE_TIMEOUT = const(1000)
_INSTALLING_WS_GET_STATE_TIMEOUT = const(6000)

def log(msg, *args, **kwargs):
    if False:
        i = 10
        return i + 15
    print('[rfcore update]', msg.format(*args, **kwargs))

class _Flash:
    _FLASH_KEY1 = 1164378403
    _FLASH_KEY2 = 3455027627
    _FLASH_CR_STRT_MASK = 1 << 16
    _FLASH_CR_LOCK_MASK = 1 << 31
    _FLASH_SR_BSY_MASK = 1 << 16

    def wait_not_busy(self):
        if False:
            for i in range(10):
                print('nop')
        while machine.mem32[stm.FLASH + stm.FLASH_SR] & _Flash._FLASH_SR_BSY_MASK:
            machine.idle()

    def unlock(self):
        if False:
            while True:
                i = 10
        if machine.mem32[stm.FLASH + stm.FLASH_CR] & _Flash._FLASH_CR_LOCK_MASK:
            machine.mem32[stm.FLASH + stm.FLASH_KEYR] = _Flash._FLASH_KEY1
            machine.mem32[stm.FLASH + stm.FLASH_KEYR] = _Flash._FLASH_KEY2
        else:
            log('Flash was already unlocked.')

    def lock(self):
        if False:
            print('Hello World!')
        machine.mem32[stm.FLASH + stm.FLASH_CR] = _Flash._FLASH_CR_LOCK_MASK

    def erase_page(self, page):
        if False:
            i = 10
            return i + 15
        assert 0 <= page <= 255
        self.wait_not_busy()
        cr = page << 3 | 1 << 1
        machine.mem32[stm.FLASH + stm.FLASH_CR] = cr
        machine.mem32[stm.FLASH + stm.FLASH_CR] = cr | _Flash._FLASH_CR_STRT_MASK
        self.wait_not_busy()
        machine.mem32[stm.FLASH + stm.FLASH_CR] = 0

    def write(self, addr, buf, sz, key=0):
        if False:
            print('Hello World!')
        assert sz % 4 == 0
        self.wait_not_busy()
        cr = 1 << 0
        machine.mem32[stm.FLASH + stm.FLASH_CR] = cr
        off = 0
        while off < sz:
            v = buf[off] | buf[off + 1] << 8 | buf[off + 2] << 16 | buf[off + 3] << 24
            machine.mem32[addr + off] = v ^ key
            off += 4
            if off % 8 == 0:
                self.wait_not_busy()
        if off % 8:
            machine.mem32[addr + off] = 0
            self.wait_not_busy()
        machine.mem32[stm.FLASH + stm.FLASH_CR] = 0

def validate_crc(f):
    if False:
        return 10
    'Should match copy of function in rfcore_makefirmware.py to confirm operation'
    f.seek(0)
    file_crc = 0
    chunk = 16 * 1024
    buff = bytearray(chunk)
    while True:
        read = f.readinto(buff)
        if read < chunk:
            file_crc = crc32(buff[0:read], file_crc)
            break
        file_crc = crc32(buff, file_crc)
    file_crc = 4294967295 & -file_crc - 1
    f.seek(0)
    return file_crc == 0

def check_file_details(filename):
    if False:
        for i in range(10):
            print('nop')
    with open(filename, 'rb') as f:
        if not validate_crc(f):
            raise ValueError('file validation failed: incorrect crc')
        f.seek(-64, 2)
        footer = f.read()
        details = struct.unpack('<37sIIIIbbbII', footer)
        (src_filename, addr_1m, addr_640k, addr_512k, addr_256k, vers_major, vers_minor, vers_patch, KEY, crc) = details
        src_filename = src_filename.strip(b'\x00').decode()
        if KEY != _OBFUSCATION_KEY:
            raise ValueError('file validation failed: incorrect key')
    return (src_filename, addr_1m, addr_640k, addr_512k, addr_256k, (vers_major, vers_minor, vers_patch))

def _copy_file_to_flash(filename):
    if False:
        return 10
    flash = _Flash()
    flash.unlock()
    _write_target_addr(0)
    try:
        (src_filename, addr_1m, addr_640k, addr_512k, addr_256k, vers) = check_file_details(filename)
        addr = load_addr = addr_1m
        log(f'Writing {src_filename} v{vers[0]}.{vers[1]}.{vers[2]} to addr: 0x{addr:x}')
        erase_addr = STAGING_AREA_START
        sfr_sfsa = machine.mem32[stm.FLASH + stm.FLASH_SFR] & 255
        erase_limit = 134217728 + sfr_sfsa * 4096
        while erase_addr < erase_limit:
            flash.erase_page((erase_addr - 134217728) // 4096)
            erase_addr += 4096
        with open(filename, 'rb') as f:
            buf = bytearray(4096)
            while 1:
                sz = f.readinto(buf)
                if sz == 0:
                    break
                flash.write(addr, buf, sz, _OBFUSCATION_KEY)
                addr += 4096
        _write_target_addr(load_addr)
    finally:
        flash.lock()

def _parse_vendor_response(data):
    if False:
        return 10
    assert len(data) >= 7
    assert data[0] == _HCI_KIND_VENDOR_RESPONSE
    assert data[1] == 14
    op = data[5] << 8 | data[4]
    return (op >> 10, op & 1023, data[6], data[7] if len(data) > 7 else 0)

def _run_sys_hci_cmd(ogf, ocf, buf=b'', timeout=0):
    if False:
        return 10
    try:
        (ogf_out, ocf_out, status, result) = _parse_vendor_response(stm.rfcore_sys_hci(ogf, ocf, buf, timeout))
    except OSError:
        return (255, 255)
    assert ogf_out == ogf
    assert ocf_out == ocf
    return (status, result)

def fus_get_state(timeout=0):
    if False:
        for i in range(10):
            print('nop')
    return _run_sys_hci_cmd(_OGF_VENDOR, _OCF_FUS_GET_STATE, timeout=timeout)

def fus_is_idle():
    if False:
        i = 10
        return i + 15
    return fus_get_state() == (0, 0)

def fus_start_ws():
    if False:
        print('Hello World!')
    return _run_sys_hci_cmd(_OGF_VENDOR, _OCF_FUS_START_WS)

def _fus_fwdelete():
    if False:
        while True:
            i = 10
    return _run_sys_hci_cmd(_OGF_VENDOR, _OCF_FUS_FW_DELETE)

def _fus_run_fwupgrade():
    if False:
        for i in range(10):
            print('nop')
    addr = _read_target_addr()
    if not addr:
        log(f'Update failed: Invalid load address: 0x{addr:x}')
        return False
    log(f'Loading to: 0x{addr:x}')
    return _run_sys_hci_cmd(_OGF_VENDOR, _OCF_FUS_FW_UPGRADE, struct.pack('<I', addr))
if stm:
    REG_RTC_STATE = stm.RTC + stm.RTC_BKP18R
    REG_RTC_REASON = stm.RTC + stm.RTC_BKP17R
    REG_RTC_ADDR = stm.RTC + stm.RTC_BKP16R

def _read_state():
    if False:
        i = 10
        return i + 15
    return machine.mem32[REG_RTC_STATE]

def _write_state(state):
    if False:
        while True:
            i = 10
    machine.mem32[REG_RTC_STATE] = state

def _read_failure_reason():
    if False:
        print('Hello World!')
    return machine.mem32[REG_RTC_REASON]

def _write_failure_state(reason):
    if False:
        i = 10
        return i + 15
    machine.mem32[REG_RTC_REASON] = reason
    _write_state(_STATE_FAILED)
    return reason

def _read_target_addr():
    if False:
        return 10
    return machine.mem32[REG_RTC_ADDR]

def _write_target_addr(addr):
    if False:
        for i in range(10):
            print('nop')
    machine.mem32[REG_RTC_ADDR] = addr

def _stat_and_start_copy(path, copying_state, copied_state):
    if False:
        while True:
            i = 10
    try:
        os.stat(path)
    except OSError:
        log('{} not found', path)
        return False
    log('{} update is available', path)
    if sum(stm.rfcore_fw_version(_FW_VERSION_WS)):
        log('Removing existing WS firmware')
        _write_state(_STATE_DELETING_WS)
        _fus_fwdelete()
    else:
        log('Copying {} to flash', path)
        _write_state(copying_state)
        _copy_file_to_flash(path)
        log('Copying complete')
        _write_state(copied_state)
    return True

def resume():
    if False:
        return 10
    log('Checking firmware update progress...')
    if stm.rfcore_status() == _MAGIC_IPCC_MEM_INCORRECT:
        return _write_failure_state(REASON_RFCORE_NOT_CONFIGURED)
    while True:
        state = _read_state()
        if state == _STATE_IDLE:
            log('Firmware update complete')
            return 0
        elif state == _STATE_FAILED:
            log('Firmware update failed')
            return _read_failure_reason()
        elif state == _STATE_WAITING_FOR_FUS:
            log('Querying FUS state')
            (status, result) = fus_get_state()
            log('FUS state: {} {}', status, result)
            if status == 255 and result == 255:
                _write_failure_state(REASON_FUS_NOT_RESPONDING)
            elif status != 0:
                log('Operation in progress. Re-querying FUS state')
            elif stm.rfcore_status() == _MAGIC_FUS_ACTIVE:
                log('FUS active')
                _write_state(_STATE_CHECK_UPDATES)
        elif state == _STATE_WAITING_FOR_WS:
            if stm.rfcore_status() != _MAGIC_FUS_ACTIVE:
                log('WS active')
                _write_state(_STATE_IDLE)
                machine.reset()
            else:
                log('Starting WS')
                (status, result) = fus_start_ws()
                if status != 0:
                    log("Can't start WS")
                    log('WS version: {}', stm.rfcore_fw_version(_FW_VERSION_WS))
                    _write_failure_state(REASON_NO_WS)
        elif state == _STATE_CHECK_UPDATES:
            log('Checking for updates')
            fus_version = stm.rfcore_fw_version(_FW_VERSION_FUS)
            log('FUS version {}', fus_version)
            if fus_version < _FUS_VERSION_102:
                log('Factory FUS detected')
                if _stat_and_start_copy(_PATH_FUS_102, _STATE_COPYING_FUS, _STATE_COPIED_FUS):
                    continue
            elif fus_version >= _FUS_VERSION_102 and fus_version < _FUS_VERSION_110:
                log('FUS 1.0.2 detected')
                if _stat_and_start_copy(_PATH_FUS_110, _STATE_COPYING_FUS, _STATE_COPIED_FUS):
                    continue
            else:
                log('FUS is up-to-date')
            if fus_version >= _FUS_VERSION_110:
                if _stat_and_start_copy(_PATH_WS_BLE_HCI, _STATE_COPYING_WS, _STATE_COPIED_WS):
                    continue
                else:
                    log('No WS updates available')
            else:
                log('Need latest FUS to install WS')
            _write_state(_STATE_WAITING_FOR_WS)
        elif state == _STATE_COPYING_FUS or state == _STATE_COPYING_WS:
            log('Flash copy failed mid-write')
            _write_failure_state(REASON_FLASH_COPY_FAILED)
        elif state == _STATE_COPIED_FUS:
            if fus_is_idle():
                log('FUS copy complete, installing')
                _write_state(_STATE_INSTALLING_FUS)
                _fus_run_fwupgrade()
            else:
                log('FUS copy bad state')
                _write_failure_state(REASON_FLASH_FUS_BAD_STATE)
        elif state == _STATE_INSTALLING_FUS:
            log('Installing FUS...')
            (status, result) = fus_get_state(_INSTALLING_FUS_GET_STATE_TIMEOUT)
            log('FUS state: {} {}', status, result)
            if 32 <= status <= 47 and result == 0:
                log('FUS still in progress...')
            elif 16 <= status <= 31 and result == 17:
                log('Attempted to install same FUS version... re-querying FUS state to resume.')
            elif status == 0:
                log('FUS update successful')
                _write_state(_STATE_CHECK_UPDATES)
            elif result == 0:
                log('Re-querying FUS state...')
            elif result == 255:
                _write_failure_state(REASON_FUS_NOT_RESPONDING_AFTER_FUS)
            else:
                _write_failure_state(REASON_FUS_VENDOR + result)
        elif state == _STATE_DELETING_WS:
            log('Deleting WS...')
            (status, result) = fus_get_state()
            log('FUS state: {} {}', status, result)
            if status == 0:
                if sum(stm.rfcore_fw_version(_FW_VERSION_WS)) == 0:
                    log('WS deletion complete')
                    _write_state(_STATE_CHECK_UPDATES)
                else:
                    log('WS deletion no effect')
                    _write_failure_state(REASON_WS_STILL_PRESENT)
            elif status == 1:
                log('WS deletion failed')
                _write_failure_state(REASON_WS_DELETION_FAILED)
        elif state == _STATE_COPIED_WS:
            if fus_is_idle():
                log('WS copy complete, installing')
                _write_state(_STATE_INSTALLING_WS)
                _fus_run_fwupgrade()
            else:
                log('WS copy bad state')
                _write_failure_state(REASON_FLASH_WS_BAD_STATE)
        elif state == _STATE_INSTALLING_WS:
            log('Installing WS...')
            (status, result) = fus_get_state(_INSTALLING_WS_GET_STATE_TIMEOUT)
            log('FUS state: {} {}', status, result)
            if 16 <= status <= 31 and result == 0:
                log('WS still in progress...')
            elif 16 <= status <= 31 and result == 17:
                log('Attempted to install same WS version... re-querying FUS state to resume.')
            elif status == 0:
                log('WS update successful')
                _write_state(_STATE_WAITING_FOR_WS)
            elif result in (0, 254):
                log('Re-querying FUS state...')
            elif result == 255:
                _write_failure_state(REASON_FUS_NOT_RESPONDING_AFTER_WS)
            else:
                _write_failure_state(REASON_WS_VENDOR + result)

def install_boot():
    if False:
        i = 10
        return i + 15
    boot_py = '/flash/boot.py'
    header = ''
    mode = 'w'
    try:
        with open(boot_py, 'r') as boot:
            header = '\n'
            mode = 'a'
            for line in boot:
                if 'rfcore_firmware.resume()' in line:
                    print('Already installed.')
                    return
        print('boot.py exists, adding upgrade handler.')
    except OSError:
        print("boot.py doesn't exists, adding with upgrade handler.")
    with open(boot_py, mode) as boot:
        boot.write(header)
        boot.write('# Handle rfcore updates.\n')
        boot.write('import rfcore_firmware\n')
        boot.write('rfcore_firmware.resume()\n')

def check_for_updates(force=False):
    if False:
        for i in range(10):
            print('nop')
    (src_filename, addr_1m, addr_640k, addr_512k, addr_256k, vers_fus) = check_file_details(_PATH_FUS_110)
    (src_filename, addr_1m, addr_640k, addr_512k, addr_256k, vers_ws) = check_file_details(_PATH_WS_BLE_HCI)
    current_version_fus = stm.rfcore_fw_version(_FW_VERSION_FUS)
    fus_uptodate = current_version_fus[0:3] == vers_fus
    current_version_ws = stm.rfcore_fw_version(_FW_VERSION_WS)
    ws_uptodate = current_version_ws[0:3] == vers_ws
    if fus_uptodate and ws_uptodate and (not force):
        log(f'Already up to date: fus: {current_version_fus}, ws: {current_version_ws}')
    else:
        log('Starting firmware update')
        log(f' - fus: {current_version_fus} -> {vers_fus}')
        log(f' - ws:  {current_version_ws} -> {vers_ws}')
        _write_state(_STATE_WAITING_FOR_FUS)
        machine.reset()