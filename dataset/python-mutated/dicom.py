import struct
from typing import Dict, Generator, Tuple, Union, cast
from ivre import utils
from ivre.types.active import NmapPort

def _gen_items(data: bytes) -> Generator[Tuple[int, bytes], None, None]:
    if False:
        print('Hello World!')
    while data:
        if len(data) < 4:
            utils.LOGGER.debug('Item too short: maybe a broken DICOM item [%r]', data)
            return
        (itype, pad, ilen) = struct.unpack('>BBH', data[:4])
        if pad:
            utils.LOGGER.debug('Non zero padding: maybe a broken DICOM item [%r]', data)
        data = data[4:]
        if ilen < len(data):
            utils.LOGGER.debug('Item too short: maybe a broken DICOM item [%r]', data)
        yield (itype, data[:ilen])
        data = data[ilen:]
_USER_INFO_ITEMS = {81: 'max_pdu_length', 82: 'implementation_class_uid', 85: 'implementation_version'}

def _parse_items(data: bytes) -> Dict[str, Union[int, str]]:
    if False:
        return 10
    res: Dict[str, Union[int, str]] = {}
    items = dict(_gen_items(data))
    if 80 not in items:
        utils.LOGGER.warning('No User Info in items [%r]', items)
        return res
    ivalue: bytes
    ivalue_parsed: Union[int, str]
    for (itype, ivalue) in _gen_items(items[80]):
        if itype == 81:
            try:
                ivalue_parsed = cast(int, struct.unpack('>I', ivalue)[0])
            except struct.error:
                utils.LOGGER.warning('Cannot convert max_pdu_length value to an integer [%r]', ivalue)
                ivalue_parsed = utils.encode_b64(ivalue).decode()
        else:
            try:
                ivalue_parsed = ivalue.decode('ascii')
            except struct.error:
                utils.LOGGER.warning('Cannot convert value to an ASCII string [%r]', ivalue)
                ivalue_parsed = utils.encode_b64(ivalue).decode()
        try:
            itype_parsed = _USER_INFO_ITEMS[itype]
        except KeyError:
            utils.LOGGER.warning('Unknown item type in User Info %02x [%r]', itype, ivalue)
            itype_parsed = 'unknown_%02x' % itype
        res[itype_parsed] = ivalue_parsed
    return res

def parse_message(data: bytes) -> NmapPort:
    if False:
        print('Hello World!')
    res: NmapPort = {}
    if len(data) < 6:
        utils.LOGGER.debug('Message too short: probably not a DICOM message [%r]', data)
        return res
    (rtype, pad, rlen) = struct.unpack('>BBI', data[:6])
    if pad:
        utils.LOGGER.debug('Non zero padding: probably not a DICOM message [%r]', data)
        return res
    if rlen > len(data) - 6:
        utils.LOGGER.debug('Message too short: probably not a DICOM message [%r]', data)
        return res
    if rtype in [2, 3]:
        res['service_name'] = 'dicom'
        extra_info: Dict[str, Union[int, str]] = {}
        if rtype == 2:
            msg = 'Any AET is accepted (Insecure)'
            if data[6:74] != b'\x00\x01\x00\x00ANY-SCP         ECHOSCU         \x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00':
                extra_info = {'info': 'Unusual accept message'}
            else:
                extra_info = _parse_items(data[74:])
        else:
            msg = 'Called AET check enabled'
            if data != b'\x03\x00\x00\x00\x00\x04\x00\x01\x01\x07':
                extra_info = {'info': 'Unusual reject message'}
        script_output = ['', 'dicom: DICOM Service Provider discovered!', 'config: %s' % msg]
        script_data: Dict[str, Union[int, str]] = {'dicom': 'DICOM Service Provider discovered!', 'config': msg}
        for (key, value) in extra_info.items():
            script_output.append('%s: %s' % (key, value))
            script_data[key] = value
        res['scripts'] = [{'id': 'dicom-ping', 'output': '\n  '.join(script_output), 'dicom-ping': script_data}]
        return res
    utils.LOGGER.debug('Unknown message type [%r]: probably not a DICOM message [%r]', rtype, data)
    return res