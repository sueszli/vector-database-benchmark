from micropython import const
import struct
import bluetooth
_ADV_TYPE_FLAGS = const(1)
_ADV_TYPE_NAME = const(9)
_ADV_TYPE_UUID16_COMPLETE = const(3)
_ADV_TYPE_UUID32_COMPLETE = const(5)
_ADV_TYPE_UUID128_COMPLETE = const(7)
_ADV_TYPE_UUID16_MORE = const(2)
_ADV_TYPE_UUID32_MORE = const(4)
_ADV_TYPE_UUID128_MORE = const(6)
_ADV_TYPE_APPEARANCE = const(25)
_ADV_MAX_PAYLOAD = const(31)

def advertising_payload(limited_disc=False, br_edr=False, name=None, services=None, appearance=0):
    if False:
        return 10
    payload = bytearray()

    def _append(adv_type, value):
        if False:
            return 10
        nonlocal payload
        payload += struct.pack('BB', len(value) + 1, adv_type) + value
    _append(_ADV_TYPE_FLAGS, struct.pack('B', (1 if limited_disc else 2) + (24 if br_edr else 4)))
    if name:
        _append(_ADV_TYPE_NAME, name)
    if services:
        for uuid in services:
            b = bytes(uuid)
            if len(b) == 2:
                _append(_ADV_TYPE_UUID16_COMPLETE, b)
            elif len(b) == 4:
                _append(_ADV_TYPE_UUID32_COMPLETE, b)
            elif len(b) == 16:
                _append(_ADV_TYPE_UUID128_COMPLETE, b)
    if appearance:
        _append(_ADV_TYPE_APPEARANCE, struct.pack('<h', appearance))
    if len(payload) > _ADV_MAX_PAYLOAD:
        raise ValueError('advertising payload too large')
    return payload

def decode_field(payload, adv_type):
    if False:
        i = 10
        return i + 15
    i = 0
    result = []
    while i + 1 < len(payload):
        if payload[i + 1] == adv_type:
            result.append(payload[i + 2:i + payload[i] + 1])
        i += 1 + payload[i]
    return result

def decode_name(payload):
    if False:
        print('Hello World!')
    n = decode_field(payload, _ADV_TYPE_NAME)
    return str(n[0], 'utf-8') if n else ''

def decode_services(payload):
    if False:
        i = 10
        return i + 15
    services = []
    for u in decode_field(payload, _ADV_TYPE_UUID16_COMPLETE):
        services.append(bluetooth.UUID(struct.unpack('<h', u)[0]))
    for u in decode_field(payload, _ADV_TYPE_UUID32_COMPLETE):
        services.append(bluetooth.UUID(struct.unpack('<d', u)[0]))
    for u in decode_field(payload, _ADV_TYPE_UUID128_COMPLETE):
        services.append(bluetooth.UUID(u))
    return services

def demo():
    if False:
        for i in range(10):
            print('nop')
    payload = advertising_payload(name='micropython', services=[bluetooth.UUID(6170), bluetooth.UUID('6E400001-B5A3-F393-E0A9-E50E24DCCA9E')])
    print(payload)
    print(decode_name(payload))
    print(decode_services(payload))
if __name__ == '__main__':
    demo()