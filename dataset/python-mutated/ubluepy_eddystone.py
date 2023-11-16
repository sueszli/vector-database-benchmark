from micropython import const
from ubluepy import Peripheral, constants
BLE_GAP_ADV_FLAG_LE_GENERAL_DISC_MODE = const(2)
BLE_GAP_ADV_FLAG_BR_EDR_NOT_SUPPORTED = const(4)
BLE_GAP_ADV_FLAGS_LE_ONLY_GENERAL_DISC_MODE = const(BLE_GAP_ADV_FLAG_LE_GENERAL_DISC_MODE | BLE_GAP_ADV_FLAG_BR_EDR_NOT_SUPPORTED)
EDDYSTONE_FRAME_TYPE_URL = const(16)
EDDYSTONE_URL_PREFIX_HTTP_WWW = const(0)
EDDYSTONE_URL_SUFFIX_DOT_COM = const(1)

def string_to_binarray(text):
    if False:
        for i in range(10):
            print('nop')
    b = bytearray([])
    for c in text:
        b.append(ord(c))
    return b

def gen_ad_type_content(ad_type, data):
    if False:
        while True:
            i = 10
    b = bytearray(1)
    b.append(ad_type)
    b.extend(data)
    b[0] = len(b) - 1
    return b

def generate_eddystone_adv_packet(url):
    if False:
        print('Hello World!')
    disc_mode = bytearray([BLE_GAP_ADV_FLAGS_LE_ONLY_GENERAL_DISC_MODE])
    packet_flags = gen_ad_type_content(constants.ad_types.AD_TYPE_FLAGS, disc_mode)
    uuid = bytearray([170, 254])
    packet_uuid16 = gen_ad_type_content(constants.ad_types.AD_TYPE_16BIT_SERVICE_UUID_COMPLETE, uuid)
    rssi = 238
    eddystone_data = bytearray([])
    eddystone_data.append(EDDYSTONE_FRAME_TYPE_URL)
    eddystone_data.append(rssi)
    eddystone_data.append(EDDYSTONE_URL_PREFIX_HTTP_WWW)
    eddystone_data.extend(string_to_binarray(url))
    eddystone_data.append(EDDYSTONE_URL_SUFFIX_DOT_COM)
    service_data = uuid + eddystone_data
    packet_service_data = gen_ad_type_content(constants.ad_types.AD_TYPE_SERVICE_DATA, service_data)
    packet = bytearray([])
    packet.extend(packet_flags)
    packet.extend(packet_uuid16)
    packet.extend(packet_service_data)
    return packet

def start():
    if False:
        while True:
            i = 10
    adv_packet = generate_eddystone_adv_packet('micropython')
    p = Peripheral()
    p.advertise(data=adv_packet, connectable=False)