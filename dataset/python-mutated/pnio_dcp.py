from scapy.compat import orb
from scapy.all import Packet, bind_layers, Padding
from scapy.fields import ByteEnumField, ConditionalField, FieldLenField, FieldListField, IPField, LenField, MACField, MultiEnumField, MultipleTypeField, PacketListField, PadField, ShortEnumField, ShortField, StrLenField, XByteField, XIntField, XShortField
MIN_PACKET_LENGTH = 44
DCP_GET_SET_FRAME_ID = 65277
DCP_IDENTIFY_REQUEST_FRAME_ID = 65278
DCP_IDENTIFY_RESPONSE_FRAME_ID = 65279
DCP_REQUEST = 0
DCP_RESPONSE = 1
DCP_SERVICE_ID_GET = 3
DCP_SERVICE_ID_SET = 4
DCP_SERVICE_ID_IDENTIFY = 5
DCP_SERVICE_ID = {0: 'reserved', 1: 'Manufacturer specific', 2: 'Manufacturer specific', 3: 'Get', 4: 'Set', 5: 'Identify', 6: 'Hello'}
DCP_SERVICE_TYPE = {0: 'Request', 1: 'Response Success', 5: 'Response - Request not supported'}
DCP_DEVICE_ROLES = {0: 'IO Supervisor', 1: 'IO Device', 2: 'IO Controller'}
DCP_OPTIONS = {0: 'reserved', 1: 'IP', 2: 'Device properties', 3: 'DHCP', 4: 'Reserved', 5: 'Control', 6: 'Device Initiative', 255: 'All Selector'}
DCP_OPTIONS.update({i: 'reserved' for i in range(7, 127)})
DCP_OPTIONS.update({i: 'Manufacturer specific' for i in range(128, 254)})
DCP_SUBOPTIONS = {1: {0: 'Reserved', 1: 'MAC Address', 2: 'IP Parameter', 3: 'Full IP Suite'}, 2: {0: 'Reserved', 1: 'Manufacturer specific (Type of Station)', 2: 'Name of Station', 3: 'Device ID', 4: 'Device Role', 5: 'Device Options', 6: 'Alias Name', 7: 'Device Instance', 8: 'OEM Device ID'}, 3: {12: 'Host name', 43: 'Vendor specific', 54: 'Server identifier', 55: 'Parameter request list', 60: 'Class identifier', 61: 'DHCP client identifier', 81: 'FQDN, Fully Qualified Domain Name', 97: 'UUID/GUID-based Client', 255: 'Control DHCP for address resolution'}, 5: {0: 'Reserved', 1: 'Start Transaction', 2: 'End Transaction', 3: 'Signal', 4: 'Response', 5: 'Reset Factory Settings', 6: 'Reset to Factory'}, 6: {0: 'Reserved', 1: 'Device Initiative'}, 255: {255: 'ALL Selector'}}
BLOCK_INFOS = {0: 'Reserved'}
BLOCK_INFOS.update({i: 'reserved' for i in range(1, 255)})
IP_BLOCK_INFOS = {0: 'IP not set', 1: 'IP set', 2: 'IP set by DHCP', 128: 'IP not set (address conflict detected)', 129: 'IP set (address conflict detected)', 130: 'IP set by DHCP (address conflict detected)'}
IP_BLOCK_INFOS.update({i: 'reserved' for i in range(3, 127)})
BLOCK_ERRORS = {0: 'Ok', 1: 'Option unsupp.', 2: 'Suboption unsupp. or no DataSet avail.', 3: 'Suboption not set', 4: 'Resource Error', 5: 'SET not possible by local reasons', 6: 'In operation, SET not possible'}
BLOCK_QUALIFIERS = {0: 'Use the value temporary', 1: 'Save the value permanent'}
BLOCK_QUALIFIERS.update({i: 'reserved' for i in range(2, 255)})

class DCPBaseBlock(Packet):
    """
        base class for all DCP Blocks
    """
    fields_desc = [ByteEnumField('option', 1, DCP_OPTIONS), MultiEnumField('sub_option', 2, DCP_SUBOPTIONS, fmt='B', depends_on=lambda p: p.option), FieldLenField('dcp_block_length', None, length_of='data'), ShortEnumField('block_info', 0, BLOCK_INFOS), StrLenField('data', '', length_from=lambda x: x.dcp_block_length)]

    def extract_padding(self, s):
        if False:
            for i in range(10):
                print('nop')
        return ('', s)

class DCPIPBlock(Packet):
    fields_desc = [ByteEnumField('option', 1, DCP_OPTIONS), MultiEnumField('sub_option', 2, DCP_SUBOPTIONS, fmt='B', depends_on=lambda p: p.option), LenField('dcp_block_length', None), ShortEnumField('block_info', 1, IP_BLOCK_INFOS), IPField('ip', '192.168.0.2'), IPField('netmask', '255.255.255.0'), IPField('gateway', '192.168.0.1'), PadField(StrLenField('padding', b'\x00', length_from=lambda p: p.dcp_block_length % 2), 1, padwith=b'\x00')]

    def extract_padding(self, s):
        if False:
            while True:
                i = 10
        return ('', s)

class DCPFullIPBlock(Packet):
    fields_desc = [ByteEnumField('option', 1, DCP_OPTIONS), MultiEnumField('sub_option', 3, DCP_SUBOPTIONS, fmt='B', depends_on=lambda p: p.option), LenField('dcp_block_length', None), ShortEnumField('block_info', 1, IP_BLOCK_INFOS), IPField('ip', '192.168.0.2'), IPField('netmask', '255.255.255.0'), IPField('gateway', '192.168.0.1'), FieldListField('dnsaddr', [], IPField('', '0.0.0.0'), count_from=lambda x: 4), PadField(StrLenField('padding', b'\x00', length_from=lambda p: p.dcp_block_length % 2), 1, padwith=b'\x00')]

    def extract_padding(self, s):
        if False:
            print('Hello World!')
        return ('', s)

class DCPMACBlock(Packet):
    fields_desc = [ByteEnumField('option', 1, DCP_OPTIONS), MultiEnumField('sub_option', 1, DCP_SUBOPTIONS, fmt='B', depends_on=lambda p: p.option), FieldLenField('dcp_block_length', None), ShortEnumField('block_info', 0, BLOCK_INFOS), MACField('mac', '00:00:00:00:00:00'), PadField(StrLenField('padding', b'\x00', length_from=lambda p: p.dcp_block_length % 2), 1, padwith=b'\x00')]

    def extract_padding(self, s):
        if False:
            for i in range(10):
                print('nop')
        return ('', s)

class DCPManufacturerSpecificBlock(Packet):
    fields_desc = [ByteEnumField('option', 2, DCP_OPTIONS), MultiEnumField('sub_option', 1, DCP_SUBOPTIONS, fmt='B', depends_on=lambda p: p.option), FieldLenField('dcp_block_length', None), ShortEnumField('block_info', 0, BLOCK_INFOS), StrLenField('device_vendor_value', 'et200sp', length_from=lambda x: x.dcp_block_length - 2), PadField(StrLenField('padding', b'\x00', length_from=lambda p: p.dcp_block_length % 2), 1, padwith=b'\x00')]

    def extract_padding(self, s):
        if False:
            i = 10
            return i + 15
        return ('', s)

class DCPNameOfStationBlock(Packet):
    fields_desc = [ByteEnumField('option', 2, DCP_OPTIONS), MultiEnumField('sub_option', 2, DCP_SUBOPTIONS, fmt='B', depends_on=lambda p: p.option), FieldLenField('dcp_block_length', None, length_of='name_of_station', adjust=lambda p, x: x + 2), ShortEnumField('block_info', 0, BLOCK_INFOS), StrLenField('name_of_station', 'et200sp', length_from=lambda x: x.dcp_block_length - 2), PadField(StrLenField('padding', b'\x00', length_from=lambda p: p.dcp_block_length % 2), 1, padwith=b'\x00')]

    def extract_padding(self, s):
        if False:
            print('Hello World!')
        return ('', s)

class DCPDeviceIDBlock(Packet):
    fields_desc = [ByteEnumField('option', 2, DCP_OPTIONS), MultiEnumField('sub_option', 3, DCP_SUBOPTIONS, fmt='B', depends_on=lambda p: p.option), LenField('dcp_block_length', None), ShortEnumField('block_info', 0, BLOCK_INFOS), XShortField('vendor_id', 42), XShortField('device_id', 787), PadField(StrLenField('padding', b'\x00', length_from=lambda p: p.dcp_block_length % 2), 1, padwith=b'\x00')]

    def extract_padding(self, s):
        if False:
            i = 10
            return i + 15
        return ('', s)

class DCPDeviceRoleBlock(Packet):
    fields_desc = [ByteEnumField('option', 2, DCP_OPTIONS), MultiEnumField('sub_option', 4, DCP_SUBOPTIONS, fmt='B', depends_on=lambda p: p.option), LenField('dcp_block_length', 4), ShortEnumField('block_info', 0, BLOCK_INFOS), ByteEnumField('device_role_details', 1, DCP_DEVICE_ROLES), XByteField('reserved', 0), PadField(StrLenField('padding', b'\x00', length_from=lambda p: p.dcp_block_length % 2), 1, padwith=b'\x00')]

    def extract_padding(self, s):
        if False:
            return 10
        return ('', s)

class DeviceOption(Packet):
    fields_desc = [ByteEnumField('option', 2, DCP_OPTIONS), MultiEnumField('sub_option', 5, DCP_SUBOPTIONS, fmt='B', depends_on=lambda p: p.option)]

    def extract_padding(self, s):
        if False:
            i = 10
            return i + 15
        return ('', s)

class DCPDeviceOptionsBlock(Packet):
    fields_desc = [ByteEnumField('option', 2, DCP_OPTIONS), MultiEnumField('sub_option', 5, DCP_SUBOPTIONS, fmt='B', depends_on=lambda p: p.option), LenField('dcp_block_length', None), ShortEnumField('block_info', 0, BLOCK_INFOS), PacketListField('device_options', [], DeviceOption, length_from=lambda p: p.dcp_block_length - 2), PadField(StrLenField('padding', b'\x00', length_from=lambda p: p.dcp_block_length % 2), 1, padwith=b'\x00')]

    def extract_padding(self, s):
        if False:
            print('Hello World!')
        return ('', s)

class DCPAliasNameBlock(Packet):
    fields_desc = [ByteEnumField('option', 2, DCP_OPTIONS), MultiEnumField('sub_option', 6, DCP_SUBOPTIONS, fmt='B', depends_on=lambda p: p.option), FieldLenField('dcp_block_length', None, length_of='alias_name', adjust=lambda p, x: x + 2), ShortEnumField('block_info', 0, BLOCK_INFOS), StrLenField('alias_name', 'et200sp', length_from=lambda x: x.dcp_block_length - 2), PadField(StrLenField('padding', b'\x00', length_from=lambda p: p.dcp_block_length % 2), 1, padwith=b'\x00')]

    def extract_padding(self, s):
        if False:
            print('Hello World!')
        return ('', s)

class DCPDeviceInstanceBlock(Packet):
    fields_desc = [ByteEnumField('option', 2, DCP_OPTIONS), MultiEnumField('sub_option', 7, DCP_SUBOPTIONS, fmt='B', depends_on=lambda p: p.option), LenField('dcp_block_length', 4), ShortEnumField('block_info', 0, BLOCK_INFOS), XByteField('device_instance_high', 0), XByteField('device_instance_low', 1), PadField(StrLenField('padding', b'\x00', length_from=lambda p: p.dcp_block_length % 2), 1, padwith=b'\x00')]

    def extract_padding(self, s):
        if False:
            while True:
                i = 10
        return ('', s)

class DCPOEMIDBlock(Packet):
    fields_desc = [ByteEnumField('option', 2, DCP_OPTIONS), MultiEnumField('sub_option', 8, DCP_SUBOPTIONS, fmt='B', depends_on=lambda p: p.option), LenField('dcp_block_length', None), ShortEnumField('block_info', 0, BLOCK_INFOS), XShortField('vendor_id', 42), XShortField('device_id', 787), PadField(StrLenField('padding', b'\x00', length_from=lambda p: p.dcp_block_length % 2), 1, padwith=b'\x00')]

    def extract_padding(self, s):
        if False:
            return 10
        return ('', s)

class DCPControlBlock(Packet):
    fields_desc = [ByteEnumField('option', 5, DCP_OPTIONS), MultiEnumField('sub_option', 4, DCP_SUBOPTIONS, fmt='B', depends_on=lambda p: p.option), LenField('dcp_block_length', 3), ByteEnumField('response', 2, DCP_OPTIONS), MultiEnumField('response_sub_option', 2, DCP_SUBOPTIONS, fmt='B', depends_on=lambda p: p.option), ByteEnumField('block_error', 0, BLOCK_ERRORS), PadField(StrLenField('padding', b'\x00', length_from=lambda p: p.dcp_block_length % 2), 1, padwith=b'\x00')]

    def extract_padding(self, s):
        if False:
            print('Hello World!')
        return ('', s)

class DCPDeviceInitiativeBlock(Packet):
    """
        device initiative DCP block
    """
    fields_desc = [ByteEnumField('option', 6, DCP_OPTIONS), MultiEnumField('sub_option', 1, DCP_SUBOPTIONS, fmt='B', depends_on=lambda p: p.option), FieldLenField('dcp_block_length', None, length_of='device_initiative'), ShortEnumField('block_info', 0, BLOCK_INFOS), ShortField('device_initiative', 1)]

    def extract_padding(self, s):
        if False:
            for i in range(10):
                print('nop')
        return ('', s)

def guess_dcp_block_class(packet, **kargs):
    if False:
        print('Hello World!')
    '\n    returns the correct dcp block class needed to dissect the current tag\n    if nothing can be found -> dcp base block will be used\n\n    :param packet: the current packet\n    :return: dcp block class\n    '
    option = orb(packet[0])
    suboption = orb(packet[1])
    class_switch_case = {1: {1: 'DCPMACBlock', 2: 'DCPIPBlock'}, 2: {1: 'DCPManufacturerSpecificBlock', 2: 'DCPNameOfStationBlock', 3: 'DCPDeviceIDBlock', 4: 'DCPDeviceRoleBlock', 5: 'DCPDeviceOptionsBlock', 6: 'DCPAliasNameBlock', 7: 'DCPDeviceInstanceBlock', 8: 'DCPOEMIDBlock'}, 3: {12: 'Host name', 43: 'Vendor specific', 54: 'Server identifier', 55: 'Parameter request list', 60: 'Class identifier', 61: 'DHCP client identifier', 81: 'FQDN, Fully Qualified Domain Name', 97: 'UUID/GUID-based Client', 255: 'Control DHCP for address resolution'}, 5: {0: 'Reserved (0x00)', 1: 'Start Transaction (0x01)', 2: 'End Transaction (0x02)', 3: 'Signal (0x03)', 4: 'DCPControlBlock', 5: 'Reset Factory Settings (0x05)', 6: 'Reset to Factory (0x06)'}, 6: {0: 'Reserved (0x00)', 1: 'DCPDeviceInitiativeBlock'}, 255: {255: 'ALL Selector (0xff)'}}
    try:
        c = class_switch_case[option][suboption]
    except KeyError:
        c = 'DCPBaseBlock'
    cls = globals()[c]
    return cls(packet, **kargs)

class ProfinetDCP(Packet):
    """
    Profinet DCP Packet

    Requests are handled via ConditionalField because here only 1 Block is used
    every time.

    Response can contain 1..n Blocks, for that you have to use one ProfinetDCP
    Layer with one or multiple DCP*Block Layers::

        ProfinetDCP / DCPNameOfStationBlock / DCPDeviceIDBlock ...

    Example for a DCP Identify All Request::

        Ether(dst="01:0e:cf:00:00:00") /
        ProfinetIO(frameID=DCP_IDENTIFY_REQUEST_FRAME_ID) /
        ProfinetDCP(service_id=DCP_SERVICE_ID_IDENTIFY,
            service_type=DCP_REQUEST, option=255, sub_option=255,
            dcp_data_length=4)

    Example for a DCP Identify Response::

        Ether(dst=dst_mac) /
        ProfinetIO(frameID=DCP_IDENTIFY_RESPONSE_FRAME_ID) /
        ProfinetDCP(
            service_id=DCP_SERVICE_ID_IDENTIFY,
            service_type=DCP_RESPONSE) /
        DCPNameOfStationBlock(name_of_station="device1")

    Example for a DCP Set Request::

        Ether(dst=mac) /
        ProfinetIO(frameID=DCP_GET_SET_FRAME_ID) /
        ProfinetDCP(service_id=DCP_SERVICE_ID_SET, service_type=DCP_REQUEST,
            option=2, sub_option=2, dcp_data_length=14, dcp_block_length=10,
            name_of_station=name, reserved=0)

    """
    name = 'Profinet DCP'
    fields_desc = [ByteEnumField('service_id', 5, DCP_SERVICE_ID), ByteEnumField('service_type', 0, DCP_SERVICE_TYPE), XIntField('xid', 16777217), ShortField('reserved', 0), LenField('dcp_data_length', None), ConditionalField(ByteEnumField('option', 2, DCP_OPTIONS), lambda pkt: pkt.service_type == 0), ConditionalField(MultiEnumField('sub_option', 3, DCP_SUBOPTIONS, fmt='B', depends_on=lambda p: p.option), lambda pkt: pkt.service_type == 0), ConditionalField(LenField('dcp_block_length', 0), lambda pkt: pkt.service_type == 0), ConditionalField(ShortEnumField('block_qualifier', 1, BLOCK_QUALIFIERS), lambda pkt: pkt.service_id == 4 and pkt.service_type == 0), ConditionalField(MultipleTypeField([(StrLenField('name_of_station', 'et200sp', length_from=lambda x: x.dcp_block_length - 2), lambda pkt: pkt.service_id == 4)], StrLenField('name_of_station', 'et200sp', length_from=lambda x: x.dcp_block_length)), lambda pkt: pkt.service_type == 0 and pkt.option == 2 and (pkt.sub_option == 2)), ConditionalField(MACField('mac', '00:00:00:00:00:00'), lambda pkt: pkt.service_id == 4 and pkt.service_type == 0 and (pkt.option == 1) and (pkt.sub_option == 1)), ConditionalField(IPField('ip', '192.168.0.2'), lambda pkt: pkt.service_id == 4 and pkt.service_type == 0 and (pkt.option == 1) and (pkt.sub_option in [2, 3])), ConditionalField(IPField('netmask', '255.255.255.0'), lambda pkt: pkt.service_id == 4 and pkt.service_type == 0 and (pkt.option == 1) and (pkt.sub_option in [2, 3])), ConditionalField(IPField('gateway', '192.168.0.1'), lambda pkt: pkt.service_id == 4 and pkt.service_type == 0 and (pkt.option == 1) and (pkt.sub_option in [2, 3])), ConditionalField(FieldListField('dnsaddr', [], IPField('', '0.0.0.0'), count_from=lambda x: 4), lambda pkt: pkt.service_id == 4 and pkt.service_type == 0 and (pkt.option == 1) and (pkt.sub_option == 3)), ConditionalField(StrLenField('alias_name', 'et200sp', length_from=lambda x: x.dcp_block_length), lambda pkt: pkt.service_id == 5 and pkt.service_type == 0 and (pkt.option == 2) and (pkt.sub_option == 6)), ConditionalField(PacketListField('dcp_blocks', [], guess_dcp_block_class, length_from=lambda p: p.dcp_data_length), lambda pkt: pkt.service_type == 1)]

    def post_build(self, pkt, pay):
        if False:
            while True:
                i = 10
        padding = MIN_PACKET_LENGTH - len(pkt + pay)
        pay += b'\x00' * padding
        return Packet.post_build(self, pkt, pay)
bind_layers(ProfinetDCP, Padding)