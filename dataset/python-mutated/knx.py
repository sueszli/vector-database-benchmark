"""
KNXNet/IP

This module provides Scapy layers for KNXNet/IP communications over UDP
according to KNX specifications v2.1 / ISO-IEC 14543-3.
Specifications can be downloaded for free here :
https://my.knx.org/en/shop/knx-specifications

Currently, the module (partially) supports the following services :
* SEARCH REQUEST/RESPONSE
* DESCRIPTION REQUEST/RESPONSE
* CONNECT, DISCONNECT, CONNECTION_STATE REQUEST/RESPONSE
* CONFIGURATION REQUEST/RESPONSE
* TUNNELING REQUEST/RESPONSE
"""
import struct
from scapy.fields import PacketField, MultipleTypeField, ByteField, XByteField, ShortEnumField, ShortField, ByteEnumField, IPField, StrFixedLenField, MACField, XBitField, PacketListField, FieldLenField, StrLenField, BitEnumField, BitField, ConditionalField
from scapy.packet import Packet, bind_layers, bind_bottom_up, Padding
from scapy.layers.inet import UDP
SERVICE_IDENTIFIER_CODES = {513: 'SEARCH_REQUEST', 514: 'SEARCH_RESPONSE', 515: 'DESCRIPTION_REQUEST', 516: 'DESCRIPTION_RESPONSE', 517: 'CONNECT_REQUEST', 518: 'CONNECT_RESPONSE', 519: 'CONNECTIONSTATE_REQUEST', 520: 'CONNECTIONSTATE_RESPONSE', 521: 'DISCONNECT_REQUEST', 522: 'DISCONNECT_RESPONSE', 784: 'CONFIGURATION_REQUEST', 785: 'CONFIGURATION_ACK', 1056: 'TUNNELING_REQUEST', 1057: 'TUNNELING_ACK'}
HOST_PROTOCOL_CODES = {1: 'IPV4_UDP', 2: 'IPV4_TCP'}
DESCRIPTION_TYPE_CODES = {1: 'DEVICE_INFO', 2: 'SUPP_SVC_FAMILIES', 3: 'IP_CONFIG', 4: 'IP_CUR_CONFIG', 5: 'KNX_ADDRESSES', 6: 'Reserved', 254: 'MFR_DATA', 255: 'not used'}
CONNECTION_TYPE_CODES = {3: 'DEVICE_MANAGEMENT_CONNECTION', 4: 'TUNNEL_CONNECTION', 6: 'REMLOG_CONNECTION', 7: 'REMCONF_CONNECTION', 8: 'OBJSVR_CONNECTION'}
MESSAGE_CODES = {17: 'L_Data.req', 46: 'L_Data.con', 252: 'M_PropRead.req', 251: 'M_PropRead.con', 246: 'M_PropWrite.req', 245: 'M_PropWrite.con'}
KNX_MEDIUM_CODES = {1: 'reserved', 2: 'TP1', 4: 'PL110', 8: 'reserved', 16: 'RF', 32: 'KNX IP'}
KNX_ACPI_CODES = {0: 'GroupValueRead', 1: 'GroupValueResp', 2: 'GroupValueWrite', 3: 'IndAddrWrite', 4: 'IndAddrRead', 5: 'IndAddrResp', 6: 'AdcRead', 7: 'AdcResp'}
CEMI_OBJECT_TYPES = {0: 'DEVICE', 11: 'IP PARAMETER_OBJECT'}
CEMI_PROPERTIES = {12: 'PID_MANUFACTURER_ID', 51: 'PID_PROJECT_INSTALLATION_ID', 52: 'PID_KNX_INDIVIDUAL_ADDRESS', 53: 'PID_ADDITIONAL_INDIVIDUAL_ADDRESSES', 54: 'PID_CURRENT_IP_ASSIGNMENT_METHOD', 55: 'PID_IP_ASSIGNMENT_METHOD', 56: 'PID_IP_CAPABILITIES', 57: 'PID_CURRENT_IP_ADDRESS', 58: 'PID_CURRENT_SUBNET_MASK', 59: 'PID_CURRENT_DEFAULT_GATEWAY', 60: 'PID_IP_ADDRESS', 61: 'PID_SUBNET_MASK', 62: 'PID_DEFAULT_GATEWAY', 63: 'PID_DHCP_BOOTP_SERVER', 64: 'PID_MAC_ADDRESS', 65: 'PID_SYSTEM_SETUP_MULTICAST_ADDRESS', 66: 'PID_ROUTING_MULTICAST_ADDRESS', 67: 'PID_TTL', 68: 'PID_KNXNETIP_DEVICE_CAPABILITIES', 69: 'PID_KNXNETIP_DEVICE_STATE', 70: 'PID_KNXNETIP_ROUTING_CAPABILITIES', 71: 'PID_PRIORITY_FIFO_ENABLED', 72: 'PID_QUEUE_OVERFLOW_TO_IP', 73: 'PID_QUEUE_OVERFLOW_TO_KNX', 74: 'PID_MSG_TRANSMIT_TO_IP', 75: 'PID_MSG_TRANSMIT_TO_KNX', 76: 'PID_FRIENDLY_NAME', 78: 'PID_ROUTING_BUSY_WAIT_TIME'}

class KNXAddressField(ShortField):

    def i2repr(self, pkt, x):
        if False:
            for i in range(10):
                print('nop')
        if x is None:
            return None
        else:
            return '%d.%d.%d' % (x >> 12 & 15, x >> 8 & 15, x & 255)

    def any2i(self, pkt, x):
        if False:
            print('Hello World!')
        if isinstance(x, str):
            try:
                (a, b, c) = map(int, x.split('.'))
                x = a << 12 | b << 8 | c
            except ValueError:
                raise ValueError(x)
        return ShortField.any2i(self, pkt, x)

class KNXGroupField(ShortField):

    def i2repr(self, pkt, x):
        if False:
            while True:
                i = 10
        return '%d/%d/%d' % (x >> 11 & 31, x >> 8 & 7, x & 255)

    def any2i(self, pkt, x):
        if False:
            print('Hello World!')
        if isinstance(x, str):
            try:
                (a, b, c) = map(int, x.split('/'))
                x = a << 11 | b << 8 | c
            except ValueError:
                raise ValueError(x)
        return ShortField.any2i(self, pkt, x)

class HPAI(Packet):
    name = 'HPAI'
    fields_desc = [ByteField('structure_length', None), ByteEnumField('host_protocol', 1, HOST_PROTOCOL_CODES), IPField('ip_address', None), ShortField('port', None)]

    def post_build(self, p, pay):
        if False:
            for i in range(10):
                print('nop')
        if self.structure_length is None:
            p = struct.pack('!B', len(p)) + p[1:]
        return p + pay

class ServiceFamily(Packet):
    name = 'Service Family'
    fields_desc = [ByteField('id', None), ByteField('version', None)]

class DIBDeviceInfo(Packet):
    name = 'DIB: DEVICE_INFO'
    fields_desc = [ByteField('structure_length', None), ByteEnumField('description_type', 1, DESCRIPTION_TYPE_CODES), ByteEnumField('knx_medium', 2, KNX_MEDIUM_CODES), ByteField('device_status', None), KNXAddressField('knx_address', None), ShortField('project_installation_identifier', None), XBitField('device_serial_number', None, 48), IPField('device_multicast_address', None), MACField('device_mac_address', None), StrFixedLenField('device_friendly_name', None, 30)]

    def post_build(self, p, pay):
        if False:
            i = 10
            return i + 15
        if self.structure_length is None:
            p = struct.pack('!B', len(p)) + p[1:]
        return p + pay

class DIBSuppSvcFamilies(Packet):
    name = 'DIB: SUPP_SVC_FAMILIES'
    fields_desc = [ByteField('structure_length', 2), ByteEnumField('description_type', 2, DESCRIPTION_TYPE_CODES), ConditionalField(PacketListField('service_family', ServiceFamily(), ServiceFamily, length_from=lambda pkt: pkt.structure_length - 2), lambda pkt: pkt.structure_length > 2)]

    def post_build(self, p, pay):
        if False:
            print('Hello World!')
        if self.structure_length is None:
            p = struct.pack('!B', len(p)) + p[1:]
        return p + pay

class TunnelingConnection(Packet):
    name = 'Tunneling Connection'
    fields_desc = [ByteField('knx_layer', 2), ByteField('reserved', None)]

class CRDTunnelingConnection(Packet):
    name = 'CRD Tunneling Connection'
    fields_desc = [KNXAddressField('knx_individual_address', None)]

class CRI(Packet):
    name = 'CRI (Connection Request Information)'
    fields_desc = [ByteField('structure_length', 2), ByteEnumField('connection_type', 3, CONNECTION_TYPE_CODES), ConditionalField(PacketField('connection_data', TunnelingConnection(), TunnelingConnection), lambda pkt: pkt.connection_type == 4)]

    def post_build(self, p, pay):
        if False:
            i = 10
            return i + 15
        if self.structure_length is None:
            p = struct.pack('!B', len(p)) + p[1:]
        return p + pay

class CRD(Packet):
    name = 'CRD (Connection Response Data)'
    fields_desc = [ByteField('structure_length', 0), ByteEnumField('connection_type', 3, CONNECTION_TYPE_CODES), ConditionalField(PacketField('connection_data', CRDTunnelingConnection(), CRDTunnelingConnection), lambda pkt: pkt.connection_type == 4)]

    def post_build(self, p, pay):
        if False:
            while True:
                i = 10
        if self.structure_length is None:
            p = struct.pack('!B', len(p)) + p[1:]
        return p + pay

class LcEMI(Packet):
    name = 'L_cEMI'
    fields_desc = [FieldLenField('additional_information_length', 0, fmt='B', length_of='additional_information'), StrLenField('additional_information', None, length_from=lambda pkt: pkt.additional_information_length), BitEnumField('frame_type', 1, 1, {1: 'standard'}), BitField('reserved_1', 0, 1), BitField('repeat_on_error', 1, 1), BitEnumField('broadcast_type', 1, 1, {1: 'domain'}), BitEnumField('priority', 3, 2, {3: 'low'}), BitField('ack_request', 0, 1), BitField('confirmation_error', 0, 1), BitEnumField('address_type', 1, 1, {1: 'group'}), BitField('hop_count', 6, 3), BitField('extended_frame_format', 0, 4), KNXAddressField('source_address', None), KNXGroupField('destination_address', '1/2/3'), FieldLenField('npdu_length', 1, fmt='B', length_of='data'), BitEnumField('packet_type', 0, 1, {0: 'data'}), BitEnumField('sequence_type', 0, 1, {0: 'unnumbered'}), BitField('reserved_2', 0, 4), BitEnumField('acpi', 2, 4, KNX_ACPI_CODES), BitField('data', 0, 6)]

class DPcEMI(Packet):
    name = 'DP_cEMI'
    fields_desc = [ShortField('object_type', None), ByteField('object_instance', 1), ByteField('property_id', None), BitField('number_of_elements', 1, 4), BitField('start_index', None, 12)]

class CEMI(Packet):
    name = 'CEMI'
    fields_desc = [ByteEnumField('message_code', None, MESSAGE_CODES), MultipleTypeField([(PacketField('cemi_data', LcEMI(), LcEMI), lambda pkt: pkt.message_code == 17), (PacketField('cemi_data', LcEMI(), LcEMI), lambda pkt: pkt.message_code == 46), (PacketField('cemi_data', DPcEMI(), DPcEMI), lambda pkt: pkt.message_code == 252), (PacketField('cemi_data', DPcEMI(), DPcEMI), lambda pkt: pkt.message_code == 251), (PacketField('cemi_data', DPcEMI(), DPcEMI), lambda pkt: pkt.message_code == 246), (PacketField('cemi_data', DPcEMI(), DPcEMI), lambda pkt: pkt.message_code == 245)], PacketField('cemi_data', LcEMI(), LcEMI))]

class KNXSearchRequest(Packet):
    name = ('SEARCH_REQUEST',)
    fields_desc = [PacketField('discovery_endpoint', HPAI(), HPAI)]

class KNXSearchResponse(Packet):
    name = ('SEARCH_RESPONSE',)
    fields_desc = [PacketField('control_endpoint', HPAI(), HPAI), PacketField('device_info', DIBDeviceInfo(), DIBDeviceInfo), PacketField('supported_service_families', DIBSuppSvcFamilies(), DIBSuppSvcFamilies)]

class KNXDescriptionRequest(Packet):
    name = 'DESCRIPTION_REQUEST'
    fields_desc = [PacketField('control_endpoint', HPAI(), HPAI)]

class KNXDescriptionResponse(Packet):
    name = 'DESCRIPTION_RESPONSE'
    fields_desc = [PacketField('device_info', DIBDeviceInfo(), DIBDeviceInfo), PacketField('supported_service_families', DIBSuppSvcFamilies(), DIBSuppSvcFamilies)]

class KNXConnectRequest(Packet):
    name = 'CONNECT_REQUEST'
    fields_desc = [PacketField('control_endpoint', HPAI(), HPAI), PacketField('data_endpoint', HPAI(), HPAI), PacketField('connection_request_information', CRI(), CRI)]

class KNXConnectResponse(Packet):
    name = 'CONNECT_RESPONSE'
    fields_desc = [ByteField('communication_channel_id', None), ByteField('status', None), PacketField('data_endpoint', HPAI(), HPAI), PacketField('connection_response_data_block', CRD(), CRD)]

class KNXConnectionstateRequest(Packet):
    name = 'CONNECTIONSTATE_REQUEST'
    fields_desc = [ByteField('communication_channel_id', None), ByteField('reserved', None), PacketField('control_endpoint', HPAI(), HPAI)]

class KNXConnectionstateResponse(Packet):
    name = 'CONNECTIONSTATE_RESPONSE'
    fields_desc = [ByteField('communication_channel_id', None), ByteField('status', 0)]

class KNXDisconnectRequest(Packet):
    name = 'DISCONNECT_REQUEST'
    fields_desc = [ByteField('communication_channel_id', 1), ByteField('reserved', None), PacketField('control_endpoint', HPAI(), HPAI)]

class KNXDisconnectResponse(Packet):
    name = 'DISCONNECT_RESPONSE'
    fields_desc = [ByteField('communication_channel_id', None), ByteField('status', 0)]

class KNXConfigurationRequest(Packet):
    name = 'CONFIGURATION_REQUEST'
    fields_desc = [ByteField('structure_length', 4), ByteField('communication_channel_id', 1), ByteField('sequence_counter', None), ByteField('reserved', None), PacketField('cemi', CEMI(), CEMI)]

    def post_build(self, p, pay):
        if False:
            print('Hello World!')
        if self.structure_length is None:
            p = struct.pack('!B', len(p[:4])) + p[1:]
        return p + pay

class KNXConfigurationACK(Packet):
    name = 'CONFIGURATION_ACK'
    fields_desc = [ByteField('structure_length', None), ByteField('communication_channel_id', 1), ByteField('sequence_counter', None), ByteField('status', None)]

    def post_build(self, p, pay):
        if False:
            print('Hello World!')
        if self.structure_length is None:
            p = struct.pack('!B', len(p)) + p[1:]
        return p + pay

class KNXTunnelingRequest(Packet):
    name = 'TUNNELING_REQUEST'
    fields_desc = [ByteField('structure_length', 4), ByteField('communication_channel_id', 1), ByteField('sequence_counter', None), ByteField('reserved', None), PacketField('cemi', CEMI(), CEMI)]

    def post_build(self, p, pay):
        if False:
            i = 10
            return i + 15
        if self.structure_length is None:
            p = struct.pack('!B', len(p[:4])) + p[1:]
        return p + pay

class KNXTunnelingACK(Packet):
    name = 'TUNNELING_ACK'
    fields_desc = [ByteField('structure_length', None), ByteField('communication_channel_id', 1), ByteField('sequence_counter', None), ByteField('status', None)]

    def post_build(self, p, pay):
        if False:
            for i in range(10):
                print('nop')
        if self.structure_length is None:
            p = struct.pack('!B', len(p)) + p[1:]
        return p + pay

class KNX(Packet):
    name = 'KNXnet/IP'
    fields_desc = [ByteField('header_length', None), XByteField('protocol_version', 16), ShortEnumField('service_identifier', None, SERVICE_IDENTIFIER_CODES), ShortField('total_length', None)]

    def post_build(self, p, pay):
        if False:
            return 10
        if self.header_length is None:
            p = struct.pack('!B', len(p)) + p[1:]
        if self.total_length is None:
            p = p[:-2] + struct.pack('!H', len(p) + len(pay))
        return p + pay
bind_bottom_up(UDP, KNX, dport=3671)
bind_bottom_up(UDP, KNX, sport=3671)
bind_layers(UDP, KNX, sport=3671, dport=3671)
bind_layers(KNX, KNXSearchRequest, service_identifier=513)
bind_layers(KNX, KNXSearchResponse, service_identifier=514)
bind_layers(KNX, KNXDescriptionRequest, service_identifier=515)
bind_layers(KNX, KNXDescriptionResponse, service_identifier=516)
bind_layers(KNX, KNXConnectRequest, service_identifier=517)
bind_layers(KNX, KNXConnectResponse, service_identifier=518)
bind_layers(KNX, KNXConnectionstateRequest, service_identifier=519)
bind_layers(KNX, KNXConnectionstateResponse, service_identifier=520)
bind_layers(KNX, KNXDisconnectResponse, service_identifier=522)
bind_layers(KNX, KNXDisconnectRequest, service_identifier=521)
bind_layers(KNX, KNXConfigurationRequest, service_identifier=784)
bind_layers(KNX, KNXConfigurationACK, service_identifier=785)
bind_layers(KNX, KNXTunnelingRequest, service_identifier=1056)
bind_layers(KNX, KNXTunnelingACK, service_identifier=1057)
bind_layers(HPAI, Padding)
bind_layers(ServiceFamily, Padding)
bind_layers(DIBDeviceInfo, Padding)
bind_layers(DIBSuppSvcFamilies, Padding)
bind_layers(TunnelingConnection, Padding)
bind_layers(CRDTunnelingConnection, Padding)
bind_layers(CRI, Padding)
bind_layers(CRD, Padding)
bind_layers(LcEMI, Padding)
bind_layers(DPcEMI, Padding)
bind_layers(CEMI, Padding)
bind_layers(KNXSearchRequest, Padding)
bind_layers(KNXSearchResponse, Padding)
bind_layers(KNXDescriptionRequest, Padding)
bind_layers(KNXDescriptionResponse, Padding)
bind_layers(KNXConnectRequest, Padding)
bind_layers(KNXConnectResponse, Padding)
bind_layers(KNXConnectionstateRequest, Padding)
bind_layers(KNXConnectionstateResponse, Padding)
bind_layers(KNXDisconnectRequest, Padding)
bind_layers(KNXDisconnectResponse, Padding)
bind_layers(KNXConfigurationRequest, Padding)
bind_layers(KNXConfigurationACK, Padding)
bind_layers(KNXTunnelingRequest, Padding)
bind_layers(KNXTunnelingACK, Padding)