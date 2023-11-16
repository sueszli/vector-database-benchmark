"""
ZigBee bindings for IEEE 802.15.4.
"""
import struct
from scapy.compat import orb
from scapy.packet import bind_layers, bind_bottom_up, Packet
from scapy.fields import BitField, ByteField, XLEIntField, ConditionalField, ByteEnumField, EnumField, BitEnumField, FieldListField, FlagsField, IntField, PacketListField, ShortField, StrField, StrFixedLenField, StrLenField, XLEShortField, XStrField
from scapy.layers.dot15d4 import dot15d4AddressField, Dot15d4Beacon, Dot15d4, Dot15d4FCS
from scapy.layers.inet import UDP
from scapy.layers.ntp import TimeStampField
_aps_profile_identifiers = {0: 'Zigbee_Device_Profile', 257: 'IPM_Industrial_Plant_Monitoring', 260: 'HA_Home_Automation', 261: 'CBA_Commercial_Building_Automation', 263: 'TA_Telecom_Applications', 264: 'HC_Health_Care', 265: 'SE_Smart_Energy_Profile'}
_zcl_cluster_identifier = {0: 'basic', 1: 'power_configuration', 2: 'device_temperature_configuration', 3: 'identify', 4: 'groups', 5: 'scenes', 6: 'on_off', 7: 'on_off_switch_configuration', 8: 'level_control', 9: 'alarms', 10: 'time', 11: 'rssi_location', 12: 'analog_input', 13: 'analog_output', 14: 'analog_value', 15: 'binary_input', 16: 'binary_output', 17: 'binary_value', 18: 'multistate_input', 19: 'multistate_output', 20: 'multistate_value', 21: 'commissioning', 256: 'shade_configuration', 512: 'pump_configuration_and_control', 513: 'thermostat', 514: 'fan_control', 515: 'dehumidification_control', 516: 'thermostat_user_interface_configuration', 768: 'color_control', 769: 'ballast_configuration', 1024: 'illuminance_measurement', 1025: 'illuminance_level_sensing', 1026: 'temperature_measurement', 1027: 'pressure_measurement', 1028: 'flow_measurement', 1029: 'relative_humidity_measurement', 1030: 'occupancy_sensing', 1280: 'ias_zone', 1281: 'ias_ace', 1282: 'ias_wd', 1536: 'generic_tunnel', 1537: 'bacnet_protocol_tunnel', 1538: 'analog_input_regular', 1539: 'analog_input_extended', 1540: 'analog_output_regular', 1541: 'analog_output_extended', 1542: 'analog_value_regular', 1543: 'analog_value_extended', 1544: 'binary_input_regular', 1545: 'binary_input_extended', 1546: 'binary_output_regular', 1547: 'binary_output_extended', 1548: 'binary_value_regular', 1549: 'binary_value_extended', 1550: 'multistate_input_regular', 1551: 'multistate_input_extended', 1552: 'multistate_output_regular', 1553: 'multistate_output_extended', 1554: 'multistate_value_regular', 1555: 'multistate_value', 1792: 'price', 1793: 'demand_response_and_load_control', 1794: 'metering', 1795: 'messaging', 1796: 'smart_energy_tunneling', 1797: 'prepayment', 2048: 'key_establishment'}
_zcl_command_frames = {0: 'read_attributes', 1: 'read_attributes_response', 2: 'write_attributes', 3: 'write_attributes_undivided', 4: 'write_attributes_response', 5: 'write_attributes_no_response', 6: 'configure_reporting', 7: 'configure_reporting_response', 8: 'read_reporting_configuration', 9: 'read_reporting_configuration_response', 10: 'report_attributes', 11: 'default_response', 12: 'discover_attributes', 13: 'discover_attributes_response', 14: 'read_attributes_structured', 15: 'write_attributes_structured', 16: 'write_attributes_structured_response', 17: 'discover_commands_received', 18: 'discover_commands_received_response', 19: 'discover_commands_generated', 20: 'discover_commands_generated_response', 21: 'discover_attributes_extended', 22: 'discover_attributes_extended_response'}
_zcl_enumerated_status_values = {0: 'SUCCESS', 1: 'FAILURE', 126: 'NOT_AUTHORIZED', 127: 'RESERVED_FIELD_NOT_ZERO', 128: 'MALFORMED_COMMAND', 129: 'UNSUP_CLUSTER_COMMAND', 130: 'UNSUP_GENERAL_COMMAND', 131: 'UNSUP_MANUF_CLUSTER_COMMAND', 132: 'UNSUP_MANUF_GENERAL_COMMAND', 133: 'INVALID_FIELD', 134: 'UNSUPPORTED_ATTRIBUTE', 135: 'INVALID_VALUE', 136: 'READ_ONLY', 137: 'INSUFFICIENT_SPACE', 138: 'DUPLICATE_EXISTS', 139: 'NOT_FOUND', 140: 'UNREPORTABLE_ATTRIBUTE', 141: 'INVALID_DATA_TYPE', 142: 'INVALID_SELECTOR', 143: 'WRITE_ONLY', 144: 'INCONSISTENT_STARTUP_STATE', 145: 'DEFINED_OUT_OF_BAND', 146: 'INCONSISTENT', 147: 'ACTION_DENIED', 148: 'TIMEOUT', 149: 'ABORT', 150: 'INVALID_IMAGE', 151: 'WAIT_FOR_DATA', 152: 'NO_IMAGE_AVAILABLE', 153: 'REQUIRE_MORE_IMAGE', 154: 'NOTIFICATION_PENDING', 192: 'HARDWARE_FAILURE', 193: 'SOFTWARE_FAILURE', 194: 'CALIBRATION_ERROR', 195: 'UNSUPPORTED_CLUSTER'}
_zcl_attribute_data_types = {0: 'no_data', 8: '8-bit_data', 9: '16-bit_data', 10: '24-bit_data', 11: '32-bit_data', 12: '40-bit_data', 13: '48-bit_data', 14: '56-bit_data', 15: '64-bit_data', 16: 'boolean', 24: '8-bit_bitmap', 25: '16-bit_bitmap', 26: '24-bit_bitmap', 27: '32-bit_bitmap', 28: '40-bit_bitmap', 29: '48-bit_bitmap', 30: '56-bit_bitmap', 31: '64-bit_bitmap', 32: 'Unsigned_8-bit_integer', 33: 'Unsigned_16-bit_integer', 34: 'Unsigned_24-bit_integer', 35: 'Unsigned_32-bit_integer', 36: 'Unsigned_40-bit_integer', 37: 'Unsigned_48-bit_integer', 38: 'Unsigned_56-bit_integer', 39: 'Unsigned_64-bit_integer', 40: 'Signed_8-bit_integer', 41: 'Signed_16-bit_integer', 42: 'Signed_24-bit_integer', 43: 'Signed_32-bit_integer', 44: 'Signed_40-bit_integer', 45: 'Signed_48-bit_integer', 46: 'Signed_56-bit_integer', 47: 'Signed_64-bit_integer', 48: '8-bit_enumeration', 49: '16-bit_enumeration', 56: 'semi_precision', 57: 'single_precision', 58: 'double_precision', 65: 'octet-string', 66: 'character_string', 67: 'long_octet_string', 68: 'long_character_string', 72: 'array', 76: 'structure', 80: 'set', 81: 'bag', 224: 'time_of_day', 225: 'date', 226: 'utc_time', 232: 'cluster_id', 233: 'attribute_id', 234: 'bacnet_oid', 240: 'ieee_address', 241: '128-bit_security_key', 255: 'unknown'}
_zcl_ias_zone_enroll_response_codes = {0: 'Success', 1: 'Not supported', 2: 'No enroll permit', 3: 'Too many zones'}
_zcl_ias_zone_zone_types = {0: 'Standard CIE', 13: 'Motion sensor', 21: 'Contact switch', 40: 'Fire sensor', 42: 'Water sensor', 43: 'Carbon Monoxide (CO) sensor', 44: 'Personal emergency device', 45: 'Vibration/Movement sensor', 271: 'Remote Control', 277: 'Key fob', 541: 'Keypad', 549: 'Standard Warning Device', 550: 'Glass break sensor', 553: 'Security repeater', 65535: 'Invalid Zone Type'}

class ZigbeeNWK(Packet):
    name = 'Zigbee Network Layer'
    fields_desc = [BitField('discover_route', 0, 2), BitField('proto_version', 2, 4), BitEnumField('frametype', 0, 2, {0: 'data', 1: 'command', 3: 'Inter-PAN'}), FlagsField('flags', 0, 8, ['multicast', 'security', 'source_route', 'extended_dst', 'extended_src', 'reserved1', 'reserved2', 'reserved3']), XLEShortField('destination', 0), XLEShortField('source', 0), ByteField('radius', 0), ByteField('seqnum', 1), ConditionalField(dot15d4AddressField('ext_dst', 0, adjust=lambda pkt, x: 8), lambda pkt: pkt.flags & 8), ConditionalField(dot15d4AddressField('ext_src', 0, adjust=lambda pkt, x: 8), lambda pkt: pkt.flags & 16), ConditionalField(ByteField('relay_count', 1), lambda pkt: pkt.flags & 4), ConditionalField(ByteField('relay_index', 0), lambda pkt: pkt.flags & 4), ConditionalField(FieldListField('relays', [], XLEShortField('', 0), count_from=lambda pkt: pkt.relay_count), lambda pkt: pkt.flags & 4)]

    @classmethod
    def dispatch_hook(cls, _pkt=None, *args, **kargs):
        if False:
            for i in range(10):
                print('nop')
        if _pkt and len(_pkt) >= 2:
            frametype = ord(_pkt[:1]) & 3
            if frametype == 3:
                return ZigbeeNWKStub
        return cls

    def guess_payload_class(self, payload):
        if False:
            return 10
        if self.flags.security:
            return ZigbeeSecurityHeader
        elif self.frametype == 0:
            return ZigbeeAppDataPayload
        elif self.frametype == 1:
            return ZigbeeNWKCommandPayload
        else:
            return Packet.guess_payload_class(self, payload)

class LinkStatusEntry(Packet):
    name = 'ZigBee Link Status Entry'
    fields_desc = [XLEShortField('neighbor_network_address', 0), BitField('reserved1', 0, 1), BitField('outgoing_cost', 0, 3), BitField('reserved2', 0, 1), BitField('incoming_cost', 0, 3)]

    def extract_padding(self, p):
        if False:
            for i in range(10):
                print('nop')
        return (b'', p)

class ZigbeeNWKCommandPayload(Packet):
    name = 'Zigbee Network Layer Command Payload'
    fields_desc = [ByteEnumField('cmd_identifier', 1, {1: 'route request', 2: 'route reply', 3: 'network status', 4: 'leave', 5: 'route record', 6: 'rejoin request', 7: 'rejoin response', 8: 'link status', 9: 'network report', 10: 'network update', 11: 'end device timeout request', 12: 'end device timeout response'}), ConditionalField(BitField('res1', 0, 1), lambda pkt: pkt.cmd_identifier in [1, 2]), ConditionalField(BitField('multicast', 0, 1), lambda pkt: pkt.cmd_identifier in [1, 2]), ConditionalField(BitField('dest_addr_bit', 0, 1), lambda pkt: pkt.cmd_identifier == 1), ConditionalField(BitEnumField('many_to_one', 0, 2, {0: 'not_m2one', 1: 'm2one_support_rrt', 2: 'm2one_no_support_rrt', 3: 'reserved'}), lambda pkt: pkt.cmd_identifier == 1), ConditionalField(BitField('res2', 0, 3), lambda pkt: pkt.cmd_identifier == 1), ConditionalField(BitField('responder_addr_bit', 0, 1), lambda pkt: pkt.cmd_identifier == 2), ConditionalField(BitField('originator_addr_bit', 0, 1), lambda pkt: pkt.cmd_identifier == 2), ConditionalField(BitField('res3', 0, 4), lambda pkt: pkt.cmd_identifier == 2), ConditionalField(ByteField('route_request_identifier', 0), lambda pkt: pkt.cmd_identifier in [1, 2]), ConditionalField(XLEShortField('originator_address', 0), lambda pkt: pkt.cmd_identifier == 2), ConditionalField(XLEShortField('responder_address', 0), lambda pkt: pkt.cmd_identifier == 2), ConditionalField(ByteEnumField('status_code', 0, {0: 'No route available', 1: 'Tree link failure', 2: 'Non-tree link failure', 3: 'Low battery level', 4: 'No routing capacity', 5: 'No indirect capacity', 6: 'Indirect transaction expiry', 7: 'Target device unavailable', 8: 'Target address unallocated', 9: 'Parent link failure', 10: 'Validate route', 11: 'Source route failure', 12: 'Many-to-one route failure', 13: 'Address conflict', 14: 'Verify addresses', 15: 'PAN identifier update', 16: 'Network address update', 17: 'Bad frame counter', 18: 'Bad key sequence number'}), lambda pkt: pkt.cmd_identifier == 3), ConditionalField(XLEShortField('destination_address', 0), lambda pkt: pkt.cmd_identifier in [1, 3]), ConditionalField(ByteField('path_cost', 0), lambda pkt: pkt.cmd_identifier in [1, 2]), ConditionalField(dot15d4AddressField('ext_dst', 0, adjust=lambda pkt, x: 8), lambda pkt: pkt.cmd_identifier == 1 and pkt.dest_addr_bit == 1), ConditionalField(dot15d4AddressField('originator_addr', 0, adjust=lambda pkt, x: 8), lambda pkt: pkt.cmd_identifier == 2 and pkt.originator_addr_bit == 1), ConditionalField(dot15d4AddressField('responder_addr', 0, adjust=lambda pkt, x: 8), lambda pkt: pkt.cmd_identifier == 2 and pkt.responder_addr_bit == 1), ConditionalField(BitField('remove_children', 0, 1), lambda pkt: pkt.cmd_identifier == 4), ConditionalField(BitField('request', 0, 1), lambda pkt: pkt.cmd_identifier == 4), ConditionalField(BitField('rejoin', 0, 1), lambda pkt: pkt.cmd_identifier == 4), ConditionalField(BitField('res4', 0, 5), lambda pkt: pkt.cmd_identifier == 4), ConditionalField(ByteField('rr_relay_count', 0), lambda pkt: pkt.cmd_identifier == 5), ConditionalField(FieldListField('rr_relay_list', [], XLEShortField('', 0), count_from=lambda pkt: pkt.rr_relay_count), lambda pkt: pkt.cmd_identifier == 5), ConditionalField(BitField('allocate_address', 0, 1), lambda pkt: pkt.cmd_identifier == 6), ConditionalField(BitField('security_capability', 0, 1), lambda pkt: pkt.cmd_identifier == 6), ConditionalField(BitField('reserved2', 0, 1), lambda pkt: pkt.cmd_identifier == 6), ConditionalField(BitField('reserved1', 0, 1), lambda pkt: pkt.cmd_identifier == 6), ConditionalField(BitField('receiver_on_when_idle', 0, 1), lambda pkt: pkt.cmd_identifier == 6), ConditionalField(BitField('power_source', 0, 1), lambda pkt: pkt.cmd_identifier == 6), ConditionalField(BitField('device_type', 0, 1), lambda pkt: pkt.cmd_identifier == 6), ConditionalField(BitField('alternate_pan_coordinator', 0, 1), lambda pkt: pkt.cmd_identifier == 6), ConditionalField(XLEShortField('network_address', 65535), lambda pkt: pkt.cmd_identifier == 7), ConditionalField(ByteField('rejoin_status', 0), lambda pkt: pkt.cmd_identifier == 7), ConditionalField(BitField('res5', 0, 1), lambda pkt: pkt.cmd_identifier == 8), ConditionalField(BitField('last_frame', 0, 1), lambda pkt: pkt.cmd_identifier == 8), ConditionalField(BitField('first_frame', 0, 1), lambda pkt: pkt.cmd_identifier == 8), ConditionalField(BitField('entry_count', 0, 5), lambda pkt: pkt.cmd_identifier == 8), ConditionalField(PacketListField('link_status_list', [], LinkStatusEntry, count_from=lambda pkt: pkt.entry_count), lambda pkt: pkt.cmd_identifier == 8), ConditionalField(BitEnumField('report_command_identifier', 0, 3, {0: 'PAN identifier conflict'}), lambda pkt: pkt.cmd_identifier == 9), ConditionalField(BitField('report_information_count', 0, 5), lambda pkt: pkt.cmd_identifier == 9), ConditionalField(BitEnumField('update_command_identifier', 0, 3, {0: 'PAN Identifier Update'}), lambda pkt: pkt.cmd_identifier == 10), ConditionalField(BitField('update_information_count', 0, 5), lambda pkt: pkt.cmd_identifier == 10), ConditionalField(dot15d4AddressField('epid', 0, adjust=lambda pkt, x: 8), lambda pkt: pkt.cmd_identifier in [9, 10]), ConditionalField(FieldListField('PAN_ID_conflict_report', [], XLEShortField('', 0), count_from=lambda pkt: pkt.report_information_count), lambda pkt: pkt.cmd_identifier == 9 and pkt.report_command_identifier == 0), ConditionalField(ByteField('update_id', 0), lambda pkt: pkt.cmd_identifier == 10), ConditionalField(XLEShortField('new_PAN_ID', 0), lambda pkt: pkt.cmd_identifier == 10 and pkt.update_command_identifier == 0), ConditionalField(ByteEnumField('req_timeout', 3, {0: '10 seconds', 1: '2 minutes', 2: '4 minutes', 3: '8 minutes', 4: '16 minutes', 5: '32 minutes', 6: '64 minutes', 7: '128 minutes', 8: '256 minutes', 9: '512 minutes', 10: '1024 minutes', 11: '2048 minutes', 12: '4096 minutes', 13: '8192 minutes', 14: '16384 minutes'}), lambda pkt: pkt.cmd_identifier == 11), ConditionalField(ByteField('ed_conf', 0), lambda pkt: pkt.cmd_identifier == 11), ConditionalField(ByteEnumField('status', 0, {0: 'Success', 1: 'Incorrect Value'}), lambda pkt: pkt.cmd_identifier == 12), ConditionalField(BitField('res6', 0, 6), lambda pkt: pkt.cmd_identifier == 12), ConditionalField(BitField('ed_timeout_req_keepalive', 0, 1), lambda pkt: pkt.cmd_identifier == 12), ConditionalField(BitField('mac_data_poll_keepalive', 0, 1), lambda pkt: pkt.cmd_identifier == 12)]

def util_mic_len(pkt):
    if False:
        print('Hello World!')
    ' Calculate the length of the attribute value field '
    if pkt.nwk_seclevel == 0:
        return 0
    elif pkt.nwk_seclevel == 1:
        return 4
    elif pkt.nwk_seclevel == 2:
        return 8
    elif pkt.nwk_seclevel == 3:
        return 16
    elif pkt.nwk_seclevel == 4:
        return 0
    elif pkt.nwk_seclevel == 5:
        return 4
    elif pkt.nwk_seclevel == 6:
        return 8
    elif pkt.nwk_seclevel == 7:
        return 16
    else:
        return 0

class ZigbeeSecurityHeader(Packet):
    name = 'Zigbee Security Header'
    fields_desc = [FlagsField('reserved1', 0, 2, ['reserved1', 'reserved2']), BitField('extended_nonce', 1, 1), BitEnumField('key_type', 1, 2, {0: 'data_key', 1: 'network_key', 2: 'key_transport_key', 3: 'key_load_key'}), BitEnumField('nwk_seclevel', 0, 3, {0: 'None', 1: 'MIC-32', 2: 'MIC-64', 3: 'MIC-128', 4: 'ENC', 5: 'ENC-MIC-32', 6: 'ENC-MIC-64', 7: 'ENC-MIC-128'}), XLEIntField('fc', 0), ConditionalField(dot15d4AddressField('source', 0, adjust=lambda pkt, x: 8), lambda pkt: pkt.extended_nonce), ConditionalField(ByteField('key_seqnum', 0), lambda pkt: pkt.getfieldval('key_type') == 1), StrField('data', ''), XStrField('mic', '')]

    def post_dissect(self, s):
        if False:
            while True:
                i = 10
        mic_length = util_mic_len(self)
        if mic_length > 0:
            (_data, _mic) = (self.data[:-mic_length], self.data[-mic_length:])
            (self.data, self.mic) = (_data, _mic)
        return s

class ZigbeeAppDataPayload(Packet):
    name = 'Zigbee Application Layer Data Payload (General APS Frame Format)'
    fields_desc = [FlagsField('frame_control', 2, 4, ['ack_format', 'security', 'ack_req', 'extended_hdr']), BitEnumField('delivery_mode', 0, 2, {0: 'unicast', 1: 'indirect', 2: 'broadcast', 3: 'group_addressing'}), BitEnumField('aps_frametype', 0, 2, {0: 'data', 1: 'command', 2: 'ack'}), ConditionalField(ByteField('dst_endpoint', 10), lambda pkt: pkt.aps_frametype == 0 and pkt.delivery_mode in [0, 2] or (pkt.aps_frametype == 2 and (not pkt.frame_control.ack_format))), ConditionalField(XLEShortField('group_addr', 0), lambda pkt: pkt.aps_frametype == 0 and pkt.delivery_mode == 3), ConditionalField(XLEShortField('cluster', 0), lambda pkt: pkt.aps_frametype == 0 or (pkt.aps_frametype == 2 and (not pkt.frame_control.ack_format))), ConditionalField(EnumField('profile', 0, _aps_profile_identifiers, fmt='<H'), lambda pkt: pkt.aps_frametype == 0 or (pkt.aps_frametype == 2 and (not pkt.frame_control.ack_format))), ConditionalField(ByteField('src_endpoint', 10), lambda pkt: pkt.aps_frametype == 0 or (pkt.aps_frametype == 2 and (not pkt.frame_control.ack_format))), ByteField('counter', 0), ConditionalField(ByteEnumField('fragmentation', 0, {0: 'none', 1: 'first_block', 2: 'middle_block'}), lambda pkt: pkt.aps_frametype in [0, 2] and pkt.frame_control.extended_hdr), ConditionalField(ByteField('block_number', 0), lambda pkt: pkt.aps_frametype in [0, 2] and pkt.fragmentation in [1, 2]), ConditionalField(ByteField('ack_bitfield', 0), lambda pkt: pkt.aps_frametype == 2 and pkt.fragmentation in [1, 2])]

    def guess_payload_class(self, payload):
        if False:
            print('Hello World!')
        if self.frame_control & 2:
            return ZigbeeSecurityHeader
        elif self.aps_frametype == 0:
            if self.profile == 0:
                return ZigbeeDeviceProfile
            else:
                return ZigbeeClusterLibrary
        elif self.aps_frametype == 1:
            return ZigbeeAppCommandPayload
        else:
            return Packet.guess_payload_class(self, payload)
_TransportKeyKeyTypes = {0: 'Trust Center Master Key', 1: 'Standard Network Key', 2: 'Application Master Key', 3: 'Application Link Key', 4: 'Trust Center Link Key', 5: 'High-Security Network Key'}
_RequestKeyKeyTypes = {2: 'Application Link Key', 4: 'Trust Center Link Key'}
_ApsStatusValues = {0: 'SUCCESS', 160: 'ASDU_TOO_LONG', 161: 'DEFRAG_DEFERRED', 162: 'DEFRAG_UNSUPPORTED', 163: 'ILLEGAL_REQUEST', 164: 'INVALID_BINDING', 165: 'INVALID_GROUP', 166: 'INVALID_PARAMETER', 167: 'NO_ACK', 168: 'NO_BOUND_DEVICE', 169: 'NO_SHORT_ADDRESS', 170: 'NOT_SUPPORTED', 171: 'SECURED_LINK_KEY', 172: 'SECURED_NWK_KEY', 173: 'SECURITY_FAIL', 174: 'TABLE_FULL', 175: 'UNSECURED', 176: 'UNSUPPORTED_ATTRIBUTE'}

class ZigbeeAppCommandPayload(Packet):
    name = 'Zigbee Application Layer Command Payload'
    fields_desc = [ByteEnumField('cmd_identifier', 1, {1: 'APS_CMD_SKKE_1', 2: 'APS_CMD_SKKE_2', 3: 'APS_CMD_SKKE_3', 4: 'APS_CMD_SKKE_4', 5: 'APS_CMD_TRANSPORT_KEY', 6: 'APS_CMD_UPDATE_DEVICE', 7: 'APS_CMD_REMOVE_DEVICE', 8: 'APS_CMD_REQUEST_KEY', 9: 'APS_CMD_SWITCH_KEY', 10: 'APS_CMD_EA_INIT_CHLNG', 11: 'APS_CMD_EA_RSP_CHLNG', 12: 'APS_CMD_EA_INIT_MAC_DATA', 13: 'APS_CMD_EA_RSP_MAC_DATA', 14: 'APS_CMD_TUNNEL', 15: 'APS_CMD_VERIFY_KEY', 16: 'APS_CMD_CONFIRM_KEY'}), ConditionalField(dot15d4AddressField('initiator', 0, adjust=lambda pkt, x: 8), lambda pkt: pkt.cmd_identifier in [1, 2, 3, 4]), ConditionalField(dot15d4AddressField('responder', 0, adjust=lambda pkt, x: 8), lambda pkt: pkt.cmd_identifier in [1, 2, 3, 4]), ConditionalField(StrFixedLenField('data', 0, length=16), lambda pkt: pkt.cmd_identifier in [1, 2, 3, 4]), ConditionalField(ByteEnumField('status', 0, _ApsStatusValues), lambda pkt: pkt.cmd_identifier == 16), ConditionalField(ByteEnumField('key_type', 0, _TransportKeyKeyTypes), lambda pkt: pkt.cmd_identifier in [5, 8, 15, 16]), ConditionalField(dot15d4AddressField('address', 0, adjust=lambda pkt, x: 8), lambda pkt: pkt.cmd_identifier in [6, 7, 15, 16]), ConditionalField(StrFixedLenField('key', None, 16), lambda pkt: pkt.cmd_identifier == 5), ConditionalField(ByteField('key_seqnum', 0), lambda pkt: pkt.cmd_identifier == 5 and pkt.key_type in [1, 5]), ConditionalField(dot15d4AddressField('dest_addr', 0, adjust=lambda pkt, x: 8), lambda pkt: pkt.cmd_identifier == 5 and pkt.key_type not in [2, 3] or pkt.cmd_identifier == 14), ConditionalField(dot15d4AddressField('src_addr', 0, adjust=lambda pkt, x: 8), lambda pkt: pkt.cmd_identifier == 5 and pkt.key_type not in [2, 3]), ConditionalField(dot15d4AddressField('partner_addr', 0, adjust=lambda pkt, x: 8), lambda pkt: pkt.cmd_identifier == 5 and pkt.key_type in [2, 3] or (pkt.cmd_identifier == 8 and pkt.key_type == 2)), ConditionalField(ByteField('initiator_flag', 0), lambda pkt: pkt.cmd_identifier == 5 and pkt.key_type in [2, 3]), ConditionalField(XLEShortField('short_address', 0), lambda pkt: pkt.cmd_identifier == 6), ConditionalField(ByteField('update_status', 0), lambda pkt: pkt.cmd_identifier == 6), ConditionalField(StrFixedLenField('seqnum', None, 8), lambda pkt: pkt.cmd_identifier == 9), ConditionalField(StrField('unimplemented', ''), lambda pkt: pkt.cmd_identifier >= 10 and pkt.cmd_identifier <= 13), ConditionalField(FlagsField('frame_control', 2, 4, ['ack_format', 'security', 'ack_req', 'extended_hdr']), lambda pkt: pkt.cmd_identifier == 14), ConditionalField(BitEnumField('delivery_mode', 0, 2, {0: 'unicast', 1: 'indirect', 2: 'broadcast', 3: 'group_addressing'}), lambda pkt: pkt.cmd_identifier == 14), ConditionalField(BitEnumField('aps_frametype', 1, 2, {0: 'data', 1: 'command', 2: 'ack'}), lambda pkt: pkt.cmd_identifier == 14), ConditionalField(ByteField('counter', 0), lambda pkt: pkt.cmd_identifier == 14), ConditionalField(StrFixedLenField('key_hash', None, 16), lambda pkt: pkt.cmd_identifier == 15)]

    def guess_payload_class(self, payload):
        if False:
            i = 10
            return i + 15
        if self.cmd_identifier == 14:
            return ZigbeeSecurityHeader
        else:
            return Packet.guess_payload_class(self, payload)

class ZigBeeBeacon(Packet):
    name = 'ZigBee Beacon Payload'
    fields_desc = [ByteField('proto_id', 0), BitField('nwkc_protocol_version', 0, 4), BitField('stack_profile', 0, 4), BitField('end_device_capacity', 0, 1), BitField('device_depth', 0, 4), BitField('router_capacity', 0, 1), BitField('reserved', 0, 2), dot15d4AddressField('extended_pan_id', 0, adjust=lambda pkt, x: 8), BitField('tx_offset', 0, 24), ByteField('update_id', 0)]

class ZigbeeNWKStub(Packet):
    name = 'Zigbee Network Layer for Inter-PAN Transmission'
    fields_desc = [BitField('res1', 0, 2), BitField('proto_version', 2, 4), BitField('frametype', 3, 2), BitField('res2', 0, 8)]

    def guess_payload_class(self, payload):
        if False:
            i = 10
            return i + 15
        if self.frametype == 3:
            return ZigbeeAppDataPayloadStub
        else:
            return Packet.guess_payload_class(self, payload)

class ZigbeeAppDataPayloadStub(Packet):
    name = 'Zigbee Application Layer Data Payload for Inter-PAN Transmission'
    fields_desc = [FlagsField('frame_control', 0, 4, ['reserved1', 'security', 'ack_req', 'extended_hdr']), BitEnumField('delivery_mode', 0, 2, {0: 'unicast', 2: 'broadcast', 3: 'group'}), BitField('frametype', 3, 2), ConditionalField(XLEShortField('group_addr', 0), lambda pkt: pkt.getfieldval('delivery_mode') == 3), XLEShortField('cluster', 0), EnumField('profile', 0, _aps_profile_identifiers, fmt='<H'), ConditionalField(StrField('data', ''), lambda pkt: pkt.frametype == 3)]

class ZDPActiveEPReq(Packet):
    name = 'ZDP Transaction Data: Active_EP_req'
    fields_desc = [XLEShortField('nwk_addr', 0)]

class ZDPDeviceAnnce(Packet):
    name = 'ZDP Transaction Data: Device_annce'
    fields_desc = [XLEShortField('nwk_addr', 0), dot15d4AddressField('ieee_addr', 0, adjust=lambda pkt, x: 8), BitField('allocate_address', 0, 1), BitField('security_capability', 0, 1), BitField('reserved2', 0, 1), BitField('reserved1', 0, 1), BitField('receiver_on_when_idle', 0, 1), BitField('power_source', 0, 1), BitField('device_type', 0, 1), BitField('alternate_pan_coordinator', 0, 1)]

class ZigbeeDeviceProfile(Packet):
    name = 'Zigbee Device Profile (ZDP) Frame'
    fields_desc = [ByteField('trans_seqnum', 0)]

    def guess_payload_class(self, payload):
        if False:
            print('Hello World!')
        if self.underlayer.cluster == 5:
            return ZDPActiveEPReq
        elif self.underlayer.cluster == 19:
            return ZDPDeviceAnnce
        return Packet.guess_payload_class(self, payload)
_ZCL_attr_length = {0: 0, 8: 1, 9: 2, 10: 3, 11: 4, 12: 5, 13: 6, 14: 7, 15: 8, 16: 1, 24: 1, 25: 2, 26: 3, 27: 4, 28: 5, 29: 6, 30: 7, 31: 8, 32: 1, 33: 2, 34: 3, 35: 4, 36: 5, 37: 6, 38: 7, 39: 8, 40: 1, 41: 2, 42: 3, 43: 4, 44: 5, 45: 6, 46: 7, 47: 8, 48: 1, 49: 2, 56: 2, 57: 4, 58: 8, 65: (1, '!B'), 66: (1, '!B'), 67: (2, '!H'), 68: (2, '!H'), 224: 4, 225: 4, 226: 4, 232: 2, 233: 2, 234: 4, 240: 8, 241: 16, 255: 0}

class _DiscreteString(StrLenField):

    def getfield(self, pkt, s):
        if False:
            print('Hello World!')
        dtype = pkt.attribute_data_type
        length = _ZCL_attr_length.get(dtype, None)
        if length is None:
            return (b'', self.m2i(pkt, s))
        elif isinstance(length, tuple):
            (size, fmt) = length
            length = struct.unpack(fmt, s[:size])[0] + size
        if isinstance(length, int):
            self.length_from = lambda x: length
            return StrLenField.getfield(self, pkt, s)
        return s

class ZCLReadAttributeStatusRecord(Packet):
    name = 'ZCL Read Attribute Status Record'
    fields_desc = [XLEShortField('attribute_identifier', 0), ByteEnumField('status', 0, _zcl_enumerated_status_values), ConditionalField(ByteEnumField('attribute_data_type', 0, _zcl_attribute_data_types), lambda pkt: pkt.status == 0), ConditionalField(_DiscreteString('attribute_value', ''), lambda pkt: pkt.status == 0)]

    def extract_padding(self, s):
        if False:
            print('Hello World!')
        return ('', s)

class ZCLWriteAttributeRecord(Packet):
    name = 'ZCL Write Attribute Record'
    fields_desc = [XLEShortField('attribute_identifier', 0), ByteEnumField('attribute_data_type', 0, _zcl_attribute_data_types), _DiscreteString('attribute_data', '')]

    def extract_padding(self, s):
        if False:
            i = 10
            return i + 15
        return ('', s)

class ZCLWriteAttributeStatusRecord(Packet):
    name = 'ZCL Write Attribute Status Record'
    fields_desc = [ByteEnumField('status', 0, _zcl_enumerated_status_values), ConditionalField(XLEShortField('attribute_identifier', 0), lambda pkt: pkt.status != 0)]

    def extract_padding(self, s):
        if False:
            i = 10
            return i + 15
        return ('', s)

class ZCLConfigureReportingRecord(Packet):
    name = 'ZCL Configure Reporting Record'
    fields_desc = [ByteField('attribute_direction', 0), XLEShortField('attribute_identifier', 0), ConditionalField(ByteEnumField('attribute_data_type', 0, _zcl_attribute_data_types), lambda pkt: pkt.attribute_direction == 0), ConditionalField(XLEShortField('min_reporting_interval', 0), lambda pkt: pkt.attribute_direction == 0), ConditionalField(XLEShortField('max_reporting_interval', 0), lambda pkt: pkt.attribute_direction == 0), ConditionalField(_DiscreteString('reportable_change', ''), lambda pkt: pkt.attribute_direction == 0), ConditionalField(XLEShortField('timeout_period', 0), lambda pkt: pkt.attribute_direction == 1)]

    def extract_padding(self, s):
        if False:
            i = 10
            return i + 15
        return ('', s)

class ZCLConfigureReportingResponseRecord(Packet):
    name = 'ZCL Configure Reporting Response Record'
    fields_desc = [ByteEnumField('status', 0, _zcl_enumerated_status_values), ConditionalField(ByteField('attribute_direction', 0), lambda pkt: pkt.status != 0), ConditionalField(XLEShortField('attribute_identifier', 0), lambda pkt: pkt.status != 0)]

    def extract_padding(self, s):
        if False:
            for i in range(10):
                print('nop')
        return ('', s)

class ZCLAttributeReport(Packet):
    name = 'ZCL Attribute Report'
    fields_desc = [XLEShortField('attribute_identifier', 0), ByteEnumField('attribute_data_type', 0, _zcl_attribute_data_types), _DiscreteString('attribute_data', '')]

    def extract_padding(self, s):
        if False:
            i = 10
            return i + 15
        return ('', s)

class ZCLGeneralReadAttributes(Packet):
    name = 'General Domain: Command Frame Payload: read_attributes'
    fields_desc = [FieldListField('attribute_identifiers', [], XLEShortField('', 0))]

class ZCLGeneralReadAttributesResponse(Packet):
    name = 'General Domain: Command Frame Payload: read_attributes_response'
    fields_desc = [PacketListField('read_attribute_status_record', [], ZCLReadAttributeStatusRecord)]

class ZCLGeneralWriteAttributes(Packet):
    name = 'General Domain: Command Frame Payload: write_attributes'
    fields_desc = [PacketListField('write_records', [], ZCLWriteAttributeRecord)]

class ZCLGeneralWriteAttributesResponse(Packet):
    name = 'General Domain: Command Frame Payload: write_attributes_response'
    fields_desc = [PacketListField('status_records', [], ZCLWriteAttributeStatusRecord)]

class ZCLGeneralConfigureReporting(Packet):
    name = 'General Domain: Command Frame Payload: configure_reporting'
    fields_desc = [PacketListField('config_records', [], ZCLConfigureReportingRecord)]

class ZCLGeneralConfigureReportingResponse(Packet):
    name = 'General Domain: Command Frame Payload: configure_reporting_response'
    fields_desc = [PacketListField('status_records', [], ZCLConfigureReportingResponseRecord)]

class ZCLGeneralReportAttributes(Packet):
    name = 'General Domain: Command Frame Payload: report_attributes'
    fields_desc = [PacketListField('attribute_reports', [], ZCLAttributeReport)]

class ZCLGeneralDefaultResponse(Packet):
    name = 'General Domain: Command Frame Payload: default_response'
    fields_desc = [ByteField('response_command_identifier', 0), ByteEnumField('status', 0, _zcl_enumerated_status_values)]

class ZCLIASZoneZoneEnrollResponse(Packet):
    name = 'IAS Zone Cluster: Zone Enroll Response Command (Server: Received)'
    fields_desc = [ByteEnumField('rsp_code', 0, _zcl_ias_zone_enroll_response_codes), ByteField('zone_id', 0)]

class ZCLIASZoneZoneStatusChangeNotification(Packet):
    name = 'IAS Zone Cluster: Zone Status Change Notification Command (Server: Generated)'
    fields_desc = [StrFixedLenField('zone_status', b'\x00\x00', length=2), StrFixedLenField('extended_status', b'\x00', length=1), ByteField('zone_id', 0), XLEShortField('delay', 0)]

class ZCLIASZoneZoneEnrollRequest(Packet):
    name = 'IAS Zone Cluster: Zone Enroll Request Command (Server: Generated)'
    fields_desc = [EnumField('zone_type', 0, _zcl_ias_zone_zone_types, fmt='<H'), XLEShortField('manuf_code', 0)]

class ZCLMeteringGetProfile(Packet):
    name = 'Metering Cluster: Get Profile Command (Server: Received)'
    fields_desc = [ByteField('Interval_Channel', 0), XLEIntField('End_Time', 0), ByteField('NumberOfPeriods', 1)]

class ZCLPriceGetCurrentPrice(Packet):
    name = 'Price Cluster: Get Current Price Command (Server: Received)'
    fields_desc = [BitField('reserved', 0, 7), BitField('Requestor_Rx_On_When_Idle', 0, 1)]

class ZCLPriceGetScheduledPrices(Packet):
    name = 'Price Cluster: Get Scheduled Prices Command (Server: Received)'
    fields_desc = [XLEIntField('start_time', 0), ByteField('number_of_events', 0)]

class ZCLPricePublishPrice(Packet):
    name = 'Price Cluster: Publish Price Command (Server: Generated)'
    fields_desc = [XLEIntField('provider_id', 0), StrLenField('rate_label', '', length_from=lambda pkt: int(pkt.rate_label[0])), XLEIntField('issuer_event_id', 0), XLEIntField('current_time', 0), ByteField('unit_of_measure', 0), XLEShortField('currency', 0), ByteField('price_trailing_digit', 0), ByteField('number_of_price_tiers', 0), XLEIntField('start_time', 0), XLEShortField('duration_in_minutes', 0), XLEIntField('price', 0), ByteField('price_ratio', 0), XLEIntField('generation_price', 0), ByteField('generation_price_ratio', 0), XLEIntField('alternate_cost_delivered', 0), ByteField('alternate_cost_unit', 0), ByteField('alternate_cost_trailing_digit', 0), ByteField('number_of_block_thresholds', 0), ByteField('price_control', 0)]

class ZigbeeClusterLibrary(Packet):
    name = 'Zigbee Cluster Library (ZCL) Frame'
    deprecated_fields = {'direction': ('command_direction', '2.5.0')}
    fields_desc = [BitField('reserved', 0, 3), BitField('disable_default_response', 0, 1), BitField('command_direction', 0, 1), BitField('manufacturer_specific', 0, 1), BitEnumField('zcl_frametype', 0, 2, {0: 'profile-wide', 1: 'cluster-specific', 2: 'reserved2', 3: 'reserved3'}), ConditionalField(XLEShortField('manufacturer_code', 0), lambda pkt: pkt.getfieldval('manufacturer_specific') == 1), ByteField('transaction_sequence', 0), ByteEnumField('command_identifier', 0, _zcl_command_frames)]

    def guess_payload_class(self, payload):
        if False:
            for i in range(10):
                print('nop')
        if self.zcl_frametype == 0:
            if self.command_identifier in {0, 1, 2, 4, 6, 7, 10, 11}:
                pass
        elif self.zcl_frametype == 1:
            if self.underlayer.cluster == 1280:
                if self.command_direction == 0:
                    if self.command_identifier == 0:
                        return ZCLIASZoneZoneEnrollResponse
                elif self.command_direction == 1:
                    if self.command_identifier == 0:
                        return ZCLIASZoneZoneStatusChangeNotification
                    elif self.command_identifier == 1:
                        return ZCLIASZoneZoneEnrollRequest
            elif self.underlayer.cluster == 1792:
                if self.command_direction == 0:
                    if self.command_identifier == 0:
                        return ZCLPriceGetCurrentPrice
                    elif self.command_identifier == 1:
                        return ZCLPriceGetScheduledPrices
                elif self.command_direction == 1:
                    if self.command_identifier == 0:
                        return ZCLPricePublishPrice
        return Packet.guess_payload_class(self, payload)
bind_layers(ZigbeeClusterLibrary, ZCLGeneralReadAttributes, zcl_frametype=0, command_identifier=0)
bind_layers(ZigbeeClusterLibrary, ZCLGeneralReadAttributesResponse, zcl_frametype=0, command_identifier=1)
bind_layers(ZigbeeClusterLibrary, ZCLGeneralWriteAttributes, zcl_frametype=0, command_identifier=2)
bind_layers(ZigbeeClusterLibrary, ZCLGeneralWriteAttributesResponse, zcl_frametype=0, command_identifier=4)
bind_layers(ZigbeeClusterLibrary, ZCLGeneralConfigureReporting, zcl_frametype=0, command_identifier=6)
bind_layers(ZigbeeClusterLibrary, ZCLGeneralConfigureReportingResponse, zcl_frametype=0, command_identifier=7)
bind_layers(ZigbeeClusterLibrary, ZCLGeneralReportAttributes, zcl_frametype=0, command_identifier=10)
bind_layers(ZigbeeClusterLibrary, ZCLGeneralDefaultResponse, zcl_frametype=0, command_identifier=11)

class ZEP2(Packet):
    name = 'Zigbee Encapsulation Protocol (V2)'
    fields_desc = [StrFixedLenField('preamble', 'EX', length=2), ByteField('ver', 0), ByteField('type', 0), ByteField('channel', 0), ShortField('device', 0), ByteField('lqi_mode', 1), ByteField('lqi_val', 0), TimeStampField('timestamp', 0), IntField('seq', 0), BitField('res', 0, 80), ByteField('length', 0)]

    @classmethod
    def dispatch_hook(cls, _pkt=b'', *args, **kargs):
        if False:
            print('Hello World!')
        if _pkt and len(_pkt) >= 4:
            v = orb(_pkt[2])
            if v == 1:
                return ZEP1
            elif v == 2:
                return ZEP2
        return cls

    def guess_payload_class(self, payload):
        if False:
            for i in range(10):
                print('nop')
        if self.lqi_mode:
            return Dot15d4
        else:
            return Dot15d4FCS

class ZEP1(ZEP2):
    name = 'Zigbee Encapsulation Protocol (V1)'
    fields_desc = [StrFixedLenField('preamble', 'EX', length=2), ByteField('ver', 0), ByteField('channel', 0), ShortField('device', 0), ByteField('lqi_mode', 0), ByteField('lqi_val', 0), BitField('res', 0, 56), ByteField('len', 0)]
bind_layers(ZigbeeAppDataPayload, ZigbeeAppCommandPayload, frametype=1)
bind_layers(Dot15d4Beacon, ZigBeeBeacon)
bind_bottom_up(UDP, ZEP2, sport=17754)
bind_bottom_up(UDP, ZEP2, sport=17754)
bind_layers(UDP, ZEP2, sport=17754, dport=17754)