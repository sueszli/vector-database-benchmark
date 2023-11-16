"""
    LLDP - Link Layer Discovery Protocol
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    :author:    Thomas Tannhaeuser, hecke@naberius.de

    :description:

        This module provides Scapy layers for the LLDP protocol.

        normative references:
            - IEEE 802.1AB 2016 - LLDP protocol, topology and MIB description

    :TODO:
        - | organization specific TLV e.g. ProfiNet
          | (see LLDPDUGenericOrganisationSpecific for a starting point)
        - Ignore everything after EndofLLDPDUTLV

    :NOTES:
        - you can find the layer configuration options at the end of this file
        - default configuration enforces standard conform:

          * | frame structure
            | (ChassisIDTLV/PortIDTLV/TimeToLiveTLV/...)
          * multiplicity of TLVs (if given by the standard)
          * min sizes of strings used by the TLVs

        - conf.contribs['LLDP'].strict_mode_disable() -> disable strict mode

"""
from scapy.config import conf
from scapy.error import Scapy_Exception
from scapy.layers.l2 import Ether, Dot1Q
from scapy.fields import MACField, IPField, IP6Field, BitField, StrLenField, ByteEnumField, BitEnumField, EnumField, ThreeBytesField, BitFieldLenField, ShortField, XStrLenField, ByteField, ConditionalField, MultipleTypeField
from scapy.packet import Packet, bind_layers
from scapy.data import ETHER_TYPES
from scapy.compat import orb
LLDP_NEAREST_BRIDGE_MAC = '01:80:c2:00:00:0e'
LLDP_NEAREST_NON_TPMR_BRIDGE_MAC = '01:80:c2:00:00:03'
LLDP_NEAREST_CUSTOMER_BRIDGE_MAC = '01:80:c2:00:00:00'
LLDP_ETHER_TYPE = 35020
ETHER_TYPES[LLDP_ETHER_TYPE] = 'LLDP'

class LLDPInvalidFrameStructure(Scapy_Exception):
    """
    basic frame structure not standard conform
    (missing TLV, invalid order or multiplicity)
    """
    pass

class LLDPMissingLowerLayer(Scapy_Exception):
    """
    first layer below first LLDPDU must be Ethernet or Dot1q
    """
    pass

class LLDPInvalidTLVCount(Scapy_Exception):
    """
    invalid number of entries for a specific TLV type
    """
    pass

class LLDPInvalidLengthField(Scapy_Exception):
    """
    invalid value of length field
    """
    pass

class LLDPDU(Packet):
    """
    base class for all LLDP data units
    """
    TYPES = {0: 'end of LLDPDU', 1: 'chassis id', 2: 'port id', 3: 'time to live', 4: 'port description', 5: 'system name', 6: 'system description', 7: 'system capabilities', 8: 'management address', 127: 'organisation specific TLV'}
    IANA_ADDRESS_FAMILY_NUMBERS = {0: 'other', 1: 'IPv4', 2: 'IPv6', 3: 'NSAP', 4: 'HDLC', 5: 'BBN', 6: '802', 7: 'E.163', 8: 'E.164', 9: 'F.69', 10: 'X.121', 11: 'IPX', 12: 'Appletalk', 13: 'Decnet IV', 14: 'Banyan Vines', 15: 'E.164 with NSAP', 16: 'DNS', 17: 'Distinguished Name', 18: 'AS Number', 19: 'XTP over IPv4', 20: 'XTP over IPv6', 21: 'XTP native mode XTP', 22: 'Fiber Channel World-Wide Port Name', 23: 'Fiber Channel World-Wide Node Name', 24: 'GWID', 25: 'AFI for L2VPN', 26: 'MPLS-TP Section Endpoint ID', 27: 'MPLS-TP LSP Endpoint ID', 28: 'MPLS-TP Pseudowire Endpoint ID', 29: 'MT IP Multi-Topology IPv4', 30: 'MT IP Multi-Topology IPv6'}
    DOT1Q_HEADER_LEN = 4
    ETHER_HEADER_LEN = 14
    ETHER_FSC_LEN = 4
    ETHER_FRAME_MIN_LEN = 64
    LAYER_STACK = []
    LAYER_MULTIPLICITIES = {}

    def guess_payload_class(self, payload):
        if False:
            for i in range(10):
                print('nop')
        try:
            lldpdu_tlv_type = orb(payload[0]) // 2
            return LLDPDU_CLASS_TYPES.get(lldpdu_tlv_type, conf.raw_layer)
        except IndexError:
            return conf.raw_layer

    @staticmethod
    def _dot1q_headers_size(layer):
        if False:
            while True:
                i = 10
        '\n        calculate size of lower dot1q layers (if present)\n        :param layer: the layer to start at\n        :return: size of vlan headers, layer below lowest vlan header\n        '
        vlan_headers_size = 0
        under_layer = layer
        while under_layer and isinstance(under_layer, Dot1Q):
            vlan_headers_size += LLDPDU.DOT1Q_HEADER_LEN
            under_layer = under_layer.underlayer
        return (vlan_headers_size, under_layer)

    def post_build(self, pkt, pay):
        if False:
            print('Hello World!')
        under_layer = self.underlayer
        if under_layer is None:
            if conf.contribs['LLDP'].strict_mode():
                raise LLDPMissingLowerLayer('No lower layer (Ethernet or Dot1Q) provided.')
            else:
                return pkt + pay
        if isinstance(under_layer, LLDPDU):
            return pkt + pay
        (frame_size, under_layer) = LLDPDU._dot1q_headers_size(under_layer)
        if not under_layer or not isinstance(under_layer, Ether):
            if conf.contribs['LLDP'].strict_mode():
                raise LLDPMissingLowerLayer('No Ethernet layer provided.')
            else:
                return pkt + pay
        frame_size += LLDPDU.ETHER_HEADER_LEN
        frame_size += len(pkt) + len(pay) + LLDPDU.ETHER_FSC_LEN
        if frame_size < LLDPDU.ETHER_FRAME_MIN_LEN:
            return pkt + pay + b'\x00' * (LLDPDU.ETHER_FRAME_MIN_LEN - frame_size)
        return pkt + pay

    @staticmethod
    def _frame_structure_check(structure_description):
        if False:
            print('Hello World!')
        '\n        check if the structure of the frame is conform to the basic\n        frame structure defined by the standard\n        :param structure_description: string-list reflecting LLDP-msg structure\n        '
        standard_frame_structure = [LLDPDUChassisID.__name__, LLDPDUPortID.__name__, LLDPDUTimeToLive.__name__, '<...>']
        if len(structure_description) < 3:
            raise LLDPInvalidFrameStructure('Invalid frame structure.\ngot: {}\nexpected: {}'.format(' '.join(structure_description), ' '.join(standard_frame_structure)))
        for (idx, layer_name) in enumerate(standard_frame_structure):
            if layer_name == '<...>':
                break
            if layer_name != structure_description[idx]:
                raise LLDPInvalidFrameStructure('Invalid frame structure.\ngot: {}\nexpected: {}'.format(' '.join(structure_description), ' '.join(standard_frame_structure)))

    @staticmethod
    def _tlv_multiplicities_check(tlv_type_count):
        if False:
            i = 10
            return i + 15
        '\n        check if multiplicity of present TLVs conforms to the standard\n        :param tlv_type_count: dict containing counte-per-TLV\n        '
        standard_multiplicities = {LLDPDUEndOfLLDPDU.__name__: '*', LLDPDUChassisID.__name__: 1, LLDPDUPortID.__name__: 1, LLDPDUTimeToLive.__name__: 1, LLDPDUPortDescription: '*', LLDPDUSystemName: '*', LLDPDUSystemDescription: '*', LLDPDUSystemCapabilities: '*', LLDPDUManagementAddress: '*'}
        for tlv_type_name in standard_multiplicities:
            standard_tlv_multiplicity = standard_multiplicities[tlv_type_name]
            if standard_tlv_multiplicity == '*':
                continue
            try:
                if tlv_type_count[tlv_type_name] != standard_tlv_multiplicity:
                    raise LLDPInvalidTLVCount('Invalid number of entries for TLV type {} - expected {} entries, got {}'.format(tlv_type_name, standard_tlv_multiplicity, tlv_type_count[tlv_type_name]))
            except KeyError:
                raise LLDPInvalidTLVCount('Missing TLV layer of type {}.'.format(tlv_type_name))

    def pre_dissect(self, s):
        if False:
            return 10
        if conf.contribs['LLDP'].strict_mode():
            if self.__class__.__name__ == 'LLDPDU':
                LLDPDU.LAYER_STACK = []
                LLDPDU.LAYER_MULTIPLICITIES = {}
            else:
                LLDPDU.LAYER_STACK.append(self.__class__.__name__)
                try:
                    LLDPDU.LAYER_MULTIPLICITIES[self.__class__.__name__] += 1
                except KeyError:
                    LLDPDU.LAYER_MULTIPLICITIES[self.__class__.__name__] = 1
        return s

    def dissection_done(self, pkt):
        if False:
            i = 10
            return i + 15
        if self.__class__.__name__ == 'LLDPDU' and conf.contribs['LLDP'].strict_mode():
            LLDPDU._frame_structure_check(LLDPDU.LAYER_STACK)
            LLDPDU._tlv_multiplicities_check(LLDPDU.LAYER_MULTIPLICITIES)
        super(LLDPDU, self).dissection_done(pkt)

    def _check(self):
        if False:
            while True:
                i = 10
        'Overwritten by LLDPU objects'
        pass

    def post_dissect(self, s):
        if False:
            while True:
                i = 10
        self._check()
        return super(LLDPDU, self).post_dissect(s)

    def do_build(self):
        if False:
            print('Hello World!')
        self._check()
        return super(LLDPDU, self).do_build()

def _ldp_id_adjustlen(pkt, x):
    if False:
        print('Hello World!')
    'Return the length of the `id` field,\n    according to its real encoded type'
    (f, v) = pkt.getfield_and_val('id')
    length = f.i2len(pkt, v) + 1
    if isinstance(pkt, LLDPDUPortID) and pkt.subtype == 4 or (isinstance(pkt, LLDPDUChassisID) and pkt.subtype == 5):
        length += 1
    return length

def _ldp_id_lengthfrom(pkt):
    if False:
        for i in range(10):
            print('nop')
    length = pkt._length
    if length is None:
        return 0
    length -= 1
    if isinstance(pkt, LLDPDUPortID) and pkt.subtype == 4 or (isinstance(pkt, LLDPDUChassisID) and pkt.subtype == 5):
        length -= 1
    return length

class LLDPDUChassisID(LLDPDU):
    """
        ieee 802.1ab-2016 - sec. 8.5.2 / p. 26
    """
    LLDP_CHASSIS_ID_TLV_SUBTYPES = {0: 'reserved', 1: 'chassis component', 2: 'interface alias', 3: 'port component', 4: 'MAC address', 5: 'network address', 6: 'interface name', 7: 'locally assigned'}
    SUBTYPE_RESERVED = 0
    SUBTYPE_CHASSIS_COMPONENT = 1
    SUBTYPE_INTERFACE_ALIAS = 2
    SUBTYPE_PORT_COMPONENT = 3
    SUBTYPE_MAC_ADDRESS = 4
    SUBTYPE_NETWORK_ADDRESS = 5
    SUBTYPE_INTERFACE_NAME = 6
    SUBTYPE_LOCALLY_ASSIGNED = 7
    fields_desc = [BitEnumField('_type', 1, 7, LLDPDU.TYPES), BitFieldLenField('_length', None, 9, length_of='id', adjust=lambda pkt, x: _ldp_id_adjustlen(pkt, x)), ByteEnumField('subtype', 0, LLDP_CHASSIS_ID_TLV_SUBTYPES), ConditionalField(ByteEnumField('family', 0, LLDPDU.IANA_ADDRESS_FAMILY_NUMBERS), lambda pkt: pkt.subtype == 5), MultipleTypeField([(MACField('id', None), lambda pkt: pkt.subtype == 4), (IPField('id', None), lambda pkt: pkt.subtype == 5 and pkt.family == 1), (IP6Field('id', None), lambda pkt: pkt.subtype == 5 and pkt.family == 2)], StrLenField('id', '', length_from=_ldp_id_lengthfrom))]

    def _check(self):
        if False:
            return 10
        '\n        run layer specific checks\n        '
        if conf.contribs['LLDP'].strict_mode() and (not self.id):
            raise LLDPInvalidLengthField('id must be >= 1 characters long')

class LLDPDUPortID(LLDPDU):
    """
        ieee 802.1ab-2016 - sec. 8.5.3 / p. 26
    """
    LLDP_PORT_ID_TLV_SUBTYPES = {0: 'reserved', 1: 'interface alias', 2: 'port component', 3: 'MAC address', 4: 'network address', 5: 'interface name', 6: 'agent circuit ID', 7: 'locally assigned'}
    SUBTYPE_RESERVED = 0
    SUBTYPE_INTERFACE_ALIAS = 1
    SUBTYPE_PORT_COMPONENT = 2
    SUBTYPE_MAC_ADDRESS = 3
    SUBTYPE_NETWORK_ADDRESS = 4
    SUBTYPE_INTERFACE_NAME = 5
    SUBTYPE_AGENT_CIRCUIT_ID = 6
    SUBTYPE_LOCALLY_ASSIGNED = 7
    fields_desc = [BitEnumField('_type', 2, 7, LLDPDU.TYPES), BitFieldLenField('_length', None, 9, length_of='id', adjust=lambda pkt, x: _ldp_id_adjustlen(pkt, x)), ByteEnumField('subtype', 0, LLDP_PORT_ID_TLV_SUBTYPES), ConditionalField(ByteEnumField('family', 0, LLDPDU.IANA_ADDRESS_FAMILY_NUMBERS), lambda pkt: pkt.subtype == 4), MultipleTypeField([(MACField('id', None), lambda pkt: pkt.subtype == 3), (IPField('id', None), lambda pkt: pkt.subtype == 4 and pkt.family == 1), (IP6Field('id', None), lambda pkt: pkt.subtype == 4 and pkt.family == 2)], StrLenField('id', '', length_from=_ldp_id_lengthfrom))]

    def _check(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        run layer specific checks\n        '
        if conf.contribs['LLDP'].strict_mode() and (not self.id):
            raise LLDPInvalidLengthField('id must be >= 1 characters long')

class LLDPDUTimeToLive(LLDPDU):
    """
        ieee 802.1ab-2016 - sec. 8.5.4 / p. 29
    """
    fields_desc = [BitEnumField('_type', 3, 7, LLDPDU.TYPES), BitField('_length', 2, 9), ShortField('ttl', 20)]

    def _check(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        run layer specific checks\n        '
        if conf.contribs['LLDP'].strict_mode() and self._length != 2:
            raise LLDPInvalidLengthField('length must be 2 - got {}'.format(self._length))

class LLDPDUEndOfLLDPDU(LLDPDU):
    """
        ieee 802.1ab-2016 - sec. 8.5.1 / p. 26
    """
    fields_desc = [BitEnumField('_type', 0, 7, LLDPDU.TYPES), BitField('_length', 0, 9)]

    def extract_padding(self, s):
        if False:
            i = 10
            return i + 15
        return ('', s)

    def _check(self):
        if False:
            print('Hello World!')
        '\n        run layer specific checks\n        '
        if conf.contribs['LLDP'].strict_mode() and self._length != 0:
            raise LLDPInvalidLengthField('length must be 0 - got {}'.format(self._length))

class LLDPDUPortDescription(LLDPDU):
    """
        ieee 802.1ab-2016 - sec. 8.5.5 / p. 29
    """
    fields_desc = [BitEnumField('_type', 4, 7, LLDPDU.TYPES), BitFieldLenField('_length', None, 9, length_of='description'), StrLenField('description', '', length_from=lambda pkt: pkt._length)]

class LLDPDUSystemName(LLDPDU):
    """
        ieee 802.1ab-2016 - sec. 8.5.6 / p. 30
    """
    fields_desc = [BitEnumField('_type', 5, 7, LLDPDU.TYPES), BitFieldLenField('_length', None, 9, length_of='system_name'), StrLenField('system_name', '', length_from=lambda pkt: pkt._length)]

class LLDPDUSystemDescription(LLDPDU):
    """
        ieee 802.1ab-2016 - sec. 8.5.7 / p. 31
    """
    fields_desc = [BitEnumField('_type', 6, 7, LLDPDU.TYPES), BitFieldLenField('_length', None, 9, length_of='description'), StrLenField('description', '', length_from=lambda pkt: pkt._length)]

class LLDPDUSystemCapabilities(LLDPDU):
    """
        ieee 802.1ab-2016 - sec. 8.5.8 / p. 31
    """
    fields_desc = [BitEnumField('_type', 7, 7, LLDPDU.TYPES), BitFieldLenField('_length', 4, 9), BitField('reserved_5_available', 0, 1), BitField('reserved_4_available', 0, 1), BitField('reserved_3_available', 0, 1), BitField('reserved_2_available', 0, 1), BitField('reserved_1_available', 0, 1), BitField('two_port_mac_relay_available', 0, 1), BitField('s_vlan_component_available', 0, 1), BitField('c_vlan_component_available', 0, 1), BitField('station_only_available', 0, 1), BitField('docsis_cable_device_available', 0, 1), BitField('telephone_available', 0, 1), BitField('router_available', 0, 1), BitField('wlan_access_point_available', 0, 1), BitField('mac_bridge_available', 0, 1), BitField('repeater_available', 0, 1), BitField('other_available', 0, 1), BitField('reserved_5_enabled', 0, 1), BitField('reserved_4_enabled', 0, 1), BitField('reserved_3_enabled', 0, 1), BitField('reserved_2_enabled', 0, 1), BitField('reserved_1_enabled', 0, 1), BitField('two_port_mac_relay_enabled', 0, 1), BitField('s_vlan_component_enabled', 0, 1), BitField('c_vlan_component_enabled', 0, 1), BitField('station_only_enabled', 0, 1), BitField('docsis_cable_device_enabled', 0, 1), BitField('telephone_enabled', 0, 1), BitField('router_enabled', 0, 1), BitField('wlan_access_point_enabled', 0, 1), BitField('mac_bridge_enabled', 0, 1), BitField('repeater_enabled', 0, 1), BitField('other_enabled', 0, 1)]

    def _check(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        run layer specific checks\n        '
        if conf.contribs['LLDP'].strict_mode() and self._length != 4:
            raise LLDPInvalidLengthField('length must be 4 - got {}'.format(self._length))

class LLDPDUManagementAddress(LLDPDU):
    """
    ieee 802.1ab-2016 - sec. 8.5.9 / p. 32

    currently only 0x00..0x1e are used by standards, no way to
    use anything > 0xff as management address subtype is only
    one octet wide

    see https://www.iana.org/assignments/address-family-numbers/address-family-numbers.xhtml  # noqa: E501
    """
    SUBTYPE_MANAGEMENT_ADDRESS_OTHER = 0
    SUBTYPE_MANAGEMENT_ADDRESS_IPV4 = 1
    SUBTYPE_MANAGEMENT_ADDRESS_IPV6 = 2
    SUBTYPE_MANAGEMENT_ADDRESS_NSAP = 3
    SUBTYPE_MANAGEMENT_ADDRESS_HDLC = 4
    SUBTYPE_MANAGEMENT_ADDRESS_BBN = 5
    SUBTYPE_MANAGEMENT_ADDRESS_802 = 6
    SUBTYPE_MANAGEMENT_ADDRESS_E_163 = 7
    SUBTYPE_MANAGEMENT_ADDRESS_E_164 = 8
    SUBTYPE_MANAGEMENT_ADDRESS_F_69 = 9
    SUBTYPE_MANAGEMENT_ADDRESS_X_121 = 10
    SUBTYPE_MANAGEMENT_ADDRESS_IPX = 11
    SUBTYPE_MANAGEMENT_ADDRESS_APPLETALK = 12
    SUBTYPE_MANAGEMENT_ADDRESS_DECNET_IV = 13
    SUBTYPE_MANAGEMENT_ADDRESS_BANYAN_VINES = 14
    SUBTYPE_MANAGEMENT_ADDRESS_E_164_WITH_NSAP = 15
    SUBTYPE_MANAGEMENT_ADDRESS_DNS = 16
    SUBTYPE_MANAGEMENT_ADDRESS_DISTINGUISHED_NAME = 17
    SUBTYPE_MANAGEMENT_ADDRESS_AS_NUMBER = 18
    SUBTYPE_MANAGEMENT_ADDRESS_XTP_OVER_IPV4 = 19
    SUBTYPE_MANAGEMENT_ADDRESS_XTP_OVER_IPV6 = 20
    SUBTYPE_MANAGEMENT_ADDRESS_XTP_NATIVE_MODE_XTP = 21
    SUBTYPE_MANAGEMENT_ADDRESS_FIBER_CHANNEL_WORLD_WIDE_PORT_NAME = 22
    SUBTYPE_MANAGEMENT_ADDRESS_FIBER_CHANNEL_WORLD_WIDE_NODE_NAME = 23
    SUBTYPE_MANAGEMENT_ADDRESS_GWID = 24
    SUBTYPE_MANAGEMENT_ADDRESS_AFI_FOR_L2VPN = 25
    SUBTYPE_MANAGEMENT_ADDRESS_MPLS_TP_SECTION_ENDPOINT_ID = 26
    SUBTYPE_MANAGEMENT_ADDRESS_MPLS_TP_LSP_ENDPOINT_ID = 27
    SUBTYPE_MANAGEMENT_ADDRESS_MPLS_TP_PSEUDOWIRE_ENDPOINT_ID = 28
    SUBTYPE_MANAGEMENT_ADDRESS_MT_IP_MULTI_TOPOLOGY_IPV4 = 29
    SUBTYPE_MANAGEMENT_ADDRESS_MT_IP_MULTI_TOPOLOGY_IPV6 = 30
    INTERFACE_NUMBERING_SUBTYPES = {1: 'unknown', 2: 'ifIndex', 3: 'system port number'}
    SUBTYPE_INTERFACE_NUMBER_UNKNOWN = 1
    SUBTYPE_INTERFACE_NUMBER_IF_INDEX = 2
    SUBTYPE_INTERFACE_NUMBER_SYSTEM_PORT_NUMBER = 3
    '\n    Note - calculation of _length field::\n\n        _length = 1@_management_address_string_length +\n                  1@management_address_subtype +\n                  management_address.len +\n                  1@interface_numbering_subtype +\n                  4@interface_number +\n                  1@_oid_string_length +\n                  object_id.len\n    '
    fields_desc = [BitEnumField('_type', 8, 7, LLDPDU.TYPES), BitFieldLenField('_length', None, 9, length_of='management_address', adjust=lambda pkt, x: 8 + len(pkt.management_address) + len(pkt.object_id)), BitFieldLenField('_management_address_string_length', None, 8, length_of='management_address', adjust=lambda pkt, x: len(pkt.management_address) + 1), ByteEnumField('management_address_subtype', 0, LLDPDU.IANA_ADDRESS_FAMILY_NUMBERS), XStrLenField('management_address', '', length_from=lambda pkt: 0 if pkt._management_address_string_length is None else pkt._management_address_string_length - 1), ByteEnumField('interface_numbering_subtype', SUBTYPE_INTERFACE_NUMBER_UNKNOWN, INTERFACE_NUMBERING_SUBTYPES), BitField('interface_number', 0, 32), BitFieldLenField('_oid_string_length', None, 8, length_of='object_id'), XStrLenField('object_id', '', length_from=lambda pkt: pkt._oid_string_length)]

    def _check(self):
        if False:
            return 10
        '\n        run layer specific checks\n        '
        if conf.contribs['LLDP'].strict_mode():
            management_address_len = len(self.management_address)
            if management_address_len == 0 or management_address_len > 31:
                raise LLDPInvalidLengthField('management address must be  1..31 characters long - got string of size {}'.format(management_address_len))

class ThreeBytesEnumField(EnumField, ThreeBytesField):

    def __init__(self, name, default, enum):
        if False:
            return 10
        EnumField.__init__(self, name, default, enum, '!I')

class LLDPDUGenericOrganisationSpecific(LLDPDU):
    ORG_UNIQUE_CODE_PNO = 3791
    ORG_UNIQUE_CODE_IEEE_802_1 = 32962
    ORG_UNIQUE_CODE_IEEE_802_3 = 4623
    ORG_UNIQUE_CODE_TIA_TR_41_MED = 4795
    ORG_UNIQUE_CODE_HYTEC = 3191318
    ORG_UNIQUE_CODES = {ORG_UNIQUE_CODE_PNO: 'PROFIBUS International (PNO)', ORG_UNIQUE_CODE_IEEE_802_1: 'IEEE 802.1', ORG_UNIQUE_CODE_IEEE_802_3: 'IEEE 802.3', ORG_UNIQUE_CODE_TIA_TR_41_MED: 'TIA TR-41 Committee . Media Endpoint Discovery', ORG_UNIQUE_CODE_HYTEC: 'Hytec Geraetebau GmbH'}
    fields_desc = [BitEnumField('_type', 127, 7, LLDPDU.TYPES), BitFieldLenField('_length', None, 9, length_of='data', adjust=lambda pkt, x: len(pkt.data) + 4), ThreeBytesEnumField('org_code', 0, ORG_UNIQUE_CODES), ByteField('subtype', 0), XStrLenField('data', '', length_from=lambda pkt: 0 if pkt._length is None else pkt._length - 4)]
LLDPDU_CLASS_TYPES = {0: LLDPDUEndOfLLDPDU, 1: LLDPDUChassisID, 2: LLDPDUPortID, 3: LLDPDUTimeToLive, 4: LLDPDUPortDescription, 5: LLDPDUSystemName, 6: LLDPDUSystemDescription, 7: LLDPDUSystemCapabilities, 8: LLDPDUManagementAddress, 127: LLDPDUGenericOrganisationSpecific}

class LLDPConfiguration(object):
    """
    basic configuration for LLDP layer
    """

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self._strict_mode = True
        self.strict_mode_enable()

    def strict_mode_enable(self):
        if False:
            i = 10
            return i + 15
        '\n        enable strict mode and dissector debugging\n        '
        self._strict_mode = True

    def strict_mode_disable(self):
        if False:
            while True:
                i = 10
        '\n        disable strict mode and dissector debugging\n        '
        self._strict_mode = False

    def strict_mode(self):
        if False:
            print('Hello World!')
        '\n        get current strict mode state\n        '
        return self._strict_mode
conf.contribs['LLDP'] = LLDPConfiguration()
bind_layers(Ether, LLDPDU, type=LLDP_ETHER_TYPE)
bind_layers(Dot1Q, LLDPDU, type=LLDP_ETHER_TYPE)