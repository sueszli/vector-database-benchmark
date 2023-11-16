"""
Real-Time Publish-Subscribe Protocol (RTPS) dissection
"""
import random
import struct
from scapy.fields import IntField, PacketField, PacketListField, ShortField, StrLenField, XIntField, XShortField, XStrFixedLenField
from scapy.packet import Packet
from scapy.contrib.rtps.common_types import STR_MAX_LEN, EField, EPacket, GUIDPacket, LeaseDurationPacket, LocatorPacket, ProductVersionPacket, ProtocolVersionPacket, TransportInfoPacket, VendorIdPacket, FORMAT_LE

class ParameterIdField(XShortField):
    _valid_ids = [2, 4, 5, 6, 7, 11, 12, 13, 14, 15, 17, 21, 22, 26, 27, 29, 30, 31, 33, 35, 37, 39, 41, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 64, 65, 67, 68, 69, 70, 72, 73, 80, 82, 83, 88, 89, 90, 96, 98, 112, 113, 119, 16404, 32768, 32769, 32783, 32784, 32790, 32791]

    def randval(self):
        if False:
            while True:
                i = 10
        return random.choice(self._valid_ids)

class PIDPacketBase(Packet):
    name = 'PID Base Packet'
    fields_desc = [EField(ParameterIdField('parameterId', 0), endianness=FORMAT_LE, endianness_from=None), EField(ShortField('parameterLength', 0), endianness=FORMAT_LE, endianness_from=None), StrLenField('parameterData', '', length_from=lambda x: x.parameterLength, max_length=STR_MAX_LEN)]

    def extract_padding(self, p):
        if False:
            while True:
                i = 10
        return (b'', p)

class PID_PAD(PIDPacketBase):
    name = 'PID_PAD'
    fields_desc = [StrLenField('parameterId', '', length_from=lambda x: 2, max_length=STR_MAX_LEN)]

class PID_SENTINEL(PIDPacketBase):
    name = 'PID_SENTINEL'

class PID_USER_DATA(PIDPacketBase):
    name = 'PID_USER_DATA'

class PID_TOPIC_NAME(PIDPacketBase):
    name = 'PID_TOPIC_NAME'

class PID_TYPE_NAME(PIDPacketBase):
    name = 'PID_TYPE_NAME'

class PID_GROUP_DATA(PIDPacketBase):
    name = 'PID_GROUP_DATA'

class PID_TOPIC_DATA(PIDPacketBase):
    name = 'PID_TOPIC_DATA'

class PID_DURABILITY(PIDPacketBase):
    name = 'PID_DURABILITY'

class PID_DURABILITY_SERVICE(PIDPacketBase):
    name = 'PID_DURABILITY_SERVICE'

class PID_DEADLINE(PIDPacketBase):
    name = 'PID_DEADLINE'

class PID_LATENCY_BUDGET(PIDPacketBase):
    name = 'PID_LATENCY_BUDGET'

class PID_LIVELINESS(PIDPacketBase):
    name = 'PID_LIVELINESS'

class PID_RELIABILITY(PIDPacketBase):
    name = 'PID_RELIABILITY'

class PID_LIFESPAN(PIDPacketBase):
    name = 'PID_LIFESPAN'

class PID_DESTINATION_ORDER(PIDPacketBase):
    name = 'PID_DESTINATION_ORDER'

class PID_HISTORY(PIDPacketBase):
    name = 'PID_HISTORY'

class PID_RESOURCE_LIMITS(PIDPacketBase):
    name = 'PID_RESOURCE_LIMITS'

class PID_OWNERSHIP(PIDPacketBase):
    name = 'PID_OWNERSHIP'

class PID_OWNERSHIP_STRENGTH(PIDPacketBase):
    name = 'PID_OWNERSHIP_STRENGTH'

class PID_PRESENTATION(PIDPacketBase):
    name = 'PID_PRESENTATION'

class PID_PARTITION(PIDPacketBase):
    name = 'PID_PARTITION'

class PID_TIME_BASED_FILTER(PIDPacketBase):
    name = 'PID_TIME_BASED_FILTER'

class PID_TRANSPORT_PRIO(PIDPacketBase):
    name = 'PID_TRANSPORT_PRIO'

class PID_PROTOCOL_VERSION(PIDPacketBase):
    name = 'PID_PROTOCOL_VERSION'
    fields_desc = [EField(ParameterIdField('parameterId', 0), endianness=FORMAT_LE, endianness_from=None), EField(ShortField('parameterLength', 0), endianness=FORMAT_LE, endianness_from=None), PacketField('protocolVersion', '', ProtocolVersionPacket), StrLenField('padding', '', length_from=lambda x: x.parameterLength - 2, max_length=STR_MAX_LEN)]

class PID_VENDOR_ID(PIDPacketBase):
    name = 'PID_VENDOR_ID'
    fields_desc = [EField(ParameterIdField('parameterId', 0), endianness=FORMAT_LE, endianness_from=None), EField(ShortField('parameterLength', 0), endianness=FORMAT_LE, endianness_from=None), PacketField('vendorId', '', VendorIdPacket), StrLenField('padding', '', length_from=lambda x: x.parameterLength - 2, max_length=STR_MAX_LEN)]

class PID_UNICAST_LOCATOR(PIDPacketBase):
    name = 'PID_UNICAST_LOCATOR'
    fields_desc = [EField(ParameterIdField('parameterId', 0), endianness=FORMAT_LE, endianness_from=None), EField(ShortField('parameterLength', 0), endianness=FORMAT_LE, endianness_from=None), PacketField('locator', '', LocatorPacket)]

class PID_MULTICAST_LOCATOR(PIDPacketBase):
    name = 'PID_MULTICAST_LOCATOR'
    fields_desc = [EField(ParameterIdField('parameterId', 0), endianness=FORMAT_LE, endianness_from=None), EField(ShortField('parameterLength', 0), endianness=FORMAT_LE, endianness_from=None), StrLenField('parameterData', '', length_from=lambda x: x.parameterLength, max_length=STR_MAX_LEN)]

class PID_MULTICAST_IPADDRESS(PIDPacketBase):
    name = 'PID_MULTICAST_IPADDRESS'

class PID_DEFAULT_UNICAST_LOCATOR(PIDPacketBase):
    name = 'PID_DEFAULT_UNICAST_LOCATOR'
    fields_desc = [EField(ParameterIdField('parameterId', 0), endianness=FORMAT_LE, endianness_from=None), EField(ShortField('parameterLength', 0), endianness=FORMAT_LE, endianness_from=None), PacketField('locator', '', LocatorPacket)]

class PID_DEFAULT_MULTICAST_LOCATOR(PIDPacketBase):
    name = 'PID_DEFAULT_MULTICAST_LOCATOR'
    fields_desc = [EField(ParameterIdField('parameterId', 0), endianness=FORMAT_LE, endianness_from=None), EField(ShortField('parameterLength', 0), endianness=FORMAT_LE, endianness_from=None), PacketField('locator', '', LocatorPacket)]

class PID_TRANSPORT_PRIORITY(PIDPacketBase):
    name = 'PID_TRANSPORT_PRIORITY'

class PID_METATRAFFIC_UNICAST_LOCATOR(PIDPacketBase):
    name = 'PID_METATRAFFIC_UNICAST_LOCATOR'
    fields_desc = [EField(ParameterIdField('parameterId', 0), endianness=FORMAT_LE, endianness_from=None), EField(ShortField('parameterLength', 0), endianness=FORMAT_LE, endianness_from=None), PacketField('locator', '', LocatorPacket)]

class PID_METATRAFFIC_MULTICAST_LOCATOR(PIDPacketBase):
    name = 'PID_METATRAFFIC_MULTICAST_LOCATOR'
    fields_desc = [EField(ParameterIdField('parameterId', 0), endianness=FORMAT_LE, endianness_from=None), EField(ShortField('parameterLength', 0), endianness=FORMAT_LE, endianness_from=None), PacketField('locator', '', LocatorPacket)]

class PID_DEFAULT_UNICAST_IPADDRESS(PIDPacketBase):
    name = 'PID_DEFAULT_UNICAST_IPADDRESS'

class PID_DEFAULT_UNICAST_PORT(PIDPacketBase):
    name = 'PID_DEFAULT_UNICAST_PORT'

class PID_METATRAFFIC_UNICAST_IPADDRESS(PIDPacketBase):
    name = 'PID_METATRAFFIC_UNICAST_IPADDRESS'

class PID_METATRAFFIC_UNICAST_PORT(PIDPacketBase):
    name = 'PID_METATRAFFIC_UNICAST_PORT'

class PID_METATRAFFIC_MULTICAST_IPADDRESS(PIDPacketBase):
    name = 'PID_METATRAFFIC_MULTICAST_IPADDRESS'

class PID_METATRAFFIC_MULTICAST_PORT(PIDPacketBase):
    name = 'PID_METATRAFFIC_MULTICAST_PORT'

class PID_EXPECTS_INLINE_QOS(PIDPacketBase):
    name = 'PID_EXPECTS_INLINE_QOS'

class PID_PARTICIPANT_MANUAL_LIVELINESS_COUNT(PIDPacketBase):
    name = 'PID_PARTICIPANT_MANUAL_LIVELINESS_COUNT'

class PID_PARTICIPANT_BUILTIN_ENDPOINTS(PIDPacketBase):
    name = 'PID_PARTICIPANT_BUILTIN_ENDPOINTS'

class PID_PARTICIPANT_LEASE_DURATION(PIDPacketBase):
    name = 'PID_PARTICIPANT_LEASE_DURATION'

class PID_CONTENT_FILTER_PROPERTY(PIDPacketBase):
    name = 'PID_CONTENT_FILTER_PROPERTY'

class PID_PARTICIPANT_GUID(PIDPacketBase):
    name = 'PID_PARTICIPANT_GUID'
    fields_desc = [EField(ParameterIdField('parameterId', 0), endianness=FORMAT_LE, endianness_from=None), EField(ShortField('parameterLength', 0), endianness=FORMAT_LE, endianness_from=None), PacketField('guid', '', GUIDPacket)]

class PID_ENDPOINT_GUID(PIDPacketBase):
    name = 'PID_ENDPOINT_GUID'
    fields_desc = [EField(ParameterIdField('parameterId', 0), endianness=FORMAT_LE, endianness_from=None), EField(ShortField('parameterLength', 0), endianness=FORMAT_LE, endianness_from=None), PacketField('guid', '', GUIDPacket)]

class PID_GROUP_GUID(PIDPacketBase):
    name = 'PID_GROUP_GUID'

class PID_GROUP_ENTITYID(PIDPacketBase):
    name = 'PID_GROUP_ENTITYID'

class PID_BUILTIN_ENDPOINT_SET(PIDPacketBase):
    name = 'PID_BUILTIN_ENDPOINT_SET'

class PID_PROPERTY_LIST(PIDPacketBase):
    name = 'PID_PROPERTY_LIST'

class PID_TYPE_MAX_SIZE_SERIALIZED(PIDPacketBase):
    name = 'PID_TYPE_MAX_SIZE_SERIALIZED'

class PID_ENTITY_NAME(PIDPacketBase):
    name = 'PID_ENTITY_NAME'

class PID_KEY_HASH(PIDPacketBase):
    name = 'PID_KEY_HASH'

class PID_STATUS_INFO(PIDPacketBase):
    name = 'PID_STATUS_INFO'

class PID_BUILTIN_ENDPOINT_QOS(PIDPacketBase):
    name = 'PID_BUILTIN_ENDPOINT_QOS'

class PID_DOMAIN_TAG(PIDPacketBase):
    name = 'PID_DOMAIN_TAG'

class PID_DOMAIN_ID(PIDPacketBase):
    name = 'PID_DOMAIN_ID'

class PID_UNKNOWN(PIDPacketBase):
    name = 'PID_UNKNOWN'

class PID_PRODUCT_VERSION(PIDPacketBase):
    name = 'PID_PRODUCT_VERSION'
    fields_desc = [EField(ParameterIdField('parameterId', 0), endianness=FORMAT_LE, endianness_from=None), EField(ShortField('parameterLength', 0), endianness=FORMAT_LE, endianness_from=None), PacketField('productVersion', '', ProductVersionPacket)]

class PID_PLUGIN_PROMISCUITY_KIND(PIDPacketBase):
    name = 'PID_PLUGIN_PROMISCUITY_KIND'
    fields_desc = [EField(ParameterIdField('parameterId', 0), endianness=FORMAT_LE, endianness_from=None), EField(ShortField('parameterLength', 0), endianness=FORMAT_LE, endianness_from=None), EField(XIntField('promiscuityKind', 0), endianness=FORMAT_LE, endianness_from=None)]

class PID_RTI_DOMAIN_ID(PIDPacketBase):
    name = 'PID_RTI_DOMAIN_ID'
    fields_desc = [EField(ParameterIdField('parameterId', 0), endianness=FORMAT_LE, endianness_from=None), EField(ShortField('parameterLength', 0), endianness=FORMAT_LE, endianness_from=None), EField(IntField('domainId', 0), endianness=FORMAT_LE, endianness_from=None)]

class PID_TRANSPORT_INFO_LIST(PIDPacketBase):
    name = 'PID_TRANSPORT_INFO_LIST'
    fields_desc = [EField(ParameterIdField('parameterId', 0), endianness=FORMAT_LE, endianness_from=None), EField(ShortField('parameterLength', 0), endianness=FORMAT_LE, endianness_from=None), XStrFixedLenField('padding', '', 4), EField(PacketListField('transportInfo', [], TransportInfoPacket, length_from=lambda p: p.parameterLength - 4))]

class PID_REACHABILITY_LEASE_DURATION(PIDPacketBase):
    name = 'PID_REACHABILITY_LEASE_DURATION'
    fields_desc = [EField(ParameterIdField('parameterId', 0), endianness=FORMAT_LE, endianness_from=None), EField(ShortField('parameterLength', 0), endianness=FORMAT_LE, endianness_from=None), PacketField('lease_duration', '', LeaseDurationPacket)]

class PID_VENDOR_BUILTIN_ENDPOINT_SET(PIDPacketBase):
    name = 'PID_VENDOR_BUILTIN_ENDPOINT_SET'
    fields_desc = [EField(ParameterIdField('parameterId', 0), endianness=FORMAT_LE, endianness_from=None), EField(ShortField('parameterLength', 0), endianness=FORMAT_LE, endianness_from=None), EField(XIntField('flags', 0), endianness=FORMAT_LE, endianness_from=None)]
_RTPSParameterIdTypes = {0: PID_PAD, 2: PID_PARTICIPANT_LEASE_DURATION, 4: PID_TIME_BASED_FILTER, 5: PID_TOPIC_NAME, 6: PID_OWNERSHIP_STRENGTH, 7: PID_TYPE_NAME, 11: PID_METATRAFFIC_MULTICAST_IPADDRESS, 12: PID_DEFAULT_UNICAST_IPADDRESS, 13: PID_METATRAFFIC_UNICAST_PORT, 14: PID_DEFAULT_UNICAST_PORT, 15: PID_DOMAIN_ID, 17: PID_MULTICAST_IPADDRESS, 21: PID_PROTOCOL_VERSION, 22: PID_VENDOR_ID, 26: PID_RELIABILITY, 27: PID_LIVELINESS, 29: PID_DURABILITY, 30: PID_DURABILITY_SERVICE, 31: PID_OWNERSHIP, 33: PID_PRESENTATION, 35: PID_DEADLINE, 37: PID_DESTINATION_ORDER, 39: PID_LATENCY_BUDGET, 41: PID_PARTITION, 43: PID_LIFESPAN, 44: PID_USER_DATA, 45: PID_GROUP_DATA, 46: PID_TOPIC_DATA, 47: PID_UNICAST_LOCATOR, 48: PID_MULTICAST_LOCATOR, 49: PID_DEFAULT_UNICAST_LOCATOR, 50: PID_METATRAFFIC_UNICAST_LOCATOR, 51: PID_METATRAFFIC_MULTICAST_LOCATOR, 52: PID_PARTICIPANT_MANUAL_LIVELINESS_COUNT, 53: PID_CONTENT_FILTER_PROPERTY, 64: PID_HISTORY, 65: PID_RESOURCE_LIMITS, 67: PID_EXPECTS_INLINE_QOS, 68: PID_PARTICIPANT_BUILTIN_ENDPOINTS, 69: PID_METATRAFFIC_UNICAST_IPADDRESS, 70: PID_METATRAFFIC_MULTICAST_PORT, 72: PID_DEFAULT_MULTICAST_LOCATOR, 73: PID_TRANSPORT_PRIORITY, 80: PID_PARTICIPANT_GUID, 82: PID_GROUP_GUID, 83: PID_GROUP_ENTITYID, 88: PID_BUILTIN_ENDPOINT_SET, 89: PID_PROPERTY_LIST, 90: PID_ENDPOINT_GUID, 96: PID_TYPE_MAX_SIZE_SERIALIZED, 98: PID_ENTITY_NAME, 112: PID_KEY_HASH, 113: PID_STATUS_INFO, 119: PID_BUILTIN_ENDPOINT_QOS, 16404: PID_DOMAIN_TAG, 32768: PID_PRODUCT_VERSION, 32769: PID_PLUGIN_PROMISCUITY_KIND, 32783: PID_RTI_DOMAIN_ID, 32784: PID_TRANSPORT_INFO_LIST, 32790: PID_REACHABILITY_LEASE_DURATION, 32791: PID_VENDOR_BUILTIN_ENDPOINT_SET}

def get_pid_class(pkt, lst, cur, remain):
    if False:
        i = 10
        return i + 15
    endianness = getattr(pkt, 'endianness', None)
    _id = struct.unpack(endianness + 'h', remain[0:2])[0]
    if _id == 1:
        return None
    _id = _id & 65535
    next_cls = _RTPSParameterIdTypes.get(_id, PID_UNKNOWN)
    next_cls.endianness = endianness
    return next_cls

class ParameterListPacket(EPacket):
    name = 'PID list'
    fields_desc = [PacketListField('parameterValues', [], next_cls_cb=get_pid_class), PacketField('sentinel', '', PID_SENTINEL)]