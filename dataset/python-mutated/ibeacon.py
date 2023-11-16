"""
scapy.contrib.ibeacon - Apple iBeacon Bluetooth LE proximity beacons.

Packet format documentation can be found at at:

* https://en.wikipedia.org/wiki/IBeacon#Packet_Structure_Byte_Map (public)
* https://developer.apple.com/ibeacon/ (official, requires license)

"""
from scapy.fields import ByteEnumField, ConditionalField, LenField, PacketListField, ShortField, SignedByteField, UUIDField
from scapy.layers.bluetooth import EIR_Hdr, EIR_Manufacturer_Specific_Data, LowEnergyBeaconHelper
from scapy.packet import bind_layers, Packet
APPLE_MFG = 76

class Apple_BLE_Submessage(Packet, LowEnergyBeaconHelper):
    """
    A basic Apple submessage.
    """
    name = 'Apple BLE submessage'
    fields_desc = [ByteEnumField('subtype', None, {1: 'overflow', 2: 'ibeacon', 5: 'airdrop', 7: 'airpods', 9: 'airplay_sink', 10: 'airplay_src', 12: 'handoff', 16: 'nearby'}), ConditionalField(LenField('len', None, fmt='B'), lambda pkt: pkt.subtype != 1)]

    def extract_padding(self, s):
        if False:
            while True:
                i = 10
        if self.subtype == 1:
            return (s[:16], s[16:])
        return (s[:self.len], s[self.len:])

    def build_frame(self):
        if False:
            for i in range(10):
                print('nop')
        'Wraps this submessage in a Apple_BLE_Frame.'
        return Apple_BLE_Frame(plist=[self])

    def build_eir(self):
        if False:
            for i in range(10):
                print('nop')
        'See Apple_BLE_Frame.build_eir.'
        return self.build_frame().build_eir()

class Apple_BLE_Frame(Packet, LowEnergyBeaconHelper):
    """
    The wrapper for a BLE manufacturer-specific data advertisement from Apple
    devices.

    Each advertisement is composed of one or multiple submessages.

    The length of this field comes from the EIR_Hdr.
    """
    name = 'Apple BLE broadcast frame'
    fields_desc = [PacketListField('plist', None, Apple_BLE_Submessage)]

    def build_eir(self):
        if False:
            while True:
                i = 10
        'Builds a list of EIR messages to wrap this frame.'
        return LowEnergyBeaconHelper.base_eir + [EIR_Hdr() / EIR_Manufacturer_Specific_Data() / self]

class IBeacon_Data(Packet):
    """
    iBeacon broadcast data frame. Composed on top of an Apple_BLE_Submessage.
    """
    name = 'iBeacon data'
    fields_desc = [UUIDField('uuid', None, uuid_fmt=UUIDField.FORMAT_BE), ShortField('major', None), ShortField('minor', None), SignedByteField('tx_power', None)]
bind_layers(EIR_Manufacturer_Specific_Data, Apple_BLE_Frame, company_id=APPLE_MFG)
bind_layers(Apple_BLE_Submessage, IBeacon_Data, subtype=2)