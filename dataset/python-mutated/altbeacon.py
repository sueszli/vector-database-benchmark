"""
scapy.contrib.altbeacon - AltBeacon Bluetooth LE proximity beacons.

The AltBeacon specification can be found at: https://github.com/AltBeacon/spec
"""
from scapy.fields import ByteField, MayEnd, ShortField, SignedByteField, StrFixedLenField
from scapy.layers.bluetooth import EIR_Hdr, EIR_Manufacturer_Specific_Data, UUIDField, LowEnergyBeaconHelper
from scapy.packet import Packet
RADIUS_NETWORKS_MFG = 280

class AltBeacon(Packet, LowEnergyBeaconHelper):
    """
    AltBeacon broadcast frame type.

    https://github.com/AltBeacon/spec
    """
    name = 'AltBeacon'
    magic = b'\xbe\xac'
    fields_desc = [StrFixedLenField('header', magic, len(magic)), UUIDField('id1', None), ShortField('id2', None), ShortField('id3', None), MayEnd(SignedByteField('tx_power', None)), ByteField('mfg_reserved', None)]

    @classmethod
    def magic_check(cls, payload):
        if False:
            while True:
                i = 10
        '\n        Checks if the given payload is for us (starts with our magic string).\n        '
        return payload.startswith(cls.magic)

    def build_eir(self):
        if False:
            i = 10
            return i + 15
        'Builds a list of EIR messages to wrap this frame.'
        return LowEnergyBeaconHelper.base_eir + [EIR_Hdr() / EIR_Manufacturer_Specific_Data(company_id=RADIUS_NETWORKS_MFG) / self]
EIR_Manufacturer_Specific_Data.register_magic_payload(AltBeacon)