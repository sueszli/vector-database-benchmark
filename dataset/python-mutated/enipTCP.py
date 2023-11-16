"""
EtherNet/IP (Industrial Protocol)

Based on https://github.com/scy-phy/scapy-cip-enip
EtherNet/IP Home: www.odva.org
"""
import struct
from scapy.packet import Packet, bind_layers
from scapy.layers.inet import TCP
from scapy.fields import LEShortField, LEShortEnumField, LEIntEnumField, LEIntField, LELongField, FieldLenField, PacketListField, ByteField, StrLenField, StrFixedLenField, XLEIntField, XLEStrLenField, LEFieldLenField, ShortField, IPField, LongField, XLEShortField
_commandIdList = {1: 'UnknownCommand', 4: 'ListServices', 99: 'ListIdentity', 100: 'ListInterfaces', 101: 'RegisterSession', 102: 'UnregisterSession', 111: 'SendRRData', 112: 'SendUnitData', 114: 'IndicateStatus', 115: 'Cancel'}
_statusList = {0: 'success', 1: 'invalid_cmd', 2: 'no_resources', 3: 'incorrect_data', 100: 'invalid_session', 101: 'invalid_length', 105: 'unsupported_prot_rev'}
_typeIdList = {0: 'Null Address Item', 12: 'CIP Identity', 134: 'CIP Security Information', 135: 'EtherNet/IP Capability', 136: 'EtherNet/IP Usage', 161: 'Connected Address Item', 177: 'Connected Data Item', 178: 'Unconnected Data Item', 256: 'List Services Response', 32768: 'Socket Address Info O->T', 32769: 'Socket Address Info T->O', 32770: 'Sequenced Address Item', 32771: 'Unconnected Message over UDP'}
_deviceTypeList = {0: 'Generic Device (deprecated)', 2: 'AC Drive', 3: 'Motor Overload', 4: 'Limit Switch', 5: 'Inductive Proximity Switch', 6: 'Photoelectric Sensor', 7: 'General Purpose Discrete I/O', 9: 'Resolver', 12: 'Communications Adapter', 14: 'Programmable Logic Controller', 16: 'Position Controller', 19: 'DC Drive', 21: 'Contactor', 22: 'Motor Starter', 23: 'Soft Start', 24: 'Human-Machine Interface', 26: 'Mass Flow Controller', 27: 'Pneumatic Valve', 28: 'Vacuum Pressure Gauge', 29: 'Process Control Value', 30: 'Residual Gas Analyzer', 31: 'DC Power Generator', 32: 'RF Power Generator', 33: 'Turbomolecular Vacuum Pump', 34: 'Encoder', 35: 'Safety Discrete I/O Device', 36: 'Fluid Flow Controller', 37: 'CIP Motion Drive', 38: 'CompoNet Repeater', 39: 'Mass Flow Controller, Enhanced', 40: 'CIP Modbus Device', 41: 'CIP Modbus Translator', 42: 'Safety Analog I/O Device', 43: 'Generic Device (keyable)', 44: 'Managed Ethernet Switch', 45: 'CIP Motion Safety Drive Device', 46: 'Safety Drive Device', 47: 'CIP Motion Encoder', 48: 'CIP Motion Converter', 49: 'CIP Motion I/O', 50: 'ControlNet Physical Layer Component', 51: 'Circuit Breaker', 52: 'HART Device', 53: 'CIP-HART Translator', 200: 'Embedded Component'}
_interfaceList = {0: 'CIP'}

class ItemData(Packet):
    """Common Packet Format"""
    name = 'Item Data'
    fields_desc = [LEShortEnumField('typeId', 0, _typeIdList), LEShortField('length', 0), XLEStrLenField('data', '', length_from=lambda pkt: pkt.length)]

    def extract_padding(self, s):
        if False:
            while True:
                i = 10
        return ('', s)

class ENIPUnknownCommand(Packet):
    """Unknown Command reply"""
    name = 'ENIPUnknownCommand'
    pass

class ENIPListServicesItem(Packet):
    """List Services Item Field"""
    name = 'ENIPListServicesItem'
    fields_desc = [LEShortEnumField('itemTypeCode', 0, _typeIdList), LEFieldLenField('itemLength', 0), LEShortField('protocolVersion', 0), XLEShortField('flag', 0), StrFixedLenField('serviceName', None, 16)]

class ENIPListServices(Packet):
    """List Services Command Field"""
    name = 'ENIPListServices'
    fields_desc = [FieldLenField('itemCount', 0, count_of='items'), PacketListField('items', None, ENIPListServicesItem)]

class ENIPListIdentityItem(Packet):
    """List Identity Item Fields"""
    name = 'ENIPListIdentityReplyItem'
    fields_desc = [LEShortEnumField('itemTypeCode', 0, _typeIdList), LEFieldLenField('itemLength', 0), LEShortField('protocolVersion', 0), ShortField('sinFamily', 0), ShortField('sinPort', 0), IPField('sinAddress', None), LongField('sinZero', 0), LEShortField('vendorId', 0), LEShortEnumField('deviceType', 0, _deviceTypeList), LEShortField('productCode', 0), ByteField('revisionMajor', 0), ByteField('revisionMinor', 0), LEShortField('status', 0), XLEIntField('serialNumber', 0), ByteField('productNameLength', 0), StrLenField('productName', None, length_from=lambda pkt: pkt.productNameLength), ByteField('state', 0)]

class ENIPListIdentity(Packet):
    """List identity request and response"""
    name = 'ENIPListIdentity'
    fields_desc = [FieldLenField('itemCount', 0, count_of='items'), PacketListField('items', None, ENIPListIdentityItem)]

class ENIPListInterfacesItem(Packet):
    """List Interfaces Item Fields"""
    name = 'ENIPListInterfacesItem'
    fields_desc = [LEShortEnumField('itemTypeCode', 0, _typeIdList), FieldLenField('itemLength', 0, length_of='itemData'), StrLenField('itemData', '', length_from=lambda pkt: pkt.itemLength)]

class ENIPListInterfaces(Packet):
    """List Interfaces Command Field"""
    name = 'ENIPListInterfaces'
    fields_desc = [FieldLenField('itemCount', 0, count_of='items'), PacketListField('items', None, ENIPListInterfacesItem)]

class ENIPRegisterSession(Packet):
    """Register Session Command Field"""
    name = 'ENIPRegisterSession'
    fields_desc = [LEShortField('protocolVersion', 1), LEShortField('options', 0)]

class ENIPUnregisterSession(Packet):
    """Unregister Session Command Field"""
    name = 'ENIPUnregisterSession'
    pass

class ENIPSendRRData(Packet):
    """Send RR Data Command Field"""
    name = 'ENIPSendRRData'
    fields_desc = [LEIntEnumField('interface', 0, _interfaceList), LEShortField('timeout', 255), LEFieldLenField('itemCount', 0, count_of='items'), PacketListField('items', None, ItemData)]

class ENIPSendUnitData(Packet):
    """Send Unit Data Command Field"""
    name = 'ENIPSendUnitData'
    fields_desc = [LEIntEnumField('interface', 0, _interfaceList), LEShortField('timeout', 255), LEFieldLenField('itemCount', 0, count_of='items'), PacketListField('items', None, ItemData)]

class ENIPTCP(Packet):
    """Ethernet/IP packet over TCP"""
    name = 'ENIPTCP'
    fields_desc = [LEShortEnumField('commandId', None, _commandIdList), LEShortField('length', 0), XLEIntField('session', 0), LEIntEnumField('status', None, _statusList), LELongField('senderContext', 0), LEIntField('options', 0)]

    def post_build(self, pkt, pay):
        if False:
            i = 10
            return i + 15
        if self.length is None and pay:
            pkt = pkt[:2] + struct.pack('<H', len(pay)) + pkt[4:]
        return pkt + pay
bind_layers(TCP, ENIPTCP, dport=44818)
bind_layers(TCP, ENIPTCP, sport=44818)
bind_layers(ENIPTCP, ENIPUnknownCommand, commandId=1)
bind_layers(ENIPTCP, ENIPListServices, commandId=4)
bind_layers(ENIPTCP, ENIPListIdentity, commandId=99)
bind_layers(ENIPTCP, ENIPListInterfaces, commandId=100)
bind_layers(ENIPTCP, ENIPRegisterSession, commandId=101)
bind_layers(ENIPTCP, ENIPUnregisterSession, commandId=102)
bind_layers(ENIPTCP, ENIPSendRRData, commandId=111)
bind_layers(ENIPTCP, ENIPSendUnitData, commandId=112)