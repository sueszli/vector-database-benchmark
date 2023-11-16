from scapy.packet import Packet, bind_layers
from scapy.fields import ByteEnumField, ByteField, FieldLenField, MACField, PacketListField, ShortField, StrFixedLenField, XIntField, PacketField
from scapy.contrib.homeplugav import HomePlugAV, QualcommTypeList
HomePlugGPTypes = {24584: 'CM_SET_KEY_REQ', 24585: 'CM_SET_KEY_CNF', 24676: 'CM_SLAC_PARM_REQ', 24677: 'CM_SLAC_PARM_CNF', 24686: 'CM_ATTEN_CHAR_IN', 24682: 'CM_START_ATTEN_CHAR_IND', 24687: 'CM_ATTEN_CHAR_RSP', 24694: 'CM_MNBC_SOUND_IND', 24700: 'CM_SLAC_MATCH_REQ', 24701: 'CM_SLAC_MATCH_CNF', 24710: 'CM_ATTENUATION_CHARACTERISTICS_MME'}
QualcommTypeList.update(HomePlugGPTypes)
HPGP_codes = {0: 'Success'}
KeyType_list = {1: 'NMK (AES-128)'}

class CM_SLAC_PARM_REQ(Packet):
    name = 'CM_SLAC_PARM_REQ'
    fields_desc = [ByteField('ApplicationType', 0), ByteField('SecurityType', 0), StrFixedLenField('RunID', b'\x00' * 8, 8)]

class CM_SLAC_PARM_CNF(Packet):
    name = 'CM_SLAC_PARM_CNF'
    fields_desc = [MACField('MSoundTargetMAC', '00:00:00:00:00:00'), ByteField('NumberMSounds', 0), ByteField('TimeOut', 0), ByteField('ResponseType', 0), MACField('ForwardingSTA', '00:00:00:00:00:00'), ByteField('ApplicationType', 0), ByteField('SecurityType', 0), StrFixedLenField('RunID', b'\x00' * 8, 8)]

class HPGP_GROUP(Packet):
    name = 'HPGP_GROUP'
    fields_desc = [ByteField('group', 0)]

    def extract_padding(self, p):
        if False:
            for i in range(10):
                print('nop')
        return ('', p)

class VS_ATTENUATION_CHARACTERISTICS_MME(Packet):
    name = 'VS_ATTENUATION_CHARACTERISTICS_MME'
    fields_desc = [MACField('EVMACAddress', '00:00:00:00:00:00'), FieldLenField('NumberOfGroups', None, count_of='Groups', fmt='B'), ByteField('NumberOfCarrierPerGroupe', 0), StrFixedLenField('Reserved', b'\x00' * 7, 7), PacketListField('Groups', '', HPGP_GROUP, length_from=lambda pkt: pkt.NumberOfGroups)]

class CM_ATTENUATION_CHARACTERISTICS_MME(Packet):
    name = 'CM_ATTENUATION_CHARACTERISTICS_MME'
    fields_desc = [MACField('EVMACAddress', '00:00:00:00:00:00'), FieldLenField('NumberOfGroups', None, count_of='Groups', fmt='B'), ByteField('NumberOfCarrierPerGroupe', 0), PacketListField('Groups', '', HPGP_GROUP, length_from=lambda pkt: pkt.NumberOfGroups)]

class CM_ATTEN_CHAR_IND(Packet):
    name = 'CM_ATTEN_CHAR_IND'
    fields_desc = [ByteField('ApplicationType', 0), ByteField('SecurityType', 0), MACField('SourceAdress', '00:00:00:00:00:00'), StrFixedLenField('RunID', b'\x00' * 8, 8), StrFixedLenField('SourceID', b'\x00' * 17, 17), StrFixedLenField('ResponseID', b'\x00' * 17, 17), ByteField('NumberOfSounds', 0), FieldLenField('NumberOfGroups', None, count_of='Groups', fmt='B'), PacketListField('Groups', '', HPGP_GROUP, length_from=lambda pkt: pkt.NumberOfGroups)]

class CM_ATTEN_CHAR_RSP(Packet):
    name = 'CM_ATTEN_CHAR_RSP'
    fields_desc = [ByteField('ApplicationType', 0), ByteField('SecurityType', 0), MACField('SourceAdress', '00:00:00:00:00:00'), StrFixedLenField('RunID', b'\x00' * 8, 8), StrFixedLenField('SourceID', b'\x00' * 17, 17), StrFixedLenField('ResponseID', b'\x00' * 17, 17), ByteEnumField('Result', 0, HPGP_codes)]

class SLAC_varfield(Packet):
    name = 'SLAC_varfield'
    fields_desc = [StrFixedLenField('EVID', b'\x00' * 17, 17), MACField('EVMAC', '00:00:00:00:00:00'), StrFixedLenField('EVSEID', b'\x00' * 17, 17), MACField('EVSEMAC', '00:00:00:00:00:00'), StrFixedLenField('RunID', b'\x00' * 8, 8), StrFixedLenField('RSVD', b'\x00' * 8, 8)]

class CM_SLAC_MATCH_REQ(Packet):
    name = 'CM_SLAC_MATCH_REQ'
    fields_desc = [ByteField('ApplicationType', 0), ByteField('SecurityType', 0), FieldLenField('MatchVariableFieldLen', None, length_of='VariableField', fmt='<H'), PacketField('VariableField', SLAC_varfield(), SLAC_varfield)]

class SLAC_varfield_cnf(Packet):
    name = 'SLAC_varfield'
    fields_desc = [StrFixedLenField('EVID', b'\x00' * 17, 17), MACField('EVMAC', '00:00:00:00:00:00'), StrFixedLenField('EVSEID', b'\x00' * 17, 17), MACField('EVSEMAC', '00:00:00:00:00:00'), StrFixedLenField('RunID', b'\x00' * 8, 8), StrFixedLenField('RSVD', b'\x00' * 8, 8), StrFixedLenField('NetworkID', b'\x00' * 7, 7), ByteField('Reserved', 0), StrFixedLenField('NMK', b'\x00' * 16, 16)]

class CM_SLAC_MATCH_CNF(Packet):
    name = 'CM_SLAC_MATCH_CNF'
    fields_desc = [ByteField('ApplicationType', 0), ByteField('SecurityType', 0), FieldLenField('MatchVariableFieldLen', None, length_of='VariableField', fmt='<H'), PacketField('VariableField', SLAC_varfield_cnf(), SLAC_varfield_cnf)]

class CM_START_ATTEN_CHAR_IND(Packet):
    name = 'CM_START_ATTEN_CHAR_IND'
    fields_desc = [ByteField('ApplicationType', 0), ByteField('SecurityType', 0), ByteField('NumberOfSounds', 0), ByteField('TimeOut', 0), ByteField('ResponseType', 0), MACField('ForwardingSTA', '00:00:00:00:00:00'), StrFixedLenField('RunID', b'\x00' * 8, 8)]

class CM_MNBC_SOUND_IND(Packet):
    name = 'CM_MNBC_SOUND_IND'
    fields_desc = [ByteField('ApplicationType', 0), ByteField('SecurityType', 0), StrFixedLenField('SenderID', b'\x00' * 17, 17), ByteField('Countdown', 0), StrFixedLenField('RunID', b'\x00' * 8, 8), StrFixedLenField('RSVD', b'\x00' * 8, 8), StrFixedLenField('RandomValue', b'\x00' * 16, 16)]

class CM_SET_KEY_REQ(Packet):
    name = 'CM_SET_KEY_REQ'
    fields_desc = [ByteEnumField('KeyType', 0, KeyType_list), XIntField('MyNonce', 0), XIntField('YourNonce', 0), ByteField('PID', 0), ShortField('ProtoRunNumber', 0), ByteField('ProtoMessNumber', 0), ByteField('CCoCapability', 0), StrFixedLenField('NetworkID', b'\x00' * 7, 7), ByteField('NewEncKeySelect', 0), StrFixedLenField('NewKey', b'\x00' * 16, 16)]

class CM_SET_KEY_CNF(Packet):
    name = 'CM_SET_KEY_CNF'
    fields_desc = [ByteEnumField('Result', 0, HPGP_codes), XIntField('MyNonce', 0), XIntField('YourNonce', 0), ByteField('PID', 0), ShortField('ProtoRunNumber', 0), ByteField('ProtoMessNumber', 0), ByteField('CCoCapability', 0)]
bind_layers(HomePlugAV, VS_ATTENUATION_CHARACTERISTICS_MME, HPtype=41294)
bind_layers(HomePlugAV, CM_SLAC_PARM_REQ, HPtype=24676)
bind_layers(HomePlugAV, CM_SLAC_PARM_CNF, HPtype=24677)
bind_layers(HomePlugAV, CM_START_ATTEN_CHAR_IND, HPtype=24682)
bind_layers(HomePlugAV, CM_ATTEN_CHAR_IND, HPtype=24686)
bind_layers(HomePlugAV, CM_ATTEN_CHAR_RSP, HPtype=24687)
bind_layers(HomePlugAV, CM_MNBC_SOUND_IND, HPtype=24694)
bind_layers(HomePlugAV, CM_SLAC_MATCH_REQ, HPtype=24700)
bind_layers(HomePlugAV, CM_SLAC_MATCH_CNF, HPtype=24701)
bind_layers(HomePlugAV, CM_SET_KEY_REQ, HPtype=24584)
bind_layers(HomePlugAV, CM_SET_KEY_CNF, HPtype=24585)
bind_layers(HomePlugAV, CM_ATTENUATION_CHARACTERISTICS_MME, HPtype=24710)