"""
.. centered::
    NOTICE
    This software/technical data was produced for the U.S. Government
    under Prime Contract No. NASA-03001 and JPL Contract No. 1295026
    and is subject to FAR 52.227-14 (6/87) Rights in Data General,
    and Article GP-51, Rights in Data  General, respectively.
    This software is publicly released under MITRE case #12-3054
"""
from scapy.packet import Packet, bind_layers
from scapy.fields import ByteEnumField, ByteField, ConditionalField, StrLenField
from scapy.contrib.sdnv import SDNV2FieldLenField, SDNV2LenField, SDNV2
from scapy.contrib.ltp import LTP, ltp_bind_payload

class BP(Packet):
    name = 'BP'
    fields_desc = [ByteField('version', 6), SDNV2('ProcFlags', 0), SDNV2LenField('BlockLen', None), SDNV2('DSO', 0), SDNV2('DSSO', 0), SDNV2('SSO', 0), SDNV2('SSSO', 0), SDNV2('RTSO', 0), SDNV2('RTSSO', 0), SDNV2('CSO', 0), SDNV2('CSSO', 0), SDNV2('CT', 0), SDNV2('CTSN', 0), SDNV2('LT', 0), SDNV2('DL', 0), ConditionalField(SDNV2('FO', 0), lambda x: x.ProcFlags & 1), ConditionalField(SDNV2('ADUL', 0), lambda x: x.ProcFlags & 1)]

    def mysummary(self):
        if False:
            while True:
                i = 10
        tmp = 'BP(%version%) flags('
        if self.ProcFlags & 1:
            tmp += ' FR'
        if self.ProcFlags & 2:
            tmp += ' AR'
        if self.ProcFlags & 4:
            tmp += ' DF'
        if self.ProcFlags & 8:
            tmp += ' CT'
        if self.ProcFlags & 16:
            tmp += ' S'
        if self.ProcFlags & 32:
            tmp += ' ACKME'
        RAWCOS = self.ProcFlags & 384
        COS = RAWCOS >> 7
        cos_tmp = ''
        if COS == 0:
            cos_tmp += 'B '
        if COS == 1:
            cos_tmp += 'N '
        if COS == 2:
            cos_tmp += 'E '
        if COS & 1040384:
            cos_tmp += 'SRR: ('
        if COS & 8192:
            cos_tmp += 'Rec '
        if COS & 16384:
            cos_tmp += 'CA '
        if COS & 32768:
            cos_tmp += 'FWD '
        if COS & 65536:
            cos_tmp += 'DLV '
        if COS & 131072:
            cos_tmp += 'DEL '
        if COS & 1040384:
            cos_tmp += ') '
        if cos_tmp:
            tmp += ' Pr: ' + cos_tmp
        tmp += ' ) len(%BlockLen%) '
        if self.DL == 0:
            tmp += 'CBHE: d[%DSO%,%DSSO%] s[%SSO%, %SSSO%] r[%RTSO%, %RTSSO%] c[%CSO%, %CSSO%] '
        else:
            tmp += 'dl[%DL%] '
        tmp += 'ct[%CT%] ctsn[%CTSN%] lt[%LT%] '
        if self.ProcFlags & 1:
            tmp += 'fo[%FO%] '
            tmp += 'tl[%ADUL%]'
        return (self.sprintf(tmp), [LTP])

class BPBLOCK(Packet):
    fields_desc = [ByteEnumField('Type', 1, {1: 'Bundle payload block'}), SDNV2('ProcFlags', 0), SDNV2FieldLenField('BlockLen', None, length_of='load'), StrLenField('load', '', length_from=lambda pkt: pkt.BlockLen, max_length=65535)]

    def mysummary(self):
        if False:
            return 10
        return self.sprintf('BPBLOCK(%Type%) Flags: %ProcFlags% Len: %BlockLen%')
ltp_bind_payload(BP, lambda pkt: pkt.DATA_ClientServiceID == 1)
bind_layers(BP, BPBLOCK)
bind_layers(BPBLOCK, BPBLOCK)