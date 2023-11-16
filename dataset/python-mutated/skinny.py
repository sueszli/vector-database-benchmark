"""
Skinny Call Control Protocol (SCCP) extension
"""
import time
import struct
from scapy.packet import Packet, bind_layers
from scapy.fields import FlagsField, IPField, LEIntEnumField, LEIntField, StrFixedLenField
from scapy.layers.inet import TCP
from scapy.volatile import RandShort
from scapy.config import conf
skinny_messages_cls = {0: 'SkinnyMessageKeepAlive', 1: 'SkinnyMessageRegister', 2: 'SkinnyMessageIpPort', 3: 'SkinnyMessageKeypadButton', 4: 'SkinnyMessageEnblocCall', 5: 'SkinnyMessageStimulus', 6: 'SkinnyMessageOffHook', 7: 'SkinnyMessageOnHook', 8: 'SkinnyMessageHookFlash', 9: 'SkinnyMessageForwardStatReq', 10: 'SkinnyMessageSpeedDialStatReq', 11: 'SkinnyMessageLineStatReq', 12: 'SkinnyMessageConfigStatReq', 13: 'SkinnyMessageTimeDateReq', 14: 'SkinnyMessageButtonTemplateReq', 15: 'SkinnyMessageVersionReq', 16: 'SkinnyMessageCapabilitiesRes', 17: 'SkinnyMessageMediaPortList', 18: 'SkinnyMessageServerReq', 32: 'SkinnyMessageAlarm', 33: 'SkinnyMessageMulticastMediaReceptionAck', 34: 'SkinnyMessageOpenReceiveChannelAck', 35: 'SkinnyMessageConnectionStatisticsRes', 36: 'SkinnyMessageOffHookWithCgpn', 37: 'SkinnyMessageSoftKeySetReq', 38: 'SkinnyMessageSoftKeyEvent', 39: 'SkinnyMessageUnregister', 40: 'SkinnyMessageSoftKeyTemplateReq', 41: 'SkinnyMessageRegisterTokenReq', 42: 'SkinnyMessageMediaTransmissionFailure', 43: 'SkinnyMessageHeadsetStatus', 44: 'SkinnyMessageMediaResourceNotification', 45: 'SkinnyMessageRegisterAvailableLines', 46: 'SkinnyMessageDeviceToUserData', 47: 'SkinnyMessageDeviceToUserDataResponse', 48: 'SkinnyMessageUpdateCapabilities', 49: 'SkinnyMessageOpenMultiMediaReceiveChannelAck', 50: 'SkinnyMessageClearConference', 51: 'SkinnyMessageServiceURLStatReq', 52: 'SkinnyMessageFeatureStatReq', 53: 'SkinnyMessageCreateConferenceRes', 54: 'SkinnyMessageDeleteConferenceRes', 55: 'SkinnyMessageModifyConferenceRes', 56: 'SkinnyMessageAddParticipantRes', 57: 'SkinnyMessageAuditConferenceRes', 64: 'SkinnyMessageAuditParticipantRes', 65: 'SkinnyMessageDeviceToUserDataVersion1', 129: 'SkinnyMessageRegisterAck', 130: 'SkinnyMessageStartTone', 131: 'SkinnyMessageStopTone', 133: 'SkinnyMessageSetRinger', 134: 'SkinnyMessageSetLamp', 135: 'SkinnyMessageSetHkFDetect', 136: 'SkinnyMessageSpeakerMode', 137: 'SkinnyMessageSetMicroMode', 138: 'SkinnyMessageStartMediaTransmission', 139: 'SkinnyMessageStopMediaTransmission', 140: 'SkinnyMessageStartMediaReception', 141: 'SkinnyMessageStopMediaReception', 143: 'SkinnyMessageCallInfo', 144: 'SkinnyMessageForwardStat', 145: 'SkinnyMessageSpeedDialStat', 146: 'SkinnyMessageLineStat', 147: 'SkinnyMessageConfigStat', 148: 'SkinnyMessageTimeDate', 149: 'SkinnyMessageStartSessionTransmission', 150: 'SkinnyMessageStopSessionTransmission', 151: 'SkinnyMessageButtonTemplate', 152: 'SkinnyMessageVersion', 153: 'SkinnyMessageDisplayText', 154: 'SkinnyMessageClearDisplay', 155: 'SkinnyMessageCapabilitiesReq', 156: 'SkinnyMessageEnunciatorCommand', 157: 'SkinnyMessageRegisterReject', 158: 'SkinnyMessageServerRes', 159: 'SkinnyMessageReset', 256: 'SkinnyMessageKeepAliveAck', 257: 'SkinnyMessageStartMulticastMediaReception', 258: 'SkinnyMessageStartMulticastMediaTransmission', 259: 'SkinnyMessageStopMulticastMediaReception', 260: 'SkinnyMessageStopMulticastMediaTransmission', 261: 'SkinnyMessageOpenReceiveChannel', 262: 'SkinnyMessageCloseReceiveChannel', 263: 'SkinnyMessageConnectionStatisticsReq', 264: 'SkinnyMessageSoftKeyTemplateRes', 265: 'SkinnyMessageSoftKeySetRes', 272: 'SkinnyMessageStationSelectSoftKeysMessage', 273: 'SkinnyMessageCallState', 274: 'SkinnyMessagePromptStatus', 275: 'SkinnyMessageClearPromptStatus', 276: 'SkinnyMessageDisplayNotify', 277: 'SkinnyMessageClearNotify', 278: 'SkinnyMessageCallPlane', 279: 'SkinnyMessageCallPlane', 280: 'SkinnyMessageUnregisterAck', 281: 'SkinnyMessageBackSpaceReq', 282: 'SkinnyMessageRegisterTokenAck', 283: 'SkinnyMessageRegisterTokenReject', 66: 'SkinnyMessageDeviceToUserDataResponseVersion1', 284: 'SkinnyMessageStartMediaFailureDetection', 285: 'SkinnyMessageDialedNumber', 286: 'SkinnyMessageUserToDeviceData', 287: 'SkinnyMessageFeatureStat', 288: 'SkinnyMessageDisplayPriNotify', 289: 'SkinnyMessageClearPriNotify', 290: 'SkinnyMessageStartAnnouncement', 291: 'SkinnyMessageStopAnnouncement', 292: 'SkinnyMessageAnnouncementFinish', 295: 'SkinnyMessageNotifyDtmfTone', 296: 'SkinnyMessageSendDtmfTone', 297: 'SkinnyMessageSubscribeDtmfPayloadReq', 298: 'SkinnyMessageSubscribeDtmfPayloadRes', 299: 'SkinnyMessageSubscribeDtmfPayloadErr', 300: 'SkinnyMessageUnSubscribeDtmfPayloadReq', 301: 'SkinnyMessageUnSubscribeDtmfPayloadRes', 302: 'SkinnyMessageUnSubscribeDtmfPayloadErr', 303: 'SkinnyMessageServiceURLStat', 304: 'SkinnyMessageCallSelectStat', 305: 'SkinnyMessageOpenMultiMediaChannel', 306: 'SkinnyMessageStartMultiMediaTransmission', 307: 'SkinnyMessageStopMultiMediaTransmission', 308: 'SkinnyMessageMiscellaneousCommand', 309: 'SkinnyMessageFlowControlCommand', 310: 'SkinnyMessageCloseMultiMediaReceiveChannel', 311: 'SkinnyMessageCreateConferenceReq', 312: 'SkinnyMessageDeleteConferenceReq', 313: 'SkinnyMessageModifyConferenceReq', 314: 'SkinnyMessageAddParticipantReq', 315: 'SkinnyMessageDropParticipantReq', 316: 'SkinnyMessageAuditConferenceReq', 317: 'SkinnyMessageAuditParticipantReq', 319: 'SkinnyMessageUserToDeviceDataVersion1'}
skinny_callstates = {1: 'Off Hook', 2: 'On Hook', 3: 'Ring out', 12: 'Proceeding'}
skinny_ring_type = {1: 'Ring off'}
skinny_speaker_modes = {1: 'Speaker on', 2: 'Speaker off'}
skinny_lamp_mode = {1: 'Off (?)', 2: 'On'}
skinny_stimulus = {9: 'Line'}

class SkinnyDateTimeField(StrFixedLenField):

    def __init__(self, name, default):
        if False:
            for i in range(10):
                print('nop')
        StrFixedLenField.__init__(self, name, default, 32)

    def m2i(self, pkt, s):
        if False:
            return 10
        (year, month, dow, day, hour, min, sec, millisecond) = struct.unpack('<8I', s)
        return (year, month, day, hour, min, sec)

    def i2m(self, pkt, val):
        if False:
            return 10
        if isinstance(val, str):
            val = self.h2i(pkt, val)
        tmp_lst = val[:2] + (0,) + val[2:7] + (0,)
        return struct.pack('<8I', *tmp_lst)

    def i2h(self, pkt, x):
        if False:
            while True:
                i = 10
        if isinstance(x, str):
            return x
        else:
            return time.ctime(time.mktime(x + (0, 0, 0)))

    def i2repr(self, pkt, x):
        if False:
            return 10
        return self.i2h(pkt, x)

    def h2i(self, pkt, s):
        if False:
            print('Hello World!')
        t = ()
        if isinstance(s, str):
            t = time.strptime(s)
            t = t[:2] + t[2:-3]
        elif not s:
            (y, m, d, h, min, sec, rest, rest, rest) = time.gmtime(time.time())
            t = (y, m, d, h, min, sec)
        else:
            t = s
        return t

class SkinnyMessageGeneric(Packet):
    name = 'Generic message'

class SkinnyMessageKeepAlive(Packet):
    name = 'keep alive'

class SkinnyMessageKeepAliveAck(Packet):
    name = 'keep alive ack'

class SkinnyMessageOffHook(Packet):
    name = 'Off Hook'
    fields_desc = [LEIntField('unknown1', 0), LEIntField('unknown2', 0)]

class SkinnyMessageOnHook(SkinnyMessageOffHook):
    name = 'On Hook'

class SkinnyMessageCallState(Packet):
    name = 'Skinny Call state message'
    fields_desc = [LEIntEnumField('state', 1, skinny_callstates), LEIntField('instance', 1), LEIntField('callid', 0), LEIntField('unknown1', 4), LEIntField('unknown2', 0), LEIntField('unknown3', 0)]

class SkinnyMessageSoftKeyEvent(Packet):
    name = 'Soft Key Event'
    fields_desc = [LEIntField('key', 0), LEIntField('instance', 1), LEIntField('callid', 0)]

class SkinnyMessageSetRinger(Packet):
    name = 'Ring message'
    fields_desc = [LEIntEnumField('ring', 1, skinny_ring_type), LEIntField('unknown1', 0), LEIntField('unknown2', 0), LEIntField('unknown3', 0)]
_skinny_tones = {33: 'Inside dial tone', 34: 'xxx', 35: 'xxx', 36: 'Alerting tone', 37: 'Reorder Tone'}

class SkinnyMessageStartTone(Packet):
    name = 'Start tone'
    fields_desc = [LEIntEnumField('tone', 33, _skinny_tones), LEIntField('unknown1', 0), LEIntField('instance', 1), LEIntField('callid', 0)]

class SkinnyMessageStopTone(SkinnyMessageGeneric):
    name = 'stop tone'
    fields_desc = [LEIntField('instance', 1), LEIntField('callid', 0)]

class SkinnyMessageSpeakerMode(Packet):
    name = 'Speaker mdoe'
    fields_desc = [LEIntEnumField('ring', 1, skinny_speaker_modes)]

class SkinnyMessageSetLamp(Packet):
    name = 'Lamp message (light of the phone)'
    fields_desc = [LEIntEnumField('stimulus', 5, skinny_stimulus), LEIntField('instance', 1), LEIntEnumField('mode', 2, skinny_lamp_mode)]

class SkinnyMessageStationSelectSoftKeysMessage(Packet):
    name = 'Station Select Soft Keys Message'
    fields_desc = [LEIntField('instance', 1), LEIntField('callid', 0), LEIntField('set', 0), LEIntField('map', 65535)]

class SkinnyMessagePromptStatus(Packet):
    name = 'Prompt status'
    fields_desc = [LEIntField('timeout', 0), StrFixedLenField('text', b'\x00' * 32, 32), LEIntField('instance', 1), LEIntField('callid', 0)]

class SkinnyMessageCallPlane(Packet):
    name = 'Activate/Deactivate Call Plane Message'
    fields_desc = [LEIntField('instance', 1)]

class SkinnyMessageTimeDate(Packet):
    name = 'Setting date and time'
    fields_desc = [SkinnyDateTimeField('settime', None), LEIntField('timestamp', 0)]

class SkinnyMessageClearPromptStatus(Packet):
    name = 'clear prompt status'
    fields_desc = [LEIntField('instance', 1), LEIntField('callid', 0)]

class SkinnyMessageKeypadButton(Packet):
    name = 'keypad button'
    fields_desc = [LEIntField('key', 0), LEIntField('instance', 1), LEIntField('callid', 0)]

class SkinnyMessageDialedNumber(Packet):
    name = 'dialed number'
    fields_desc = [StrFixedLenField('number', '1337', 24), LEIntField('instance', 1), LEIntField('callid', 0)]
_skinny_message_callinfo_restrictions = ['CallerName', 'CallerNumber', 'CalledName', 'CalledNumber', 'OriginalCalledName', 'OriginalCalledNumber', 'LastRedirectName', 'LastRedirectNumber'] + ['Bit%d' % i for i in range(8, 15)]

class SkinnyMessageCallInfo(Packet):
    name = 'call information'
    fields_desc = [StrFixedLenField('callername', 'Jean Valjean', 40), StrFixedLenField('callernum', '1337', 24), StrFixedLenField('calledname', 'Causette', 40), StrFixedLenField('callednum', '1034', 24), LEIntField('lineinstance', 1), LEIntField('callid', 0), StrFixedLenField('originalcalledname', 'Causette', 40), StrFixedLenField('originalcallednum', '1034', 24), StrFixedLenField('lastredirectingname', 'Causette', 40), StrFixedLenField('lastredirectingnum', '1034', 24), LEIntField('originalredirectreason', 0), LEIntField('lastredirectreason', 0), StrFixedLenField('voicemailboxG', b'\x00' * 24, 24), StrFixedLenField('voicemailboxD', b'\x00' * 24, 24), StrFixedLenField('originalvoicemailboxD', b'\x00' * 24, 24), StrFixedLenField('lastvoicemailboxD', b'\x00' * 24, 24), LEIntField('security', 0), FlagsField('restriction', 0, 16, _skinny_message_callinfo_restrictions), LEIntField('unknown', 0)]

class SkinnyRateField(LEIntField):

    def i2repr(self, pkt, x):
        if False:
            while True:
                i = 10
        if x is None:
            x = 0
        return '%d ms/pkt' % x
_skinny_codecs = {0: 'xxx', 1: 'xxx', 2: 'xxx', 3: 'xxx', 4: 'G711 ulaw 64k'}
_skinny_echo = {0: 'echo cancellation off', 1: 'echo cancellation on'}

class SkinnyMessageOpenReceiveChannel(Packet):
    name = 'open receive channel'
    fields_desc = [LEIntField('conference', 0), LEIntField('passthru', 0), SkinnyRateField('rate', 20), LEIntEnumField('codec', 4, _skinny_codecs), LEIntEnumField('echo', 0, _skinny_echo), LEIntField('unknown1', 0), LEIntField('callid', 0)]

    def guess_payload_class(self, p):
        if False:
            i = 10
            return i + 15
        return conf.padding_layer
_skinny_receive_channel_status = {0: 'ok', 1: 'ko'}

class SkinnyMessageOpenReceiveChannelAck(Packet):
    name = 'open receive channel'
    fields_desc = [LEIntEnumField('status', 0, _skinny_receive_channel_status), IPField('remote', '0.0.0.0'), LEIntField('port', RandShort()), LEIntField('passthru', 0), LEIntField('callid', 0)]
_skinny_silence = {0: 'silence suppression off', 1: 'silence suppression on'}

class SkinnyFramePerPacketField(LEIntField):

    def i2repr(self, pkt, x):
        if False:
            print('Hello World!')
        if x is None:
            x = 0
        return '%d frames/pkt' % x

class SkinnyMessageStartMediaTransmission(Packet):
    name = 'start multimedia transmission'
    fields_desc = [LEIntField('conference', 0), LEIntField('passthru', 0), IPField('remote', '0.0.0.0'), LEIntField('port', RandShort()), SkinnyRateField('rate', 20), LEIntEnumField('codec', 4, _skinny_codecs), LEIntField('precedence', 200), LEIntEnumField('silence', 0, _skinny_silence), SkinnyFramePerPacketField('maxframes', 0), LEIntField('unknown1', 0), LEIntField('callid', 0)]

    def guess_payload_class(self, p):
        if False:
            return 10
        return conf.padding_layer

class SkinnyMessageCloseReceiveChannel(Packet):
    name = 'close receive channel'
    fields_desc = [LEIntField('conference', 0), LEIntField('passthru', 0), IPField('remote', '0.0.0.0'), LEIntField('port', RandShort()), SkinnyRateField('rate', 20), LEIntEnumField('codec', 4, _skinny_codecs), LEIntField('precedence', 200), LEIntEnumField('silence', 0, _skinny_silence), LEIntField('callid', 0)]

class SkinnyMessageStopMultiMediaTransmission(Packet):
    name = 'stop multimedia transmission'
    fields_desc = [LEIntField('conference', 0), LEIntField('passthru', 0), LEIntField('callid', 0)]

class Skinny(Packet):
    name = 'Skinny'
    fields_desc = [LEIntField('len', None), LEIntField('res', 0), LEIntEnumField('msg', 0, skinny_messages_cls)]

    def post_build(self, pkt, p):
        if False:
            return 10
        if self.len is None:
            tmp_len = len(p) + len(pkt) - 8
            pkt = struct.pack('@I', tmp_len) + pkt[4:]
        return pkt + p

def get_cls(name, fallback_cls):
    if False:
        return 10
    return globals().get(name, fallback_cls)
for (msgid, strcls) in skinny_messages_cls.items():
    cls = get_cls(strcls, SkinnyMessageGeneric)
    bind_layers(Skinny, cls, {'msg': msgid})
bind_layers(TCP, Skinny, {'dport': 2000})
bind_layers(TCP, Skinny, {'sport': 2000})
if __name__ == '__main__':
    from scapy.main import interact
    interact(mydict=globals(), mybanner='Welcome to Skinny add-on')