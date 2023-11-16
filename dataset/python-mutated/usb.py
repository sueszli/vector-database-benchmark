"""
Default USB frames & Basic implementation
"""
from scapy.config import conf
from scapy.compat import chb
from scapy.data import DLT_USBPCAP
from scapy.fields import ByteField, XByteField, ByteEnumField, LEShortField, LEShortEnumField, LEIntField, LEIntEnumField, XLELongField, LenField
from scapy.packet import Packet, bind_top_down
_usbd_status_codes = {0: 'Success', 1073741824: 'Pending', 3221225472: 'Halted', 2147483648: 'Error'}
_transfer_types = {0: 'Isochronous', 1: 'Interrupt', 2: 'Control'}
_urb_functions = {8: 'URB_FUNCTION_CONTROL_TRANSFER', 9: 'URB_FUNCTION_BULK_OR_INTERRUPT_TRANSFER', 10: 'URB_FUNCTION_ISOCH_TRANSFER', 11: 'URB_FUNCTION_GET_DESCRIPTOR_FROM_DEVICE', 12: 'URB_FUNCTION_SET_DESCRIPTOR_TO_DEVICE', 13: 'URB_FUNCTION_SET_FEATURE_TO_DEVICE', 14: 'URB_FUNCTION_SET_FEATURE_TO_INTERFACE', 15: 'URB_FUNCTION_SET_FEATURE_TO_ENDPOINT', 16: 'URB_FUNCTION_CLEAR_FEATURE_TO_DEVICE', 17: 'URB_FUNCTION_CLEAR_FEATURE_TO_INTERFACE', 18: 'URB_FUNCTION_CLEAR_FEATURE_TO_ENDPOINT', 19: 'URB_FUNCTION_GET_STATUS_FROM_DEVICE', 20: 'URB_FUNCTION_GET_STATUS_FROM_INTERFACE', 21: 'URB_FUNCTION_GET_STATUS_FROM_ENDPOINT', 23: 'URB_FUNCTION_VENDOR_DEVICE', 24: 'URB_FUNCTION_VENDOR_INTERFACE', 25: 'URB_FUNCTION_VENDOR_ENDPOINT', 26: 'URB_FUNCTION_CLASS_DEVICE', 27: 'URB_FUNCTION_CLASS_INTERFACE', 28: 'URB_FUNCTION_CLASS_ENDPOINT', 31: 'URB_FUNCTION_CLASS_OTHER', 32: 'URB_FUNCTION_VENDOR_OTHER', 33: 'URB_FUNCTION_GET_STATUS_FROM_OTHER', 34: 'URB_FUNCTION_CLEAR_FEATURE_TO_OTHER', 35: 'URB_FUNCTION_SET_FEATURE_TO_OTHER', 36: 'URB_FUNCTION_GET_DESCRIPTOR_FROM_ENDPOINT', 37: 'URB_FUNCTION_SET_DESCRIPTOR_TO_ENDPOINT', 38: 'URB_FUNCTION_GET_CONFIGURATION', 39: 'URB_FUNCTION_GET_INTERFACE', 40: 'URB_FUNCTION_GET_DESCRIPTOR_FROM_INTERFACE', 41: 'URB_FUNCTION_SET_DESCRIPTOR_TO_INTERFACE', 42: 'URB_FUNCTION_GET_MS_FEATURE_DESCRIPTOR', 50: 'URB_FUNCTION_CONTROL_TRANSFER_EX', 55: 'URB_FUNCTION_BULK_OR_INTERRUPT_TRANSFER_USING_CHAINED_MDL', 2: 'URB_FUNCTION_ABORT_PIPE', 30: 'URB_FUNCTION_SYNC_RESET_PIPE_AND_CLEAR_STALL', 48: 'URB_FUNCTION_SYNC_RESET_PIPE', 49: 'URB_FUNCTION_SYNC_CLEAR_STALL'}

class USBpcap(Packet):
    name = 'USBpcap URB'
    fields_desc = [ByteField('headerLen', None), ByteField('res', 0), XLELongField('irpId', 0), LEIntEnumField('usbd_status', 0, _usbd_status_codes), LEShortEnumField('function', 0, _urb_functions), XByteField('info', 0), LEShortField('bus', 0), LEShortField('device', 0), XByteField('endpoint', 0), ByteEnumField('transfer', 0, _transfer_types), LenField('dataLength', None, fmt='<I')]

    def post_build(self, p, pay):
        if False:
            i = 10
            return i + 15
        if self.headerLen is None:
            headerLen = len(p)
            if isinstance(self.payload, (USBpcapTransferIsochronous, USBpcapTransferInterrupt, USBpcapTransferControl)):
                headerLen += len(self.payload) - len(self.payload.payload)
            p = chb(headerLen) + p[1:]
        return p + pay

    def guess_payload_class(self, payload):
        if False:
            i = 10
            return i + 15
        if self.headerLen == 27:
            return super(USBpcap, self).guess_payload_class(payload)
        if self.transfer == 0:
            return USBpcapTransferIsochronous
        elif self.transfer == 1:
            return USBpcapTransferInterrupt
        elif self.transfer == 2:
            return USBpcapTransferControl
        return super(USBpcap, self).guess_payload_class(payload)

class USBpcapTransferIsochronous(Packet):
    name = 'USBpcap Transfer Isochronous'
    fields_desc = [LEIntField('offset', 0), LEIntField('length', 0), LEIntEnumField('usbd_status', 0, _usbd_status_codes)]

class USBpcapTransferInterrupt(Packet):
    name = 'USBpcap Transfer Interrupt'
    fields_desc = [LEIntField('startFrame', 0), LEIntField('numberOfPackets', 0), LEIntField('errorCount', 0)]

class USBpcapTransferControl(Packet):
    name = 'USBpcap Transfer Control'
    fields_desc = [ByteField('stage', 0)]
bind_top_down(USBpcap, USBpcapTransferIsochronous, transfer=0)
bind_top_down(USBpcap, USBpcapTransferInterrupt, transfer=1)
bind_top_down(USBpcap, USBpcapTransferControl, transfer=2)
conf.l2types.register(DLT_USBPCAP, USBpcap)