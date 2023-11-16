import os
import sys
import time
import json
import ctypes
import struct

class AEDesc(ctypes.Structure):
    _fields_ = [('descKey', ctypes.c_int), ('descContent', ctypes.c_void_p)]

class EventTypeSpec(ctypes.Structure):
    _fields_ = [('eventClass', ctypes.c_int), ('eventKind', ctypes.c_uint)]

def _ctypes_setup():
    if False:
        return 10
    carbon = ctypes.CDLL('/System/Library/Carbon.framework/Carbon')
    ae_callback = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)
    carbon.AEInstallEventHandler.argtypes = [ctypes.c_int, ctypes.c_int, ae_callback, ctypes.c_void_p, ctypes.c_char]
    carbon.AERemoveEventHandler.argtypes = [ctypes.c_int, ctypes.c_int, ae_callback, ctypes.c_char]
    carbon.AEProcessEvent.restype = ctypes.c_int
    carbon.AEProcessEvent.argtypes = [ctypes.c_void_p]
    carbon.ReceiveNextEvent.restype = ctypes.c_int
    carbon.ReceiveNextEvent.argtypes = [ctypes.c_long, ctypes.POINTER(EventTypeSpec), ctypes.c_double, ctypes.c_char, ctypes.POINTER(ctypes.c_void_p)]
    carbon.AEGetParamDesc.restype = ctypes.c_int
    carbon.AEGetParamDesc.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.POINTER(AEDesc)]
    carbon.AECountItems.restype = ctypes.c_int
    carbon.AECountItems.argtypes = [ctypes.POINTER(AEDesc), ctypes.POINTER(ctypes.c_long)]
    carbon.AEGetNthDesc.restype = ctypes.c_int
    carbon.AEGetNthDesc.argtypes = [ctypes.c_void_p, ctypes.c_long, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]
    carbon.AEGetDescDataSize.restype = ctypes.c_int
    carbon.AEGetDescDataSize.argtypes = [ctypes.POINTER(AEDesc)]
    carbon.AEGetDescData.restype = ctypes.c_int
    carbon.AEGetDescData.argtypes = [ctypes.POINTER(AEDesc), ctypes.c_void_p, ctypes.c_int]
    carbon.FSRefMakePath.restype = ctypes.c_int
    carbon.FSRefMakePath.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint]
    return carbon
carbon = _ctypes_setup()
(kAEInternetSuite,) = struct.unpack('>i', b'GURL')
(kAEGetURL,) = struct.unpack('>i', b'GURL')
(kCoreEventClass,) = struct.unpack('>i', b'aevt')
(kAEOpenApplication,) = struct.unpack('>i', b'oapp')
(kAEReOpenApplication,) = struct.unpack('>i', b'rapp')
(kAEActivate,) = struct.unpack('>i', b'actv')
(kAEOpenDocuments,) = struct.unpack('>i', b'odoc')
(keyDirectObject,) = struct.unpack('>i', b'----')
(typeAEList,) = struct.unpack('>i', b'list')
(typeChar,) = struct.unpack('>i', b'TEXT')
(typeFSRef,) = struct.unpack('>i', b'fsrf')
FALSE = b'\x00'
TRUE = b'\x01'
eventLoopTimedOutErr = -9875
(kEventClassAppleEvent,) = struct.unpack('>i', b'eppc')
kEventAppleEvent = 1
ae_callback = carbon.AEInstallEventHandler.argtypes[2]

class Application:

    def __init__(self):
        if False:
            print('Hello World!')
        self.runtime = 15
        filtered_args = [arg for arg in sys.argv[1:] if not arg.startswith('-psn')]
        if filtered_args:
            try:
                self.runtime = float(filtered_args[0])
            except Exception:
                pass
        self.activation_count = 0
        self.logfile = open(self._get_logfile_path(), 'w')
        self.ae_handlers = {'oapp': self.open_app_handler, 'odoc': self.open_document_handler, 'GURL': self.open_url_handler, 'rapp': self.reopen_app_handler, 'actv': self.activate_app_handler}

    def _get_logfile_path(self):
        if False:
            return 10
        if getattr(sys, 'frozen', False):
            basedir = os.path.dirname(sys.executable)
            if os.path.basename(basedir) == 'MacOS':
                basedir = os.path.abspath(os.path.join(basedir, os.pardir, os.pardir, os.pardir))
        else:
            basedir = os.path.dirname(__file__)
        return os.path.join(basedir, 'events.log')

    def log_error(self, message):
        if False:
            return 10
        self.logfile.write(f'ERROR {message}\n')
        self.logfile.flush()

    def log_event(self, event_id, event_data={}):
        if False:
            for i in range(10):
                print('nop')
        self.logfile.write(f'{event_id} {json.dumps(event_data)}\n')
        self.logfile.flush()

    def main(self):
        if False:
            for i in range(10):
                print('nop')
        self.log_event('started', {'args': sys.argv[1:]})

        @ae_callback
        def _ae_handler(message, reply, refcon):
            if False:
                return 10
            event_id = struct.pack('>i', refcon).decode('utf8')
            print('Event handler called with event ID: %s' % (event_id,))
            try:
                handler = self.ae_handlers.get(event_id, None)
                assert handler, 'No handler available!'
                event_data = handler(message, reply, refcon)
                self.log_event(f'ae {event_id}', event_data)
            except Exception as e:
                print('Failed to handle event %s: %s!' % (event_id, e))
                self.log_error(f"Failed to handle event '{event_id}': {e}")
            return 0
        carbon.AEInstallEventHandler(kCoreEventClass, kAEOpenApplication, _ae_handler, kAEOpenApplication, FALSE)
        carbon.AEInstallEventHandler(kCoreEventClass, kAEOpenDocuments, _ae_handler, kAEOpenDocuments, FALSE)
        carbon.AEInstallEventHandler(kAEInternetSuite, kAEGetURL, _ae_handler, kAEGetURL, FALSE)
        carbon.AEInstallEventHandler(kCoreEventClass, kAEReOpenApplication, _ae_handler, kAEReOpenApplication, FALSE)
        carbon.AEInstallEventHandler(kCoreEventClass, kAEActivate, _ae_handler, kAEActivate, FALSE)
        start = time.time()
        eventType = EventTypeSpec()
        eventType.eventClass = kEventClassAppleEvent
        eventType.eventKind = kEventAppleEvent
        while time.time() < start + self.runtime:
            event = ctypes.c_void_p()
            status = carbon.ReceiveNextEvent(1, ctypes.byref(eventType), max(start + self.runtime - time.time(), 0), TRUE, ctypes.byref(event))
            if status == eventLoopTimedOutErr:
                break
            elif status != 0:
                self.log_error(f'Failed to fetch events: {status}!')
                break
            status = carbon.AEProcessEvent(event)
            if status != 0:
                self.log_error(f'Failed to process event: {status}!')
                break
        carbon.AERemoveEventHandler(kCoreEventClass, kAEOpenApplication, _ae_handler, FALSE)
        carbon.AERemoveEventHandler(kCoreEventClass, kAEOpenDocuments, _ae_handler, FALSE)
        carbon.AERemoveEventHandler(kAEInternetSuite, kAEGetURL, _ae_handler, FALSE)
        carbon.AERemoveEventHandler(kCoreEventClass, kAEReOpenApplication, _ae_handler, FALSE)
        carbon.AERemoveEventHandler(kCoreEventClass, kAEActivate, _ae_handler, FALSE)
        self.log_event('finished', {'activation_count': self.activation_count})
        self.logfile.close()
        self.logfile = None

    def open_app_handler(self, message, reply, refcon):
        if False:
            for i in range(10):
                print('nop')
        self.activation_count += 1
        return {}

    def reopen_app_handler(self, message, reply, refcon):
        if False:
            while True:
                i = 10
        self.activation_count += 1
        return {}

    def activate_app_handler(self, message, reply, refcon):
        if False:
            return 10
        self.activation_count += 1
        return {}

    def open_document_handler(self, message, reply, refcon):
        if False:
            return 10
        listdesc = AEDesc()
        status = carbon.AEGetParamDesc(message, keyDirectObject, typeAEList, ctypes.byref(listdesc))
        assert status == 0, f'Could not retrieve descriptor list: {status}!'
        item_count = ctypes.c_long()
        status = carbon.AECountItems(ctypes.byref(listdesc), ctypes.byref(item_count))
        assert status == 0, f'Could not count number of items in descriptor list: {status}!'
        desc = AEDesc()
        paths = []
        for i in range(item_count.value):
            status = carbon.AEGetNthDesc(ctypes.byref(listdesc), i + 1, typeFSRef, 0, ctypes.byref(desc))
            assert status == 0, f'Could not retrieve descriptor #{i}: {status}!'
            sz = carbon.AEGetDescDataSize(ctypes.byref(desc))
            buf = ctypes.create_string_buffer(sz)
            status = carbon.AEGetDescData(ctypes.byref(desc), buf, sz)
            assert status == 0, f'Could not retrieve data for descriptor #{i}: {status}!'
            fsref = buf
            buf = ctypes.create_string_buffer(4096)
            status = carbon.FSRefMakePath(ctypes.byref(fsref), buf, 4095)
            assert status == 0, f'Could not convert data for descriptor #{i} to path: {status}!'
            paths.append(buf.value.decode('utf-8'))
        return paths

    def open_url_handler(self, message, reply, refcon):
        if False:
            print('Hello World!')
        listdesc = AEDesc()
        status = carbon.AEGetParamDesc(message, keyDirectObject, typeAEList, ctypes.byref(listdesc))
        assert status == 0, f'Could not retrieve descriptor list: {status}!'
        item_count = ctypes.c_long()
        status = carbon.AECountItems(ctypes.byref(listdesc), ctypes.byref(item_count))
        assert status == 0, f'Could not count number of items in descriptor list: {status}!'
        desc = AEDesc()
        urls = []
        for i in range(item_count.value):
            status = carbon.AEGetNthDesc(ctypes.byref(listdesc), i + 1, typeChar, 0, ctypes.byref(desc))
            assert status == 0, f'Could not retrieve descriptor #{i}: {status}!'
            sz = carbon.AEGetDescDataSize(ctypes.byref(desc))
            buf = ctypes.create_string_buffer(sz)
            status = carbon.AEGetDescData(ctypes.byref(desc), buf, sz)
            assert status == 0, f'Could not retrieve data for descriptor #{i}: {status}!'
            urls.append(buf.value.decode('utf-8'))
        return urls
if __name__ == '__main__':
    app = Application()
    app.main()