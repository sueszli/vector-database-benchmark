from queue import Queue
from threading import Thread, Event
from time import time
from typing import Optional
from zeroconf import Zeroconf, ServiceBrowser, ServiceStateChange, ServiceInfo
from UM.Logger import Logger
from UM.Signal import Signal
from cura.CuraApplication import CuraApplication

class ZeroConfClient:
    """The ZeroConfClient handles all network discovery logic.

    It emits signals when new network services were found or disappeared.
    """
    ZERO_CONF_NAME = u'_ultimaker._tcp.local.'
    addedNetworkCluster = Signal()
    removedNetworkCluster = Signal()

    def __init__(self) -> None:
        if False:
            i = 10
            return i + 15
        self._zero_conf = None
        self._zero_conf_browser = None
        self._service_changed_request_queue = None
        self._service_changed_request_event = None
        self._service_changed_request_thread = None

    def start(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        "The ZeroConf service changed requests are handled in a separate thread so we don't block the UI.\n\n        We can also re-schedule the requests when they fail to get detailed service info.\n        Any new or re-reschedule requests will be appended to the request queue and the thread will process them.\n        "
        self._service_changed_request_queue = Queue()
        self._service_changed_request_event = Event()
        try:
            self._zero_conf = Zeroconf()
        except OSError:
            Logger.logException('e', 'Failed to create zeroconf instance.')
            return
        self._service_changed_request_thread = Thread(target=self._handleOnServiceChangedRequests, daemon=True, name='ZeroConfServiceChangedThread')
        self._service_changed_request_thread.start()
        self._zero_conf_browser = ServiceBrowser(self._zero_conf, self.ZERO_CONF_NAME, [self._queueService])

    def stop(self) -> None:
        if False:
            print('Hello World!')
        if self._zero_conf is not None:
            self._zero_conf.close()
            self._zero_conf = None
        if self._zero_conf_browser is not None:
            self._zero_conf_browser.cancel()
            self._zero_conf_browser = None

    def _queueService(self, zeroconf: Zeroconf, service_type, name: str, state_change: ServiceStateChange) -> None:
        if False:
            return 10
        'Handles a change is discovered network services.'
        item = (zeroconf, service_type, name, state_change)
        if not self._service_changed_request_queue or not self._service_changed_request_event:
            return
        self._service_changed_request_queue.put(item)
        self._service_changed_request_event.set()

    def _handleOnServiceChangedRequests(self) -> None:
        if False:
            i = 10
            return i + 15
        'Callback for when a ZeroConf service has changes.'
        if not self._service_changed_request_queue or not self._service_changed_request_event:
            return
        while True:
            self._service_changed_request_event.wait(timeout=5.0)
            if CuraApplication.getInstance().isShuttingDown():
                return
            self._service_changed_request_event.clear()
            reschedule_requests = []
            while not self._service_changed_request_queue.empty():
                request = self._service_changed_request_queue.get()
                (zeroconf, service_type, name, state_change) = request
                try:
                    result = self._onServiceChanged(zeroconf, service_type, name, state_change)
                    if not result:
                        reschedule_requests.append(request)
                except Exception:
                    Logger.logException('e', 'Failed to get service info for [%s] [%s], the request will be rescheduled', service_type, name)
                    reschedule_requests.append(request)
            if reschedule_requests:
                for request in reschedule_requests:
                    self._service_changed_request_queue.put(request)

    def _onServiceChanged(self, zero_conf: Zeroconf, service_type: str, name: str, state_change: ServiceStateChange) -> bool:
        if False:
            print('Hello World!')
        'Handler for zeroConf detection.\n\n        Return True or False indicating if the process succeeded.\n        Note that this function can take over 3 seconds to complete. Be careful calling it from the main thread.\n        '
        if state_change == ServiceStateChange.Added:
            return self._onServiceAdded(zero_conf, service_type, name)
        elif state_change == ServiceStateChange.Removed:
            return self._onServiceRemoved(name)
        return True

    def _onServiceAdded(self, zero_conf: Zeroconf, service_type: str, name: str) -> bool:
        if False:
            for i in range(10):
                print('nop')
        'Handler for when a ZeroConf service was added.'
        info = ServiceInfo(service_type, name, properties={})
        for record in zero_conf.cache.entries_with_name(name.lower()):
            info.update_record(zero_conf, time(), record)
        for record in zero_conf.cache.entries_with_name(info.server):
            info.update_record(zero_conf, time(), record)
            if hasattr(info, 'addresses') and info.addresses:
                break
        if not hasattr(info, 'addresses') or not info.addresses:
            new_info = zero_conf.get_service_info(service_type, name)
            if new_info is not None:
                info = new_info
        if info and hasattr(info, 'addresses') and info.addresses:
            type_of_device = info.properties.get(b'type', None)
            if type_of_device:
                if type_of_device == b'printer':
                    address = '.'.join(map(str, info.addresses[0]))
                    self.addedNetworkCluster.emit(str(name), address, info.properties)
                else:
                    Logger.log('w', "The type of the found device is '%s', not 'printer'." % type_of_device)
        else:
            Logger.log('w', 'Could not get information about %s' % name)
            return False
        return True

    def _onServiceRemoved(self, name: str) -> bool:
        if False:
            return 10
        'Handler for when a ZeroConf service was removed.'
        Logger.log('d', 'ZeroConf service removed: %s' % name)
        self.removedNetworkCluster.emit(str(name))
        return True