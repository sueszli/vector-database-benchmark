import servicemanager
import win32con
import win32event
import win32gui
import win32gui_struct
import win32service
import win32serviceutil
GUID_DEVINTERFACE_USB_DEVICE = '{A5DCBF10-6530-11D2-901F-00C04FB951ED}'

class EventDemoService(win32serviceutil.ServiceFramework):
    _svc_name_ = 'PyServiceEventDemo'
    _svc_display_name_ = 'Python Service Event Demo'
    _svc_description_ = 'Demonstrates a Python service which takes advantage of the extra notifications'

    def __init__(self, args):
        if False:
            for i in range(10):
                print('nop')
        win32serviceutil.ServiceFramework.__init__(self, args)
        self.hWaitStop = win32event.CreateEvent(None, 0, 0, None)
        filter = win32gui_struct.PackDEV_BROADCAST_DEVICEINTERFACE(GUID_DEVINTERFACE_USB_DEVICE)
        self.hdn = win32gui.RegisterDeviceNotification(self.ssh, filter, win32con.DEVICE_NOTIFY_SERVICE_HANDLE)

    def GetAcceptedControls(self):
        if False:
            for i in range(10):
                print('nop')
        rc = win32serviceutil.ServiceFramework.GetAcceptedControls(self)
        rc |= win32service.SERVICE_ACCEPT_PARAMCHANGE | win32service.SERVICE_ACCEPT_NETBINDCHANGE | win32service.SERVICE_CONTROL_DEVICEEVENT | win32service.SERVICE_ACCEPT_HARDWAREPROFILECHANGE | win32service.SERVICE_ACCEPT_POWEREVENT | win32service.SERVICE_ACCEPT_SESSIONCHANGE
        return rc

    def SvcOtherEx(self, control, event_type, data):
        if False:
            i = 10
            return i + 15
        if control == win32service.SERVICE_CONTROL_DEVICEEVENT:
            info = win32gui_struct.UnpackDEV_BROADCAST(data)
            msg = f'A device event occurred: {event_type:x} - {info}'
        elif control == win32service.SERVICE_CONTROL_HARDWAREPROFILECHANGE:
            msg = f'A hardware profile changed: type={event_type}, data={data}'
        elif control == win32service.SERVICE_CONTROL_POWEREVENT:
            msg = 'A power event: setting %s' % data
        elif control == win32service.SERVICE_CONTROL_SESSIONCHANGE:
            msg = f'Session event: type={event_type}, data={data}'
        else:
            msg = 'Other event: code=%d, type=%s, data=%s' % (control, event_type, data)
        servicemanager.LogMsg(servicemanager.EVENTLOG_INFORMATION_TYPE, 61440, (msg, ''))

    def SvcStop(self):
        if False:
            print('Hello World!')
        self.ReportServiceStatus(win32service.SERVICE_STOP_PENDING)
        win32event.SetEvent(self.hWaitStop)

    def SvcDoRun(self):
        if False:
            for i in range(10):
                print('nop')
        win32event.WaitForSingleObject(self.hWaitStop, win32event.INFINITE)
        servicemanager.LogMsg(servicemanager.EVENTLOG_INFORMATION_TYPE, servicemanager.PYS_SERVICE_STOPPED, (self._svc_name_, ''))
if __name__ == '__main__':
    win32serviceutil.HandleCommandLine(EventDemoService)