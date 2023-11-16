import msvcrt
import sys
import threading
import win32con
from win32event import *
from win32file import *

def FindModem():
    if False:
        while True:
            i = 10
    for i in range(1, 5):
        port = 'COM%d' % (i,)
        try:
            handle = CreateFile(port, win32con.GENERIC_READ | win32con.GENERIC_WRITE, 0, None, win32con.OPEN_EXISTING, win32con.FILE_ATTRIBUTE_NORMAL, None)
            if GetCommModemStatus(handle) != 0:
                return port
        except error:
            pass
    return None

class SerialTTY:

    def __init__(self, port):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(port, int):
            port = 'COM%d' % (port,)
        self.handle = CreateFile(port, win32con.GENERIC_READ | win32con.GENERIC_WRITE, 0, None, win32con.OPEN_EXISTING, win32con.FILE_ATTRIBUTE_NORMAL | win32con.FILE_FLAG_OVERLAPPED, None)
        SetCommMask(self.handle, EV_RXCHAR)
        SetupComm(self.handle, 4096, 4096)
        PurgeComm(self.handle, PURGE_TXABORT | PURGE_RXABORT | PURGE_TXCLEAR | PURGE_RXCLEAR)
        timeouts = (4294967295, 0, 1000, 0, 1000)
        SetCommTimeouts(self.handle, timeouts)
        dcb = GetCommState(self.handle)
        dcb.BaudRate = CBR_115200
        dcb.ByteSize = 8
        dcb.Parity = NOPARITY
        dcb.StopBits = ONESTOPBIT
        SetCommState(self.handle, dcb)
        print(f'Connected to {port} at {dcb.BaudRate} baud')

    def _UserInputReaderThread(self):
        if False:
            while True:
                i = 10
        overlapped = OVERLAPPED()
        overlapped.hEvent = CreateEvent(None, 1, 0, None)
        try:
            while 1:
                ch = msvcrt.getch()
                if ord(ch) == 3:
                    break
                WriteFile(self.handle, ch, overlapped)
                WaitForSingleObject(overlapped.hEvent, INFINITE)
        finally:
            SetEvent(self.eventStop)

    def _ComPortThread(self):
        if False:
            i = 10
            return i + 15
        overlapped = OVERLAPPED()
        overlapped.hEvent = CreateEvent(None, 1, 0, None)
        while 1:
            (rc, mask) = WaitCommEvent(self.handle, overlapped)
            if rc == 0:
                SetEvent(overlapped.hEvent)
            rc = WaitForMultipleObjects([overlapped.hEvent, self.eventStop], 0, INFINITE)
            if rc == WAIT_OBJECT_0:
                (flags, comstat) = ClearCommError(self.handle)
                (rc, data) = ReadFile(self.handle, comstat.cbInQue, overlapped)
                WaitForSingleObject(overlapped.hEvent, INFINITE)
                sys.stdout.write(data)
            else:
                sys.stdout.close()
                break

    def Run(self):
        if False:
            while True:
                i = 10
        self.eventStop = CreateEvent(None, 0, 0, None)
        user_thread = threading.Thread(target=self._UserInputReaderThread)
        user_thread.start()
        com_thread = threading.Thread(target=self._ComPortThread)
        com_thread.start()
        user_thread.join()
        com_thread.join()
if __name__ == '__main__':
    print('Serial port terminal demo - press Ctrl+C to exit')
    if len(sys.argv) <= 1:
        port = FindModem()
        if port is None:
            print('No COM port specified, and no modem could be found')
            print('Please re-run this script with the name of a COM port (eg COM3)')
            sys.exit(1)
    else:
        port = sys.argv[1]
    tty = SerialTTY(port)
    tty.Run()