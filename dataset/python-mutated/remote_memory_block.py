"""Module containing wrapper around VirtualAllocEx/VirtualFreeEx
Win32 API functions to perform custom marshalling
"""
from __future__ import print_function
import sys
from ctypes import wintypes
from ctypes import c_void_p
from ctypes import pointer
from ctypes import sizeof
from ctypes import byref
from ctypes import c_size_t
from ctypes import WinError
import win32api
from ..windows import win32functions
from ..windows import win32defines
from ..windows import win32structures
from ..actionlogger import ActionLogger

class AccessDenied(RuntimeError):
    """Raised when we cannot allocate memory in the control's process"""
    pass

class RemoteMemoryBlock(object):
    """Class that enables reading and writing memory in a different process"""

    def __init__(self, ctrl, size=4096):
        if False:
            while True:
                i = 10
        'Allocate the memory'
        self.mem_address = 0
        self.size = size
        self.process = 0
        self.handle = ctrl.handle
        if self.handle == 18446744071562067968:
            raise Exception('Incorrect handle: ' + str(self.handle))
        self._as_parameter_ = self.mem_address
        pid = wintypes.DWORD()
        win32functions.GetWindowThreadProcessId(self.handle, byref(pid))
        process_id = pid.value
        if not process_id:
            raise AccessDenied(str(WinError()) + ' Cannot get process ID from handle.')
        self.process = win32functions.OpenProcess(win32defines.PROCESS_VM_OPERATION | win32defines.PROCESS_VM_READ | win32defines.PROCESS_VM_WRITE, 0, process_id)
        if not self.process:
            raise AccessDenied(str(WinError()) + 'process: %d', process_id)
        self.mem_address = win32functions.VirtualAllocEx(c_void_p(self.process), c_void_p(0), win32structures.ULONG_PTR(self.size + 4), win32defines.MEM_RESERVE | win32defines.MEM_COMMIT, win32defines.PAGE_READWRITE)
        if hasattr(self.mem_address, 'value'):
            self.mem_address = self.mem_address.value
        if self.mem_address == 0:
            raise WinError()
        if hex(self.mem_address) == '0xffffffff80000000' or hex(self.mem_address).upper() == '0xFFFFFFFF00000000':
            raise Exception('Incorrect allocation: ' + hex(self.mem_address))
        self._as_parameter_ = self.mem_address
        signature = wintypes.LONG(1717986918)
        ret = win32functions.WriteProcessMemory(c_void_p(self.process), c_void_p(self.mem_address + self.size), pointer(signature), win32structures.ULONG_PTR(4), win32structures.ULONG_PTR(0))
        if ret == 0:
            ActionLogger().log('================== Error: Failed to write guard signature: address = ' + hex(self.mem_address) + ', size = ' + str(self.size))
            last_error = win32api.GetLastError()
            ActionLogger().log('LastError = ' + str(last_error) + ': ' + win32api.FormatMessage(last_error).rstrip())

    def _CloseHandle(self):
        if False:
            while True:
                i = 10
        'Close the handle to the process.'
        ret = win32functions.CloseHandle(self.process)
        if ret == 0:
            ActionLogger().log('Warning: cannot close process handle!')

    def CleanUp(self):
        if False:
            return 10
        'Free Memory and the process handle'
        if self.process != 0 and self.mem_address != 0:
            self.CheckGuardSignature()
            ret = win32functions.VirtualFreeEx(c_void_p(self.process), c_void_p(self.mem_address), win32structures.ULONG_PTR(0), wintypes.DWORD(win32defines.MEM_RELEASE))
            if ret == 0:
                print('Error: CleanUp: VirtualFreeEx() returned zero for address ', hex(self.mem_address))
                last_error = win32api.GetLastError()
                print('LastError = ', last_error, ': ', win32api.FormatMessage(last_error).rstrip())
                sys.stdout.flush()
                self._CloseHandle()
                raise WinError()
            self.mem_address = 0
            self._CloseHandle()
        else:
            pass

    def __del__(self):
        if False:
            print('Hello World!')
        'Ensure that the memory is Freed'
        self.CleanUp()

    def Address(self):
        if False:
            for i in range(10):
                print('nop')
        'Return the address of the memory block'
        return self.mem_address

    def Write(self, data, address=None, size=None):
        if False:
            for i in range(10):
                print('nop')
        'Write data into the memory block'
        if not address:
            address = self.mem_address
        if hasattr(address, 'value'):
            address = address.value
        if size:
            nSize = win32structures.ULONG_PTR(size)
        else:
            nSize = win32structures.ULONG_PTR(sizeof(data))
        if self.size < nSize.value:
            raise Exception(('Write: RemoteMemoryBlock is too small ({0} bytes),' + ' {1} is required.').format(self.size, nSize.value))
        if hex(address).lower().startswith('0xffffff'):
            raise Exception('Write: RemoteMemoryBlock has incorrect address = ' + hex(address))
        ret = win32functions.WriteProcessMemory(c_void_p(self.process), c_void_p(address), pointer(data), nSize, win32structures.ULONG_PTR(0))
        if ret == 0:
            ActionLogger().log('Error: Write failed: address = ' + str(address))
            last_error = win32api.GetLastError()
            ActionLogger().log('Error: LastError = ' + str(last_error) + ': ' + win32api.FormatMessage(last_error).rstrip())
            raise WinError()
        self.CheckGuardSignature()

    def Read(self, data, address=None, size=None):
        if False:
            while True:
                i = 10
        'Read data from the memory block'
        if not address:
            address = self.mem_address
        if hasattr(address, 'value'):
            address = address.value
        if size:
            nSize = win32structures.ULONG_PTR(size)
        else:
            nSize = win32structures.ULONG_PTR(sizeof(data))
        if self.size < nSize.value:
            raise Exception(('Read: RemoteMemoryBlock is too small ({0} bytes),' + ' {1} is required.').format(self.size, nSize.value))
        if hex(address).lower().startswith('0xffffff'):
            raise Exception('Read: RemoteMemoryBlock has incorrect address =' + hex(address))
        lpNumberOfBytesRead = c_size_t(0)
        ret = win32functions.ReadProcessMemory(c_void_p(self.process), c_void_p(address), byref(data), nSize, byref(lpNumberOfBytesRead))
        if ret == 0:
            ret = win32functions.ReadProcessMemory(c_void_p(self.process), c_void_p(address), byref(data), nSize, byref(lpNumberOfBytesRead))
            if ret == 0:
                last_error = win32api.GetLastError()
                if last_error != win32defines.ERROR_PARTIAL_COPY:
                    ActionLogger().log('Read: WARNING! self.mem_address =' + hex(self.mem_address) + ' data address =' + str(byref(data)))
                    ActionLogger().log('LastError = ' + str(last_error) + ': ' + win32api.FormatMessage(last_error).rstrip())
                else:
                    ActionLogger().log('Error: ERROR_PARTIAL_COPY')
                    ActionLogger().log('\nRead: WARNING! self.mem_address =' + hex(self.mem_address) + ' data address =' + str(byref(data)))
                ActionLogger().log('lpNumberOfBytesRead =' + str(lpNumberOfBytesRead) + ' nSize =' + str(nSize))
                raise WinError()
            else:
                ActionLogger().log('Warning! Read OK: 2nd attempt!')
        self.CheckGuardSignature()
        return data

    def CheckGuardSignature(self):
        if False:
            return 10
        'read guard signature at the end of memory block'
        signature = win32structures.LONG(0)
        lpNumberOfBytesRead = c_size_t(0)
        ret = win32functions.ReadProcessMemory(c_void_p(self.process), c_void_p(self.mem_address + self.size), pointer(signature), win32structures.ULONG_PTR(4), byref(lpNumberOfBytesRead))
        if ret == 0:
            ActionLogger().log('Error: Failed to read guard signature: address = ' + hex(self.mem_address) + ', size = ' + str(self.size) + ', lpNumberOfBytesRead = ' + str(lpNumberOfBytesRead))
            raise WinError()
        elif hex(signature.value) != '0x66666666':
            raise Exception('----------------------------------------   ' + 'Error: read incorrect guard signature = ' + hex(signature.value))