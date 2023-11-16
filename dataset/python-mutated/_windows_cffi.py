from __future__ import annotations
import enum
import re
from typing import TYPE_CHECKING, NewType, NoReturn, Protocol, cast
if TYPE_CHECKING:
    from typing_extensions import TypeAlias
import cffi
LIB = "\n// https://msdn.microsoft.com/en-us/library/windows/desktop/aa383751(v=vs.85).aspx\ntypedef int BOOL;\ntypedef unsigned char BYTE;\ntypedef BYTE BOOLEAN;\ntypedef void* PVOID;\ntypedef PVOID HANDLE;\ntypedef unsigned long DWORD;\ntypedef unsigned long ULONG;\ntypedef unsigned int NTSTATUS;\ntypedef unsigned long u_long;\ntypedef ULONG *PULONG;\ntypedef const void *LPCVOID;\ntypedef void *LPVOID;\ntypedef const wchar_t *LPCWSTR;\n\ntypedef uintptr_t ULONG_PTR;\ntypedef uintptr_t UINT_PTR;\n\ntypedef UINT_PTR SOCKET;\n\ntypedef struct _OVERLAPPED {\n    ULONG_PTR Internal;\n    ULONG_PTR InternalHigh;\n    union {\n        struct {\n            DWORD Offset;\n            DWORD OffsetHigh;\n        } DUMMYSTRUCTNAME;\n        PVOID Pointer;\n    } DUMMYUNIONNAME;\n\n    HANDLE  hEvent;\n} OVERLAPPED, *LPOVERLAPPED;\n\ntypedef OVERLAPPED WSAOVERLAPPED;\ntypedef LPOVERLAPPED LPWSAOVERLAPPED;\ntypedef PVOID LPSECURITY_ATTRIBUTES;\ntypedef PVOID LPCSTR;\n\ntypedef struct _OVERLAPPED_ENTRY {\n    ULONG_PTR lpCompletionKey;\n    LPOVERLAPPED lpOverlapped;\n    ULONG_PTR Internal;\n    DWORD dwNumberOfBytesTransferred;\n} OVERLAPPED_ENTRY, *LPOVERLAPPED_ENTRY;\n\n// kernel32.dll\nHANDLE WINAPI CreateIoCompletionPort(\n  _In_     HANDLE    FileHandle,\n  _In_opt_ HANDLE    ExistingCompletionPort,\n  _In_     ULONG_PTR CompletionKey,\n  _In_     DWORD     NumberOfConcurrentThreads\n);\n\nBOOL SetFileCompletionNotificationModes(\n  HANDLE FileHandle,\n  UCHAR  Flags\n);\n\nHANDLE CreateFileW(\n  LPCWSTR               lpFileName,\n  DWORD                 dwDesiredAccess,\n  DWORD                 dwShareMode,\n  LPSECURITY_ATTRIBUTES lpSecurityAttributes,\n  DWORD                 dwCreationDisposition,\n  DWORD                 dwFlagsAndAttributes,\n  HANDLE                hTemplateFile\n);\n\nBOOL WINAPI CloseHandle(\n  _In_ HANDLE hObject\n);\n\nBOOL WINAPI PostQueuedCompletionStatus(\n  _In_     HANDLE       CompletionPort,\n  _In_     DWORD        dwNumberOfBytesTransferred,\n  _In_     ULONG_PTR    dwCompletionKey,\n  _In_opt_ LPOVERLAPPED lpOverlapped\n);\n\nBOOL WINAPI GetQueuedCompletionStatusEx(\n  _In_  HANDLE             CompletionPort,\n  _Out_ LPOVERLAPPED_ENTRY lpCompletionPortEntries,\n  _In_  ULONG              ulCount,\n  _Out_ PULONG             ulNumEntriesRemoved,\n  _In_  DWORD              dwMilliseconds,\n  _In_  BOOL               fAlertable\n);\n\nBOOL WINAPI CancelIoEx(\n  _In_     HANDLE       hFile,\n  _In_opt_ LPOVERLAPPED lpOverlapped\n);\n\nBOOL WriteFile(\n  HANDLE       hFile,\n  LPCVOID      lpBuffer,\n  DWORD        nNumberOfBytesToWrite,\n  LPDWORD      lpNumberOfBytesWritten,\n  LPOVERLAPPED lpOverlapped\n);\n\nBOOL ReadFile(\n  HANDLE       hFile,\n  LPVOID       lpBuffer,\n  DWORD        nNumberOfBytesToRead,\n  LPDWORD      lpNumberOfBytesRead,\n  LPOVERLAPPED lpOverlapped\n);\n\nBOOL WINAPI SetConsoleCtrlHandler(\n  _In_opt_ void*            HandlerRoutine,\n  _In_     BOOL             Add\n);\n\nHANDLE CreateEventA(\n  LPSECURITY_ATTRIBUTES lpEventAttributes,\n  BOOL                  bManualReset,\n  BOOL                  bInitialState,\n  LPCSTR                lpName\n);\n\nBOOL SetEvent(\n  HANDLE hEvent\n);\n\nBOOL ResetEvent(\n  HANDLE hEvent\n);\n\nDWORD WaitForSingleObject(\n  HANDLE hHandle,\n  DWORD  dwMilliseconds\n);\n\nDWORD WaitForMultipleObjects(\n  DWORD        nCount,\n  HANDLE       *lpHandles,\n  BOOL         bWaitAll,\n  DWORD        dwMilliseconds\n);\n\nULONG RtlNtStatusToDosError(\n  NTSTATUS Status\n);\n\nint WSAIoctl(\n  SOCKET                             s,\n  DWORD                              dwIoControlCode,\n  LPVOID                             lpvInBuffer,\n  DWORD                              cbInBuffer,\n  LPVOID                             lpvOutBuffer,\n  DWORD                              cbOutBuffer,\n  LPDWORD                            lpcbBytesReturned,\n  LPWSAOVERLAPPED                    lpOverlapped,\n  // actually LPWSAOVERLAPPED_COMPLETION_ROUTINE\n  void* lpCompletionRoutine\n);\n\nint WSAGetLastError();\n\nBOOL DeviceIoControl(\n  HANDLE       hDevice,\n  DWORD        dwIoControlCode,\n  LPVOID       lpInBuffer,\n  DWORD        nInBufferSize,\n  LPVOID       lpOutBuffer,\n  DWORD        nOutBufferSize,\n  LPDWORD      lpBytesReturned,\n  LPOVERLAPPED lpOverlapped\n);\n\n// From https://github.com/piscisaureus/wepoll/blob/master/src/afd.h\ntypedef struct _AFD_POLL_HANDLE_INFO {\n  HANDLE Handle;\n  ULONG Events;\n  NTSTATUS Status;\n} AFD_POLL_HANDLE_INFO, *PAFD_POLL_HANDLE_INFO;\n\n// This is really defined as a messy union to allow stuff like\n// i.DUMMYSTRUCTNAME.LowPart, but we don't need those complications.\n// Under all that it's just an int64.\ntypedef int64_t LARGE_INTEGER;\n\ntypedef struct _AFD_POLL_INFO {\n  LARGE_INTEGER Timeout;\n  ULONG NumberOfHandles;\n  ULONG Exclusive;\n  AFD_POLL_HANDLE_INFO Handles[1];\n} AFD_POLL_INFO, *PAFD_POLL_INFO;\n\n"
REGEX_SAL_ANNOTATION = re.compile('\\b(_In_|_Inout_|_Out_|_Outptr_|_Reserved_)(opt_)?\\b')
LIB = REGEX_SAL_ANNOTATION.sub(' ', LIB)
LIB = re.sub('\\bFAR\\b', ' ', LIB)
LIB = re.sub('\\bPASCAL\\b', '__stdcall', LIB)
ffi = cffi.api.FFI()
ffi.cdef(LIB)
CData: TypeAlias = cffi.api.FFI.CData
CType: TypeAlias = cffi.api.FFI.CType
AlwaysNull: TypeAlias = CType
Handle = NewType('Handle', CData)
HandleArray = NewType('HandleArray', CData)

class _Kernel32(Protocol):
    """Statically typed version of the kernel32.dll functions we use."""

    def CreateIoCompletionPort(self, FileHandle: Handle, ExistingCompletionPort: CData | AlwaysNull, CompletionKey: int, NumberOfConcurrentThreads: int, /) -> Handle:
        if False:
            print('Hello World!')
        ...

    def CreateEventA(self, lpEventAttributes: AlwaysNull, bManualReset: bool, bInitialState: bool, lpName: AlwaysNull, /) -> Handle:
        if False:
            print('Hello World!')
        ...

    def SetFileCompletionNotificationModes(self, handle: Handle, flags: CompletionModes, /) -> int:
        if False:
            print('Hello World!')
        ...

    def PostQueuedCompletionStatus(self, CompletionPort: Handle, dwNumberOfBytesTransferred: int, dwCompletionKey: int, lpOverlapped: CData | AlwaysNull, /) -> bool:
        if False:
            for i in range(10):
                print('nop')
        ...

    def CancelIoEx(self, hFile: Handle, lpOverlapped: CData | AlwaysNull, /) -> bool:
        if False:
            print('Hello World!')
        ...

    def WriteFile(self, hFile: Handle, lpBuffer: CData, nNumberOfBytesToWrite: int, lpNumberOfBytesWritten: AlwaysNull, lpOverlapped: _Overlapped, /) -> bool:
        if False:
            i = 10
            return i + 15
        ...

    def ReadFile(self, hFile: Handle, lpBuffer: CData, nNumberOfBytesToRead: int, lpNumberOfBytesRead: AlwaysNull, lpOverlapped: _Overlapped, /) -> bool:
        if False:
            i = 10
            return i + 15
        ...

    def GetQueuedCompletionStatusEx(self, CompletionPort: Handle, lpCompletionPortEntries: CData, ulCount: int, ulNumEntriesRemoved: CData, dwMilliseconds: int, fAlertable: bool | int, /) -> CData:
        if False:
            print('Hello World!')
        ...

    def CreateFileW(self, lpFileName: CData, dwDesiredAccess: FileFlags, dwShareMode: FileFlags, lpSecurityAttributes: AlwaysNull, dwCreationDisposition: FileFlags, dwFlagsAndAttributes: FileFlags, hTemplateFile: AlwaysNull, /) -> Handle:
        if False:
            while True:
                i = 10
        ...

    def WaitForSingleObject(self, hHandle: Handle, dwMilliseconds: int, /) -> CData:
        if False:
            for i in range(10):
                print('nop')
        ...

    def WaitForMultipleObjects(self, nCount: int, lpHandles: HandleArray, bWaitAll: bool, dwMilliseconds: int, /) -> ErrorCodes:
        if False:
            print('Hello World!')
        ...

    def SetEvent(self, handle: Handle, /) -> None:
        if False:
            for i in range(10):
                print('nop')
        ...

    def CloseHandle(self, handle: Handle, /) -> bool:
        if False:
            for i in range(10):
                print('nop')
        ...

    def DeviceIoControl(self, hDevice: Handle, dwIoControlCode: int, lpInBuffer: AlwaysNull, nInBufferSize: int, lpOutBuffer: AlwaysNull, nOutBufferSize: int, lpBytesReturned: AlwaysNull, lpOverlapped: CData, /) -> bool:
        if False:
            return 10
        ...

class _Nt(Protocol):
    """Statically typed version of the dtdll.dll functions we use."""

    def RtlNtStatusToDosError(self, status: int, /) -> ErrorCodes:
        if False:
            print('Hello World!')
        ...

class _Ws2(Protocol):
    """Statically typed version of the ws2_32.dll functions we use."""

    def WSAGetLastError(self) -> int:
        if False:
            return 10
        ...

    def WSAIoctl(self, socket: CData, dwIoControlCode: WSAIoctls, lpvInBuffer: AlwaysNull, cbInBuffer: int, lpvOutBuffer: CData, cbOutBuffer: int, lpcbBytesReturned: CData, lpOverlapped: AlwaysNull, lpCompletionRoutine: AlwaysNull, /) -> int:
        if False:
            for i in range(10):
                print('nop')
        ...

class _DummyStruct(Protocol):
    Offset: int
    OffsetHigh: int

class _DummyUnion(Protocol):
    DUMMYSTRUCTNAME: _DummyStruct
    Pointer: object

class _Overlapped(Protocol):
    Internal: int
    InternalHigh: int
    DUMMYUNIONNAME: _DummyUnion
    hEvent: Handle
kernel32 = cast(_Kernel32, ffi.dlopen('kernel32.dll'))
ntdll = cast(_Nt, ffi.dlopen('ntdll.dll'))
ws2_32 = cast(_Ws2, ffi.dlopen('ws2_32.dll'))
INVALID_HANDLE_VALUE = Handle(ffi.cast('HANDLE', -1))

class ErrorCodes(enum.IntEnum):
    STATUS_TIMEOUT = 258
    WAIT_TIMEOUT = 258
    WAIT_ABANDONED = 128
    WAIT_OBJECT_0 = 0
    WAIT_FAILED = 4294967295
    ERROR_IO_PENDING = 997
    ERROR_OPERATION_ABORTED = 995
    ERROR_ABANDONED_WAIT_0 = 735
    ERROR_INVALID_HANDLE = 6
    ERROR_INVALID_PARMETER = 87
    ERROR_NOT_FOUND = 1168
    ERROR_NOT_SOCKET = 10038

class FileFlags(enum.IntFlag):
    GENERIC_READ = 2147483648
    SYNCHRONIZE = 1048576
    FILE_FLAG_OVERLAPPED = 1073741824
    FILE_SHARE_READ = 1
    FILE_SHARE_WRITE = 2
    FILE_SHARE_DELETE = 4
    CREATE_NEW = 1
    CREATE_ALWAYS = 2
    OPEN_EXISTING = 3
    OPEN_ALWAYS = 4
    TRUNCATE_EXISTING = 5

class AFDPollFlags(enum.IntFlag):
    AFD_POLL_RECEIVE = 1
    AFD_POLL_RECEIVE_EXPEDITED = 2
    AFD_POLL_SEND = 4
    AFD_POLL_DISCONNECT = 8
    AFD_POLL_ABORT = 16
    AFD_POLL_LOCAL_CLOSE = 32
    AFD_POLL_CONNECT = 64
    AFD_POLL_ACCEPT = 128
    AFD_POLL_CONNECT_FAIL = 256
    AFD_POLL_QOS = 512
    AFD_POLL_GROUP_QOS = 1024
    AFD_POLL_ROUTING_INTERFACE_CHANGE = 2048
    AFD_POLL_EVENT_ADDRESS_LIST_CHANGE = 4096

class WSAIoctls(enum.IntEnum):
    SIO_BASE_HANDLE = 1207959586
    SIO_BSP_HANDLE_SELECT = 1207959580
    SIO_BSP_HANDLE_POLL = 1207959581

class CompletionModes(enum.IntFlag):
    FILE_SKIP_COMPLETION_PORT_ON_SUCCESS = 1
    FILE_SKIP_SET_EVENT_ON_HANDLE = 2

class IoControlCodes(enum.IntEnum):
    IOCTL_AFD_POLL = 73764

def _handle(obj: int | CData) -> Handle:
    if False:
        while True:
            i = 10
    if isinstance(obj, int):
        return Handle(ffi.cast('HANDLE', obj))
    return Handle(obj)

def handle_array(count: int) -> HandleArray:
    if False:
        print('Hello World!')
    'Make an array of handles.'
    return HandleArray(ffi.new(f'HANDLE[{count}]'))

def raise_winerror(winerror: int | None=None, *, filename: str | None=None, filename2: str | None=None) -> NoReturn:
    if False:
        i = 10
        return i + 15
    if winerror is None:
        err = ffi.getwinerror()
        if err is None:
            raise RuntimeError('No error set?')
        (winerror, msg) = err
    else:
        err = ffi.getwinerror(winerror)
        if err is None:
            raise RuntimeError('No error set?')
        (_, msg) = err
    raise OSError(0, msg, filename, winerror, filename2)